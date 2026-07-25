"""经济:生产(训练完成落地)、建造(到场开工/完工落地)、采集一体循环。

同 tick 竞争的仲裁纪律(调研报告 §1/§5):
- 入驻抢槽/开工抢点:按资源点逐点处理(Nn=6 的 Python 循环,编译期展开),
  点内排序用槽号(同一玩家内部无公平性问题),跨玩家用每 tick 随机先手位防偏置。
- 出矿:无条件弹回入口格,允许暂时叠格(防「排队工人站满入口」死锁,见函数内注释)。
- 同玩家同 tick 多笔建造支出:按点序对账扣费,余额不足的工地顺延(不欠账)。

v1.0 生产假设:每家只有 HQ 一个生产建筑,同玩家同 tick 至多一个训练完成,
落地无冲突(v1.1 加兵营时须把落地改成逐生产者仲裁——见文内 NOTE)。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    BTASK_UPGRADE,
    LINE_INFANTRY,
    LINE_WORKER,
    RES_WATER,
    TYPE_CAMP,
    TYPE_HQ,
    TYPE_INFANTRY,
    TYPE_MINE,
    TYPE_PUMP,
    TYPE_WORKER,
    Config,
)
from .map import MapData
from .state import (
    ORDER_BUILD,
    ORDER_HARVEST,
    ORDER_IDLE,
    PH_MINING,
    PH_TO_HQ,
    PH_TO_NODE,
    WorldState,
    hq_slot,
)

# 落地用的 8 邻。**斜角优先**:正邻 4 格是卸货格(4-邻接判定),新单位落在正邻格
# 会堵死运矿工人的入家路——实测 4 个待命步兵恰好站满 4 个卸货格,把经济锁死。
_SPAWN_DIRS = jnp.asarray(
    [[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 1], [1, 0], [0, -1], [-1, 0]], jnp.int32)


def occupancy_grid(state: WorldState, cfg: Config) -> jax.Array:
    """bool [H,W]:在场实体(alive 且不在矿内)占用图。每 tick 重算,不进 state。"""
    on = state.alive & ~state.inside
    occ = jnp.zeros((cfg.grid_h, cfg.grid_w), bool)
    return occ.at[state.pos[:, 0], state.pos[:, 1]].max(on)


def inside_counts(state: WorldState, cfg: Config) -> jax.Array:
    """int32 [Nn]:各资源点当前驻内工人数。"""
    tn = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    return (jnp.zeros(cfg.n_nodes, jnp.int32)
            .at[tn].add((state.inside).astype(jnp.int32)))


def upgrade_cost_of(state, cfg: Config) -> jax.Array:
    """int32 [N,2]:每个实体「从当前等级升一级」的(ore,water)成本,按类型查表。
    定价唯一定义处(legality 乐观检查与 paid_orders_pass 实扣都用它)。
    非可升级类型返回 0(调用方用类型掩码门控)。"""
    lv = jnp.clip(state.level.astype(jnp.int32), 0, 7)
    is_hq = state.etype == TYPE_HQ
    is_camp = state.etype == TYPE_CAMP
    ore = jnp.where(is_hq, jnp.asarray(cfg.base_up_cost_ore)[lv],
                    jnp.asarray(cfg.node_up_cost_ore)[lv])
    ore = jnp.where(is_camp, jnp.asarray(cfg.camp_up_cost_ore)[lv], ore)
    wat = jnp.where(is_hq, jnp.asarray(cfg.base_up_cost_water)[lv],
                    jnp.asarray(cfg.node_up_cost_water)[lv])
    wat = jnp.where(is_camp, jnp.asarray(cfg.camp_up_cost_water)[lv], wat)
    return jnp.stack([ore, wat], axis=-1)


def upgrade_time_of(state, cfg: Config) -> jax.Array:
    """int32 [N]:每个实体升一级的耗时,按类型查表。"""
    lv = jnp.clip(state.level.astype(jnp.int32), 0, 7)
    is_hq = state.etype == TYPE_HQ
    is_camp = state.etype == TYPE_CAMP
    t = jnp.where(is_hq, jnp.asarray(cfg.base_up_time)[lv],
                  jnp.asarray(cfg.node_up_time)[lv])
    return jnp.where(is_camp, jnp.asarray(cfg.camp_up_time)[lv], t)


def paid_orders_pass(state: WorldState, act: jax.Array, cfg: Config,
                     mapdata: MapData, owner: jax.Array) -> WorldState:
    """同 tick 可能多笔的付费指令(升级/研发/建营)的**顺序对账扣费**
    (critic B-1):同玩家按槽号累计支出,超出库存的笔自动 no-op,库存恒 ≥0。
    act 必须是 apply_orders 返回的合法化动作。
    结算顺序(固定,写死防漂移):①升级+研发按槽号 cumsum 对账 ②建营
    (同玩家同 tick 至多批准一座,用①之后的余额)。"""
    from .actions import a_build_camp, a_research, a_upgrade
    from .config import (
        BTASK_BUILD_CAMP,
        BTASK_RESEARCH_INF,
        BTASK_RESEARCH_WORKER,
        LINE_INFANTRY,
        LINE_WORKER,
        TYPE_CAMP,
    )

    own_i = owner.astype(jnp.int32)
    st = state

    # ---- ① 升级 + 研发:统一 cumsum 对账 ----
    w_up = act == a_upgrade(cfg)
    w_ri = act == a_research(LINE_INFANTRY, cfg)
    w_rw = act == a_research(LINE_WORKER, cfg)

    # 同玩家同 tick 对同一条线的并发研发申请去重,只批槽号最小者
    # (audit v1.1 P0-1:掩码的 busy_same 只见已写入的 btype,看不见同 tick
    # 并发,两营同时下单会双倍扣费只得一级;去重必须在扣费之前)
    slots_arr = jnp.arange(cfg.n_total)
    for p in (0, 1):  # 编译期展开
        for w in ("ri", "rw"):
            cand = (w_ri if w == "ri" else w_rw) & (own_i == p)
            keep = cand & (slots_arr == jnp.argmax(cand))
            if w == "ri":
                w_ri = jnp.where(own_i == p, keep, w_ri)
            else:
                w_rw = jnp.where(own_i == p, keep, w_rw)
    cur_i = st.upgrades[own_i, LINE_INFANTRY].astype(jnp.int32)
    cur_w = st.upgrades[own_i, LINE_WORKER].astype(jnp.int32)
    cost = upgrade_cost_of(st, cfg) * w_up[:, None]
    cost += (jnp.stack([jnp.asarray(cfg.inf_res_cost_ore)[cur_i],
                        jnp.asarray(cfg.inf_res_cost_water)[cur_i]], -1)
             * w_ri[:, None])
    cost += (jnp.stack([jnp.asarray(cfg.worker_res_cost_ore)[cur_w],
                        jnp.asarray(cfg.worker_res_cost_water)[cur_w]], -1)
             * w_rw[:, None])
    want = w_up | w_ri | w_rw

    afford = jnp.zeros(cfg.n_total, bool)
    stock = st.resources
    for p in (0, 1):  # 编译期展开
        cum = jnp.cumsum(cost * (own_i == p)[:, None], axis=0)  # 含本行
        ok = jnp.all(cum <= stock[p][None, :], axis=-1)
        afford = jnp.where(own_i == p, ok, afford)
    do = want & afford

    pay = jnp.zeros_like(stock).at[own_i].add(jnp.where(do[:, None], cost, 0))
    stock = stock - pay
    task = jnp.where(w_ri, BTASK_RESEARCH_INF,
                     jnp.where(w_rw, BTASK_RESEARCH_WORKER, BTASK_UPGRADE))
    t = jnp.where(w_ri, jnp.asarray(cfg.inf_res_time)[cur_i],
                  jnp.where(w_rw, jnp.asarray(cfg.worker_res_time)[cur_w],
                            upgrade_time_of(st, cfg)))
    st = st._replace(
        resources=stock,
        btype=jnp.where(do, task, st.btype).astype(jnp.int8),
        btimer=jnp.where(do, t, st.btimer).astype(jnp.int16),
    )

    # ---- ② 建营:同玩家取槽号最小的申请者,落其相邻第一空闲格 ----
    camp_cost = jnp.asarray([cfg.camp_cost_ore, cfg.camp_cost_water], jnp.int32)
    occ = occupancy_grid(st, cfg)
    passable = jnp.asarray(mapdata.passable)
    h, w = cfg.grid_h, cfg.grid_w
    for p in (0, 1):  # 编译期展开
        cand = (act == a_build_camp(cfg)) & (own_i == p)
        bidx = jnp.argmax(cand)                # 全 False 返 0,has 门控
        has = jnp.any(cand) & jnp.all(st.resources[p] >= camp_cost)

        cells = st.pos[bidx] + _SPAWN_DIRS
        inb = ((cells[:, 0] >= 0) & (cells[:, 0] < h)
               & (cells[:, 1] >= 0) & (cells[:, 1] < w))
        cc = jnp.clip(cells, 0, jnp.asarray([h - 1, w - 1]))
        ok = inb & passable[cc[:, 0], cc[:, 1]] & ~occ[cc[:, 0], cc[:, 1]]
        ci = jnp.argmax(ok)
        base = p * cfg.e_max
        free = jax.lax.dynamic_slice(~st.alive, (base,), (cfg.e_max,))
        slot = base + jnp.argmax(free)
        do_c = has & jnp.any(ok) & jnp.any(free)
        cell = cc[ci]
        start_hp = jnp.asarray(cfg.camp_hp_by_level)[2] // 10

        st = st._replace(
            resources=st.resources.at[p].add(jnp.where(do_c, -camp_cost, 0)),
            alive=st.alive.at[slot].set(jnp.where(do_c, True, st.alive[slot])),
            etype=st.etype.at[slot].set(
                jnp.where(do_c, TYPE_CAMP, st.etype[slot]).astype(jnp.int8)),
            pos=st.pos.at[slot].set(jnp.where(do_c, cell, st.pos[slot])),
            hp=st.hp.at[slot].set(jnp.where(do_c, start_hp, st.hp[slot])),
            level=st.level.at[slot].set(
                jnp.where(do_c, 1, st.level[slot]).astype(jnp.int8)),
            btype=st.btype.at[slot].set(
                jnp.where(do_c, BTASK_BUILD_CAMP, st.btype[slot]).astype(jnp.int8)),
            btimer=st.btimer.at[slot].set(
                jnp.where(do_c, cfg.camp_build_time, st.btimer[slot]).astype(jnp.int16)),
        )
        occ = occ.at[cell[0], cell[1]].max(do_c)  # 防两家同格(远隔,保险)
    return st


def special_tasks_tick(state: WorldState, cfg: Config,
                       owner: jax.Array) -> WorldState:
    """负数 btype 任务的推进与完成结算(解码唯一集中处;完成分支必须 btype←0,
    否则「btype<0 & btimer==0」下一 tick 重复触发,level 每 tick+1 直到溢出)。
    btimer 的递减复用 production_tick(它对所有 btimer>0 无差别倒数)。"""
    from .config import (
        BTASK_BUILD_CAMP,
        BTASK_RESEARCH_INF,
        BTASK_RESEARCH_WORKER,
    )
    st = state
    own_i = owner.astype(jnp.int32)

    # ---- 在建营的 hp 线性成长(增量式,伤害得以保留;完成拍补齐整数余数,
    # 使「未挨打的建成营 = 满血」恒成立)----
    full = int(cfg.camp_hp_by_level[2])
    start = full // 10
    t_build = cfg.camp_build_time
    g = (full - start) // t_build  # 每 tick 增量(可为 0,余数在完成拍补)
    growing = st.alive & (st.btype == BTASK_BUILD_CAMP) & (st.btimer > 0)
    hp = st.hp + jnp.where(growing, g, 0)

    done = st.alive & (st.btype < 0) & (st.btimer == 0)

    # 建营完成:level=2(建成即 2 级,issue v1.1)+ 补 hp 余数。
    # 成长只发生在 btimer>0 的 T-1 个 tick(完成拍 btimer 已归 0),余数按 T-1 算,
    # 保证「未挨打的建成营 = 满血」恒成立。
    camp_done = done & (st.btype == BTASK_BUILD_CAMP)
    remainder = (full - start) - g * (t_build - 1)
    hp = hp + jnp.where(camp_done, remainder, 0)
    level = jnp.where(camp_done, 2, st.level)

    # 自升级完成
    upg = done & (st.btype == BTASK_UPGRADE)
    level = jnp.where(upg, level + 1, level).astype(jnp.int8)

    # 研发完成:该玩家对应线 +1(legality 保证同线同 tick 至多一营在研),
    # 存量单位 hp 补上限差额(plan:不缩放,保伤痕)
    upgrades = st.upgrades
    for line, code, hp_table, ut in (
        (LINE_INFANTRY, BTASK_RESEARCH_INF, cfg.inf_hp_by_level, TYPE_INFANTRY),
        (LINE_WORKER, BTASK_RESEARCH_WORKER, cfg.worker_hp_by_level, TYPE_WORKER),
    ):
        rdone = done & (st.btype == code)
        for p in (0, 1):  # 编译期展开
            hit = jnp.any(rdone & (own_i == p))
            old = upgrades[p, line].astype(jnp.int32)
            new = jnp.minimum(old + 1, 7)
            upgrades = upgrades.at[p, line].set(
                jnp.where(hit, new, old).astype(jnp.int8))
            delta = jnp.asarray(hp_table)[new] - jnp.asarray(hp_table)[old]
            bump = st.alive & (own_i == p) & (st.etype == ut)
            hp = hp + jnp.where(hit & bump, delta, 0)

    btype = jnp.where(done, 0, st.btype).astype(jnp.int8)
    return st._replace(hp=hp.astype(jnp.int32), level=level,
                       upgrades=upgrades, btype=btype)


def _unit_spawn_hp(cfg: Config, ut: jax.Array, upgrades_p: jax.Array) -> jax.Array:
    """新单位出生血量:按其所属玩家的升级线等级查表(v1.1 起,全局线生效)。"""
    whp = jnp.asarray(cfg.worker_hp_by_level)[upgrades_p[LINE_WORKER]]
    ihp = jnp.asarray(cfg.inf_hp_by_level)[upgrades_p[LINE_INFANTRY]]
    return jnp.where(ut == TYPE_WORKER, whp, ihp)


def production_tick(state: WorldState, cfg: Config, mapdata: MapData) -> WorldState:
    """训练倒计时与落地。落不下(周围无空格/半区无空槽)则停在 1,下 tick 重试。

    NOTE(v1.1):此实现假设同玩家同 tick 至多一个生产建筑完成(v1.0 仅 HQ)。
    引入兵营后须改为对完成者逐个仲裁落地格与槽位。
    """
    btimer = jnp.where(state.btimer > 0, state.btimer - 1, state.btimer)
    st = state._replace(btimer=btimer.astype(jnp.int16))

    occ = occupancy_grid(st, cfg)
    passable = jnp.asarray(mapdata.passable)
    h, w = cfg.grid_h, cfg.grid_w

    for p in (0, 1):  # 编译期展开
        base = hq_slot(p, cfg)
        producing = st.alive & (st.btype > 0) & (st.btimer == 0)
        half = jnp.zeros(cfg.n_total, bool).at[base:base + cfg.e_max].set(True)
        cand = producing & half
        pidx = jnp.argmax(cand)          # 全无效返回 0,用 any 门控
        has_p = jnp.any(cand)

        # 落地格:生产者 8 邻中第一个界内、可通行、无人格
        cells = st.pos[pidx] + _SPAWN_DIRS                       # [8,2]
        inb = ((cells[:, 0] >= 0) & (cells[:, 0] < h)
               & (cells[:, 1] >= 0) & (cells[:, 1] < w))
        cc = jnp.clip(cells, 0, jnp.asarray([h - 1, w - 1]))
        ok = inb & passable[cc[:, 0], cc[:, 1]] & ~occ[cc[:, 0], cc[:, 1]]
        ci = jnp.argmax(ok)
        has_cell = jnp.any(ok)

        # 槽位:本半区第一个空槽
        free = ~st.alive[base:base + cfg.e_max]
        slot = base + jnp.argmax(free)
        has_slot = jnp.any(free)

        do = has_p & has_cell & has_slot
        ut = st.btype[pidx]
        uhp = _unit_spawn_hp(cfg, ut, st.upgrades[p])
        spawn_cell = cc[ci]

        st = st._replace(
            alive=st.alive.at[slot].set(jnp.where(do, True, st.alive[slot])),
            etype=st.etype.at[slot].set(jnp.where(do, ut, st.etype[slot])),
            pos=st.pos.at[slot].set(jnp.where(do, spawn_cell, st.pos[slot])),
            hp=st.hp.at[slot].set(jnp.where(do, uhp, st.hp[slot])),
            cooldown=st.cooldown.at[slot].set(jnp.where(do, 0, st.cooldown[slot])),
            order=st.order.at[slot].set(jnp.where(do, ORDER_IDLE, st.order[slot])),
            phase=st.phase.at[slot].set(jnp.where(do, PH_TO_NODE, st.phase[slot])),
            target_node=st.target_node.at[slot].set(
                jnp.where(do, -1, st.target_node[slot])),
            target_cell=st.target_cell.at[slot].set(
                jnp.where(do, spawn_cell, st.target_cell[slot])),
            cargo=st.cargo.at[slot].set(jnp.where(do, 0, st.cargo[slot])),
            mine_timer=st.mine_timer.at[slot].set(jnp.where(do, 0, st.mine_timer[slot])),
            inside=st.inside.at[slot].set(jnp.where(do, False, st.inside[slot])),
            btype=st.btype.at[slot].set(jnp.where(do, 0, st.btype[slot])),
            btimer=st.btimer.at[slot].set(jnp.where(do, 0, st.btimer[slot])),
            node_id=st.node_id.at[slot].set(jnp.where(do, -1, st.node_id[slot])),
        )
        # 生产者收尾:成功清 btype;失败(有产出但落不下)btimer 回 1 顺延
        st = st._replace(
            btype=st.btype.at[pidx].set(
                jnp.where(has_p & do, 0, st.btype[pidx])),
            btimer=st.btimer.at[pidx].set(
                jnp.where(has_p & ~do, 1, st.btimer[pidx])),
        )
        occ = occ.at[spawn_cell[0], spawn_cell[1]].max(do)  # 防两家同格(理论不可能)
    return st


def start_constructions(state: WorldState, cfg: Config, mapdata: MapData,
                        owner: jax.Array, key: jax.Array) -> WorldState:
    """工人到场(与点 4 邻)后开工:抢点仲裁 → 按点序对账扣费 → 立 timer。
    跨玩家同 tick 抢同一点:每 tick 随机先手玩家,防下标恒赢的对称性偏置。"""
    dist = jnp.asarray(mapdata.dist_fields)                # [G,H,W]
    tn = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    my_d = dist[tn, state.pos[:, 0], state.pos[:, 1]]      # [N] 到各自目标点的距离
    arrived = (state.alive & ~state.inside & (state.order == ORDER_BUILD)
               & (state.target_node >= 0) & (my_d == 1))

    first = jax.random.bernoulli(key).astype(jnp.int8)     # 本 tick 先手玩家
    from .actions import node_costs
    ncost = node_costs(cfg, mapdata)                       # [Nn,2]

    st = state
    stock = state.resources                                # [2,2] 顺序对账
    slots = jnp.arange(cfg.n_total)
    for k in range(cfg.n_nodes):  # 编译期展开
        claimable = (st.node_owner[k] == -1) & (st.node_build_timer[k] == 0)
        cand = arrived & (st.target_node == k)
        # 先手玩家优先,再按槽号
        score = slots + cfg.n_total * (owner != first)
        score = jnp.where(cand, score, jnp.iinfo(jnp.int32).max)
        widx = jnp.argmin(score)
        has = jnp.any(cand) & claimable
        wowner = owner[widx].astype(jnp.int32)
        afford = jnp.all(stock[wowner] >= ncost[k])
        build_t = jnp.where(jnp.asarray(mapdata.node_type)[k] == RES_WATER,
                            cfg.pump_time_build, cfg.mine_time_build)
        do = has & afford
        stock = stock.at[wowner].add(jnp.where(do, -ncost[k], 0))
        st = st._replace(
            node_owner=st.node_owner.at[k].set(
                jnp.where(do, owner[widx], st.node_owner[k])),
            node_build_timer=st.node_build_timer.at[k].set(
                jnp.where(do, build_t, st.node_build_timer[k]).astype(jnp.int16)),
            node_builder=st.node_builder.at[k].set(
                jnp.where(do, widx, st.node_builder[k]).astype(jnp.int16)),
        )
    return st._replace(resources=stock)


def construction_tick(state: WorldState, cfg: Config, mapdata: MapData,
                      owner: jax.Array) -> WorldState:
    """工地倒计时与结构落地(矿/泵实体入表);半区无空槽则停在 1 顺延。
    完工后施工工人转 IDLE(它站在相邻格,可被改派)。"""
    nbt = jnp.where(state.node_build_timer > 0,
                    state.node_build_timer - 1, state.node_build_timer)
    st = state._replace(node_build_timer=nbt.astype(jnp.int16))

    node_pos = jnp.asarray(mapdata.node_pos)
    node_type = jnp.asarray(mapdata.node_type)
    for k in range(cfg.n_nodes):  # 编译期展开
        done_k = ((st.node_build_timer[k] == 0) & (st.node_owner[k] >= 0)
                  & (st.node_ent[k] == -1) & (st.node_builder[k] >= 0))
        p = jnp.clip(st.node_owner[k].astype(jnp.int32), 0, 1)
        base = p * cfg.e_max
        free = jax.lax.dynamic_slice(~st.alive, (base,), (cfg.e_max,))
        slot = base + jnp.argmax(free)
        has_slot = jnp.any(free)
        do = done_k & has_slot
        stype = jnp.where(node_type[k] == RES_WATER, TYPE_PUMP, TYPE_MINE)
        st = st._replace(
            alive=st.alive.at[slot].set(jnp.where(do, True, st.alive[slot])),
            etype=st.etype.at[slot].set(
                jnp.where(do, stype, st.etype[slot]).astype(jnp.int8)),
            pos=st.pos.at[slot].set(jnp.where(do, node_pos[k], st.pos[slot])),
            hp=st.hp.at[slot].set(jnp.where(do, cfg.node_struct_hp, st.hp[slot])),
            inside=st.inside.at[slot].set(jnp.where(do, False, st.inside[slot])),
            order=st.order.at[slot].set(jnp.where(do, ORDER_IDLE, st.order[slot])),
            btype=st.btype.at[slot].set(jnp.where(do, 0, st.btype[slot])),
            btimer=st.btimer.at[slot].set(jnp.where(do, 0, st.btimer[slot])),
            cargo=st.cargo.at[slot].set(jnp.where(do, 0, st.cargo[slot])),
            node_id=st.node_id.at[slot].set(
                jnp.where(do, k, st.node_id[slot]).astype(jnp.int8)),
            node_ent=st.node_ent.at[k].set(
                jnp.where(do, slot, st.node_ent[k]).astype(jnp.int16)),
            # 无空槽:工地顺延在 1
            node_build_timer=st.node_build_timer.at[k].set(
                jnp.where(done_k & ~has_slot, 1, st.node_build_timer[k])),
        )
        # 施工工人转 IDLE
        b = jnp.clip(st.node_builder[k].astype(jnp.int32), 0, cfg.n_total - 1)
        st = st._replace(
            order=st.order.at[b].set(jnp.where(do, ORDER_IDLE, st.order[b])),
            target_node=st.target_node.at[b].set(
                jnp.where(do, -1, st.target_node[b])),
            node_builder=st.node_builder.at[k].set(
                jnp.where(do, -1, st.node_builder[k])),
        )
    return st


def harvest_tick(state: WorldState, cfg: Config, mapdata: MapData,
                 owner: jax.Array) -> WorldState:
    """采集一体循环:入驻(抢槽)→ 矿内倒计时 → 出矿(抢出口格)→ 到家卸货。"""
    dist = jnp.asarray(mapdata.dist_fields)
    node_type = jnp.asarray(mapdata.node_type)
    tn = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    harv = state.alive & (state.order == ORDER_HARVEST) & (state.target_node >= 0)

    st = state
    # ---- 入驻:相邻 + 点归属己方且结构在 + 容量余额,同点按槽号排队 ----
    d_node = dist[tn, st.pos[:, 0], st.pos[:, 1]]
    want_in = (harv & ~st.inside & (st.phase == PH_TO_NODE) & (d_node == 1)
               & (st.node_owner[tn] == owner) & (st.node_ent[tn] >= 0))
    counts = inside_counts(st, cfg)                        # [Nn]
    # 开采耗时按各工人所属玩家的工人线等级查表(v1.1)
    wl = st.upgrades[owner.astype(jnp.int32), LINE_WORKER]           # [N]
    my_mine_time = jnp.asarray(cfg.worker_mine_time_by_level)[wl]    # [N]
    inside = st.inside
    mine_timer = st.mine_timer
    phase = st.phase
    for k in range(cfg.n_nodes):  # 编译期展开
        cand = want_in & (st.target_node == k)
        rank = jnp.cumsum(cand.astype(jnp.int32)) - 1      # 槽号序内的名次
        enter = cand & (counts[k] + rank < cfg.node_capacity)
        inside = inside | enter
        mine_timer = jnp.where(enter, my_mine_time, mine_timer)
        phase = jnp.where(enter, PH_MINING, phase)
    st = st._replace(inside=inside, mine_timer=mine_timer.astype(jnp.int16),
                     phase=phase.astype(jnp.int8))

    # ---- 矿内倒计时 ----
    mt = jnp.where(st.inside & (st.mine_timer > 0), st.mine_timer - 1, st.mine_timer)
    st = st._replace(mine_timer=mt.astype(jnp.int16))

    # ---- 出矿:回到入口格(pos 一直保留)。**允许暂时叠格**:若要求入口格为空,
    # 满员矿点外排队的工人恰好站满全部入口格时会互相锁死(里面出不来 → 外面
    # 进不去)。叠格后占用图按「格上有人」算,叠格者的后续移动会自然散开
    # (与矿被拆时的弹出同一约定,见 docs/DECISIONS.md)。----
    win = st.inside & (st.mine_timer == 0) & (st.phase == PH_MINING)
    # 一趟入账公式(plan/critic S-2 定案):carry[工人线级] + yield_bonus[矿泵级]
    my_carry = jnp.asarray(cfg.worker_carry_by_level)[wl]            # [N]
    ent_idx = jnp.clip(st.node_ent[tn].astype(jnp.int32), 0, cfg.n_total - 1)
    node_lv = jnp.where(st.node_ent[tn] >= 0, st.level[ent_idx], 1)  # [N]
    trip = my_carry + jnp.asarray(cfg.node_yield_bonus)[node_lv]
    st = st._replace(
        inside=jnp.where(win, False, st.inside),
        cargo=jnp.where(win, trip, st.cargo).astype(jnp.int16),
        cargo_type=jnp.where(win, node_type[tn], st.cargo_type).astype(jnp.int8),
        phase=jnp.where(win, PH_TO_HQ, st.phase).astype(jnp.int8),
    )

    # ---- 卸货:与己方 HQ 相邻即入账(资源类型 = 目标点类型)----
    hq_goal = cfg.n_nodes + owner.astype(jnp.int32)
    d_hq = dist[hq_goal, st.pos[:, 0], st.pos[:, 1]]
    dep = (harv & ~st.inside & (st.phase == PH_TO_HQ) & (st.cargo > 0) & (d_hq == 1))
    rtype = st.cargo_type.astype(jnp.int32)                # 出矿时定格的载荷类型
    gain = (jnp.zeros_like(st.resources)
            .at[owner.astype(jnp.int32), rtype]
            .add(jnp.where(dep, st.cargo.astype(jnp.int32), 0)))
    st = st._replace(
        resources=st.resources + gain,
        cargo=jnp.where(dep, 0, st.cargo).astype(jnp.int16),
        phase=jnp.where(dep, PH_TO_NODE, st.phase).astype(jnp.int8),
    )
    return st
