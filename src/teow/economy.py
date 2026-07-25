"""经济:生产(训练完成落地)、建造(到场开工/完工落地)、采集一体循环。

同 tick 竞争的仲裁纪律(调研报告 §1/§5):
- 指派抢名额(v1.3,在 actions.apply_orders)/开工抢点:按资源点逐点处理
  (Nn=8 的 Python 循环,编译期展开),点内排序用槽号(同一玩家内部无公平性
  问题),开工跨玩家用每 tick 随机先手位防偏置;入驻不再仲裁(名额指派侧已封顶)。
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
    N_LINES,
    N_TYPES,
    RES_WATER,
    TYPE_CAMP,
    TYPE_HQ,
    TYPE_MINE,
    TYPE_PUMP,
    Config,
    btask_research,
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
    cell_of,
    hq_slot,
)

# 落地用的 8 邻。**斜角优先**:正邻 4 格是卸货格(4-邻接判定),新单位落在正邻格
# 会堵死运矿工人的入家路——实测 4 个待命步兵恰好站满 4 个卸货格,把经济锁死。
_SPAWN_DIRS = jnp.asarray(
    [[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 1], [1, 0], [0, -1], [-1, 0]], jnp.int32)


def occupancy_grid(state: WorldState, cfg: Config) -> jax.Array:
    """bool [H,W]:在场实体(alive 且不在矿内)占用图。每 tick 重算,不进 state。"""
    on = state.alive & ~state.inside
    cell = cell_of(state.pos)
    occ = jnp.zeros((cfg.grid_h, cfg.grid_w), bool)
    return occ.at[cell[:, 0], cell[:, 1]].max(on)


def assigned_counts(state: WorldState, cfg: Config) -> jax.Array:
    """int32 [Nn]:各点已指派工人数(alive & order==HARVEST,含在途/矿内/运输段)。
    名额占用的唯一口径(v1.3 指派即占用):残留 target_node 被 order 门控。"""
    tn = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    assigned = (state.alive & (state.order == ORDER_HARVEST)
                & (state.target_node >= 0))
    return (jnp.zeros(cfg.n_nodes, jnp.int32)
            .at[tn].add(assigned.astype(jnp.int32)))


def upgrade_cost_of(state, cfg: Config) -> jax.Array:
    """int32 [N,2]:每个实体「从当前等级升一级」的(ore,water)成本,按类型查表。
    定价唯一定义处(legality 乐观检查与 paid_orders_pass 实扣都用它)。
    非可升级类型返回 0(调用方用类型掩码门控)。"""
    from .config import TYPE_BARRACKS, TYPE_TOWER
    lv = jnp.clip(state.level.astype(jnp.int32), 0, 7)
    is_hq = state.etype == TYPE_HQ
    is_camp = state.etype == TYPE_CAMP
    is_tower = state.etype == TYPE_TOWER
    is_bar = state.etype == TYPE_BARRACKS
    ore = jnp.where(is_hq, jnp.asarray(cfg.base_up_cost_ore)[lv],
                    jnp.asarray(cfg.node_up_cost_ore)[lv])
    ore = jnp.where(is_camp, jnp.asarray(cfg.camp_up_cost_ore)[lv], ore)
    ore = jnp.where(is_tower, jnp.asarray(cfg.tower_up_cost_ore)[lv], ore)
    ore = jnp.where(is_bar, jnp.asarray(cfg.barracks_up_cost_ore)[lv], ore)
    wat = jnp.where(is_hq, jnp.asarray(cfg.base_up_cost_water)[lv],
                    jnp.asarray(cfg.node_up_cost_water)[lv])
    wat = jnp.where(is_camp, jnp.asarray(cfg.camp_up_cost_water)[lv], wat)
    wat = jnp.where(is_tower, jnp.asarray(cfg.tower_up_cost_water)[lv], wat)
    wat = jnp.where(is_bar, jnp.asarray(cfg.barracks_up_cost_water)[lv], wat)
    return jnp.stack([ore, wat], axis=-1)


def upgrade_time_of(state, cfg: Config) -> jax.Array:
    """int32 [N]:每个实体升一级的耗时,按类型查表。"""
    lv = jnp.clip(state.level.astype(jnp.int32), 0, 7)
    is_hq = state.etype == TYPE_HQ
    from .config import TYPE_BARRACKS, TYPE_TOWER
    is_camp = state.etype == TYPE_CAMP
    t = jnp.where(is_hq, jnp.asarray(cfg.base_up_time)[lv],
                  jnp.asarray(cfg.node_up_time)[lv])
    t = jnp.where(is_camp, jnp.asarray(cfg.camp_up_time)[lv], t)
    t = jnp.where(state.etype == TYPE_BARRACKS,
                  jnp.asarray(cfg.barracks_up_time)[lv], t)
    return jnp.where(state.etype == TYPE_TOWER,
                     jnp.asarray(cfg.tower_up_time)[lv], t)


def paid_orders_pass(state: WorldState, act: jax.Array, cfg: Config,
                     mapdata: MapData, owner: jax.Array) -> WorldState:
    """同 tick 可能多笔的付费指令(升级/研发/建营)的**顺序对账扣费**
    (critic B-1):同玩家按槽号累计支出,超出库存的笔自动 no-op,库存恒 ≥0。
    act 必须是 apply_orders 返回的合法化动作。
    结算顺序(固定,写死防漂移):①升级+研发按槽号 cumsum 对账 ②建营
    (同玩家同 tick 至多批准一座,用①之后的余额)。"""
    from .actions import (
        a_build_barracks,
        a_build_camp,
        a_build_fence,
        a_build_mortar,
        a_build_tower,
        a_research_line,
        a_train_dog,
        a_upgrade,
    )
    from .config import (
        BTASK_BUILD_BARRACKS,
        BTASK_BUILD_CAMP,
        BTASK_BUILD_FENCE_IRON,
        BTASK_BUILD_FENCE_STONE,
        BTASK_BUILD_FENCE_WOOD,
        BTASK_BUILD_MORTAR,
        BTASK_BUILD_TOWER,
        TYPE_BARRACKS,
        TYPE_CAMP,
        TYPE_DOG,
        TYPE_FENCE_IRON,
        TYPE_FENCE_STONE,
        TYPE_FENCE_WOOD,
        TYPE_MORTAR,
        TYPE_TOWER,
    )

    own_i = owner.astype(jnp.int32)
    st = state

    # ---- ① 升级 + 研发 + 训狗:统一 cumsum 对账 ----
    w_up = act == a_upgrade(cfg)
    # 八线研发(v1.4):同玩家同 tick 对同一条线的并发申请去重,只批槽号最小者
    # (audit v1.1 P0-1:掩码的 busy_same 只见已写入的 btype,看不见同 tick
    # 并发,两营同时下单会双倍扣费只得一级;去重必须在扣费之前)
    slots_arr = jnp.arange(cfg.n_total)
    w_res = jnp.zeros(cfg.n_total, bool)     # 任一线研发(去重后)
    res_line = jnp.zeros(cfg.n_total, jnp.int32)  # 该行研的线号(w_res 门控)
    for line in range(N_LINES):  # 编译期展开
        w_l = act == a_research_line(line, cfg)
        for p in range(cfg.n_players):  # 编译期展开
            cand = w_l & (own_i == p)
            keep = cand & (slots_arr == jnp.argmax(cand))
            w_l = jnp.where(own_i == p, keep, w_l)
        w_res = w_res | w_l
        res_line = jnp.where(w_l, line, res_line)
    cur_l = st.upgrades[own_i, jnp.clip(res_line, 0, N_LINES - 1)].astype(jnp.int32)
    res_cost = jnp.stack([jnp.asarray(cfg.line_res_cost_ore)[cur_l],
                          jnp.asarray(cfg.line_res_cost_water)[cur_l]], -1)
    cost = upgrade_cost_of(st, cfg) * w_up[:, None]
    cost += res_cost * w_res[:, None]
    # 兵营训练(v1.4:狗+六新兵种):同 tick 多座兵营可同时下单,
    # 与升级/研发同一 cumsum 对账;成本按 train_cost 表 gather
    from .actions import TRAIN_ORDER, a_train_unit
    from .config import TYPE_ARCHER
    w_btr = act == a_train_dog(cfg)
    btr_t = jnp.where(w_btr, TYPE_DOG, 0)
    for t in TRAIN_ORDER[TRAIN_ORDER.index(TYPE_ARCHER):]:  # 兵营系六兵种
        w_t = act == a_train_unit(t, cfg)
        w_btr = w_btr | w_t
        btr_t = jnp.where(w_t, t, btr_t)
    tco = jnp.asarray(cfg.train_cost_ore_by_type, jnp.int32)
    tcw = jnp.asarray(cfg.train_cost_water_by_type, jnp.int32)
    cost += jnp.stack([tco[btr_t], tcw[btr_t]], -1) * w_btr[:, None]
    want = w_up | w_res | w_btr

    afford = jnp.zeros(cfg.n_total, bool)
    stock = st.resources
    for p in range(cfg.n_players):  # 编译期展开
        cum = jnp.cumsum(cost * (own_i == p)[:, None], axis=0)  # 含本行
        ok = jnp.all(cum <= stock[p][None, :], axis=-1)
        afford = jnp.where(own_i == p, ok, afford)
    do = want & afford

    pay = jnp.zeros_like(stock).at[own_i].add(jnp.where(do[:, None], cost, 0))
    stock = stock - pay
    task = jnp.where(w_res, btask_research(0) - res_line,
                     jnp.where(w_btr, btr_t, BTASK_UPGRADE))
    ttime = jnp.asarray(cfg.train_time_by_type, jnp.int32)
    t = jnp.where(w_res, jnp.asarray(cfg.line_res_time)[cur_l],
                  jnp.where(w_btr, ttime[btr_t],
                            upgrade_time_of(st, cfg)))
    st = st._replace(
        resources=stock,
        btype=jnp.where(do, task, st.btype).astype(jnp.int8),
        btimer=jnp.where(do, t, st.btimer).astype(jnp.int16),
    )

    # ---- ② 自由格建筑(营/兵营):同玩家每种取槽号最小的申请者,
    # 落其相邻第一空闲格 ----
    occ = occupancy_grid(st, cfg)
    # 矿内单位的入口格(pos 保留)也视为占用:占用图只算在场实体,矿内单位
    # 隐身——建筑恰落在其入口格上时,出矿弹回即被永久活埋(困在硬障碍格内
    # 场梯度归零,实测卡满全场 1800 tick 且吊死采集名额;v1.4 审计发现)。
    # 单位落地(production)不用此加强版:单位间短暂叠格由互推自然散开。
    in_cells = cell_of(st.pos)
    occ = occ.at[in_cells[:, 0], in_cells[:, 1]].max(st.alive & st.inside)
    passable = jnp.asarray(mapdata.passable)
    h, w = cfg.grid_h, cfg.grid_w
    structs = (
        (a_build_camp(cfg), TYPE_CAMP, BTASK_BUILD_CAMP,
         jnp.asarray([cfg.camp_cost_ore, cfg.camp_cost_water], jnp.int32),
         cfg.camp_build_time, jnp.asarray(cfg.camp_hp_by_level)[2] // 10),
        (a_build_barracks(cfg), TYPE_BARRACKS, BTASK_BUILD_BARRACKS,
         jnp.asarray([cfg.barracks_cost_ore, cfg.barracks_cost_water], jnp.int32),
         cfg.barracks_build_time, int(cfg.barracks_hp_by_level[1]) // 10),
        (a_build_tower(cfg), TYPE_TOWER, BTASK_BUILD_TOWER,
         jnp.asarray([cfg.tower_cost_ore, cfg.tower_cost_water], jnp.int32),
         cfg.tower_build_time, int(cfg.tower_hp_by_level[1]) // 10),
        (a_build_mortar(cfg), TYPE_MORTAR, BTASK_BUILD_MORTAR,
         jnp.asarray([cfg.mortar_cost_ore, cfg.mortar_cost_water], jnp.int32),
         cfg.mortar_build_time, cfg.mortar_hp // 10),
        (a_build_fence(TYPE_FENCE_WOOD, cfg), TYPE_FENCE_WOOD,
         BTASK_BUILD_FENCE_WOOD,
         jnp.asarray([cfg.fence_wood_cost_ore, cfg.fence_wood_cost_water],
                     jnp.int32),
         cfg.fence_wood_build_time, cfg.fence_wood_hp // 10),
        (a_build_fence(TYPE_FENCE_STONE, cfg), TYPE_FENCE_STONE,
         BTASK_BUILD_FENCE_STONE,
         jnp.asarray([cfg.fence_stone_cost_ore, cfg.fence_stone_cost_water],
                     jnp.int32),
         cfg.fence_stone_build_time, cfg.fence_stone_hp // 10),
        (a_build_fence(TYPE_FENCE_IRON, cfg), TYPE_FENCE_IRON,
         BTASK_BUILD_FENCE_IRON,
         jnp.asarray([cfg.fence_iron_cost_ore, cfg.fence_iron_cost_water],
                     jnp.int32),
         cfg.fence_iron_build_time, cfg.fence_iron_hp // 10),
    )
    for a_id, stype, btask, camp_cost, build_t, start_hp0 in structs:
      for p in range(cfg.n_players):  # 编译期展开
        cand = (act == a_id) & (own_i == p)
        bidx = jnp.argmax(cand)                # 全 False 返 0,has 门控
        has = jnp.any(cand) & jnp.all(st.resources[p] >= camp_cost)

        cells = cell_of(st.pos[bidx])[None, :] + _SPAWN_DIRS
        cells = cells[0] if cells.ndim == 3 else cells
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
        start_hp = start_hp0

        st = st._replace(
            resources=st.resources.at[p].add(jnp.where(do_c, -camp_cost, 0)),
            alive=st.alive.at[slot].set(jnp.where(do_c, True, st.alive[slot])),
            etype=st.etype.at[slot].set(
                jnp.where(do_c, stype, st.etype[slot]).astype(jnp.int8)),
            pos=st.pos.at[slot].set(
                jnp.where(do_c, cell.astype(jnp.float32), st.pos[slot])),
            hp=st.hp.at[slot].set(jnp.where(do_c, start_hp, st.hp[slot])),
            level=st.level.at[slot].set(
                jnp.where(do_c, 1, st.level[slot]).astype(jnp.int8)),
            btype=st.btype.at[slot].set(
                jnp.where(do_c, btask, st.btype[slot]).astype(jnp.int8)),
            btimer=st.btimer.at[slot].set(
                jnp.where(do_c, build_t, st.btimer[slot]).astype(jnp.int16)),
        )
        occ = occ.at[cell[0], cell[1]].max(do_c)  # 防两家同格(远隔,保险)
    return st


def special_tasks_tick(state: WorldState, cfg: Config,
                       owner: jax.Array) -> WorldState:
    """负数 btype 任务的推进与完成结算(解码唯一集中处;完成分支必须 btype←0,
    否则「btype<0 & btimer==0」下一 tick 重复触发,level 每 tick+1 直到溢出)。
    btimer 的递减复用 production_tick(它对所有 btimer>0 无差别倒数)。"""
    from .config import (
        BTASK_BUILD_BARRACKS,
        BTASK_BUILD_CAMP,
        BTASK_BUILD_FENCE_IRON,
        BTASK_BUILD_FENCE_STONE,
        BTASK_BUILD_FENCE_WOOD,
        BTASK_BUILD_MORTAR,
        BTASK_BUILD_TOWER,
        TYPE_TOWER,
    )
    st = state
    own_i = owner.astype(jnp.int32)

    # ---- 在建自由格建筑的 hp 线性成长(增量式,伤害得以保留;完成拍补齐整数
    # 余数,使「未挨打的建成建筑 = 满血」恒成立。成长只发生在 btimer>0 的 T-1
    # 个 tick——完成拍 btimer 已归 0,余数按 T-1 算)。v1.4 起统一描述符循环,
    # 新建筑照抄一行(plan Phase 3)。----
    done = st.alive & (st.btype < 0) & (st.btimer == 0)
    hp = st.hp
    grow_specs = (
        (BTASK_BUILD_CAMP, int(cfg.camp_hp_by_level[2]), cfg.camp_build_time),
        (BTASK_BUILD_BARRACKS, int(cfg.barracks_hp_by_level[1]),
         cfg.barracks_build_time),
        (BTASK_BUILD_TOWER, int(cfg.tower_hp_by_level[1]), cfg.tower_build_time),
        (BTASK_BUILD_MORTAR, int(cfg.mortar_hp), cfg.mortar_build_time),
        (BTASK_BUILD_FENCE_WOOD, int(cfg.fence_wood_hp),
         cfg.fence_wood_build_time),
        (BTASK_BUILD_FENCE_STONE, int(cfg.fence_stone_hp),
         cfg.fence_stone_build_time),
        (BTASK_BUILD_FENCE_IRON, int(cfg.fence_iron_hp),
         cfg.fence_iron_build_time),
    )
    for btask, full, t_build in grow_specs:  # 编译期展开
        start = full // 10
        g = (full - start) // t_build  # 每 tick 增量(可为 0,余数在完成拍补)
        growing = st.alive & (st.btype == btask) & (st.btimer > 0)
        hp = hp + jnp.where(growing, g, 0)
        b_done = done & (st.btype == btask)
        rem = (full - start) - g * (t_build - 1)
        hp = hp + jnp.where(b_done, rem, 0)
    # 建营完成:level=2(建成即 2 级,issue v1.1;其余建筑建成保持 1 级)
    camp_done = done & (st.btype == BTASK_BUILD_CAMP)
    level = jnp.where(camp_done, 2, st.level)

    # 自升级完成;哨塔/兵营升级补血量上限差额(升级提升血量;v1.4 兵营加入)
    upg = done & (st.btype == BTASK_UPGRADE)
    lv0 = jnp.clip(st.level.astype(jnp.int32), 0, 6)
    thp = jnp.asarray(cfg.tower_hp_by_level)
    tower_up = upg & (st.etype == TYPE_TOWER)
    hp = hp + jnp.where(tower_up, thp[lv0 + 1] - thp[lv0], 0)
    from .config import TYPE_BARRACKS as _TB
    bhp = jnp.asarray(cfg.barracks_hp_by_level)
    bar_up = upg & (st.etype == _TB)
    hp = hp + jnp.where(bar_up, bhp[lv0 + 1] - bhp[lv0], 0)
    level = jnp.where(upg, level + 1, level).astype(jnp.int8)

    # 研发完成:该玩家对应线 +1(legality+paid 去重保证同线同 tick 至多一营在研),
    # 存量该线单位 hp 补上限差额(不缩放,保伤痕)。v1.4 八线,每线恰惠及一个
    # 兵种(line_of_type 反查),补血按 stats hp_table 向量化 gather。
    from .stats import hp_table as _hp_table
    upgrades = st.upgrades
    htab = _hp_table(cfg)                                   # [N_TYPES, 8]
    et32 = jnp.clip(st.etype.astype(jnp.int32), 0, N_TYPES - 1)
    line_of = jnp.asarray(cfg.line_of_type, jnp.int32)[et32]  # [N]
    for line in range(N_LINES):  # 编译期展开
        rdone = done & (st.btype == btask_research(line))
        for p in range(cfg.n_players):  # 编译期展开
            hit = jnp.any(rdone & (own_i == p))
            old = upgrades[p, line].astype(jnp.int32)
            new = jnp.minimum(old + 1, 7)
            upgrades = upgrades.at[p, line].set(
                jnp.where(hit, new, old).astype(jnp.int8))
            delta = htab[et32, new] - htab[et32, old]       # [N] 各实体差额
            bump = st.alive & (own_i == p) & (line_of == line)
            hp = hp + jnp.where(hit & bump, delta, 0)

    btype = jnp.where(done, 0, st.btype).astype(jnp.int8)
    return st._replace(hp=hp.astype(jnp.int32), level=level,
                       upgrades=upgrades, btype=btype)


def _unit_spawn_hp(cfg: Config, ut: jax.Array, upgrades_p: jax.Array) -> jax.Array:
    """新单位出生血量:类型 × 线级查 stats 表(v1.4;采集单位无线,恒 1 级行)。"""
    from .stats import hp_table as _hp_table
    ut32 = jnp.clip(ut.astype(jnp.int32), 0, N_TYPES - 1)
    line = jnp.asarray(cfg.line_of_type, jnp.int32)[ut32]
    lv = jnp.where(line >= 0,
                   upgrades_p[jnp.clip(line, 0, N_LINES - 1)].astype(jnp.int32), 1)
    return _hp_table(cfg)[ut32, jnp.clip(lv, 0, 7)]


def production_tick(state: WorldState, cfg: Config, mapdata: MapData) -> WorldState:
    """训练倒计时与落地。落不下(周围无空格/半区无空槽)则停在 1,下 tick 重试。

    v1.2:同玩家同 tick 可有多个生产建筑完成(HQ+兵营),每玩家循环处理
    至多 1+max_barracks 个完成者,逐个仲裁落地格与槽位(critic B-1 兑现)。
    """
    btimer = jnp.where(state.btimer > 0, state.btimer - 1, state.btimer)
    st = state._replace(btimer=btimer.astype(jnp.int16))

    occ = occupancy_grid(st, cfg)
    passable = jnp.asarray(mapdata.passable)
    h, w = cfg.grid_h, cfg.grid_w

    for p in range(cfg.n_players):  # 编译期展开
      # 每玩家至多 HQ + max_barracks 座兵营同 tick 完成(编译期展开;plan D8)
      for _round in range(1 + cfg.max_barracks):
        base = hq_slot(p, cfg)
        producing = st.alive & (st.btype > 0) & (st.btimer == 0)
        half = jnp.zeros(cfg.n_total, bool).at[base:base + cfg.e_max].set(True)
        cand = producing & half
        pidx = jnp.argmax(cand)          # 全无效返回 0,用 any 门控
        has_p = jnp.any(cand)

        # 落地格:生产者 8 邻中第一个界内、可通行、无人格
        cells = cell_of(st.pos[pidx])[None, :] + _SPAWN_DIRS
        cells = (cells[0] if cells.ndim == 3 else cells)             # [8,2]
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
            pos=st.pos.at[slot].set(
                jnp.where(do, spawn_cell.astype(jnp.float32), st.pos[slot])),
            hp=st.hp.at[slot].set(jnp.where(do, uhp, st.hp[slot])),
            order=st.order.at[slot].set(jnp.where(do, ORDER_IDLE, st.order[slot])),
            phase=st.phase.at[slot].set(jnp.where(do, PH_TO_NODE, st.phase[slot])),
            target_node=st.target_node.at[slot].set(
                jnp.where(do, -1, st.target_node[slot])),
            target_cell=st.target_cell.at[slot].set(
                jnp.where(do, spawn_cell.astype(jnp.float32), st.target_cell[slot])),
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
    tn = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    npos = jnp.asarray(mapdata.node_pos, jnp.float32)
    my_d = jnp.linalg.norm(state.pos - npos[tn], axis=-1)  # 欧氏(v1.2)
    arrived = (state.alive & ~state.inside & (state.order == ORDER_BUILD)
               & (state.target_node >= 0) & (my_d <= cfg.reach_radius))

    # 本 tick 先手玩家(v1.5 critic B-2:bernoulli 只出 0/1,四人下玩家 2/3
    # 跨玩家平票永败——改均匀 randint,P 家等概率)
    first = jax.random.randint(key, (), 0, cfg.n_players).astype(jnp.int8)
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
        p = jnp.clip(st.node_owner[k].astype(jnp.int32), 0, cfg.n_players - 1)
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
            pos=st.pos.at[slot].set(
                jnp.where(do, node_pos[k].astype(jnp.float32), st.pos[slot])),
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
    """采集一体循环:入驻 → 矿内倒计时 → 出矿(弹回入口格)→ 到家卸货。"""
    node_type = jnp.asarray(mapdata.node_type)
    npos = jnp.asarray(mapdata.node_pos, jnp.float32)
    tn = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    harv = state.alive & (state.order == ORDER_HARVEST) & (state.target_node >= 0)

    st = state
    # ---- 入驻:到达半径内 + 点归属己方且结构在。v1.3 名额制下不再做容量
    # 仲裁——名额在指派侧(掩码 + apply_orders 同 tick 仲裁)已保证每点
    # order==HARVEST 数 ≤ 等级名额,驻内数不可能超。----
    d_node = jnp.linalg.norm(st.pos - npos[tn], axis=-1)
    enter = (harv & ~st.inside & (st.phase == PH_TO_NODE)
             & (d_node <= cfg.reach_radius)
             & (st.node_owner[tn] == owner) & (st.node_ent[tn] >= 0))
    # 开采耗时按采集单位类型查表(v1.4:工人经济线取消,大力士/马车各有参数)
    et32 = jnp.clip(st.etype.astype(jnp.int32), 0, N_TYPES - 1)
    my_mine_time = jnp.asarray(cfg.mine_time_by_type, jnp.int32)[et32]  # [N]
    st = st._replace(
        inside=st.inside | enter,
        mine_timer=jnp.where(enter, my_mine_time, st.mine_timer).astype(jnp.int16),
        phase=jnp.where(enter, PH_MINING, st.phase).astype(jnp.int8))

    # ---- 矿内倒计时 ----
    mt = jnp.where(st.inside & (st.mine_timer > 0), st.mine_timer - 1, st.mine_timer)
    st = st._replace(mine_timer=mt.astype(jnp.int16))

    # ---- 出矿:回到入口格(pos 一直保留)。**允许暂时叠格**:若要求入口格为空,
    # 满员矿点外排队的工人恰好站满全部入口格时会互相锁死(里面出不来 → 外面
    # 进不去)。叠格后占用图按「格上有人」算,叠格者的后续移动会自然散开
    # (与矿被拆时的弹出同一约定,见 docs/DECISIONS.md)。----
    win = st.inside & (st.mine_timer == 0) & (st.phase == PH_MINING)
    # 一趟入账公式:carry[采集单位类型] + yield_bonus[矿泵级](v1.4 per-type)
    my_carry = jnp.asarray(cfg.carry_by_type, jnp.int32)[et32]       # [N]
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
    hqp = jnp.asarray(mapdata.hq_pos, jnp.float32)[owner.astype(jnp.int32)]
    d_hq = jnp.linalg.norm(st.pos - hqp, axis=-1)
    dep = (harv & ~st.inside & (st.phase == PH_TO_HQ) & (st.cargo > 0)
           & (d_hq <= cfg.reach_radius))
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
