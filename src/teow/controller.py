"""控制器:random 与 scripted 两个纯 jnp 实现,统一接口
`fn(state, key) -> actions[N]`(只给自家实体出招,别家一律 NOOP)。

这是 v2 RL 策略的接口占位:PPO 策略实现同一签名即可替换,环境零改动。
random 依赖 legality_mask 在合法动作里均匀采样(Gumbel-argmax);
scripted 是极简运营 AI:补工人→建矿泵→采集→攒兵→过阈值全军进攻。
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .actions import (
    A_ATTACK,
    A_NOOP,
    a_build,
    a_build_camp,
    a_garrison_flag,
    a_garrison_node,
    a_harvest,
    a_plant_flag,
    a_recall_flag,
    a_research_line,
    a_train_infantry,
    a_train_worker,
    a_upgrade,
    legality_mask,
)
from .config import TYPE_CAMP, TYPE_HQ, TYPE_INFANTRY, TYPE_MINE, TYPE_PUMP, TYPE_WORKER, Config
from .map import BIG_DIST, MapData
from .state import ORDER_BUILD, ORDER_HARVEST, ORDER_IDLE, WorldState, owner_of_slots


def merge_actions(owner: jax.Array, a0: jax.Array, a1: jax.Array) -> jax.Array:
    return jnp.where(owner == 0, a0, a1)


def random_actions(state: WorldState, cfg: Config, mapdata: MapData,
                   owner: jax.Array, player: int, key: jax.Array) -> jax.Array:
    """在各自合法动作集里均匀采样(Gumbel-argmax;非法位 -inf)。"""
    legal = legality_mask(state, cfg, mapdata, owner)      # [N,A]
    logits = jnp.where(legal, 0.0, -jnp.inf)
    g = jax.random.gumbel(key, logits.shape)
    act = jnp.argmax(logits + g, axis=-1)
    mine = state.alive & (owner == player)
    return jnp.where(mine, act, A_NOOP).astype(jnp.int32)


def scripted_actions(state: WorldState, cfg: Config, mapdata: MapData,
                     owner: jax.Array, player: int, key: jax.Array) -> jax.Array:
    """极简规则 AI(纯 jnp,可进 scan)。非法选择靠 apply_orders 的掩码兜底成
    NOOP,本函数只负责「想做什么」。"""
    del key
    st = state
    n = cfg.n_total
    mine = st.alive & (owner == player)
    dist = jnp.asarray(mapdata.dist_fields)                # [G,H,W]
    from .state import cell_of
    cl = cell_of(st.pos)                                   # 连续坐标归格(v1.2)

    # ---- HQ:缺工人补工人 → 富余则升本(到 ai_base_level_target 止)→ 否则爆兵 ----
    from .config import TYPE_STRONGMAN as _TS
    from .config import TYPE_WAGON as _TW
    n_workers = jnp.sum(mine & ((st.etype == TYPE_WORKER)
                                | (st.etype == _TS) | (st.etype == _TW)))
    hq_slot_p = player * cfg.e_max
    base_lv = st.level[hq_slot_p].astype(jnp.int32)
    up_cost = jnp.asarray(
        [cfg.base_up_cost_ore, cfg.base_up_cost_water], jnp.int32)[:, base_lv]
    rich_for_base = jnp.all(st.resources[player]
                            >= up_cost + cfg.ai_upgrade_reserve)
    # v1.4:HQ3 起补大力士(≤2)、HQ5 起补马车(1),优先级在补工人之后、
    # 爆步兵之前(采集单位计入 n_workers 配额,不会无限膨胀)
    from .actions import a_train_unit
    n_sm = jnp.sum(mine & (st.etype == _TS))
    n_wg = jnp.sum(mine & (st.etype == _TW))
    tl_hq = cfg.train_level_by_type
    want_sm = (base_lv >= tl_hq[_TS]) & (n_sm < 2)
    want_wg = (base_lv >= tl_hq[_TW]) & (n_wg < 1)
    hq_act = jnp.where(n_workers < cfg.ai_worker_target,
                       a_train_worker(cfg),
                       jnp.where(want_wg, a_train_unit(_TW, cfg),
                                 jnp.where(want_sm, a_train_unit(_TS, cfg),
                                           a_train_infantry(cfg))))
    # 升级在途门:升级完成拍控制器还看着旧等级,会再下一单把基地顶超目标
    # 一级(600/400 白烧,晚期训练被饿——v1.4 覆盖局实测);只挡「升级任务
    # 在途」,不挡训练(训练中下升级单本就被掩码 NOOP,完成后自然轮到)
    from .config import BTASK_UPGRADE as _BU
    hq_act = jnp.where(rich_for_base & (base_lv < cfg.ai_base_level_target)
                       & (st.btype[hq_slot_p] != _BU),
                       a_upgrade(cfg), hq_act)

    # ---- 工人:一个去建(最近的无主点),其余采(最近的有余位己方点)----
    node_d = dist[:cfg.n_nodes, cl[:, 0], cl[:, 1]]        # [Nn,N]
    # 按「已指派数」(驻内 + 在途)控制派工,而不是只看驻内数——否则会把一堆
    # 工人堆到同一个点门口排队
    tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    assigned = (jnp.zeros(cfg.n_nodes, jnp.int32)
                .at[tn].add((mine & (st.order == ORDER_HARVEST)).astype(jnp.int32)))
    # 有己方工人在途去建的点不再重复派人(否则每 tick 都会把下一个空闲工人
    # 派去同一个点,制造成群结队的无效建造工)
    pending = (jnp.zeros(cfg.n_nodes, jnp.int32)
               .at[tn].add((mine & (st.order == ORDER_BUILD)).astype(jnp.int32)))
    claimable = ((st.node_owner == -1) & (st.node_build_timer == 0)
                 & (pending == 0))                                          # [Nn]
    ent_n = jnp.clip(st.node_ent.astype(jnp.int32), 0, cfg.n_total - 1)     # [Nn]
    caps = jnp.asarray(cfg.harvest_slots_by_level, jnp.int32)[
        jnp.clip(jnp.where(st.node_ent >= 0, st.level[ent_n], 0), 0, 7)]
    harvestable = ((st.node_owner == player) & (st.node_ent >= 0)
                   & (assigned < caps))                                     # [Nn]

    # 付得起才派建造者(audit v1.1 P2 活锁根因:无主点+付不起 → 同一空闲工人
    # 每 tick 被重复指派、动作被掩码、永不落入采集分支,水收入归零 1200+ tick。
    # 加可负担门控后,付不起时该工人自然滑入采集分支,收入恢复即可再派)
    from .actions import node_costs
    ncost = node_costs(cfg, mapdata)                       # [Nn,2]
    # 扩张分层预算(v1.3 双角 8 点地图):每类资源的**第一个**点是生存必需,
    # 裸成本就抢(预备金门控会掐死首座水泵→水收入归零,v1.1 活锁翻版);
    # 该类已有点后的**额外扩张**才要求留 ai_upgrade_reserve,否则扩张支出
    # 会把研发/升级永远挤在可负担线以下(实测全场 0 研发)
    from .config import RES_ORE
    ntype = jnp.asarray(mapdata.node_type)                 # [Nn]
    owned_t = st.node_owner == player
    have_same = jnp.where(ntype == RES_ORE,
                          jnp.any(owned_t & (ntype == RES_ORE)),
                          jnp.any(owned_t & (ntype != RES_ORE)))  # [Nn]
    need = ncost + jnp.where(have_same, cfg.ai_upgrade_reserve, 0)[:, None]
    afford_node = jnp.all(st.resources[player][None, :] >= need, axis=-1)  # [Nn]
    claimable = claimable & afford_node
    bd = jnp.where(claimable[:, None], node_d, BIG_DIST)
    build_k = jnp.argmin(bd, axis=0)
    has_build = jnp.any(claimable)
    hd = jnp.where(harvestable[:, None], node_d, BIG_DIST)
    harv_k = jnp.argmin(hd, axis=0)
    has_harv = jnp.any(harvestable)

    # v1.4:采集单位三类(工人/大力士/马车)同权派工
    from .config import TYPE_STRONGMAN, TYPE_WAGON
    is_harv_u = ((st.etype == TYPE_WORKER) | (st.etype == TYPE_STRONGMAN)
                 | (st.etype == TYPE_WAGON))
    idle_worker = mine & is_harv_u & (st.order == ORDER_IDLE) & ~st.inside
    # 「征调一个建造者」:优先空闲工人,没有就从采集线上拉一个(不拉矿内的)。
    # 只靠闲人的老逻辑在「afford 门控 + 满员派工」后会饿死——人人有活干,
    # 营/扩张永远没人去建(Phase 0 工具债的连带修正)。
    can_pull = (mine & is_harv_u & ~st.inside
                & ((st.order == ORDER_IDLE) | (st.order == ORDER_HARVEST)))
    pull_score = jnp.where(can_pull,
                           jnp.arange(n) + n * (st.order != ORDER_IDLE),
                           jnp.iinfo(jnp.int32).max)
    builder = jnp.argmin(pull_score)
    is_builder = (jnp.arange(n) == builder) & jnp.any(can_pull) & has_build

    worker_act = jnp.where(has_harv, a_harvest(0, cfg) + harv_k, A_NOOP)
    worker_act = jnp.where(is_builder, a_build(0) + build_k, worker_act)
    worker_act = jnp.where(idle_worker, worker_act, A_NOOP)

    # ---- 军队:全部战斗兵种(line_of_type≥0)攒够阈值全军 attack-move ----
    from .config import TYPE_DOG
    is_army = jnp.asarray(cfg.line_of_type, jnp.int32)[
        jnp.clip(st.etype.astype(jnp.int32), 0, 31)] >= 0
    n_army = jnp.sum(mine & is_army)
    attack_on = n_army >= cfg.ai_attack_threshold
    inf_act = jnp.where(attack_on, A_ATTACK, A_NOOP)

    # ---- 训练营(v1.4 八线):在**已解锁**的线里研最低的(掩码兜底非法),
    # 全部到顶则升营 ----
    from .config import TYPE_BARRACKS as _TB2
    from .config import TYPE_INFANTRY as _TI
    from .config import TYPE_OF_LINE
    line_lv = st.upgrades[player].astype(jnp.int32)        # [N_LINES]
    tl = jnp.asarray(cfg.train_level_by_type, jnp.int32)
    tol = jnp.asarray(TYPE_OF_LINE, jnp.int32)
    bar_lv_max = jnp.max(jnp.where(
        mine & (st.etype == _TB2) & (st.btype >= 0),
        st.level.astype(jnp.int32), 0))
    line_unlocked = (tol == _TI) | (bar_lv_max >= tl[tol])   # [N_LINES]
    line_score = jnp.where(line_unlocked, line_lv, 99)
    low_line = jnp.argmin(line_score).astype(jnp.int32)
    camp_act = jnp.where(
        line_score[low_line] < st.level.astype(jnp.int32),
        a_research_line(0, cfg) + low_line,
        a_upgrade(cfg))                                    # [N](逐营取自身 level 比较)

    # ---- 科技优先预留(v1.3):研发在排队(有营且低线未及营级)时,练兵先给
    # 研发留出该线的实际研发成本——否则练兵扣费(apply_orders)每 tick 都排在
    # 研发对账(paid_orders_pass)之前,水永远差一口,8 点地图实测全场 0 研发 ----
    from .config import TYPE_CAMP as _TC
    camp_lv_max = jnp.max(jnp.where(mine & (st.etype == _TC),
                                    st.level.astype(jnp.int32), 0))
    research_pending = ((line_score[low_line] < camp_lv_max)
                        & line_unlocked[low_line])
    lc = jnp.clip(line_lv[low_line], 0, 7)
    res_cost = jnp.asarray([jnp.asarray(cfg.line_res_cost_ore)[lc],
                            jnp.asarray(cfg.line_res_cost_water)[lc]])
    # 升本预留(v1.3 名额制配套):单点采集压到 3 人后练兵不再有余粮自然溢出,
    # 基地未到 ai_base_level_target 时练兵先留出升本成本+预备金,否则矿石永远
    # 在练兵线以下打转,3000 tick 全程升不了本(与研发预留同根同款)
    base_pending = base_lv < cfg.ai_base_level_target
    reserve = (jnp.where(research_pending, res_cost, 0)
               + jnp.where(base_pending, up_cost + cfg.ai_upgrade_reserve, 0))
    spare = st.resources[player] - reserve
    inf_ok = jnp.all(spare >= jnp.asarray(
        [cfg.infantry_cost_ore, cfg.infantry_cost_water]))
    hq_act = jnp.where((hq_act == a_train_infantry(cfg)) & ~inf_ok,
                       A_NOOP, hq_act)

    # ---- 兵营行为(v1.4):落后基地就升级(富余),否则训「己方数量最少的
    # 已解锁兵种」(healer/ram 各封顶 2,防一屋子辅助);成本过 spare 预留门 ----
    from .actions import a_train_dog
    from .actions import a_train_unit as _atu
    from .config import (
        TYPE_ARCHER,
        TYPE_HEALER,
        TYPE_HEAVY,
        TYPE_LCAV,
        TYPE_MAGE,
        TYPE_RAM,
    )
    # healer 排在 mage 前:argmin 平手取先者,否则「mage 阵亡归零」与 healer
    # 永远同分,healer 永不入队(覆盖局实测)
    bar_types = (TYPE_DOG, TYPE_ARCHER, TYPE_LCAV, TYPE_HEAVY,
                 TYPE_HEALER, TYPE_MAGE, TYPE_RAM)
    bar_caps = (99, 99, 99, 99, 2, 99, 2)
    tlv = cfg.train_level_by_type
    counts = jnp.stack([jnp.sum(mine & (st.etype == t)) for t in bar_types])
    counts = counts + jnp.where(
        counts >= jnp.asarray(bar_caps), 999, 0)             # 封顶型惩罚
    lv_vec = st.level.astype(jnp.int32)
    unlocked_t = jnp.stack(
        [lv_vec >= tlv[t] for t in bar_types], axis=1)       # [N, T]
    # 平手偏置(×8 保证计数差主导):高阶兵种优先——狗在乱战里反复归零,
    # 纯 argmin 平手恒取狗,ram(尾位)永不入队(覆盖局实测);healer 先于 mage
    tie_bias = jnp.asarray((6, 5, 4, 3, 1, 2, 0), jnp.int32)  # 对齐 bar_types
    t_score = (counts[None, :] * 8 + tie_bias[None, :]
               + jnp.where(unlocked_t, 0, 9999))
    choice = jnp.argmin(t_score, axis=1)                     # [N] 每兵营选型
    train_ids = jnp.asarray(
        [a_train_dog(cfg)] + [_atu(t, cfg) for t in bar_types[1:]], jnp.int32)
    tco_v = jnp.asarray(cfg.train_cost_ore_by_type, jnp.int32)
    tcw_v = jnp.asarray(cfg.train_cost_water_by_type, jnp.int32)
    chosen_t = jnp.asarray(bar_types, jnp.int32)[choice]
    train_ok = ((spare[0] >= tco_v[chosen_t]) & (spare[1] >= tcw_v[chosen_t]))
    bar_train_act = jnp.where(train_ok, train_ids[choice], A_NOOP)
    # 升级优先:兵营级 < 基地级且富余(含预备金)
    bar_up_cost = jnp.stack(
        [jnp.asarray(cfg.barracks_up_cost_ore)[jnp.clip(lv_vec, 0, 7)],
         jnp.asarray(cfg.barracks_up_cost_water)[jnp.clip(lv_vec, 0, 7)]], -1)
    rich_for_barup = jnp.all(
        st.resources[player][None, :] >= bar_up_cost + cfg.ai_upgrade_reserve,
        axis=-1)
    bar_act_full = jnp.where(rich_for_barup & (lv_vec < base_lv)
                             & (st.btype != _BU),
                             a_upgrade(cfg), bar_train_act)

    # ---- 建营:基地达标且没有己方营(含在建)时,征调一个工人就地起营
    # (同一征调池;若与扩张建造者撞同一人,建营优先——它在 act 覆盖链更后)----
    has_camp = jnp.any(mine & (st.etype == TYPE_CAMP))
    unlock = jnp.asarray(cfg.unlock_level_by_type, jnp.int32)
    base_ok = st.level[player * cfg.e_max] >= unlock[TYPE_CAMP]
    camp_afford = jnp.all(st.resources[player]
                          >= jnp.asarray([cfg.camp_cost_ore, cfg.camp_cost_water]))
    camp_builder = jnp.argmin(pull_score)
    is_camp_builder = ((jnp.arange(n) == camp_builder) & jnp.any(can_pull)
                       & base_ok & ~has_camp & camp_afford)

    # ---- 兵营:有了营再建兵营(科技优先),兵营空闲就训狗 ----
    from .actions import a_build_barracks, a_train_dog
    from .config import TYPE_BARRACKS
    has_bar = jnp.any(mine & (st.etype == TYPE_BARRACKS))
    bar_afford = jnp.all(st.resources[player] >= jnp.asarray(
        [cfg.barracks_cost_ore, cfg.barracks_cost_water]))
    bar_builder = jnp.argmin(pull_score)
    is_bar_builder = ((jnp.arange(n) == bar_builder) & jnp.any(can_pull)
                      & (st.level[player * cfg.e_max] >= unlock[TYPE_BARRACKS])
                      & has_camp & ~has_bar & bar_afford & ~is_camp_builder)

    # ---- 迫击炮(v1.4):基地 3 级、有兵营后建一座(cap 1)----
    from .actions import a_build_mortar
    from .config import TYPE_MORTAR
    has_mortar = jnp.any(mine & (st.etype == TYPE_MORTAR))
    mor_afford = jnp.all(st.resources[player] >= jnp.asarray(
        [cfg.mortar_cost_ore, cfg.mortar_cost_water]))
    mr_builder = jnp.argmin(pull_score)
    is_mr_builder = ((jnp.arange(n) == mr_builder) & jnp.any(can_pull)
                     & (st.level[player * cfg.e_max] >= unlock[TYPE_MORTAR])
                     & has_bar & ~has_mortar & mor_afford
                     & ~is_camp_builder & ~is_bar_builder)

    # ---- 哨塔:有兵营后建到数量上限为止(v1.4 多塔:上限挂基地等级)----
    from .actions import a_build_tower
    from .config import TYPE_TOWER
    n_towers = jnp.sum(mine & (st.etype == TYPE_TOWER))
    twr_cap = jnp.asarray(cfg.tower_cap_by_hq_level, jnp.int32)[
        jnp.clip(base_lv, 0, 7)]
    tower_afford = jnp.all(st.resources[player] >= jnp.asarray(
        [cfg.tower_cost_ore, cfg.tower_cost_water]))
    tw_builder = jnp.argmin(pull_score)
    is_tw_builder = ((jnp.arange(n) == tw_builder) & jnp.any(can_pull)
                     & (st.level[player * cfg.e_max] >= unlock[TYPE_TOWER])
                     & has_bar & (n_towers < twr_cap) & tower_afford
                     & ~is_camp_builder & ~is_bar_builder & ~is_mr_builder)

    # ---- 驻守/军旗(v1.3):有兵营后,槽号最小的 2 只狗驻守离家最远的已建
    # 矿泵点;第 3 只狗在当前位置插旗(驻守途中/圈内插,位置自然靠前线),
    # 此后新狗驻守旗 0;总攻期间兵营撤旗 0、全军照旧 A_ATTACK(act 覆盖链里
    # ATTACK 在后,驻守被覆盖)----
    is_my_dog = mine & (st.etype == TYPE_DOG)
    n_dogs = jnp.sum(is_my_dog)
    dog_rank = jnp.cumsum(is_my_dog.astype(jnp.int32)) - 1   # 槽号序内第几只狗
    npos = jnp.asarray(mapdata.node_pos)
    nd_home = dist[cfg.n_nodes + player, npos[:, 0], npos[:, 1]]  # [Nn] 到己方 HQ
    owned_built = (st.node_owner == player) & (st.node_ent >= 0)
    far_k = jnp.argmax(jnp.where(owned_built, nd_home, -1)).astype(jnp.int32)
    has_far = jnp.any(owned_built)
    any_flag = jnp.any(st.flag_active[player])
    dog_act = jnp.where(is_my_dog & (dog_rank < 2) & has_bar & (n_dogs >= 2)
                        & has_far, a_garrison_node(0, cfg) + far_k, A_NOOP)
    # 第一只狗先插旗(v1.4 鲁棒化:旧「第 3 只狗插旗」在总攻波次里凑不齐
    # 3 只并发狗,插旗时点贴着终局、对 jit 融合浮点微差过敏;插旗免费即时,
    # 一拍完成后该狗下 tick 自然回到驻守分支)
    dog_act = jnp.where(is_my_dog & (dog_rank == 0) & ~any_flag,
                        a_plant_flag(cfg), dog_act)
    dog_act = jnp.where(is_my_dog & (dog_rank >= 2) & any_flag,
                        a_garrison_flag(0, cfg), dog_act)

    # ---- 矿/泵:库存 ≥ 升级成本+储备 才升(audit v1.1 P2:只查储备不查成本
    # 会把「留军费」的意图打穿,曾是 seed12 水危机的助燃剂)----
    is_node_b = (st.etype == TYPE_MINE) | (st.etype == TYPE_PUMP)
    node_lv = st.level.astype(jnp.int32)
    node_up_cost = jnp.stack(
        [jnp.asarray(cfg.node_up_cost_ore)[node_lv],
         jnp.asarray(cfg.node_up_cost_water)[node_lv]], -1)     # [N,2]
    rich_for_node = jnp.all(
        st.resources[player][None, :] >= node_up_cost + cfg.ai_upgrade_reserve,
        axis=-1)                                                # [N]
    node_act = jnp.where(rich_for_node, a_upgrade(cfg), A_NOOP)

    act = jnp.full(n, A_NOOP, jnp.int32)
    act = jnp.where(mine & (st.etype == TYPE_HQ), hq_act, act)
    act = jnp.where(idle_worker, worker_act, act)
    act = jnp.where(is_camp_builder, a_build_camp(cfg), act)
    act = jnp.where(is_bar_builder, a_build_barracks(cfg), act)
    act = jnp.where(is_tw_builder, a_build_tower(cfg), act)
    act = jnp.where(is_mr_builder, a_build_mortar(cfg), act)
    act = jnp.where(mine & (st.etype == TYPE_BARRACKS), bar_act_full, act)
    act = jnp.where(mine & (st.etype == TYPE_CAMP), camp_act, act)
    act = jnp.where(mine & is_node_b, node_act, act)
    act = jnp.where(mine & is_army, inf_act, act)
    # 驻守只在非总攻期下达(总攻 tick 起 ATTACK 覆盖全军);**插旗例外**:
    # 免费即时、只损失一拍行军,总攻期也照插——否则「第 3 只狗凑齐时恰逢
    # 全军总攻」的时间线永远不插旗,涌现测试对浮点微差(jit 融合)过敏
    # (v1.4 保绿修正,scripted 行为微调不属引擎规则)。撤旗挂兵营,
    # 覆盖训狗一拍(下 tick 旗已灭、掩码自动放行训狗)
    is_plant_a = dog_act == a_plant_flag(cfg)
    act = jnp.where((dog_act != A_NOOP) & (~attack_on | is_plant_a),
                    dog_act, act)
    act = jnp.where(mine & (st.etype == TYPE_BARRACKS) & attack_on
                    & st.flag_active[player, 0], a_recall_flag(0, cfg), act)

    # ---- 让路:空闲单位贴在自家 HQ 正邻格(=卸货格)上会堵死运矿工人,
    # 命令它朝「远离 HQ」的方向挪一格(实测 4 个待命步兵站满卸货环锁死经济)----
    hq_field = dist[cfg.n_nodes + player]                  # [H,W] 静态
    on_ring = hq_field[cl[:, 0], cl[:, 1]] == 1
    is_idle_unit = (mine & ~st.inside & (st.order == ORDER_IDLE)
                    & (is_harv_u | (st.etype == TYPE_INFANTRY)))
    dirs = jnp.asarray([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)
    cand = cl[:, None, :] + dirs[None]                     # [N,4,2]
    cc = jnp.clip(cand, 0, jnp.asarray([cfg.grid_h - 1, cfg.grid_w - 1]))
    away = hq_field[cc[:, :, 0], cc[:, :, 1]]              # 越大越远
    best_dir = jnp.argmax(away, axis=1)
    step_off = is_idle_unit & on_ring & (act == A_NOOP)
    act = jnp.where(step_off, 3 + best_dir, act)           # 3=A_MOVE0
    return act


def make_controller(name: str, player: int, cfg: Config, mapdata: MapData):
    """按名字构建 `fn(state, key) -> actions[N]`。"""
    owner = owner_of_slots(cfg)
    if name == "random":
        return functools.partial(random_actions, cfg=cfg, mapdata=mapdata,
                                 owner=owner, player=player)
    if name == "scripted":
        return functools.partial(scripted_actions, cfg=cfg, mapdata=mapdata,
                                 owner=owner, player=player)
    raise ValueError(f"未知控制器: {name!r}(可选 random / scripted)")


def make_joint_controller(name0: str, name1: str, cfg: Config, mapdata: MapData):
    """两家控制器合并成 `fn(state, key) -> actions[N]`(供 make_scan 使用)。"""
    owner = owner_of_slots(cfg)
    c0 = make_controller(name0, 0, cfg, mapdata)
    c1 = make_controller(name1, 1, cfg, mapdata)

    def joint(state: WorldState, key: jax.Array) -> jax.Array:
        k0, k1 = jax.random.split(key)
        return merge_actions(owner, c0(state, key=k0), c1(state, key=k1))

    return joint
