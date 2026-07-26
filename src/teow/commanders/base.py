"""StrategyProfile 驱动的多风格脚本指挥官(v1.8 核心)。

`commander_actions(state,cfg,mapdata,owner,player,key,profile)->int32[N]` 是
`controller.scripted_actions` 的**参数化泛化**:同一份 branchless-JAX 宏观管线,
由静态 profile(bake 进 make_controller 的 partial,与 player 同)选出不同 jnp 计算。

关键纪律:
- **profile 是静态闭包**——对 profile 字段的 Python `if/elif` 在 trace 期解析,
  不产生 traced 控制流;只有依赖 state 的量走 `jnp.where`。这让「多风格」既丰富又
  可进 `jax.lax.scan`/vmap。
- profile == PROFILES["balanced"](= cfg.ai_* 默认值)时,本函数逐算子复现
  scripted_actions(见每处 lever 的「balanced 恒等」注释);"scripted" 名仍路由到
  原 scripted_actions,回归测试 test_scripted_v16 不受影响。
- stochastic>0 时用线程 key(不 del),同 seed 仍复现;否则 del key 保持决定论。

指挥官启发式常量是**版本控制代码常量**(AI 策略非引擎平衡数字,不进 config;
profile.py 同款约定),集中在下方 `_` 前缀模块常量,便于 grep/校准。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..actions import (
    A_ATTACK,
    A_MOVE0,
    A_NOOP,
    a_build,
    a_build_barracks,
    a_build_camp,
    a_build_defense,
    a_build_mortar,
    a_build_tower,
    a_garrison_flag,
    a_garrison_node,
    a_harvest,
    a_plant_flag,
    a_recall_flag,
    a_research_line,
    a_train_dog,
    a_train_infantry,
    a_train_unit,
    a_train_v16,
    a_train_worker,
    a_upgrade,
    node_costs,
)
from ..config import (
    BTASK_UPGRADE,
    RES_ORE,
    TYPE_AIRSHIP,
    TYPE_ARCHER,
    TYPE_BARRACKS,
    TYPE_CAMP,
    TYPE_CATAPULT,
    TYPE_DOG,
    TYPE_DRAGON,
    TYPE_FLAMER,
    TYPE_HEALER,
    TYPE_HEAVY,
    TYPE_HQ,
    TYPE_INFANTRY,
    TYPE_LANDMINE,
    TYPE_LASER,
    TYPE_LCAV,
    TYPE_MAGE,
    TYPE_MAGETOWER,
    TYPE_MINE,
    TYPE_MORTAR,
    TYPE_PUMP,
    TYPE_RAM,
    TYPE_STRONGMAN,
    TYPE_TOWER,
    TYPE_WAGON,
    TYPE_WORKER,
    Config,
)
from ..map import BIG_DIST, MapData
from ..state import ORDER_BUILD, ORDER_HARVEST, ORDER_IDLE, WorldState, cell_of
from .profile import StrategyProfile

# ---- 指挥官启发式常量(非引擎平衡数字;版本控制代码常量,与 profile 同类)----
_ADAPT_WIPE_ARMY = 2       # 战斗单位 < 此数视为「主力被清空」(自适应/allin 触发口径)
_ADAPT_WORKER_BUMP = 4     # setback 后 effective worker_target 提升(转宏观运营)
_ADAPT_THRESH_BUMP = 6     # setback 后 effective attack_threshold 提升(降激进,停止空转)
_ADAPT_EARLY_FRAC = 3      # 早期窗口 = gate_open_tick // 此值;过窗口才允许自适应回退
_COMP_BIAS_MAG = 16        # comp_bias 家族偏置量(≈2 单位计数差;仍保留混编不塌缩)
_COUNTER_BIAS_MAG = 16     # counter 最佳响应偏置量
_STOCH_NOISE = 6.0         # chaos 兵营选型噪声幅度(≈计数差量级,不越过封顶/未解锁罚)
_DEF_HOLD_FRAC = 3         # defensive 延迟出击到 gate_open_tick // 此值 之后(定性「守」)
_ECON_EXPAND_RESERVE_CAP = 20   # economy 扩张预备金上限(压低→多铺矿泵)
_TECH_BASEUP_RESERVE_CAP = 40   # air/defense 升本预备金上限(压低→快 climb 高级兵种)

# 兵营选型的兵种顺序(与 base 内 bar_types 对齐;family bias 用其下标)
_BAR_TYPES = (TYPE_DOG, TYPE_ARCHER, TYPE_LCAV, TYPE_HEAVY, TYPE_HEALER,
              TYPE_MAGE, TYPE_RAM, TYPE_CATAPULT, TYPE_AIRSHIP, TYPE_DRAGON)
_BAR_IDX = {t: i for i, t in enumerate(_BAR_TYPES)}

# comp_bias 家族偏好:favored 兵种在 tie 上加负偏置(argmin→更优先)。
# 静态家族(counter 走动态最佳响应,单列)。
_COMP_FAMILY = {
    "dogs": (TYPE_DOG, TYPE_LCAV),        # 快速骚扰
    "ranged": (TYPE_ARCHER, TYPE_MAGE),   # 风筝
    "heavy": (TYPE_HEAVY, TYPE_LCAV),     # 硬近战(重甲肉盾 + 轻骑冲锋)
    "air": (TYPE_AIRSHIP, TYPE_DRAGON),   # 制空
    "siege": (TYPE_RAM, TYPE_CATAPULT),   # 拆家
}

# tech_focus → 建造征调「组优先级」。camp/barracks 恒居首(科技依赖:营→兵营),
# 仅重排 {mortar, def, tower}。balanced 组序 == scripted 现链(逐算子恒等)。
_BUILD_ORDER = {
    "balanced": ("camp", "barracks", "mortar", "def", "tower"),
    "military": ("camp", "barracks", "mortar", "tower", "def"),
    "air": ("camp", "barracks", "tower", "def", "mortar"),
    "defense": ("camp", "barracks", "tower", "mortar", "def"),
    "economy": ("camp", "barracks", "mortar", "def", "tower"),
}


def _family_delta(profile: StrategyProfile, st: WorldState, owner: jax.Array,
                  player: int) -> jax.Array:
    """int32 [10]:兵营选型的家族偏置向量(对齐 _BAR_TYPES)。
    balanced→全 0(t_score 逐算子恒等 scripted);counter→按敌方最强家族动态克制。"""
    zero = jnp.zeros(len(_BAR_TYPES), jnp.int32)
    cb = profile.comp_bias
    if cb == "balanced":
        return zero
    if cb == "counter":
        # 上帝视角读全场敌方存活单位,数四个家族桶,偏向其克制。全 branchless。
        enemy = st.alive & (owner != player)
        et = st.etype
        c_air = jnp.sum(enemy & ((et == TYPE_AIRSHIP) | (et == TYPE_DRAGON)))
        c_heavy = jnp.sum(enemy & ((et == TYPE_HEAVY) | (et == TYPE_RAM)))
        c_melee = jnp.sum(enemy & ((et == TYPE_DOG) | (et == TYPE_LCAV)
                                   | (et == TYPE_INFANTRY)))
        c_ranged = jnp.sum(enemy & ((et == TYPE_ARCHER) | (et == TYPE_MAGE)))
        counts4 = jnp.stack([c_air, c_heavy, c_melee, c_ranged])
        best = jnp.argmax(counts4)
        has_enemy = jnp.max(counts4) > 0
        m = _COUNTER_BIAS_MAG
        # 行对齐 _BAR_TYPES:DOG0 ARCHER1 LCAV2 HEAVY3 HEALER4 MAGE5 RAM6 CAT7 AIR8 DRA9
        cbm = jnp.asarray([
            [0, -m, 0, 0, 0, 0, 0, 0, 0, 0],    # vs 空军 → 弓箭手(对空)
            [0, 0, 0, 0, 0, -m, 0, 0, 0, 0],    # vs 重甲 → 法师(魔法无视护甲)
            [0, -m, 0, 0, 0, 0, 0, -m, 0, 0],   # vs 群体近战 → 弓 + 投石(远程/AoE)
            [0, 0, -m, -m, 0, 0, 0, 0, 0, 0],   # vs 远程 → 轻骑 + 重甲(贴脸/抗打)
        ], jnp.int32)
        return jnp.where(has_enemy, cbm[best], zero)
    # 静态家族:trace 期构造常量偏置向量
    arr = [0] * len(_BAR_TYPES)
    for t in _COMP_FAMILY[cb]:
        arr[_BAR_IDX[t]] = -_COMP_BIAS_MAG
    return jnp.asarray(arr, jnp.int32)


def commander_actions(state: WorldState, cfg: Config, mapdata: MapData,
                      owner: jax.Array, player: int, key: jax.Array,
                      profile: StrategyProfile) -> jax.Array:
    """profile 参数化的脚本指挥官(纯 jnp,可进 scan)。非法选择靠 apply_orders
    掩码兜底成 NOOP,本函数只表达「想做什么」。"""
    st = state
    n = cfg.n_total
    mine = st.alive & (owner == player)
    dist = jnp.asarray(mapdata.dist_fields)                # [G,H,W]
    cl = cell_of(st.pos)                                   # 连续坐标归格
    et = jnp.clip(st.etype.astype(jnp.int32), 0, 31)

    # ---- profile 静态量(trace 期解析)----
    worker_target = profile.worker_target
    attack_threshold = profile.attack_threshold
    base_level_target = profile.base_level_target
    upgrade_reserve = profile.upgrade_reserve
    focus = profile.tech_focus
    # tech_focus 派生预备金(balanced 恒等于 upgrade_reserve):
    #   economy 扩张更便宜(多铺矿泵);air/defense 升本更便宜(快爬高级兵种)
    expansion_reserve = (min(upgrade_reserve, _ECON_EXPAND_RESERVE_CAP)
                         if focus == "economy" else upgrade_reserve)
    base_up_reserve = (min(upgrade_reserve, _TECH_BASEUP_RESERVE_CAP)
                       if focus in ("air", "defense") else upgrade_reserve)
    use_key = profile.stochastic > 0
    if not use_key:
        del key                                            # 保持决定论(现 scripted 行为)

    # ---- 军队规模(自适应/攻击触发都要,提前算;值与位置无关)----
    is_army = jnp.asarray(cfg.is_combat_by_type, bool)[et]
    n_army = jnp.sum(mine & is_army)

    # ---- lever 4 自适应回退(反「无意义空转」):主力被清空且过早期窗口 → 抬
    # 工人目标 + 降激进,转宏观运营。adaptive 静态门,setback 依赖 state。----
    if profile.adaptive:
        past_early = st.tick > (cfg.gate_open_tick // _ADAPT_EARLY_FRAC)
        setback = past_early & (n_army < _ADAPT_WIPE_ARMY)
        wt_bump = jnp.where(setback, _ADAPT_WORKER_BUMP, 0)
        at_bump = jnp.where(setback, _ADAPT_THRESH_BUMP, 0)
    else:
        wt_bump = 0
        at_bump = 0
    eff_worker_target = worker_target + wt_bump
    eff_attack_threshold = attack_threshold + at_bump

    # ---- HQ:缺工人补工人 → 富余则升本(到 base_level_target 止)→ 否则爆兵 ----
    n_workers = jnp.sum(mine & ((st.etype == TYPE_WORKER)
                                | (st.etype == TYPE_STRONGMAN)
                                | (st.etype == TYPE_WAGON)))
    hq_slot_p = player * cfg.e_max
    base_lv = st.level[hq_slot_p].astype(jnp.int32)
    up_cost = jnp.asarray(
        [cfg.base_up_cost_ore, cfg.base_up_cost_water], jnp.int32)[:, base_lv]
    rich_for_base = jnp.all(st.resources[player] >= up_cost + base_up_reserve)
    n_sm = jnp.sum(mine & (st.etype == TYPE_STRONGMAN))
    n_wg = jnp.sum(mine & (st.etype == TYPE_WAGON))
    tl_hq = cfg.train_level_by_type
    want_sm = (base_lv >= tl_hq[TYPE_STRONGMAN]) & (n_sm < 2)
    want_wg = (base_lv >= tl_hq[TYPE_WAGON]) & (n_wg < 1)
    hq_act = jnp.where(n_workers < eff_worker_target,
                       a_train_worker(cfg),
                       jnp.where(want_wg, a_train_unit(TYPE_WAGON, cfg),
                                 jnp.where(want_sm, a_train_unit(TYPE_STRONGMAN, cfg),
                                           a_train_infantry(cfg))))
    hq_act = jnp.where(rich_for_base & (base_lv < base_level_target)
                       & (st.btype[hq_slot_p] != BTASK_UPGRADE),
                       a_upgrade(cfg), hq_act)

    # ---- 工人:一个去建(最近的无主点),其余采(最近的有余位己方点)----
    node_d = dist[:cfg.n_nodes, cl[:, 0], cl[:, 1]]        # [Nn,N]
    tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    assigned = (jnp.zeros(cfg.n_nodes, jnp.int32)
                .at[tn].add((mine & (st.order == ORDER_HARVEST)).astype(jnp.int32)))
    pending = (jnp.zeros(cfg.n_nodes, jnp.int32)
               .at[tn].add((mine & (st.order == ORDER_BUILD)).astype(jnp.int32)))
    claimable = ((st.node_owner == -1) & (st.node_build_timer == 0)
                 & (pending == 0))                                          # [Nn]
    ent_n = jnp.clip(st.node_ent.astype(jnp.int32), 0, cfg.n_total - 1)     # [Nn]
    caps = jnp.asarray(cfg.harvest_slots_by_level, jnp.int32)[
        jnp.clip(jnp.where(st.node_ent >= 0, st.level[ent_n], 0), 0, 7)]
    harvestable = ((st.node_owner == player) & (st.node_ent >= 0)
                   & (assigned < caps))                                     # [Nn]
    ncost = node_costs(cfg, mapdata)                       # [Nn,2]
    ntype = jnp.asarray(mapdata.node_type)                 # [Nn]
    owned_t = st.node_owner == player
    have_same = jnp.where(ntype == RES_ORE,
                          jnp.any(owned_t & (ntype == RES_ORE)),
                          jnp.any(owned_t & (ntype != RES_ORE)))  # [Nn]
    # lever 2:扩张预备金随 tech_focus(economy 压低→多铺;balanced 恒等 upgrade_reserve)
    need = ncost + jnp.where(have_same, expansion_reserve, 0)[:, None]
    afford_node = jnp.all(st.resources[player][None, :] >= need, axis=-1)  # [Nn]
    claimable = claimable & afford_node
    bd = jnp.where(claimable[:, None], node_d, BIG_DIST)
    build_k = jnp.argmin(bd, axis=0)
    has_build = jnp.any(claimable)
    hd = jnp.where(harvestable[:, None], node_d, BIG_DIST)
    harv_k = jnp.argmin(hd, axis=0)
    has_harv = jnp.any(harvestable)

    is_harv_u = ((st.etype == TYPE_WORKER) | (st.etype == TYPE_STRONGMAN)
                 | (st.etype == TYPE_WAGON))
    idle_worker = mine & is_harv_u & (st.order == ORDER_IDLE) & ~st.inside
    can_pull = (mine & is_harv_u & ~st.inside
                & ((st.order == ORDER_IDLE) | (st.order == ORDER_HARVEST)))
    pull_score = jnp.where(can_pull,
                           jnp.arange(n) + n * (st.order != ORDER_IDLE),
                           jnp.iinfo(jnp.int32).max)
    builder = jnp.argmin(pull_score)
    is_builder_slot = (jnp.arange(n) == builder) & jnp.any(can_pull)
    # 经济扩张建造者:征调工人去建最近的可占矿泵(与 camp/兵营链共用同一征调
    # 工人;冲突时 camp/兵营在 act 覆盖链更后→建筑优先,矿泵扩张顺延)
    is_builder = is_builder_slot & has_build

    worker_act = jnp.where(has_harv, a_harvest(0, cfg) + harv_k, A_NOOP)
    worker_act = jnp.where(is_builder, a_build(0) + build_k, worker_act)
    worker_act = jnp.where(idle_worker, worker_act, A_NOOP)

    # ---- lever 5 军队攻击触发:eff_attack_threshold + aggression 定性 ----
    attack_on = n_army >= eff_attack_threshold
    agg = profile.aggression
    if agg == "defensive":
        # 守:攒够大军(高 threshold)且过 hold 窗口才出击,否则守家
        attack_on = attack_on & (st.tick >= cfg.gate_open_tick // _DEF_HOLD_FRAC)
    elif agg == "allin":
        # 倾家一波:门开(sudden-death)后只要还有兵就无条件压上(硬 commit)
        attack_on = attack_on | ((st.tick >= cfg.gate_open_tick)
                                 & (n_army >= _ADAPT_WIPE_ARMY))
    inf_act = jnp.where(attack_on, A_ATTACK, A_NOOP)

    # ---- 训练营(v1.4 八线):已解锁里研最低的,全到顶则升营 ----
    from ..config import TYPE_OF_LINE
    line_lv = st.upgrades[player].astype(jnp.int32)        # [N_LINES]
    tl = jnp.asarray(cfg.train_level_by_type, jnp.int32)
    tol = jnp.asarray(TYPE_OF_LINE, jnp.int32)
    bar_lv_max = jnp.max(jnp.where(
        mine & (st.etype == TYPE_BARRACKS) & (st.btype >= 0),
        st.level.astype(jnp.int32), 0))
    line_cap = jnp.asarray(cfg.line_cap_by_line, jnp.int32)
    line_unlocked = (((tol == TYPE_INFANTRY) | (bar_lv_max >= tl[tol]))
                     & (line_lv < line_cap))
    line_score = jnp.where(line_unlocked, line_lv, 99)
    low_line = jnp.argmin(line_score).astype(jnp.int32)
    rid_tbl = jnp.asarray([a_research_line(li, cfg) for li in range(len(TYPE_OF_LINE))],
                          jnp.int32)
    camp_act = jnp.where(
        line_score[low_line] < st.level.astype(jnp.int32),
        rid_tbl[low_line],
        a_upgrade(cfg))                                    # [N] 逐营取自身 level 比较

    # ---- 科技/升本预留(练兵先给研发/升本留成本+预备金)----
    camp_lv_max = jnp.max(jnp.where(mine & (st.etype == TYPE_CAMP),
                                    st.level.astype(jnp.int32), 0))
    research_pending = ((line_score[low_line] < camp_lv_max)
                        & line_unlocked[low_line])
    lc = jnp.clip(line_lv[low_line], 0, 7)
    res_cost = jnp.asarray([jnp.asarray(cfg.line_res_cost_ore)[lc],
                            jnp.asarray(cfg.line_res_cost_water)[lc]])
    base_pending = base_lv < base_level_target
    reserve = (jnp.where(research_pending, res_cost, 0)
               + jnp.where(base_pending, up_cost + upgrade_reserve, 0))
    spare = st.resources[player] - reserve
    inf_ok = jnp.all(spare >= jnp.asarray(
        [cfg.infantry_cost_ore, cfg.infantry_cost_water]))
    hq_act = jnp.where((hq_act == a_train_infantry(cfg)) & ~inf_ok,
                       A_NOOP, hq_act)

    # ---- 兵营行为:落后基地就升级(富余),否则训「己方数量最少的已解锁兵种」
    # (lever 3 家族偏置 + lever 6 stochastic 噪声叠加到选型 t_score)----
    bar_caps = (99, 99, 99, 99, 2, 99, 2, 2, 1, 1)
    tlv = cfg.train_level_by_type
    counts = jnp.stack([jnp.sum(mine & (st.etype == t)) for t in _BAR_TYPES])
    counts = counts + jnp.where(
        counts >= jnp.asarray(bar_caps), 999, 0)             # 封顶型惩罚
    lv_vec = st.level.astype(jnp.int32)
    unlocked_t = jnp.stack(
        [lv_vec >= tlv[t] for t in _BAR_TYPES], axis=1)      # [N, T]
    tie_bias = jnp.asarray((9, 8, 7, 6, 4, 5, 3, 2, 1, 0), jnp.int32)  # 高阶优先
    family_delta = _family_delta(profile, st, owner, player)  # [10];balanced→0
    t_score = (counts[None, :] * 8 + tie_bias[None, :] + family_delta[None, :]
               + jnp.where(unlocked_t, 0, 9999))
    if use_key:
        # chaos:选型加均匀噪声,重复对局(不同 seed)分化兵种构成
        t_score = t_score.astype(jnp.float32) + jax.random.uniform(
            key, t_score.shape, dtype=jnp.float32) * _STOCH_NOISE
    choice = jnp.argmin(t_score, axis=1)                     # [N] 每兵营选型
    train_ids = jnp.asarray(
        [a_train_dog(cfg)] + [a_train_unit(t, cfg) for t in _BAR_TYPES[1:7]]
        + [a_train_v16(t, cfg) for t in _BAR_TYPES[7:]], jnp.int32)
    tco_v = jnp.asarray(cfg.train_cost_ore_by_type, jnp.int32)
    tcw_v = jnp.asarray(cfg.train_cost_water_by_type, jnp.int32)
    chosen_t = jnp.asarray(_BAR_TYPES, jnp.int32)[choice]
    train_ok = ((spare[0] >= tco_v[chosen_t]) & (spare[1] >= tcw_v[chosen_t]))
    bar_train_act = jnp.where(train_ok, train_ids[choice], A_NOOP)
    bar_up_cost = jnp.stack(
        [jnp.asarray(cfg.barracks_up_cost_ore)[jnp.clip(lv_vec, 0, 7)],
         jnp.asarray(cfg.barracks_up_cost_water)[jnp.clip(lv_vec, 0, 7)]], -1)
    rich_for_barup = jnp.all(
        st.resources[player][None, :] >= bar_up_cost + upgrade_reserve,
        axis=-1)
    bar_act_full = jnp.where(rich_for_barup & (lv_vec < base_lv)
                             & (st.btype != BTASK_UPGRADE),
                             a_upgrade(cfg), bar_train_act)

    # ---- lever 2 建造征调链:同一征调工人,按 tech_focus 组优先级派 ----
    unlock = jnp.asarray(cfg.unlock_level_by_type, jnp.int32)
    base_lv_slot = st.level[player * cfg.e_max]             # int8 标量(解锁门用)
    has_camp = jnp.any(mine & (st.etype == TYPE_CAMP))
    has_bar = jnp.any(mine & (st.etype == TYPE_BARRACKS))
    has_mortar = jnp.any(mine & (st.etype == TYPE_MORTAR))
    n_towers = jnp.sum(mine & (st.etype == TYPE_TOWER))
    twr_cap = jnp.asarray(cfg.tower_cap_by_hq_level, jnp.int32)[
        jnp.clip(base_lv, 0, 7)]

    def _afford(co, cw):
        return jnp.all(st.resources[player] >= jnp.asarray([co, cw]))

    # 各建筑「裸资格」(不含互斥;互斥由下方 claimed 累积器统一处理,
    # 逐算子复现 scripted 现链的 ~is_X_builder 排除语义)。
    raw_camp = (is_builder_slot & (base_lv_slot >= unlock[TYPE_CAMP])
                & ~has_camp & _afford(cfg.camp_cost_ore, cfg.camp_cost_water))
    if focus == "economy":
        # economy:经济成型前不开军事(先铺满工人再动兵营链);但加 tick 兜底——
        # 工人被压制时纯 worker 门会**永久锁死**军事(反「无意义空转」铁律),
        # 到 gate//2 无条件放行军事,保证 boomer 不会因骚扰而躺死。
        raw_camp = raw_camp & ((n_workers >= worker_target)
                               | (st.tick >= cfg.gate_open_tick // 2))
    raw_bar = (is_builder_slot & (base_lv_slot >= unlock[TYPE_BARRACKS])
               & has_camp & ~has_bar
               & _afford(cfg.barracks_cost_ore, cfg.barracks_cost_water))
    raw_mor = (is_builder_slot & (base_lv_slot >= unlock[TYPE_MORTAR])
               & has_bar & ~has_mortar
               & _afford(cfg.mortar_cost_ore, cfg.mortar_cost_water))
    raw_tw = (is_builder_slot & (base_lv_slot >= unlock[TYPE_TOWER])
              & has_bar & (n_towers < twr_cap)
              & _afford(cfg.tower_cost_ore, cfg.tower_cost_water))
    def_specs = ((TYPE_MAGETOWER, 1, (cfg.magetower_cost_ore, cfg.magetower_cost_water)),
                 (TYPE_LANDMINE, 5, (cfg.landmine_cost_ore, cfg.landmine_cost_water)),
                 (TYPE_FLAMER, 1, (cfg.flamer_cost_ore, cfg.flamer_cost_water)),
                 (TYPE_LASER, 1, (cfg.laser_cost_ore, cfg.laser_cost_water)))
    def_entries = []
    for dt, cap_d, (co, cw) in def_specs:
        n_dt = jnp.sum(mine & (st.etype == dt))
        raw_d = (is_builder_slot & (base_lv_slot >= unlock[dt])
                 & (n_dt < cap_d) & _afford(co, cw))
        def_entries.append((a_build_defense(dt, cfg), raw_d))
    groups = {
        "camp": [(a_build_camp(cfg), raw_camp)],
        "barracks": [(a_build_barracks(cfg), raw_bar)],
        "mortar": [(a_build_mortar(cfg), raw_mor)],
        "def": def_entries,
        "tower": [(a_build_tower(cfg), raw_tw)],
    }
    order = _BUILD_ORDER[focus]
    claimed = jnp.zeros(n, bool)
    build_act = jnp.full(n, A_NOOP, jnp.int32)
    for g in order:
        for action_id, raw in groups[g]:
            pick = raw & ~claimed
            build_act = jnp.where(pick, action_id, build_act)
            claimed = claimed | pick
    is_conscript = claimed

    # ---- 驻守/军旗(v1.3;不受 profile 影响,沿用 scripted)----
    is_my_dog = mine & (st.etype == TYPE_DOG)
    n_dogs = jnp.sum(is_my_dog)
    dog_rank = jnp.cumsum(is_my_dog.astype(jnp.int32)) - 1
    npos = jnp.asarray(mapdata.node_pos)
    nd_home = dist[cfg.n_nodes + player, npos[:, 0], npos[:, 1]]  # [Nn] 到己方 HQ
    owned_built = (st.node_owner == player) & (st.node_ent >= 0)
    far_k = jnp.argmax(jnp.where(owned_built, nd_home, -1)).astype(jnp.int32)
    has_far = jnp.any(owned_built)
    any_flag = jnp.any(st.flag_active[player])
    dog_act = jnp.where(is_my_dog & (dog_rank < 2) & has_bar & (n_dogs >= 2)
                        & has_far, a_garrison_node(0, cfg) + far_k, A_NOOP)
    dog_act = jnp.where(is_my_dog & (dog_rank == 0) & ~any_flag,
                        a_plant_flag(cfg), dog_act)
    dog_act = jnp.where(is_my_dog & (dog_rank >= 2) & any_flag,
                        a_garrison_flag(0, cfg), dog_act)

    # ---- 矿/泵:库存 ≥ 升级成本+储备 才升 ----
    is_node_b = (st.etype == TYPE_MINE) | (st.etype == TYPE_PUMP)
    node_lv = st.level.astype(jnp.int32)
    node_up_cost = jnp.stack(
        [jnp.asarray(cfg.node_up_cost_ore)[node_lv],
         jnp.asarray(cfg.node_up_cost_water)[node_lv]], -1)     # [N,2]
    rich_for_node = jnp.all(
        st.resources[player][None, :] >= node_up_cost + upgrade_reserve,
        axis=-1)                                                # [N]
    node_act = jnp.where(rich_for_node, a_upgrade(cfg), A_NOOP)

    # ---- 组合(覆盖链顺序与 scripted 一致;build 5 行合并为 is_conscript 单覆盖)----
    act = jnp.full(n, A_NOOP, jnp.int32)
    act = jnp.where(mine & (st.etype == TYPE_HQ), hq_act, act)
    act = jnp.where(idle_worker, worker_act, act)
    act = jnp.where(is_conscript, build_act, act)
    act = jnp.where(mine & (st.etype == TYPE_BARRACKS), bar_act_full, act)
    act = jnp.where(mine & (st.etype == TYPE_CAMP), camp_act, act)
    act = jnp.where(mine & is_node_b, node_act, act)
    act = jnp.where(mine & is_army, inf_act, act)
    is_plant_a = dog_act == a_plant_flag(cfg)
    act = jnp.where((dog_act != A_NOOP) & (~attack_on | is_plant_a),
                    dog_act, act)
    act = jnp.where(mine & (st.etype == TYPE_BARRACKS) & attack_on
                    & st.flag_active[player, 0], a_recall_flag(0, cfg), act)

    # ---- 让路:空闲单位堵在自家 HQ 卸货环上则朝远离 HQ 挪一格 ----
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
    act = jnp.where(step_off, A_MOVE0 + best_dir, act)
    return act
