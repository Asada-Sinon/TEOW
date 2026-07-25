"""动作编码、合法性掩码、下达指令。

每 tick 控制器给每个实体一个离散动作 id(非法动作被掩成 no-op,控制器可以放心
乱给)。动作只是「改写常驻指令」——真正的行动(移动/开采/生产)由 step 各阶段按
常驻指令推进,所以 no-op ≠ 发呆。

动作表(v1.4:n_actions = 18 + 3*Nn + 2*F + 17;Nn=8、F=3 时共 65,
完整 id 布局见各 a_*() 定义):
  0                 NOOP     维持现状
  1                 STOP     清除常驻指令(矿内工人不可用,先等它出来)
  2                 ATTACK   attack-move 向敌方 HQ(战斗单位)
  3..6              MOVE     N/E/S/W 走一格(到达即转 IDLE)
  7..7+Nn-1         BUILD_k  去资源点 k 建矿/泵(类型由点决定;采集单位)
  7+Nn..7+2Nn-1     HARV_k   指派到资源点 k 的采集循环(采集单位,点须己方已建
                             且名额未满——v1.3 指派即占用,重复下达豁免)
  7+2Nn             TRAIN_W  训练工人(仅 HQ)
  8+2Nn             TRAIN_I  训练近战步兵(仅 HQ)
  9+2Nn             UPGRADE  建筑自升级
  10+2Nn            BUILD_CAMP
  11+2Nn..12+2Nn    (退役 v1.3 双线研发 id,永久非法保号——旧轨迹可解)
  13+2Nn            BUILD_BARRACKS
  14+2Nn            TRAIN_DOG
  15+2Nn            BUILD_TOWER
  16+2Nn            GARR_HQ  驻守己方 HQ(仅战斗单位,v1.3)
  17+2Nn..17+3Nn-1  GARR_k   驻守己方资源点 k 的矿泵(仅战斗单位,v1.3)
  17+3Nn..17+3Nn+F-1  GARR_FLAG_j  驻守己方 j 号旗(仅战斗单位,v1.3)
  17+3Nn+F          PLANT    在脚下插旗(仅战斗单位;免费即时,须拥有建成兵营)
  18+3Nn+F..17+3Nn+2F  RECALL_j  撤 j 号旗(挂在建成兵营上,远程即时)
  ---- v1.4 追加块(B = 18+3Nn+2F 起)----
  B+0..B+7          TRAIN_t  训练新兵种(大力士/马车=HQ;弓箭手/轻骑/重甲/法师/
                             奶妈/攻城车=兵营,门槛 train_level_by_type)
  B+8               BUILD_MORTAR  工人起迫击炮(HQ3 解锁,每玩家限 1)
  B+9..B+16         RESEARCH_l    训练营研发第 l 条兵种线(l=0..7)

合法性掩码从第一天就是引擎输出(调研报告 §5.8):v2 的 RL invalid-action-masking
直接复用,random 控制器也靠它只在合法动作里采样。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    N_LINES,
    RES_WATER,
    TYPE_AIRSHIP,
    TYPE_ARCHER,
    TYPE_CAMP,
    TYPE_CATAPULT,
    TYPE_DRAGON,
    TYPE_FENCE_IRON,
    TYPE_FENCE_STONE,
    TYPE_FENCE_WOOD,
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
    TYPE_OF_LINE,
    TYPE_PUMP,
    TYPE_RAM,
    TYPE_STRONGMAN,
    TYPE_WAGON,
    TYPE_WORKER,
    Config,
    btask_research,
)
from .map import MapData
from .state import (
    ORDER_ATTACK,
    ORDER_BUILD,
    ORDER_GARRISON,
    ORDER_HARVEST,
    ORDER_IDLE,
    ORDER_MOVE,
    PH_TO_HQ,
    PH_TO_NODE,
    WorldState,
)

A_NOOP = 0
A_STOP = 1
A_ATTACK = 2
A_MOVE0 = 3  # N,E,S,W = +0..+3
_DIRS = jnp.asarray([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)


def a_build(k: int) -> int:
    return 7 + k


def a_harvest(k: int, cfg: Config) -> int:
    return 7 + cfg.n_nodes + k


def a_train_worker(cfg: Config) -> int:
    return 7 + 2 * cfg.n_nodes


def a_train_infantry(cfg: Config) -> int:
    return 8 + 2 * cfg.n_nodes


def a_upgrade(cfg: Config) -> int:
    """建筑通用自升级(v1.1):HQ/矿/泵/营。扣费不在 apply_orders,
    在 economy.paid_orders_pass 按槽号顺序对账(同 tick 多笔支出防透支)。"""
    return 9 + 2 * cfg.n_nodes


def a_build_camp(cfg: Config) -> int:
    """工人在自身相邻空闲格起技能训练营(解锁表 TYPE_CAMP)。
    扣费+落格+占槽都在 paid_orders_pass(同 tick 同玩家至多批准一座)。"""
    return 10 + 2 * cfg.n_nodes


def a_research_legacy(line: int, cfg: Config) -> int:
    """v1.3 双线研发 id(11/12+2Nn)。已退役:永久非法、保号不复用,
    使旧轨迹语义可解。新研发走 a_research_line。"""
    return 11 + line + 2 * cfg.n_nodes


def a_build_barracks(cfg: Config) -> int:
    """工人起兵营(v1.2;解锁表 TYPE_BARRACKS,机制同建营)。"""
    return 13 + 2 * cfg.n_nodes


def a_train_dog(cfg: Config) -> int:
    """兵营训练狗子(v1.2)。"""
    return 14 + 2 * cfg.n_nodes


def a_build_tower(cfg: Config) -> int:
    """工人起哨塔(v1.2;解锁表 TYPE_TOWER,机制同建营)。"""
    return 15 + 2 * cfg.n_nodes


def a_garrison_hq(cfg: Config) -> int:
    """驻守己方 HQ(v1.3;仅战斗单位)。"""
    return 16 + 2 * cfg.n_nodes


def a_garrison_node(k: int, cfg: Config) -> int:
    """驻守己方资源点 k 的矿泵(v1.3;仅战斗单位,点须己方已建)。"""
    return 17 + 2 * cfg.n_nodes + k


def a_garrison_flag(j: int, cfg: Config) -> int:
    """驻守己方 j 号旗(v1.3;仅战斗单位,旗须激活)。"""
    return 17 + 3 * cfg.n_nodes + j


def a_plant_flag(cfg: Config) -> int:
    """在自身脚下插旗(v1.3;仅战斗单位。免费、即时,须拥有建成兵营,
    每玩家至多 max_flags 面,脚下格须非资源点/建筑占用格)。"""
    return 17 + 3 * cfg.n_nodes + cfg.max_flags


def a_recall_flag(j: int, cfg: Config) -> int:
    """撤 j 号旗(v1.3;挂在建成兵营上,远程即时;原驻守该旗的兵原地转 IDLE)。"""
    return 18 + 3 * cfg.n_nodes + cfg.max_flags + j


# ---- v1.4 追加块(只追加不插入,旧 id 全部保号)----

# 新训练动作的兵种顺序(B+0..B+7;HQ 系在前,兵营系在后)
TRAIN_ORDER = (TYPE_STRONGMAN, TYPE_WAGON, TYPE_ARCHER, TYPE_LCAV,
               TYPE_HEAVY, TYPE_MAGE, TYPE_HEALER, TYPE_RAM)


def _v14_base(cfg: Config) -> int:
    return 18 + 3 * cfg.n_nodes + 2 * cfg.max_flags


def a_train_unit(t: int, cfg: Config) -> int:
    """训练 v1.4 新兵种 t(TRAIN_ORDER 内);工人/步兵/狗沿用旧 id。"""
    return _v14_base(cfg) + TRAIN_ORDER.index(t)


def a_build_mortar(cfg: Config) -> int:
    """工人起迫击炮(v1.4;HQ3 解锁,build_cap 限 1,机制同塔)。"""
    return _v14_base(cfg) + 8


def a_research_line(line: int, cfg: Config) -> int:
    """训练营研发第 line 条兵种线(v1.4;line=0..7,兵种见 TYPE_OF_LINE)。"""
    return _v14_base(cfg) + 9 + line


# ---- v1.5 追加块(栅栏三档;B2 = v1.4 块末尾)----
FENCE_ORDER = (TYPE_FENCE_WOOD, TYPE_FENCE_STONE, TYPE_FENCE_IRON)


def _v15_base(cfg: Config) -> int:
    return _v14_base(cfg) + 9 + N_LINES


def a_build_fence(t: int, cfg: Config) -> int:
    """采集单位在相邻空闲格起栅栏(v1.5;t ∈ FENCE_ORDER,解锁 木2/石3/铁5,
    无数量上限;落位与扣费走 paid_orders_pass 的自由格建筑管线)。"""
    return _v15_base(cfg) + FENCE_ORDER.index(t)


# ---- v1.6 追加块(B3 = v1.5 块末尾)----
TRAIN_ORDER_V16 = (TYPE_CATAPULT, TYPE_AIRSHIP, TYPE_DRAGON)
BUILD_ORDER_V16 = (TYPE_MAGETOWER, TYPE_LANDMINE, TYPE_FLAMER, TYPE_LASER)


def _v16_base(cfg: Config) -> int:
    return _v15_base(cfg) + len(FENCE_ORDER)


def a_train_v16(t: int, cfg: Config) -> int:
    """兵营训练 v1.6 兵种(投石车/飞艇=兵营6,龙骑兵=兵营7)。"""
    return _v16_base(cfg) + TRAIN_ORDER_V16.index(t)


def a_build_defense(t: int, cfg: Config) -> int:
    """采集单位起 v1.6 防御建筑(法师塔 HQ3/地雷 HQ4/喷火器 HQ6/激光炮 HQ7;
    除地雷限 5 外每种限 1,机制同迫击炮)。"""
    return _v16_base(cfg) + 3 + BUILD_ORDER_V16.index(t)


def a_board(cfg: Config) -> int:
    """上艇(v1.6):地面战斗单位登上 reach 半径内最近的己方有空位飞艇,
    即时;敌方攻击范围内禁止,开火后 reboard_lockout 内禁止。"""
    return _v16_base(cfg) + 7


def a_drop_all(cfg: Config) -> int:
    """空降(v1.6):飞艇全体乘员落到艇当前位置(暂时叠格互推散开)。"""
    return _v16_base(cfg) + 8


def n_actions(cfg: Config) -> int:
    return _v16_base(cfg) + 9


def unit_costs(cfg: Config) -> jax.Array:
    """int32 [2(单位:工人/步兵), 2(资源)]。"""
    return jnp.asarray(
        [[cfg.worker_cost_ore, cfg.worker_cost_water],
         [cfg.infantry_cost_ore, cfg.infantry_cost_water]], jnp.int32)


def node_costs(cfg: Config, mapdata: MapData) -> jax.Array:
    """int32 [Nn,2]:在点 k 建结构的成本(矿点=矿成本,水点=泵成本)。"""
    is_water = jnp.asarray(mapdata.node_type) == RES_WATER
    ore_c = jnp.where(is_water, cfg.pump_cost_ore, cfg.mine_cost_ore)
    wat_c = jnp.where(is_water, cfg.pump_cost_water, cfg.mine_cost_water)
    return jnp.stack([ore_c, wat_c], axis=-1).astype(jnp.int32)


def legality_mask(state: WorldState, cfg: Config, mapdata: MapData,
                  owner: jax.Array) -> jax.Array:
    """bool [N, n_actions]。只查静态可判定项(资源够不够、点归属、槽满没满);
    动态竞争(同 tick 抢格/抢点/抢驻槽)由 step 内仲裁,输了自动顺延,不算非法。"""
    n = cfg.n_total
    nn = cfg.n_nodes
    h, w = cfg.grid_h, cfg.grid_w
    passable = jnp.asarray(mapdata.passable)

    from .stats import etype_idx
    et = etype_idx(state)
    # 采集单位(工人/大力士/马车):建造+采集;战斗单位(line_of_type≥0):
    # ATTACK/驻守/插旗。v1.4 规格:采集单位不能攻击、不能驻守、不能驻旗。
    is_harvester = ((state.etype == TYPE_WORKER)
                    | (state.etype == TYPE_STRONGMAN)
                    | (state.etype == TYPE_WAGON))
    is_worker = is_harvester
    # 战斗单位判据(v1.6 D7:八线 ∪ {投石车,龙};飞艇不算——无攻击)
    is_inf = jnp.asarray(cfg.is_combat_by_type, bool)[et]
    is_airship = state.etype == TYPE_AIRSHIP
    is_hq = state.etype == TYPE_HQ
    # 矿内/舱内只许 NOOP(v1.6:舱内=离场)
    actable = state.alive & ~state.inside & (state.aboard < 0)

    mask = jnp.zeros((n, n_actions(cfg)), bool)
    mask = mask.at[:, A_NOOP].set(True)  # 死槽也「合法 NOOP」,apply 处再兜底
    mask = mask.at[:, A_STOP].set(actable & (is_worker | is_inf | is_airship))
    mask = mask.at[:, A_ATTACK].set(actable & is_inf)

    # 移动:目标格在界内且静态可通行(v1.2 连续坐标,经 cell_of 归格)
    from .state import cell_of
    for d in range(4):
        nxt = state.pos + _DIRS[d].astype(jnp.float32)
        nc = cell_of(nxt)
        ok = ((nc[:, 0] >= 0) & (nc[:, 0] < h) & (nc[:, 1] >= 0) & (nc[:, 1] < w))
        nc = jnp.clip(nc, 0, jnp.asarray([h - 1, w - 1]))
        ok = ok & passable[nc[:, 0], nc[:, 1]]
        mask = mask.at[:, A_MOVE0 + d].set(
            actable & (is_worker | is_inf | is_airship) & ok)

    # 建造/采集:逐点([Nn,N] 小矩阵,Nn=8)
    from .economy import assigned_counts  # 名额口径唯一定义处,避免双真源
    ncost = node_costs(cfg, mapdata)                      # [Nn,2]
    stock = state.resources[owner.astype(jnp.int32)]      # [N,2] 各实体所属玩家的库存
    acounts = assigned_counts(state, cfg)                 # [Nn] 指派即占用(v1.3)
    slots_tbl = jnp.asarray(cfg.harvest_slots_by_level, jnp.int32)
    for k in range(nn):
        unclaimed = (state.node_owner[k] == -1) & (state.node_build_timer[k] == 0)
        afford = jnp.all(stock >= ncost[k], axis=-1)
        mask = mask.at[:, a_build(k)].set(actable & is_worker & unclaimed & afford)
        mine_up = (state.node_owner[k] == owner) & (state.node_ent[k] >= 0)
        # 名额未满才可新指派;已指派到 k 者豁免(重复下同一指令 no-op 化)
        ent_k = jnp.clip(state.node_ent[k].astype(jnp.int32), 0, cfg.n_total - 1)
        cap_k = slots_tbl[jnp.clip(state.level[ent_k].astype(jnp.int32), 0, 7)]
        already_k = (state.order == ORDER_HARVEST) & (state.target_node == k)
        mask = mask.at[:, a_harvest(k, cfg)].set(
            actable & is_worker & mine_up & ((acounts[k] < cap_k) | already_k))
        # 驻守点 k(v1.3):战斗单位 + 点须己方已建
        mask = mask.at[:, a_garrison_node(k, cfg)].set(actable & is_inf & mine_up)

    # 驻守己方 HQ(v1.3):目标恒存在(HQ 亡即终局)
    mask = mask.at[:, a_garrison_hq(cfg)].set(actable & is_inf)

    # 训练:HQ 空闲 + 本玩家行块有空槽 + 付得起
    # per-player 计数统一 scatter-add(v1.5 P 泛化;旧 stack([p0,p1])[half]
    # 模式只认两家)
    ucost = unit_costs(cfg)
    half = owner.astype(jnp.int32)

    def _pcount(cond):
        """bool [N] → int32 [N]:各实体所属玩家的 cond 计数(广播回逐实体)。"""
        return jnp.zeros(cfg.n_players, jnp.int32).at[half].add(
            cond.astype(jnp.int32))[half]

    free_in_half = _pcount(~state.alive) > 0              # [N]
    hq_idle = (actable & is_hq & (state.btimer == 0) & (state.btype == 0)
               & free_in_half)
    mask = mask.at[:, a_train_worker(cfg)].set(
        hq_idle & jnp.all(stock >= ucost[0], axis=-1))
    mask = mask.at[:, a_train_infantry(cfg)].set(
        hq_idle & jnp.all(stock >= ucost[1], axis=-1))
    # HQ 高级采集单位(v1.4):大力士(HQ3)/马车(HQ5),门=基地等级
    tl = cfg.train_level_by_type
    hq_lv0 = state.level[half * cfg.e_max].astype(jnp.int32)  # 所属玩家基地级
    for t in (TYPE_STRONGMAN, TYPE_WAGON):
        c = jnp.asarray([cfg.train_cost_ore_by_type[t],
                         cfg.train_cost_water_by_type[t]], jnp.int32)
        ok_t = hq_idle & (hq_lv0 >= tl[t]) & jnp.all(stock >= c, axis=-1)
        mask = mask.at[:, a_train_unit(t, cfg)].set(ok_t)

    # 升级(v1.1):建筑空闲(btimer==0,critic S-1:否则覆写在训单位)+ 上限链
    # + 乐观付得起(最终扣费在 paid_orders_pass 顺序对账,同 tick 多笔不透支)
    from .economy import upgrade_cost_of  # 集中定价,避免双真源
    lv = state.level.astype(jnp.int32)
    hq_lv = state.level[half * cfg.e_max].astype(jnp.int32)   # 各实体所属玩家的基地级
    is_node_b = (state.etype == TYPE_MINE) | (state.etype == TYPE_PUMP)
    is_camp = state.etype == TYPE_CAMP
    from .config import TYPE_BARRACKS as _TB0
    from .config import TYPE_TOWER as _TT
    is_tower = state.etype == _TT
    is_bar_u = state.etype == _TB0     # 兵营可升级(v1.4;上限=基地等级)
    cap_ok = jnp.where(is_hq, lv < cfg.base_max_level, lv < hq_lv)
    up_cost = upgrade_cost_of(state, cfg)                     # [N,2] 按类型/等级
    can_up = (actable & (is_hq | is_node_b | is_camp | is_tower | is_bar_u)
              & (state.btimer == 0) & cap_ok
              & jnp.all(stock >= up_cost, axis=-1))
    mask = mask.at[:, a_upgrade(cfg)].set(can_up)

    # 自由格建筑(营/兵营):工人 + 解锁表 + 半区有空槽 + 乐观付得起
    # (落格与最终扣费在 paid_orders_pass;同玩家同 tick 每种至多批准一座)
    from .config import TYPE_BARRACKS
    unlock = jnp.asarray(cfg.unlock_level_by_type, jnp.int32)
    camp_cost = jnp.asarray([cfg.camp_cost_ore, cfg.camp_cost_water], jnp.int32)
    can_camp = (actable & is_worker & (hq_lv >= unlock[TYPE_CAMP])
                & free_in_half & jnp.all(stock >= camp_cost, axis=-1))
    mask = mask.at[:, a_build_camp(cfg)].set(can_camp)
    bar_cost = jnp.asarray([cfg.barracks_cost_ore, cfg.barracks_cost_water], jnp.int32)
    n_bar = _pcount(state.alive & (state.etype == TYPE_BARRACKS))
    can_bar = (actable & is_worker & (hq_lv >= unlock[TYPE_BARRACKS])
               & (n_bar < cfg.max_barracks) & free_in_half
               & jnp.all(stock >= bar_cost, axis=-1))
    mask = mask.at[:, a_build_barracks(cfg)].set(can_bar)
    from .config import TYPE_TOWER
    tower_cost = jnp.asarray([cfg.tower_cost_ore, cfg.tower_cost_water], jnp.int32)
    # v1.4 多塔:数量挂基地等级(在建也 alive 计数,防重复下单;paid pass
    # 每 tick 每玩家每种至多批一座,掩码时 <cap ⇒ 终态 ≤cap)
    n_twr = _pcount(state.alive & (state.etype == TYPE_TOWER))
    twr_cap = jnp.asarray(cfg.tower_cap_by_hq_level, jnp.int32)[
        jnp.clip(hq_lv, 0, 7)]
    can_tower = (actable & is_worker & (hq_lv >= unlock[TYPE_TOWER])
                 & (n_twr < twr_cap) & free_in_half
                 & jnp.all(stock >= tower_cost, axis=-1))
    mask = mask.at[:, a_build_tower(cfg)].set(can_tower)
    # 迫击炮(v1.4):HQ3 解锁,build_cap 限 1(在建计数,机制同塔)
    from .config import TYPE_MORTAR
    mor_cost = jnp.asarray([cfg.mortar_cost_ore, cfg.mortar_cost_water], jnp.int32)
    n_mor = _pcount(state.alive & (state.etype == TYPE_MORTAR))
    mor_cap = jnp.asarray(cfg.build_cap_by_type, jnp.int32)[TYPE_MORTAR]
    can_mor = (actable & is_worker & (hq_lv >= unlock[TYPE_MORTAR])
               & (n_mor < mor_cap) & free_in_half
               & jnp.all(stock >= mor_cost, axis=-1))
    mask = mask.at[:, a_build_mortar(cfg)].set(can_mor)
    # 栅栏三档(v1.5):解锁门独立(高级解锁后低级仍可造),无数量上限
    fence_costs = {
        TYPE_FENCE_WOOD: (cfg.fence_wood_cost_ore, cfg.fence_wood_cost_water),
        TYPE_FENCE_STONE: (cfg.fence_stone_cost_ore, cfg.fence_stone_cost_water),
        TYPE_FENCE_IRON: (cfg.fence_iron_cost_ore, cfg.fence_iron_cost_water),
    }
    for ft in FENCE_ORDER:
        fc = jnp.asarray(fence_costs[ft], jnp.int32)
        can_f = (actable & is_worker & (hq_lv >= unlock[ft]) & free_in_half
                 & jnp.all(stock >= fc, axis=-1))
        mask = mask.at[:, a_build_fence(ft, cfg)].set(can_f)
    # v1.6 防御建筑四种(cap:法师塔/喷火器/激光炮 1,地雷 5;机制同迫击炮)
    def_costs = {
        TYPE_MAGETOWER: (cfg.magetower_cost_ore, cfg.magetower_cost_water),
        TYPE_LANDMINE: (cfg.landmine_cost_ore, cfg.landmine_cost_water),
        TYPE_FLAMER: (cfg.flamer_cost_ore, cfg.flamer_cost_water),
        TYPE_LASER: (cfg.laser_cost_ore, cfg.laser_cost_water),
    }
    cap_tbl = jnp.asarray(cfg.build_cap_by_type, jnp.int32)
    for dt in BUILD_ORDER_V16:
        dc = jnp.asarray(def_costs[dt], jnp.int32)
        n_dt = _pcount(state.alive & (state.etype == dt))
        can_d = (actable & is_worker & (hq_lv >= unlock[dt])
                 & (n_dt < cap_tbl[dt]) & free_in_half
                 & jnp.all(stock >= dc, axis=-1))
        mask = mask.at[:, a_build_defense(dt, cfg)].set(can_d)

    # 训狗(v1.2):建成兵营空闲 + 半区有空槽 + 付得起
    is_bar = state.etype == TYPE_BARRACKS
    dog_cost = jnp.asarray([cfg.dog_cost_ore, cfg.dog_cost_water], jnp.int32)
    bar_idle = (actable & is_bar & (state.btimer == 0) & (state.btype == 0)
                & free_in_half)
    mask = mask.at[:, a_train_dog(cfg)].set(
        bar_idle & jnp.all(stock >= dog_cost, axis=-1))
    # 兵营高级兵种(v1.4):门=**本座兵营**自身等级 ≥ train_level_by_type
    for t in (TYPE_ARCHER, TYPE_LCAV, TYPE_HEAVY, TYPE_MAGE,
              TYPE_HEALER, TYPE_RAM):
        c = jnp.asarray([cfg.train_cost_ore_by_type[t],
                         cfg.train_cost_water_by_type[t]], jnp.int32)
        ok_t = bar_idle & (lv >= tl[t]) & jnp.all(stock >= c, axis=-1)
        mask = mask.at[:, a_train_unit(t, cfg)].set(ok_t)
    # 兵营 v1.6 兵种(投石车/飞艇 6 级,龙 7 级)
    for t in TRAIN_ORDER_V16:
        c = jnp.asarray([cfg.train_cost_ore_by_type[t],
                         cfg.train_cost_water_by_type[t]], jnp.int32)
        ok_t = bar_idle & (lv >= tl[t]) & jnp.all(stock >= c, axis=-1)
        mask = mask.at[:, a_train_v16(t, cfg)].set(ok_t)

    # 研发(v1.4 八线):建成的营(level>=2)空闲 + 线级<营级 + 付得起
    # + 本玩家没有别的营在研同一条线(防同线并研双倍跳级)
    # + **兵种已解锁**:该线兵种可被训练(步兵线恒解锁——HQ 恒在;
    #   兵营系兵种须拥有 level ≥ train_level_by_type 的建成兵营)
    from .config import TYPE_BARRACKS as _TB
    line_lv = state.upgrades[half]                            # [N,N_LINES]
    tl = cfg.train_level_by_type
    bar_built_lv = jnp.where(
        state.alive & (state.etype == _TB) & (state.btype >= 0),
        state.level.astype(jnp.int32), 0)                     # [N] 建成兵营等级
    for line in range(N_LINES):
        t = TYPE_OF_LINE[line]
        cur = line_lv[:, line].astype(jnp.int32)
        rcost = jnp.stack([jnp.asarray(cfg.line_res_cost_ore)[cur],
                           jnp.asarray(cfg.line_res_cost_water)[cur]], -1)
        code = btask_research(line)
        busy_same = _pcount(state.alive & (state.btype == code)) > 0
        if t == TYPE_INFANTRY:
            unlocked = jnp.ones(n, bool)
        else:
            unlocked = _pcount(bar_built_lv >= tl[t]) > 0
        ok = (actable & is_camp & (lv >= 2) & (state.btimer == 0)
              & (cur < lv) & ~busy_same & unlocked
              & jnp.all(stock >= rcost, axis=-1))
        mask = mask.at[:, a_research_line(line, cfg)].set(ok)

    # ---- 军旗(v1.3):插旗/驻守旗/撤旗 ----
    # 「拥有兵营」= 建成(btype>=0 排除在建负值任务;critic m-2:自由格建筑
    # 开工瞬间即 alive,在建不算)。规格明文 1 级即可、无空闲要求——正在训狗
    # 的兵营(btype==TYPE_DOG)仍算「拥有」,也仍可撤旗(撤旗免费即时,
    # 不占生产位)。
    bar_built = state.alive & is_bar & (state.btype >= 0)
    has_bar = _pcount(bar_built) > 0
    my_flags = jnp.sum(state.flag_active.astype(jnp.int32), axis=1)[half]
    from .movement import building_cells
    bcells = building_cells(state, cfg)
    mycell = jnp.clip(cell_of(state.pos), 0, jnp.asarray([h - 1, w - 1]))
    # 雷格不可插旗(critic MINOR-2:地雷已从 building_cells 排除,单列检查)
    is_mine_e = state.alive & (state.etype == TYPE_LANDMINE)
    mine_cells = (jnp.zeros((h, w), bool)
                  .at[mycell[:, 0], mycell[:, 1]].max(is_mine_e))
    cell_free = (passable[mycell[:, 0], mycell[:, 1]]
                 & ~bcells[mycell[:, 0], mycell[:, 1]]
                 & ~mine_cells[mycell[:, 0], mycell[:, 1]])
    mask = mask.at[:, a_plant_flag(cfg)].set(
        actable & is_inf & has_bar & (my_flags < cfg.max_flags) & cell_free)
    for j in range(cfg.max_flags):
        fj = state.flag_active[half, j]
        mask = mask.at[:, a_garrison_flag(j, cfg)].set(actable & is_inf & fj)
        mask = mask.at[:, a_recall_flag(j, cfg)].set(actable & bar_built & fj)

    # ---- 上艇/空降(v1.6)----
    is_air_e = jnp.asarray(cfg.is_air_by_type, bool)[et]
    pos_d = jnp.linalg.norm(state.pos[:, None, :] - state.pos[None, :, :],
                            axis=-1)
    # 己方建成飞艇的现载员数(scatter 到载具槽)
    carrier = jnp.clip(state.aboard.astype(jnp.int32), 0, n - 1)
    pax = (jnp.zeros(n, jnp.int32)
           .at[carrier].add((state.aboard >= 0).astype(jnp.int32)))
    ship_ok = (state.alive & is_airship & (state.btype >= 0)
               & (pax < cfg.airship_capacity))                 # [N] 可载的艇
    near_ship = jnp.any(ship_ok[None, :] & (owner[:, None] == owner[None, :])
                        & (pos_d <= cfg.reach_radius), axis=1)
    # 威胁禁区(critic M-2:与 combat 攻击者门同源——(atk>0 & rng>0) 的
    # 射程圆 ∪ 喷火器/龙的自心圈;奶妈治疗射程不算威胁)
    from .stats import atk_of, type_tables
    tt_l = type_tables(cfg)
    atk_l = atk_of(state, cfg, owner)
    rng_l = tt_l["rng"][et]
    sa_l = tt_l["self_aoe"][et]
    threat_r = jnp.maximum(jnp.where((atk_l > 0) & (rng_l > 0), rng_l, 0.0),
                           sa_l)
    threat_src = (state.alive & ~state.inside & (state.aboard < 0)
                  & (threat_r > 0))
    in_threat = jnp.any(threat_src[None, :]
                        & (owner[:, None] != owner[None, :])
                        & (pos_d <= threat_r[None, :]), axis=1)
    can_board = (actable & is_inf & ~is_air_e & (state.reboard_lock == 0)
                 & near_ship & ~in_threat)
    mask = mask.at[:, a_board(cfg)].set(can_board)
    # 空降:建成飞艇 + 舱内有人
    mask = mask.at[:, a_drop_all(cfg)].set(
        actable & is_airship & (state.btype >= 0) & (pax > 0))
    return mask


def apply_orders(state: WorldState, actions: jax.Array, cfg: Config,
                 mapdata: MapData, owner: jax.Array) -> tuple[WorldState, jax.Array]:
    """把本 tick 的动作写成常驻指令,返回 (state, 合法化后的动作)。非法 → NOOP。
    训练在此立刻扣费开工(每家只有 1 座 HQ,同 tick 同玩家至多一笔训练支出,
    对库存的检查是安全的);**升级/研发/建营这类可同 tick 多笔的支出不在本函数**,
    由 economy.paid_orders_pass 拿返回的合法化动作按槽号顺序对账(critic B-1);
    资源点建造扣费在工人到场开工时(economy.start_constructions)。"""
    legal = legality_mask(state, cfg, mapdata, owner)
    act = jnp.where(legal[jnp.arange(cfg.n_total), actions], actions, A_NOOP)

    # ---- 采集名额同 tick 超发仲裁(v1.3):掩码只见已写入的指派,看不见同 tick
    # 并发申请(仿 v1.1 研发去重先例),按槽号 rank 裁到等级名额;被裁者动作退回
    # NOOP、保持原指令(与训练落地顺延同哲学)。重复下达当前指派(no-op 化)
    # 不占新名额、不参与仲裁。----
    is_harv_a = (act >= a_harvest(0, cfg)) & (act < a_harvest(cfg.n_nodes, cfg))
    harv_k_a = act - a_harvest(0, cfg)                     # 仅 is_harv_a 位有效
    reissue = (is_harv_a & (state.order == ORDER_HARVEST)
               & (state.target_node == harv_k_a))
    new_cmd_a = ((act == A_STOP) | (act == A_ATTACK)
                 | ((act >= A_MOVE0) & (act < A_MOVE0 + 4))
                 | ((act >= a_build(0)) & (act < a_build(cfg.n_nodes)))
                 | is_harv_a)
    # 现存指派数:非 HARVEST 新指令(STOP/ATTACK/MOVE/BUILD)必然生效,旧名额
    # 立即释放;HARVEST 改派(A 点 → k 点)可能被下方仲裁按 rank 拒绝,旧名额
    # 保守持有——「新指派成功才释放」,否则同 tick「改派被拒 + 空位被新人抢走」
    # 会把 A 点顶到 cap+1 且两名持有者都不动就一直超额(v1.3 终审 P1-1)。
    # 代价:「A 满、B 有空位」的单向让位多卡一拍(下 tick 掩码即放行);
    # 满-满对换本就被掩码挡住(需 STOP 中转),与本修复无关。
    hold = (state.alive & (state.order == ORDER_HARVEST)
            & (state.target_node >= 0) & (~new_cmd_a | is_harv_a))
    tn_a = jnp.clip(state.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    assigned_excl = (jnp.zeros(cfg.n_nodes, jnp.int32)
                     .at[tn_a].add(hold.astype(jnp.int32)))
    slots_tbl = jnp.asarray(cfg.harvest_slots_by_level, jnp.int32)
    for k in range(cfg.n_nodes):  # 编译期展开
        cand = is_harv_a & (harv_k_a == k) & ~reissue
        ent_k = jnp.clip(state.node_ent[k].astype(jnp.int32), 0, cfg.n_total - 1)
        cap_k = slots_tbl[jnp.clip(state.level[ent_k].astype(jnp.int32), 0, 7)]
        rank = jnp.cumsum(cand.astype(jnp.int32)) - 1      # 槽号序内的名次
        act = jnp.where(cand & (assigned_excl[k] + rank >= cap_k), A_NOOP, act)

    is_stop = act == A_STOP
    is_att = act == A_ATTACK
    is_move = (act >= A_MOVE0) & (act < A_MOVE0 + 4)
    is_build = (act >= a_build(0)) & (act < a_build(cfg.n_nodes))
    is_harv = (act >= a_harvest(0, cfg)) & (act < a_harvest(cfg.n_nodes, cfg))
    is_tw = act == a_train_worker(cfg)
    is_ti = act == a_train_infantry(cfg)
    is_tsm = act == a_train_unit(TYPE_STRONGMAN, cfg)   # v1.4 HQ 系
    is_twg = act == a_train_unit(TYPE_WAGON, cfg)
    is_ghq = act == a_garrison_hq(cfg)
    is_gnode = ((act >= a_garrison_node(0, cfg))
                & (act < a_garrison_node(cfg.n_nodes, cfg)))
    gar_k = jnp.clip(act - a_garrison_node(0, cfg), 0, cfg.n_nodes - 1)
    is_gflag = ((act >= a_garrison_flag(0, cfg))
                & (act < a_garrison_flag(cfg.max_flags, cfg)))
    gf_j = jnp.clip(act - a_garrison_flag(0, cfg), 0, cfg.max_flags - 1)
    is_plant = act == a_plant_flag(cfg)
    # 训狗不在此扣费:同 tick 多座兵营可同时下单,走 paid_orders_pass 对账

    build_k = jnp.where(is_build, act - a_build(0), -1).astype(jnp.int8)
    harv_k = jnp.where(is_harv, act - a_harvest(0, cfg), -1).astype(jnp.int8)
    mdir = jnp.clip(act - A_MOVE0, 0, 3)

    order = state.order
    order = jnp.where(is_stop, ORDER_IDLE, order)
    order = jnp.where(is_att, ORDER_ATTACK, order)
    order = jnp.where(is_move, ORDER_MOVE, order)
    order = jnp.where(is_build, ORDER_BUILD, order)
    order = jnp.where(is_harv, ORDER_HARVEST, order)
    order = jnp.where(is_ghq | is_gnode | is_gflag, ORDER_GARRISON, order)

    # 驻守锚点 id(v1.3):A_STOP 顺手清残值;其余改派指令留残值无害——
    # garrison_id 的一切消费方都门控在 order==ORDER_GARRISON 上
    garrison_id = jnp.where(is_stop, -1, state.garrison_id)
    garrison_id = jnp.where(is_ghq, 0, garrison_id)
    garrison_id = jnp.where(is_gnode, gar_k + 1, garrison_id)
    garrison_id = jnp.where(is_gflag, cfg.n_nodes + 1 + gf_j,
                            garrison_id).astype(jnp.int8)

    target_node = jnp.where(is_build, build_k, state.target_node)
    target_node = jnp.where(is_harv, harv_k, target_node)

    # ATTACK 的 target_cell = 最近存活敌方 HQ(欧氏;仅显示/到达兜底口径,
    # 实际寻路目标由 movement 每 tick 按场值动态重选——v1.5 D3)
    from .state import hq_slot as _hq_slot
    P = cfg.n_players
    hq_pos_all = jnp.asarray(mapdata.hq_pos, jnp.float32)          # [P,2]
    hq_alive = jnp.stack([state.alive[_hq_slot(p, cfg)] for p in range(P)])
    d_hq = jnp.linalg.norm(state.pos[:, None, :] - hq_pos_all[None, :, :],
                           axis=-1)                                # [N,P]
    own_i32a = owner.astype(jnp.int32)
    d_hq = jnp.where((jnp.arange(P)[None, :] == own_i32a[:, None])
                     | ~hq_alive[None, :], 1e9, d_hq)
    enemy_hq = hq_pos_all[jnp.argmin(d_hq, axis=1)]                # [N,2]
    target_cell = jnp.where(is_att[:, None], enemy_hq, state.target_cell)
    target_cell = jnp.where(is_move[:, None],
                            state.pos + _DIRS[mdir].astype(jnp.float32), target_cell)
    # 驻守锚点格(v1.3):HQ 位置 / node_pos[k];回岗距离一律对 target_cell 判
    own_hq = jnp.asarray(mapdata.hq_pos, jnp.float32)[owner.astype(jnp.int32)]
    node_pos_f = jnp.asarray(mapdata.node_pos, jnp.float32)
    target_cell = jnp.where(is_ghq[:, None], own_hq, target_cell)
    target_cell = jnp.where(is_gnode[:, None], node_pos_f[gar_k], target_cell)
    # 驻守旗:锚点=旗位(撤旗即失效,由下方撤旗清理兜底,movement 无需特判)
    target_cell = jnp.where(
        is_gflag[:, None],
        state.flag_pos[owner.astype(jnp.int32), gf_j], target_cell)

    # 换指令时相位复位;带着货被改派采集 → 先回家卸货再进循环
    # (「满载」按该工人当前线级的载荷判;历史低级载荷也 >0 即回家,用 >0 更稳)
    new_cmd = is_stop | is_att | is_move | is_build | is_harv
    phase = jnp.where(new_cmd, PH_TO_NODE, state.phase).astype(jnp.int8)
    phase = jnp.where(is_harv & (state.cargo > 0), PH_TO_HQ, phase)

    # 训练扣费 + 开工(HQ 系即扣:每家仅 1 座 HQ,同 tick 至多一笔,安全;
    # 兵营系训练走 paid_orders_pass cumsum 对账)
    ucost = unit_costs(cfg)
    sm_cost = jnp.asarray([cfg.strongman_cost_ore, cfg.strongman_cost_water],
                          jnp.int32)
    wg_cost = jnp.asarray([cfg.wagon_cost_ore, cfg.wagon_cost_water], jnp.int32)
    spend = (is_tw[:, None] * ucost[0] + is_ti[:, None] * ucost[1]
             + is_tsm[:, None] * sm_cost + is_twg[:, None] * wg_cost)  # [N,2]
    pay = jnp.zeros_like(state.resources).at[owner.astype(jnp.int32)].add(spend)
    btype = jnp.where(is_tw, TYPE_WORKER, state.btype)
    btype = jnp.where(is_ti, TYPE_INFANTRY, btype)
    btype = jnp.where(is_tsm, TYPE_STRONGMAN, btype)
    btype = jnp.where(is_twg, TYPE_WAGON, btype).astype(jnp.int8)
    btimer = jnp.where(is_tw, cfg.worker_time, state.btimer)
    btimer = jnp.where(is_ti, cfg.infantry_time, btimer)
    btimer = jnp.where(is_tsm, cfg.strongman_time, btimer)
    btimer = jnp.where(is_twg, cfg.wagon_time, btimer).astype(jnp.int16)

    # ---- 军旗表更新(v1.3;免费无扣费,留在本函数不进 paid_orders_pass)。
    # 求值顺序(critic M-2):撤旗对单位的清理在个体 order 覆写**之后**——
    # 同 tick 既被撤旗又收到新指令(如 ATTACK)者,新指令生效、不被打回 IDLE,
    # 只清「覆写后仍为 GARRISON 指向该旗」者。撤旗先于插旗结算,腾出的旗位
    # 同 tick 可复用。----
    own_i32 = owner.astype(jnp.int32)
    flag_pos = state.flag_pos
    flag_active = state.flag_active
    for p in range(cfg.n_players):  # 编译期展开
        for j in range(cfg.max_flags):
            rec = jnp.any((act == a_recall_flag(j, cfg)) & (own_i32 == p))
            flag_active = flag_active.at[p, j].set(
                jnp.where(rec, False, flag_active[p, j]))
            flag_pos = flag_pos.at[p, j].set(
                jnp.where(rec, -1.0, flag_pos[p, j]))
            ref = (rec & (own_i32 == p)
                   & (garrison_id == cfg.n_nodes + 1 + j))
            order = jnp.where(ref & (order == ORDER_GARRISON),
                              ORDER_IDLE, order)
            # 指向被撤旗的 garrison_id 残值一并清(含同 tick 被改派 ATTACK 者
            # ——order 已是新指令,不打回 IDLE;规格「引擎保证无悬空引用」)
            garrison_id = jnp.where(ref, -1, garrison_id).astype(jnp.int8)
    # 插旗:同 tick 多申请按槽号逐个批(每轮批槽号最小者进最小空旗位),
    # 旗位耗尽后其余申请自动 no-op(与训练落地顺延同哲学)
    slots_p = jnp.arange(cfg.n_total)
    for p in range(cfg.n_players):  # 编译期展开
        cand = is_plant & (own_i32 == p)
        for _ in range(cfg.max_flags):
            widx = jnp.argmax(cand)                # 全 False 返 0,do 门控
            free_j = jnp.argmax(~flag_active[p])
            do = jnp.any(cand) & jnp.any(~flag_active[p])
            cell = jnp.round(state.pos[widx])      # 旗落申请者脚下格心
            flag_pos = flag_pos.at[p, free_j].set(
                jnp.where(do, cell, flag_pos[p, free_j]))
            flag_active = flag_active.at[p, free_j].set(
                jnp.where(do, True, flag_active[p, free_j]))
            cand = cand & (slots_p != widx)

    # ---- 空降/上艇(v1.6;先 DROP 后 BOARD——腾位同拍可用,与撤旗先于
    # 插旗同哲学,记 DECISIONS)----
    is_drop = act == a_drop_all(cfg)
    carrier_i = jnp.clip(state.aboard.astype(jnp.int32), 0, cfg.n_total - 1)
    dropped = (state.aboard >= 0) & is_drop[carrier_i]
    aboard = jnp.where(dropped, -1, state.aboard).astype(jnp.int16)
    # 上艇:目标=reach 内最近己方建成可载艇;同拍超发按槽号 rank 裁到剩余容量
    is_board2 = act == a_board(cfg)
    is_airship_e = (state.alive & (state.etype == TYPE_AIRSHIP)
                    & (state.btype >= 0))
    pos_d2 = jnp.linalg.norm(state.pos[:, None, :] - state.pos[None, :, :],
                             axis=-1)
    own_eq = owner[:, None] == owner[None, :]
    ship_d = jnp.where(is_airship_e[None, :] & own_eq
                       & (pos_d2 <= cfg.reach_radius), pos_d2, 1e9)
    my_ship = jnp.argmin(ship_d, axis=1)
    has_ship = jnp.any(ship_d < 1e9, axis=1)
    pax_now = (jnp.zeros(cfg.n_total, jnp.int32)
               .at[jnp.clip(aboard.astype(jnp.int32), 0, cfg.n_total - 1)]
               .add((aboard >= 0).astype(jnp.int32)))
    cand_b = is_board2 & has_ship
    tri = jnp.tril(jnp.ones((cfg.n_total, cfg.n_total), bool), k=-1)
    same_ship = (cand_b[:, None] & cand_b[None, :]
                 & (my_ship[:, None] == my_ship[None, :]))
    rank_b = jnp.sum(same_ship & tri, axis=1)      # 槽号序内名次
    free_cap = cfg.airship_capacity - pax_now[my_ship]
    ok_board = cand_b & (rank_b < free_cap)
    aboard = jnp.where(ok_board, my_ship, aboard).astype(jnp.int16)
    # 登艇卫生:清常驻指令与锚点(防舱内悬空引用/名额吊死)
    order = jnp.where(ok_board, ORDER_IDLE, order)
    target_node = jnp.where(ok_board, -1, target_node).astype(jnp.int8)
    garrison_id = jnp.where(ok_board, -1, garrison_id).astype(jnp.int8)
    phase = jnp.where(ok_board, PH_TO_NODE, phase).astype(jnp.int8)

    return state._replace(
        order=order.astype(jnp.int8),
        phase=phase,
        target_node=target_node,
        target_cell=target_cell,
        garrison_id=garrison_id,
        aboard=aboard,
        flag_pos=flag_pos,
        flag_active=flag_active,
        btype=btype,
        btimer=btimer,
        resources=state.resources - pay,
    ), act
