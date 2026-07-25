"""动作编码、合法性掩码、下达指令。

每 tick 控制器给每个实体一个离散动作 id(非法动作被掩成 no-op,控制器可以放心
乱给)。动作只是「改写常驻指令」——真正的行动(移动/开采/生产)由 step 各阶段按
常驻指令推进,所以 no-op ≠ 发呆。

动作表(n_actions = 9 + 2*Nn;Nn=6 时共 21):
  0                 NOOP     维持现状
  1                 STOP     清除常驻指令(矿内工人不可用,先等它出来)
  2                 ATTACK   attack-move 向敌方 HQ(仅步兵)
  3..6              MOVE     N/E/S/W 走一格(到达即转 IDLE)
  7..7+Nn-1         BUILD_k  去资源点 k 建矿/泵(类型由点决定;仅工人)
  7+Nn..7+2Nn-1     HARV_k   指派到资源点 k 的采集循环(仅工人,点须己方已建)
  7+2Nn             TRAIN_W  训练工人(仅 HQ)
  8+2Nn             TRAIN_I  训练步兵(仅 HQ)

合法性掩码从第一天就是引擎输出(调研报告 §5.8):v2 的 RL invalid-action-masking
直接复用,random 控制器也靠它只在合法动作里采样。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
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
    ORDER_ATTACK,
    ORDER_BUILD,
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
    """工人在自身相邻空闲格起技能训练营(基地≥camp_unlock_level 解锁)。
    扣费+落格+占槽都在 paid_orders_pass(同 tick 同玩家至多批准一座)。"""
    return 10 + 2 * cfg.n_nodes


def a_research(line: int, cfg: Config) -> int:
    """训练营研发:line 0=步兵捆绑线,1=工人经济线。"""
    return 11 + line + 2 * cfg.n_nodes


def n_actions(cfg: Config) -> int:
    return 13 + 2 * cfg.n_nodes


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

    is_worker = state.etype == TYPE_WORKER
    is_inf = state.etype == TYPE_INFANTRY
    is_hq = state.etype == TYPE_HQ
    actable = state.alive & ~state.inside  # 矿内工人只许 NOOP

    mask = jnp.zeros((n, n_actions(cfg)), bool)
    mask = mask.at[:, A_NOOP].set(True)  # 死槽也「合法 NOOP」,apply 处再兜底
    mask = mask.at[:, A_STOP].set(actable & (is_worker | is_inf))
    mask = mask.at[:, A_ATTACK].set(actable & is_inf)

    # 移动:目标格在界内且静态可通行(v1.2 连续坐标,经 cell_of 归格)
    from .state import cell_of
    for d in range(4):
        nxt = state.pos + _DIRS[d].astype(jnp.float32)
        nc = cell_of(nxt)
        ok = ((nc[:, 0] >= 0) & (nc[:, 0] < h) & (nc[:, 1] >= 0) & (nc[:, 1] < w))
        nc = jnp.clip(nc, 0, jnp.asarray([h - 1, w - 1]))
        ok = ok & passable[nc[:, 0], nc[:, 1]]
        mask = mask.at[:, A_MOVE0 + d].set(actable & (is_worker | is_inf) & ok)

    # 建造/采集:逐点([Nn,N] 小矩阵,Nn=6)
    ncost = node_costs(cfg, mapdata)                      # [Nn,2]
    stock = state.resources[owner.astype(jnp.int32)]      # [N,2] 各实体所属玩家的库存
    for k in range(nn):
        unclaimed = (state.node_owner[k] == -1) & (state.node_build_timer[k] == 0)
        afford = jnp.all(stock >= ncost[k], axis=-1)
        mask = mask.at[:, a_build(k)].set(actable & is_worker & unclaimed & afford)
        mine_up = (state.node_owner[k] == owner) & (state.node_ent[k] >= 0)
        mask = mask.at[:, a_harvest(k, cfg)].set(actable & is_worker & mine_up)

    # 训练:HQ 空闲 + 本半区有空槽 + 付得起
    ucost = unit_costs(cfg)
    half = owner.astype(jnp.int32)
    free_in_half = jnp.stack([
        jnp.any(~state.alive[:cfg.e_max]),
        jnp.any(~state.alive[cfg.e_max:]),
    ])[half]                                              # [N]
    hq_idle = actable & is_hq & (state.btimer == 0) & free_in_half
    mask = mask.at[:, a_train_worker(cfg)].set(
        hq_idle & jnp.all(stock >= ucost[0], axis=-1))
    mask = mask.at[:, a_train_infantry(cfg)].set(
        hq_idle & jnp.all(stock >= ucost[1], axis=-1))

    # 升级(v1.1):建筑空闲(btimer==0,critic S-1:否则覆写在训单位)+ 上限链
    # + 乐观付得起(最终扣费在 paid_orders_pass 顺序对账,同 tick 多笔不透支)
    from .economy import upgrade_cost_of  # 集中定价,避免双真源
    lv = state.level.astype(jnp.int32)
    hq_lv = state.level[half * cfg.e_max].astype(jnp.int32)   # 各实体所属玩家的基地级
    is_node_b = (state.etype == TYPE_MINE) | (state.etype == TYPE_PUMP)
    is_camp = state.etype == TYPE_CAMP
    cap_ok = jnp.where(is_hq, lv < cfg.base_max_level, lv < hq_lv)
    up_cost = upgrade_cost_of(state, cfg)                     # [N,2] 按类型/等级
    can_up = (actable & (is_hq | is_node_b | is_camp) & (state.btimer == 0) & cap_ok
              & jnp.all(stock >= up_cost, axis=-1))
    mask = mask.at[:, a_upgrade(cfg)].set(can_up)

    # 建训练营(v1.1):工人 + 基地解锁 + 半区有空槽 + 乐观付得起
    # (落格与最终扣费在 paid_orders_pass;同玩家同 tick 至多批准一座)
    camp_cost = jnp.asarray([cfg.camp_cost_ore, cfg.camp_cost_water], jnp.int32)
    can_camp = (actable & is_worker & (hq_lv >= cfg.camp_unlock_level)
                & free_in_half & jnp.all(stock >= camp_cost, axis=-1))
    mask = mask.at[:, a_build_camp(cfg)].set(can_camp)

    # 研发(v1.1):建成的营(level>=2)空闲 + 线级<营级 + 付得起
    # + 本玩家没有别的营在研同一条线(防同线并研双倍跳级)
    from .config import (
        BTASK_RESEARCH_INF,
        BTASK_RESEARCH_WORKER,
        LINE_INFANTRY,
        LINE_WORKER,
    )
    line_lv = state.upgrades[half]                            # [N,2]
    for line, code, cost_o, cost_w in (
        (LINE_INFANTRY, BTASK_RESEARCH_INF,
         cfg.inf_res_cost_ore, cfg.inf_res_cost_water),
        (LINE_WORKER, BTASK_RESEARCH_WORKER,
         cfg.worker_res_cost_ore, cfg.worker_res_cost_water),
    ):
        cur = line_lv[:, line].astype(jnp.int32)
        rcost = jnp.stack([jnp.asarray(cost_o)[cur], jnp.asarray(cost_w)[cur]], -1)
        busy_same = jnp.stack([
            jnp.any((owner == 0) & state.alive & (state.btype == code)),
            jnp.any((owner == 1) & state.alive & (state.btype == code)),
        ])[half]
        ok = (actable & is_camp & (lv >= 2) & (state.btimer == 0)
              & (cur < lv) & ~busy_same & jnp.all(stock >= rcost, axis=-1))
        mask = mask.at[:, a_research(line, cfg)].set(ok)
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

    is_stop = act == A_STOP
    is_att = act == A_ATTACK
    is_move = (act >= A_MOVE0) & (act < A_MOVE0 + 4)
    is_build = (act >= a_build(0)) & (act < a_build(cfg.n_nodes))
    is_harv = (act >= a_harvest(0, cfg)) & (act < a_harvest(cfg.n_nodes, cfg))
    is_tw = act == a_train_worker(cfg)
    is_ti = act == a_train_infantry(cfg)

    build_k = jnp.where(is_build, act - a_build(0), -1).astype(jnp.int8)
    harv_k = jnp.where(is_harv, act - a_harvest(0, cfg), -1).astype(jnp.int8)
    mdir = jnp.clip(act - A_MOVE0, 0, 3)

    order = state.order
    order = jnp.where(is_stop, ORDER_IDLE, order)
    order = jnp.where(is_att, ORDER_ATTACK, order)
    order = jnp.where(is_move, ORDER_MOVE, order)
    order = jnp.where(is_build, ORDER_BUILD, order)
    order = jnp.where(is_harv, ORDER_HARVEST, order)

    target_node = jnp.where(is_build, build_k, state.target_node)
    target_node = jnp.where(is_harv, harv_k, target_node)

    enemy_hq = jnp.asarray(mapdata.hq_pos, jnp.float32)[1 - owner.astype(jnp.int32)]
    target_cell = jnp.where(is_att[:, None], enemy_hq, state.target_cell)
    target_cell = jnp.where(is_move[:, None],
                            state.pos + _DIRS[mdir].astype(jnp.float32), target_cell)

    # 换指令时相位复位;带着货被改派采集 → 先回家卸货再进循环
    # (「满载」按该工人当前线级的载荷判;历史低级载荷也 >0 即回家,用 >0 更稳)
    new_cmd = is_stop | is_att | is_move | is_build | is_harv
    phase = jnp.where(new_cmd, PH_TO_NODE, state.phase).astype(jnp.int8)
    phase = jnp.where(is_harv & (state.cargo > 0), PH_TO_HQ, phase)

    # 训练扣费 + 开工
    ucost = unit_costs(cfg)
    spend = (is_tw[:, None] * ucost[0] + is_ti[:, None] * ucost[1])  # [N,2]
    pay = jnp.zeros_like(state.resources).at[owner.astype(jnp.int32)].add(spend)
    btype = jnp.where(is_tw, TYPE_WORKER, state.btype)
    btype = jnp.where(is_ti, TYPE_INFANTRY, btype).astype(jnp.int8)
    btimer = jnp.where(is_tw, cfg.worker_time, state.btimer)
    btimer = jnp.where(is_ti, cfg.infantry_time, btimer).astype(jnp.int16)

    return state._replace(
        order=order.astype(jnp.int8),
        phase=phase,
        target_node=target_node,
        target_cell=target_cell,
        btype=btype,
        btimer=btimer,
        resources=state.resources - pay,
    ), act
