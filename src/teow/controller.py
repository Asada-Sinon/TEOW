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
    a_harvest,
    a_research,
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

    # ---- HQ:缺工人补工人 → 富余则升本(到 ai_base_level_target 止)→ 否则爆兵 ----
    n_workers = jnp.sum(mine & (st.etype == TYPE_WORKER))
    hq_slot_p = player * cfg.e_max
    base_lv = st.level[hq_slot_p].astype(jnp.int32)
    up_cost = jnp.asarray(
        [cfg.base_up_cost_ore, cfg.base_up_cost_water], jnp.int32)[:, base_lv]
    rich_for_base = jnp.all(st.resources[player]
                            >= up_cost + cfg.ai_upgrade_reserve)
    hq_act = jnp.where(n_workers < cfg.ai_worker_target,
                       a_train_worker(cfg), a_train_infantry(cfg))
    hq_act = jnp.where(rich_for_base & (base_lv < cfg.ai_base_level_target),
                       a_upgrade(cfg), hq_act)

    # ---- 工人:一个去建(最近的无主点),其余采(最近的有余位己方点)----
    node_d = dist[:cfg.n_nodes, st.pos[:, 0], st.pos[:, 1]]  # [Nn,N]
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
    harvestable = ((st.node_owner == player) & (st.node_ent >= 0)
                   & (assigned < cfg.node_capacity))                        # [Nn]

    bd = jnp.where(claimable[:, None], node_d, BIG_DIST)
    build_k = jnp.argmin(bd, axis=0)                       # [N] 最近无主点
    has_build = jnp.any(claimable)
    hd = jnp.where(harvestable[:, None], node_d, BIG_DIST)
    harv_k = jnp.argmin(hd, axis=0)
    has_harv = jnp.any(harvestable)

    idle_worker = mine & (st.etype == TYPE_WORKER) & (st.order == ORDER_IDLE) & ~st.inside
    # 「派一个建造者」:空闲工人中槽号最小的那个;全 False 时 argmax 返 0,用 has 门控
    builder = jnp.argmax(idle_worker)
    is_builder = (jnp.arange(n) == builder) & jnp.any(idle_worker) & has_build

    worker_act = jnp.where(has_harv, a_harvest(0, cfg) + harv_k, A_NOOP)
    worker_act = jnp.where(is_builder, a_build(0) + build_k, worker_act)
    worker_act = jnp.where(idle_worker, worker_act, A_NOOP)

    # ---- 步兵:攒够阈值全军 attack-move(重复下达无害)----
    n_inf = jnp.sum(mine & (st.etype == TYPE_INFANTRY))
    inf_act = jnp.where(n_inf >= cfg.ai_attack_threshold, A_ATTACK, A_NOOP)

    # ---- 训练营(v1.1):优先研低的那条线(掩码兜底非法),双线到顶则升营 ----
    line_lv = st.upgrades[player]                          # [2]
    low_line = jnp.argmin(line_lv).astype(jnp.int32)       # 平级取步兵线(0)
    camp_act = jnp.where(
        line_lv[low_line] < st.level.astype(jnp.int32), a_research(0, cfg) + low_line,
        a_upgrade(cfg))                                    # [N](逐营取自身 level 比较)

    # ---- 建营:基地达标且没有己方营(含在建)时,派一个空闲工人就地起营 ----
    has_camp = jnp.any(mine & (st.etype == TYPE_CAMP))
    base_ok = st.level[player * cfg.e_max] >= cfg.camp_unlock_level
    camp_builder = jnp.argmax(idle_worker)
    is_camp_builder = ((jnp.arange(n) == camp_builder) & jnp.any(idle_worker)
                       & base_ok & ~has_camp)

    # ---- 矿/泵:库存富余就升(掩码兜底上限链/施工中)----
    is_node_b = (st.etype == TYPE_MINE) | (st.etype == TYPE_PUMP)
    rich_for_node = jnp.all(
        st.resources[player] >= cfg.ai_upgrade_reserve)
    node_act = jnp.where(rich_for_node, a_upgrade(cfg), A_NOOP)

    act = jnp.full(n, A_NOOP, jnp.int32)
    act = jnp.where(mine & (st.etype == TYPE_HQ), hq_act, act)
    act = jnp.where(idle_worker, worker_act, act)
    act = jnp.where(is_camp_builder, a_build_camp(cfg), act)
    act = jnp.where(mine & (st.etype == TYPE_CAMP), camp_act, act)
    act = jnp.where(mine & is_node_b, node_act, act)
    act = jnp.where(mine & (st.etype == TYPE_INFANTRY), inf_act, act)

    # ---- 让路:空闲单位贴在自家 HQ 正邻格(=卸货格)上会堵死运矿工人,
    # 命令它朝「远离 HQ」的方向挪一格(实测 4 个待命步兵站满卸货环锁死经济)----
    hq_field = dist[cfg.n_nodes + player]                  # [H,W] 静态
    on_ring = hq_field[st.pos[:, 0], st.pos[:, 1]] == 1
    is_idle_unit = (mine & ~st.inside & (st.order == ORDER_IDLE)
                    & ((st.etype == TYPE_WORKER) | (st.etype == TYPE_INFANTRY)))
    dirs = jnp.asarray([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)
    cand = st.pos[:, None, :] + dirs[None]                 # [N,4,2]
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
