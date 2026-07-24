"""移动:动态距离场下降 + scatter-min 抢格仲裁。

两套距离场,各司其职:
- **静态 BFS 场**(map.py 预计算):只用于「到达/入驻/卸货/开工」的相邻判定——
  语义是与目标建筑相邻,与单位占用无关。
- **动态场**(本文件,每 tick min-plus 松弛重算):用于选路。障碍 = 静态不可通行
  + **静止单位**(没有移动类指令、或已到达目标的在场单位)。纯静态场会被站桩
  单位永久堵死(实测:满载工人被 HQ 旁发呆的初始工人堵在唯一下坡格上,循环
  卡死)——动态场让路径绕开他们。移动中的单位不算障碍(转瞬即离,算了会抖)。
  松弛 K=H+W 次覆盖全图;24×24×8 场每 tick ~90k 元素运算,单环境无感。
  NOTE(v2):vmap 数千 env 训练时此项是每 tick 主要开销,届时考虑降频重算/
  减少迭代数/增量更新。

其余纪律:
- 任意格目标(MOVE 一格)用 L1 贪心,不走场。
- 候选格排除「tick 初被占」的格(反正仲裁会拒,提前排除让单位选次优绕行);
  同 tick 抢同一空格用每 tick 随机优先级 scatter-min 仲裁,防 self-play 下标
  偏置(调研报告 §1)。
- 方向候选顺序对玩家 1 取反,保证 180° 旋转对称下两家行为互为镜像。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import TYPE_INFANTRY, TYPE_WORKER, Config
from .economy import occupancy_grid
from .map import BIG_DIST, MapData
from .state import (
    ORDER_ATTACK,
    ORDER_BUILD,
    ORDER_HARVEST,
    ORDER_IDLE,
    ORDER_MOVE,
    PH_MINING,
    PH_TO_HQ,
    WorldState,
)

_DIRS = jnp.asarray([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)  # N,E,S,W


def _relax_fields(goal_seeds: jax.Array, blocked: jax.Array,
                  cell_cost: jax.Array, k_iters: int) -> jax.Array:
    """min-plus 松弛出「绕开 blocked、按 cell_cost 计价」的距离场。
    goal_seeds bool [G,H,W];blocked bool [H,W] 硬障碍(目标格本身可以是障碍,
    种子强制为 0);cell_cost int32 [H,W] 进入该格的代价(空地 1,被移动单位
    占的格 1+congestion_cost——软障碍,让对向流互相绕道而不是对头冻结,
    见 audit P0-1)。"""
    big = jnp.int32(BIG_DIST)
    d = jnp.where(goal_seeds, 0, big)

    def body(d, _):
        pad = ((0, 0), (0, 0))
        up = jnp.pad(d[:, 1:, :], (pad[0], (0, 1), pad[1]), constant_values=big)
        dn = jnp.pad(d[:, :-1, :], (pad[0], (1, 0), pad[1]), constant_values=big)
        lf = jnp.pad(d[:, :, 1:], (pad[0], pad[1], (0, 1)), constant_values=big)
        rt = jnp.pad(d[:, :, :-1], (pad[0], pad[1], (1, 0)), constant_values=big)
        nd = jnp.minimum(jnp.minimum(up, dn), jnp.minimum(lf, rt)) + cell_cost[None]
        nd = jnp.minimum(d, nd)
        nd = jnp.where(blocked[None], big, nd)
        nd = jnp.where(goal_seeds, 0, nd)
        return nd, None

    d, _ = jax.lax.scan(body, d, None, length=k_iters)
    return d


def movement_tick(state: WorldState, cfg: Config, mapdata: MapData,
                  owner: jax.Array, key: jax.Array) -> WorldState:
    n = cfg.n_total
    h, w = cfg.grid_h, cfg.grid_w
    static_dist = jnp.asarray(mapdata.dist_fields)         # [G,H,W] 相邻判定用
    passable = jnp.asarray(mapdata.passable)
    st = state

    is_unit = (st.etype == TYPE_WORKER) | (st.etype == TYPE_INFANTRY)
    on_board = st.alive & ~st.inside
    tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    own_i = owner.astype(jnp.int32)

    # ---- 目标场编号与是否用场 ----
    goal = jnp.where(st.order == ORDER_ATTACK, cfg.n_nodes + (1 - own_i), tn)
    goal = jnp.where((st.order == ORDER_HARVEST) & (st.phase == PH_TO_HQ),
                     cfg.n_nodes + own_i, goal)
    use_field = ((st.order == ORDER_HARVEST) | (st.order == ORDER_BUILD)
                 | (st.order == ORDER_ATTACK))

    # ---- 到达判定(静态相邻语义)----
    here_static = static_dist[goal, st.pos[:, 0], st.pos[:, 1]]
    l1_here = jnp.sum(jnp.abs(st.pos - st.target_cell), axis=-1)
    arrived = jnp.where(use_field, here_static <= 1, l1_here == 0)
    # 步兵 attack-move:身边已有敌人就地开打,不再挤路
    cheb = jnp.max(jnp.abs(st.pos[:, None, :] - st.pos[None, :, :]), axis=-1)
    enemy_adj = jnp.any(
        (cheb <= 1) & (owner[None, :] != owner[:, None])
        & (st.alive & ~st.inside)[None, :], axis=-1)
    arrived = arrived | ((st.order == ORDER_ATTACK) & enemy_adj)

    moving_order = ((st.order == ORDER_MOVE) | (st.order == ORDER_ATTACK)
                    | (st.order == ORDER_BUILD)
                    | ((st.order == ORDER_HARVEST) & (st.phase != PH_MINING)))
    wants = on_board & is_unit & moving_order & ~arrived & (st.cooldown == 0)

    # ---- 动态场:静止单位=硬障碍(站桩会永久堵路);移动单位=软障碍
    # (穿其格 +congestion_cost),对向工人流据此互相绕道,消除对头死锁
    # (audit P0-1:两拨采集工在走廊对头相遇后全体永久冻结)----
    moving_unit = on_board & is_unit & moving_order & ~arrived
    stationary = on_board & ~moving_unit
    occ_stat = (jnp.zeros((h, w), bool)
                .at[st.pos[:, 0], st.pos[:, 1]].max(stationary))
    occ_mov = (jnp.zeros((h, w), bool)
               .at[st.pos[:, 0], st.pos[:, 1]].max(moving_unit))
    blocked = ~passable | occ_stat
    cell_cost = 1 + cfg.congestion_cost * occ_mov.astype(jnp.int32)
    dyn = _relax_fields(jnp.asarray(mapdata.goal_seeds), blocked, cell_cost, h + w)

    def fval(cell: jax.Array) -> jax.Array:
        fv = dyn[goal, cell[:, 0], cell[:, 1]]
        l1 = jnp.sum(jnp.abs(cell - st.target_cell), axis=-1)
        return jnp.where(use_field, fv, l1)

    here = fval(st.pos)

    # ---- 候选格评分:4 邻(玩家 1 方向序取反),排除界外/静态障碍/tick 初被占 ----
    occ = occupancy_grid(st, cfg)
    dirs_pp = jnp.where((owner == 0)[:, None, None], _DIRS[None], -_DIRS[None])
    cands = st.pos[:, None, :] + dirs_pp                   # [N,4,2]
    inb = ((cands[:, :, 0] >= 0) & (cands[:, :, 0] < h)
           & (cands[:, :, 1] >= 0) & (cands[:, :, 1] < w))
    cc = jnp.clip(cands, 0, jnp.asarray([h - 1, w - 1]))
    ok = inb & passable[cc[:, :, 0], cc[:, :, 1]] & ~occ[cc[:, :, 0], cc[:, :, 1]]
    cand_f = jnp.stack([fval(cc[:, d]) for d in range(4)], axis=1)  # [N,4]
    cand_f = jnp.where(ok, cand_f, BIG_DIST)
    best = jnp.argmin(cand_f, axis=1)
    best_f = jnp.take_along_axis(cand_f, best[:, None], axis=1)[:, 0]
    # 严格改善才走(等值滑动会来回振荡);全无效时 argmin 返 0,由 best_f 门控
    do_move = wants & (best_f < here)
    prop = jnp.where(do_move[:, None],
                     jnp.take_along_axis(cc, best[:, None, None].repeat(2, -1),
                                         axis=1)[:, 0], st.pos)

    # ---- 抢格仲裁:每 tick 随机优先级 scatter-min ----
    prio = jax.random.permutation(key, n).astype(jnp.int32)
    cell_id = prop[:, 0] * w + prop[:, 1]
    claim = (jnp.full(cfg.n_cells, jnp.iinfo(jnp.int32).max)
             .at[cell_id].min(jnp.where(do_move, prio, jnp.iinfo(jnp.int32).max)))
    win = do_move & (claim[cell_id] == prio)

    new_pos = jnp.where(win[:, None], prop, st.pos)
    cooldown = jnp.where(win, cfg.move_cooldown, st.cooldown).astype(jnp.int8)

    # MOVE 到达(含本 tick 走到)→ 转 IDLE
    l1_after = jnp.sum(jnp.abs(new_pos - st.target_cell), axis=-1)
    move_done = (st.order == ORDER_MOVE) & (l1_after == 0)
    order = jnp.where(move_done, ORDER_IDLE, st.order).astype(jnp.int8)

    return st._replace(pos=new_pos, cooldown=cooldown, order=order)
