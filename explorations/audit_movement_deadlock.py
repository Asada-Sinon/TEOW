"""v1.0 审计:诊断采集工人永久卡死的机制。
重放 seed 1 scripted vs scripted 到 600 tick,对末态复算 movement_tick 内部量:
每个 HARVEST 工人 wants/arrived/here/候选格值/占用者是谁,判定是否「对向对撞死锁」
(A 的唯一改善格被 B 占,B 的唯一改善格被 A 占,движ规则不允许换位)。
并输出每个工人的位置从哪个 tick 起不再变化。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_movement_deadlock.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from teow.config import TYPE_INFANTRY, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.economy import occupancy_grid
from teow.movement import _relax_fields
from teow.state import (
    ORDER_ATTACK,
    ORDER_BUILD,
    ORDER_HARVEST,
    ORDER_MOVE,
    PH_MINING,
    PH_TO_HQ,
    owner_of_slots,
)
from teow.step import new_world

T_END = 600
cfg = Config(seed=1)
state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller("scripted", "scripted", cfg, m))
owner = np.asarray(owner_of_slots(cfg))

last_moved = np.full(cfg.n_total, -1)
prev_pos = np.asarray(state.pos).copy()
for t in range(T_END):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    pos = np.asarray(state.pos)
    moved = (pos != prev_pos).any(axis=1)
    last_moved[moved] = t
    prev_pos = pos.copy()

st = state
n = cfg.n_total
own_i = jnp.asarray(owner, jnp.int32)
is_unit = (st.etype == TYPE_WORKER) | (st.etype == TYPE_INFANTRY)
on_board = st.alive & ~st.inside
tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
static_dist = jnp.asarray(m.dist_fields)

goal = jnp.where(st.order == ORDER_ATTACK, cfg.n_nodes + (1 - own_i), tn)
goal = jnp.where((st.order == ORDER_HARVEST) & (st.phase == PH_TO_HQ),
                 cfg.n_nodes + own_i, goal)
use_field = ((st.order == ORDER_HARVEST) | (st.order == ORDER_BUILD)
             | (st.order == ORDER_ATTACK))
here_static = static_dist[goal, st.pos[:, 0], st.pos[:, 1]]
arrived = jnp.where(use_field, here_static <= 1,
                    jnp.sum(jnp.abs(st.pos - st.target_cell), axis=-1) == 0)
moving_order = ((st.order == ORDER_MOVE) | (st.order == ORDER_ATTACK)
                | (st.order == ORDER_BUILD)
                | ((st.order == ORDER_HARVEST) & (st.phase != PH_MINING)))
wants = on_board & is_unit & moving_order & ~arrived

stationary = on_board & (~(is_unit & moving_order & ~arrived))
occ_stat = jnp.zeros((cfg.grid_h, cfg.grid_w), bool).at[
    st.pos[:, 0], st.pos[:, 1]].max(stationary)
blocked = ~jnp.asarray(m.passable) | occ_stat
dyn = _relax_fields(jnp.asarray(m.goal_seeds), blocked, cfg.grid_h + cfg.grid_w)
occ = occupancy_grid(st, cfg)

pos_np = np.asarray(st.pos)
who = {}  # cell -> slot
for i in range(n):
    if bool(on_board[i]):
        who[(int(pos_np[i, 0]), int(pos_np[i, 1]))] = i

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
print(f"t={T_END} res={np.asarray(st.resources).tolist()}")
print("HARVEST 工人逐个(wants 但走不动的,列出改善候选格被谁占):")
blocker_pairs = []
for i in range(n):
    if not (bool(st.alive[i]) and int(st.order[i]) == ORDER_HARVEST
            and not bool(st.inside[i])):
        continue
    r, c = int(pos_np[i, 0]), int(pos_np[i, 1])
    g = int(goal[i])
    here = int(dyn[g, r, c])
    cands = []
    for dr, dc in DIRS:
        rr, cc2 = r + dr, c + dc
        if not (0 <= rr < cfg.grid_h and 0 <= cc2 < cfg.grid_w):
            continue
        f = int(dyn[g, rr, cc2])
        occ_by = who.get((rr, cc2))
        cands.append((f, (rr, cc2), occ_by, bool(occ[rr, cc2]),
                      bool(m.passable[rr, cc2])))
    improving = [x for x in cands if x[0] < here and x[4]]
    free_improving = [x for x in improving if not x[3]]
    tag = ("卡死" if (bool(wants[i]) and not free_improving)
           else ("wants" if bool(wants[i]) else "到达/停"))
    print(f"  w{i} p{owner[i]} pos=({r},{c}) ph={int(st.phase[i])} tn={int(st.target_node[i])} "
          f"goal={g} here={here} last_moved={last_moved[i]} [{tag}] "
          f"改善格被占者={[(x[1], x[2]) for x in improving]}")
    for x in improving:
        if x[2] is not None:
            blocker_pairs.append((i, x[2]))

# 对撞环:i 的改善格被 j 占,且 j 的改善格被 i 占
pairset = set(blocker_pairs)
mutual = [(a, b) for (a, b) in pairset if (b, a) in pairset and a < b]
print(f"\n互相堵死(swap 死锁)对: {mutual}")
