"""诊断:scripted 对局矿石收入为何 ~0(水正常)。
用法:JAX_PLATFORMS=cpu <python> explorations/debug_ore_starvation.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp

from teow.config import TYPE_INFANTRY, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.state import ORDER_HARVEST, owner_of_slots
from teow.step import new_world

cfg = Config()
state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller("scripted", "scripted", cfg, m))
owner = owner_of_slots(cfg)

for t in range(1200):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    if t % 200 == 0 or t == 1199:
        st = state
        p0 = owner == 0
        nw = int(jnp.sum(p0 & st.alive & (st.etype == TYPE_WORKER)))
        ni = int(jnp.sum(p0 & st.alive & (st.etype == TYPE_INFANTRY)))
        tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
        assigned = [int(jnp.sum(p0 & st.alive & (st.order == ORDER_HARVEST)
                                & (st.target_node == k))) for k in range(6)]
        inside = [int(jnp.sum(p0 & st.inside & (st.target_node == k)))
                  for k in range(6)]
        print(f"t={t:4d} res_p0={st.resources[0].tolist()} workers={nw} inf={ni} "
              f"node_owner={st.node_owner.tolist()} assigned={assigned} inside={inside} "
              f"hq_btype={int(st.btype[0])} hq_btimer={int(st.btimer[0])}")

# 追加:细看 p0 指派到 node0 的工人卡在什么状态
st = state
sel = (owner == 0) & st.alive & (st.order == ORDER_HARVEST) & (st.target_node == 0)
idx = [int(i) for i in jnp.nonzero(sel)[0]]
d0 = m.dist_fields[0]
dh = m.dist_fields[cfg.n_nodes]
for i in idx:
    r, c = int(st.pos[i][0]), int(st.pos[i][1])
    print(f"slot={i} pos=({r},{c}) d_node={int(d0[r,c])} d_hq={int(dh[r,c])} "
          f"phase={int(st.phase[i])} inside={bool(st.inside[i])} "
          f"cargo={int(st.cargo[i])} timer={int(st.mine_timer[i])} "
          f"cooldown={int(st.cooldown[i])}")

# 追加2:复算 movement 内部量,看动态场/占用为何锁死
import numpy as np
from teow.economy import occupancy_grid
from teow.movement import _relax_fields
from teow.config import TYPE_WORKER, TYPE_HQ
from teow.state import ORDER_MOVE, ORDER_ATTACK, ORDER_BUILD, PH_MINING

st = state
is_unit = (st.etype == TYPE_WORKER) | (st.etype == TYPE_INFANTRY)
on_board = st.alive & ~st.inside
moving_order = ((st.order == ORDER_MOVE) | (st.order == ORDER_ATTACK)
                | (st.order == ORDER_BUILD)
                | ((st.order == ORDER_HARVEST) & (st.phase != PH_MINING)))
goal = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
here_static = jnp.asarray(m.dist_fields)[goal, st.pos[:, 0], st.pos[:, 1]]
arrived_f = here_static <= 1
stationary = on_board & (~(is_unit & moving_order & ~arrived_f))
occ_stat = jnp.zeros((cfg.grid_h, cfg.grid_w), bool).at[st.pos[:, 0], st.pos[:, 1]].max(stationary)
blocked = ~jnp.asarray(m.passable) | occ_stat
dyn = _relax_fields(jnp.asarray(m.goal_seeds), blocked, cfg.grid_h + cfg.grid_w)

print("\np0 全部在场单位:")
for i in range(cfg.e_max):
    if bool(st.alive[i]) and not bool(st.inside[i]):
        print(f"  slot={i} type={int(st.etype[i])} pos={st.pos[i].tolist()} "
              f"order={int(st.order[i])} phase={int(st.phase[i])} "
              f"stationary={bool(stationary[i])}")
print("\n矿0周边 dyn[goal=0] (行0-6,列5-11):")
print(np.asarray(dyn[0])[0:7, 5:12])
print("blocked 同窗口:")
print(np.asarray(blocked)[0:7, 5:12].astype(int))
print("HQ 场 dyn[6] 同窗口:")
print(np.asarray(dyn[6])[0:7, 5:12])
