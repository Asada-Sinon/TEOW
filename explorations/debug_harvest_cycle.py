"""诊断:采集一体循环为何零产出(test_build_mine_then_harvest_cycle 失败)。
用法:JAX_PLATFORMS=cpu <python> explorations/debug_harvest_cycle.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP, a_build, a_harvest
from teow.config import Config
from teow.step import new_world

cfg = Config()
state, _, step_fn, m = new_world(cfg)
W0 = 1
node = 0

key = jax.random.PRNGKey(0)
# 建矿
for t in range(200):
    acts = jnp.full(cfg.n_total, A_NOOP, jnp.int32)
    if t == 0:
        acts = acts.at[W0].set(a_build(node))
    key, sub = jax.random.split(key)
    state = step_fn(state, acts, sub)
print(f"建矿后: node_owner={int(state.node_owner[node])} "
      f"ent={int(state.node_ent[node])} worker_pos={state.pos[W0].tolist()} "
      f"order={int(state.order[W0])}")

# 采集,逐 tick 打印前 80 拍
key = jax.random.PRNGKey(1)
for t in range(80):
    acts = jnp.full(cfg.n_total, A_NOOP, jnp.int32)
    if t == 0:
        acts = acts.at[W0].set(a_harvest(node, cfg))
    key, sub = jax.random.split(key)
    state = step_fn(state, acts, sub)
    if t < 5 or t % 10 == 0 or bool(state.cargo[W0] > 0):
        d0 = int(m.dist_fields[node][int(state.pos[W0][0]), int(state.pos[W0][1])])
        dh = int(m.dist_fields[cfg.n_nodes][int(state.pos[W0][0]), int(state.pos[W0][1])])
        print(f"t={t:3d} pos={state.pos[W0].tolist()} d_node={d0} d_hq={dh} "
              f"order={int(state.order[W0])} phase={int(state.phase[W0])} "
              f"inside={bool(state.inside[W0])} timer={int(state.mine_timer[W0])} "
              f"cargo={int(state.cargo[W0])} res={state.resources[0].tolist()}")
