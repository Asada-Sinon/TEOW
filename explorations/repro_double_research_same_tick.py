"""复现:同玩家两座建成营,同 tick 对同一条线下研发 → 是否双扣费单跳级?"""
import sys
sys.path.insert(0, "/home/michael/workspace/pi05/temp/TEOW/src")
import jax, jax.numpy as jnp
import numpy as np
from teow.config import TYPE_CAMP, Config
from teow.actions import a_research
from teow.step import new_world
from teow.state import hq_slot

cfg = Config(seed=0)
state, key, step_fn, m = new_world(cfg)
# 手搓:基地2级,两座建成营(slot 2,3),资源充足
st = state._replace(
    level=state.level.at[0].set(2).at[2].set(2).at[3].set(2),
    alive=state.alive.at[2].set(True).at[3].set(True),
    etype=state.etype.at[2].set(TYPE_CAMP).at[3].set(TYPE_CAMP),
    pos=state.pos.at[2].set(jnp.asarray([10, 3])).at[3].set(jnp.asarray([10, 5])),
    hp=state.hp.at[2].set(150).at[3].set(150),
    resources=state.resources.at[0].set(jnp.asarray([500, 500])),
)
acts = jnp.zeros(cfg.n_total, jnp.int32)
acts = acts.at[2].set(a_research(0, cfg)).at[3].set(a_research(0, cfg))  # 两营同下步兵线
r0 = np.asarray(st.resources[0])
st1 = step_fn(st, acts, jax.random.PRNGKey(1))
r1 = np.asarray(st1.resources[0])
print(f"下单前 res={r0.tolist()} 下单后 res={r1.tolist()} 单价=({cfg.inf_res_cost_ore[1]},{cfg.inf_res_cost_water[1]})")
print(f"营2 btype={int(st1.btype[2])} btimer={int(st1.btimer[2])};营3 btype={int(st1.btype[3])} btimer={int(st1.btimer[3])}")
# 跑到完成
noop = jnp.zeros(cfg.n_total, jnp.int32)
s = st1
for t in range(cfg.inf_res_time[1] + 2):
    s = step_fn(s, noop, jax.random.PRNGKey(2 + t))
print(f"完成后 步兵线={int(s.upgrades[0, 0])} res={np.asarray(s.resources[0]).tolist()}")
paid = (r0 - r1).tolist()
print(f"结论: 扣了 {paid},线到 {int(s.upgrades[0,0])} 级"
      f"(单研一次应扣 [60,40] 得 2 级)")
