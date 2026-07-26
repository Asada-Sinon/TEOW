"""v1.8 审计:10 风格无一退化(必造军事 + 非无限 NOOP)。
两场 4 家局覆盖 8 风格(airtech/counter 已由 coverage 局覆盖),默认 gate。
每家报:峰值军队数、产出过的军事单位类型数、非 NOOP 动作占比。"""
import jax, jax.numpy as jnp, numpy as np
from teow.config import Config
from teow.step import new_world
from teow.controller import make_joint_controller, make_controller
from teow.state import owner_of_slots

A_NOOP = 0

def smoke(names, seed=1, ep=1500):
    cfg = Config(seed=seed, episode_len=ep)
    state, key, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    ctrls = [jax.jit(make_controller(nm, p, cfg, m)) for p, nm in enumerate(names)]
    is_army = np.asarray(cfg.is_combat_by_type, bool)
    E = cfg.e_max
    peak_army = [0]*4
    army_types = [set() for _ in range(4)]
    nonnoop = [0]*4; frames = 0
    st = state
    for t in range(ep):
        key, *ks = jax.random.split(key, 5)
        acts = []
        for p in range(4):
            a = ctrls[p](st, key=ks[p])
            acts.append(a)
            mine = slice(p*E, (p+1)*E)
            amine = np.asarray(a)[mine]
            nonnoop[p] += int((amine != A_NOOP).sum())
        frames += 1
        # merge
        merged = jnp.stack(acts)[owner.astype(jnp.int32), jnp.arange(cfg.n_total)]
        st = step_fn(st, merged, ks[0])
        et = np.asarray(st.etype); al = np.asarray(st.alive)
        for p in range(4):
            mine = slice(p*E, (p+1)*E)
            am = al[mine] & is_army[np.clip(et[mine],0,31)]
            peak_army[p] = max(peak_army[p], int(am.sum()))
            for ty in np.unique(et[mine][am]):
                army_types[p].add(int(ty))
        if bool(st.done): break
    for p, nm in enumerate(names):
        print(f"  {nm:9s} 峰值军队={peak_army[p]:3d}  军种数={len(army_types[p])}  "
              f"非NOOP动作累计={nonnoop[p]:5d}  {'退化!' if peak_army[p]==0 or nonnoop[p]==0 else 'ok'}")

print("=== 局A: rusher/timing/harasser/tempo ===")
smoke(("rusher","timing","harasser","tempo"))
print("=== 局B: balanced/boomer/turtle/chaos ===")
smoke(("balanced","boomer","turtle","chaos"))
