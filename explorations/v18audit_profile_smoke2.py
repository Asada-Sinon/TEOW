"""v1.8 审计续:富开局(8000/8000,消除经济瓶颈)下,'慢' 风格是否确实造军事。
若富开局仍 0 军事 → 退化;若造军事 → 上一测 0 只是默认经济慢。"""
import jax, jax.numpy as jnp, numpy as np
from teow.config import Config
from teow.step import new_world
from teow.controller import make_controller
from teow.state import owner_of_slots

def smoke(names, seed=1, ep=900):
    cfg = Config(seed=seed, episode_len=ep, start_ore=8000, start_water=8000)
    state, key, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    ctrls = [jax.jit(make_controller(nm, p, cfg, m)) for p, nm in enumerate(names)]
    is_army = np.asarray(cfg.is_combat_by_type, bool); E = cfg.e_max
    peak=[0]*4; types=[set() for _ in range(4)]; base_peak=[1]*4
    st=state
    for t in range(ep):
        key,*ks = jax.random.split(key,5)
        acts=[ctrls[p](st,key=ks[p]) for p in range(4)]
        merged=jnp.stack(acts)[owner.astype(jnp.int32), jnp.arange(cfg.n_total)]
        st=step_fn(st,merged,ks[0])
        et=np.asarray(st.etype); al=np.asarray(st.alive); lv=np.asarray(st.level)
        for p in range(4):
            sl=slice(p*E,(p+1)*E)
            am=al[sl]&is_army[np.clip(et[sl],0,31)]
            peak[p]=max(peak[p],int(am.sum()))
            base_peak[p]=max(base_peak[p], int(lv[p*E]))
            for ty in np.unique(et[sl][am]): types[p].add(int(ty))
        if bool(st.done): break
    for p,nm in enumerate(names):
        tag = '退化!' if peak[p]==0 else 'ok'
        print(f"  {nm:9s} 峰值军队={peak[p]:3d} 军种数={len(types[p])} 峰值基地={base_peak[p]} {tag}")

print("=== 富开局 balanced/timing/tempo/turtle ===")
smoke(("balanced","timing","tempo","turtle"))
print("=== 富开局 chaos/boomer/counter/airtech ===")
smoke(("chaos","boomer","counter","airtech"))
