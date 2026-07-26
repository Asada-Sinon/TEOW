"""v1.8 审计:seed 是否真的进入 rollout(movement/claim 仲裁 key)。
默认资源(非富开局)+ 默认 gate,多 seed 比较末态差异叶子数。"""
import jax, jax.numpy as jnp
from teow.config import Config
from teow.step import new_world, make_scan
from teow.controller import make_joint_controller

def run(names, seed, ep, **kw):
    cfg = Config(seed=seed, episode_len=ep, **kw)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller(*names, cfg=cfg, mapdata=m)
    st, _, _ = make_scan(step_fn, joint)(state, key, ep)
    return st

def ndiff(a, b):
    return sum(0 if bool(jnp.array_equal(xa, xb)) else 1
              for (_, xa), (_, xb) in zip(a._asdict().items(), b._asdict().items()))

# 场景1:默认资源,scripted x4,短局(有真实工人抢点/抢格)
s1a = run(("scripted",)*4, 1, 500)
s1b = run(("scripted",)*4, 2, 500)
print("default scripted x4, 500t: seed1 vs seed2 differ in", ndiff(s1a, s1b), "leaves")

# 场景2:含 chaos(显式用 key)
s2a = run(("chaos","scripted","scripted","scripted"), 1, 500)
s2b = run(("chaos","scripted","scripted","scripted"), 2, 500)
print("chaos+3scripted, 500t: seed1 vs seed2 differ in", ndiff(s2a, s2b), "leaves")

# 场景3:富开局(coverage 同款)是否 seed 不敏感
s3a = run(("airtech","turtle","boomer","counter"), 3, 500, start_ore=8000, start_water=8000)
s3b = run(("airtech","turtle","boomer","counter"), 5, 500, start_ore=8000, start_water=8000)
print("rich coverage, 500t: seed3 vs seed5 differ in", ndiff(s3a, s3b), "leaves")

# 场景4:两 random 对抗(纯 key 驱动)
s4a = run(("random",)*4, 1, 300)
s4b = run(("random",)*4, 2, 300)
print("random x4, 300t: seed1 vs seed2 differ in", ndiff(s4a, s4b), "leaves")
