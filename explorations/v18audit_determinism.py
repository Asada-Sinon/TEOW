"""v1.8 审计:同 (init_state, seed) 两次 rollout 是否逐位一致(含怪物子表),
跑进异界之门 overtime。回答:monster 子系统是否破坏 tick 决定论。"""
import sys
import jax, jax.numpy as jnp
from teow.config import Config
from teow.step import new_world, make_scan
from teow.controller import make_joint_controller

def run(names, seed, gate, ep):
    cfg = Config(seed=seed, gate_open_tick=gate, episode_len=ep,
                 start_ore=8000, start_water=8000)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller(*names, cfg=cfg, mapdata=m)
    st, _, dones = make_scan(step_fn, joint)(state, key, ep)
    return st, cfg

names = ("airtech", "turtle", "boomer", "counter")
# gate 低,episode 短 → 保证进 overtime 且出怪、监测怪物子表
a, ca = run(names, 3, 400, 900)
b, cb = run(names, 3, 400, 900)

bad = []
for (ka, xa), (kb, xb) in zip(a._asdict().items(), b._asdict().items()):
    if not bool(jnp.array_equal(xa, xb)):
        bad.append(ka)
print("done_a=", bool(a.done), "winner_a=", int(a.winner), "tick_a=", int(a.tick))
print("done_b=", bool(b.done), "winner_b=", int(b.winner), "tick_b=", int(b.tick))
print("monsters_alive_total_a=", int(a.monster_alive.sum()))
print("mismatch_leaves=", bad)
# 交叉 seed:不同 seed 应不同(否则 seed 没进 key)——反向健全性
c, _ = run(names, 7, 400, 900)
diff = [k for (k, xa), (_, xc) in zip(a._asdict().items(), c._asdict().items())
        if not bool(jnp.array_equal(xa, xc))]
print("seed3_vs_seed7_differs_in", len(diff), "leaves (>0 表示 seed 确实进了 key)")
sys.exit(1 if bad else 0)
