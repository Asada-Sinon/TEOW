"""v1.8 审计:balanced profile 是否逐位复现 scripted(向后兼容)。
两条 controller 各驱一场同 seed 同 config 局,比较每帧动作 + 末态叶子。"""
import jax, jax.numpy as jnp
from teow.config import Config
from teow.step import new_world, build_step
from teow.controller import make_joint_controller

def rollout(names, seed, ep):
    cfg = Config(seed=seed, episode_len=ep)
    state, key, step_fn, m = new_world(cfg)
    joint = jax.jit(make_joint_controller(*names, cfg=cfg, mapdata=m))
    acts_mismatch = 0
    # 需要两条链共用同一 state 序列来比动作:改为分别跑,比末态
    st = state
    for t in range(ep):
        key, ka, ks = jax.random.split(key, 3)
        a = joint(st, ka)
        st = step_fn(st, a, ks)
        if bool(st.done): break
    return st

def per_tick_action_diff(seed, ep):
    """同一 state 序列(用 scripted 推进),每帧比较 scripted vs balanced 的动作。"""
    cfg = Config(seed=seed, episode_len=ep)
    state, key, step_fn, m = new_world(cfg)
    js = jax.jit(make_joint_controller("scripted","scripted","scripted","scripted", cfg=cfg, mapdata=m))
    jb = jax.jit(make_joint_controller("balanced","balanced","balanced","balanced", cfg=cfg, mapdata=m))
    st = state; total = 0; mism = 0
    for t in range(ep):
        key, ka, ks = jax.random.split(key, 3)
        a_s = js(st, ka); a_b = jb(st, ka)     # 同 state 同 key
        total += 1
        if not bool(jnp.array_equal(a_s, a_b)): mism += 1
        st = step_fn(st, a_s, ks)              # 用 scripted 推进
        if bool(st.done): break
    return total, mism

tot, mism = per_tick_action_diff(1, 400)
print(f"同 state 序列 400 拍: scripted vs balanced 动作不一致帧数 = {mism}/{tot}")

# 末态一致(各自独立驱动整局)
sa = rollout(("scripted",)*4, 1, 400)
sb = rollout(("balanced",)*4, 1, 400)
bad = [k for (k,x),(_,y) in zip(sa._asdict().items(), sb._asdict().items())
       if not bool(jnp.array_equal(x,y))]
print("独立整局末态不一致叶子:", bad)
print("scripted 末态 tick/winner:", int(sa.tick), int(sa.winner),
      " | balanced:", int(sb.tick), int(sb.winner))
