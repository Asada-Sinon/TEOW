"""v1.6 覆盖局定标(critic M-4):扫富开局参数,找「投石车/飞艇/龙/四防御建筑
全出现」的最小预算;target 7 的升本预留若挤死练兵线则记录并回退方案。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/calibrate_v16_coverage.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import jax.numpy as jnp

from teow.config import (
    TYPE_AIRSHIP,
    TYPE_CATAPULT,
    TYPE_DRAGON,
    TYPE_FLAMER,
    TYPE_LANDMINE,
    TYPE_LASER,
    TYPE_MAGETOWER,
    Config,
)
from teow.controller import make_joint_controller
from teow.step import new_world

WATCH = (("catapult", TYPE_CATAPULT), ("airship", TYPE_AIRSHIP),
         ("dragon", TYPE_DRAGON), ("magetower", TYPE_MAGETOWER),
         ("landmine", TYPE_LANDMINE), ("flamer", TYPE_FLAMER),
         ("laser", TYPE_LASER))

for ore, water, target, thr in ((8000, 8000, 7, 25), (12000, 12000, 7, 25)):
    cfg = Config(seed=0, start_ore=ore, start_water=water,
                 ai_base_level_target=target, ai_worker_target=6,
                 ai_attack_threshold=thr)
    state, key, step_fn, m = new_world(cfg)
    joint = jax.jit(make_joint_controller(*(["scripted"] * 4), cfg=cfg,
                                          mapdata=m))
    st = state
    flags: dict = {}
    for t in range(cfg.episode_len):
        key, ka, ks = jax.random.split(key, 3)
        st = step_fn(st, joint(st, ka), ks)
        if t % 25 == 0:
            al, e = st.alive, st.etype
            for nm, tp in WATCH:
                if nm not in flags and bool(jnp.any(al & (e == tp))):
                    flags[nm] = t
            if "boom" not in flags:
                # 地雷爆炸事件:曾出现过雷、当前雷数下降(粗判)
                pass
        if bool(st.done):
            break
    base_lvs = [int(st.level[p * cfg.e_max]) for p in range(4)]
    missing = [nm for nm, _ in WATCH if nm not in flags]
    print(f"ore={ore} water={water} target={target} thr={thr} "
          f"done={int(st.tick)} win={int(st.winner)} base={base_lvs}")
    print(f"  flags={flags}")
    print(f"  MISSING={missing}")
    if not missing:
        print("  ---- 全覆盖,采用该配置 ----")
        break
