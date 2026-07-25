"""决定论:同 seed 同控制器 → 全轨迹逐位一致(引擎硬约束,CLAUDE.md 项目特定)。"""

import jax

from teow.config import Config
from teow.controller import make_joint_controller
from teow.step import make_scan, new_world


def rollout(seed: int, n: int = 300):
    cfg = Config(seed=seed)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller("scripted", "random", "scripted", "random", cfg=cfg, mapdata=m)
    scan = make_scan(step_fn, joint)
    final, _, dones = scan(state, key, n)
    return final


def test_bitwise_identical_same_seed():
    a = rollout(0)
    b = rollout(0)
    for la, lb in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert (la == lb).all()


def test_scan_runs_with_other_seed():
    # 换 seed 只要求能跑完不炸(轨迹大概率不同,但不作硬断言——
    # 早期 tick 的随机仲裁可能根本没被触发)
    rollout(1)
