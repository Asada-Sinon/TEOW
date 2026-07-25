"""v1.6 涌现验证:富开局覆盖局里 scripted AI 用上整个 v1.6 内容集。

配置=explorations/calibrate_v16_coverage.py 定标值(critic M-4:water 是
兵营 5→6 的历史卡点,8000/8000 才解;seed 0 实测时间线 laser 1950 →
airship 3650 → dragon 3875 → catapult 4225 → 4456 分胜负)。
断言全部走事件旗,不做时间线断言。"""

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

COVER_CFG = dict(start_ore=8000, start_water=8000, ai_base_level_target=7,
                 ai_worker_target=6, ai_attack_threshold=25)


def test_scripted_exercises_v16_content():
    cfg = Config(seed=0, **COVER_CFG)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller(*(["scripted"] * cfg.n_players),
                                  cfg=cfg, mapdata=m)

    watch = (TYPE_MAGETOWER, TYPE_LANDMINE, TYPE_FLAMER, TYPE_LASER,
             TYPE_CATAPULT, TYPE_AIRSHIP, TYPE_DRAGON)

    def body(carry, _):
        st, k = carry
        k, k_act, k_step = jax.random.split(k, 3)
        st = step_fn(st, joint(st, k_act), k_step)
        al = st.alive
        seen = jnp.stack([jnp.any(al & (st.etype == t)) for t in watch])
        return (st, k), seen

    (fin, _), seen = jax.lax.scan(body, (state, key), None,
                                  length=cfg.episode_len)
    seen_any = jnp.any(seen, axis=0)
    names = ("法师塔", "地雷", "喷火器", "激光炮", "投石车", "飞艇", "龙骑兵")
    for i, nm in enumerate(names):
        assert bool(seen_any[i]), f"整局未出现 {nm}"
    assert int(fin.winner) != -1, "覆盖局应分出结果(胜或和)"
