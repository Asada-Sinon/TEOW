"""v1.3 Phase 5:scripted AI 真的用上驻守与军旗(对局涌现验证)。

视野取 episode_len(plan critic m-4 后备口径):v1.3 名额制放慢攒狗节奏,
seed 0 时间线为 狗x2→驻守 966 tick、狗x3→插旗 1006 tick、终局 1116 tick
(explorations/debug_v13_scripted_garrison.py 量得),900 tick 定长视野不够。
"""

import jax
import jax.numpy as jnp

from teow.config import Config
from teow.controller import make_joint_controller
from teow.state import ORDER_GARRISON
from teow.step import new_world


def test_scripted_uses_garrison_and_flag():
    cfg = Config(seed=0)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller("scripted", "scripted", cfg=cfg, mapdata=m)

    def body(carry, _):
        st, k = carry
        k, k_act, k_step = jax.random.split(k, 3)
        st = step_fn(st, joint(st, k_act), k_step)
        return (st, k), (jnp.any(st.alive & (st.order == ORDER_GARRISON)),
                         jnp.any(st.flag_active))

    (_, _), (seen_gar, seen_flag) = jax.lax.scan(
        body, (state, key), None, length=cfg.episode_len)
    assert bool(jnp.any(seen_gar)), "整局内从未出现 ORDER_GARRISON 单位"
    assert bool(jnp.any(seen_flag)), "整局内从未出现激活军旗"
