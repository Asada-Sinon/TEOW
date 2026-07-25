"""诊断:Phase 5 涌现测试假红——scripted 对局里驻守/插旗分支为何从未触发?

逐 tick 记录 p0 视角:狗数、军队数、attack_on、has_bar(含在建)、旗数、
是否有 GARRISON 单位,打印各条件首次成立的 tick 与末态摘要。
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from teow.config import TYPE_BARRACKS, TYPE_DOG, TYPE_INFANTRY, Config
from teow.controller import make_joint_controller
from teow.state import ORDER_GARRISON, owner_of_slots
from teow.step import new_world

SEED = 0  # 与 tests/test_scripted_v13.py 一致
TICKS = 3000


def main():
    cfg = Config(seed=SEED)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller("scripted", "scripted", cfg, m)
    owner = owner_of_slots(cfg)

    def body(carry, _):
        st, k = carry
        k, k_act, k_step = jax.random.split(k, 3)
        st = step_fn(st, joint(st, k_act), k_step)
        is_army = (st.etype == TYPE_INFANTRY) | (st.etype == TYPE_DOG)
        per_p = []
        for p in (0, 1):
            mine = st.alive & (owner == p)
            per_p += [
                jnp.sum(mine & (st.etype == TYPE_DOG)),
                jnp.sum(mine & is_army),
                jnp.any(mine & (st.etype == TYPE_BARRACKS)),
                jnp.sum(st.flag_active[p].astype(jnp.int32)),
                jnp.any(mine & (st.order == ORDER_GARRISON)),
            ]
        return (st, k), tuple(per_p) + (st.done, st.winner)

    (_, _), recs = jax.lax.scan(body, (state, key), None, length=TICKS)

    def first_tick(cond):
        idx = jnp.argmax(cond)
        return int(idx) if bool(cond[idx]) else None

    print(f"seed={SEED} ticks={TICKS}")
    for p in (0, 1):
        dogs, army, bar, flags, gar = recs[p * 5:p * 5 + 5]
        print(f"-- p{p}: 兵营 tick {first_tick(bar)}, 狗1/2/3 tick "
              f"{first_tick(dogs >= 1)}/{first_tick(dogs >= 2)}/{first_tick(dogs >= 3)}, "
              f"军队>=6 tick {first_tick(army >= cfg.ai_attack_threshold)}, "
              f"旗 tick {first_tick(flags > 0)}, GARRISON tick {first_tick(gar)}, "
              f"max dogs {int(jnp.max(dogs))}, max army {int(jnp.max(army))}")
    done, winner = recs[10], recs[11]
    dt = first_tick(done)
    print("done tick:", dt, "winner:", int(winner[dt]) if dt is not None else "-")


if __name__ == "__main__":
    main()
