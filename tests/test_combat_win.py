"""战斗与胜负:相邻互砍、同时结算、HQ 摧毁终局、终局冻结。"""

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP
from teow.config import TYPE_INFANTRY, Config
from teow.state import hq_slot
from teow.step import new_world


def spawn_inf(st, cfg, player, slot_off, rc, hp=None):
    """手工把一个步兵放进指定槽(测试专用的状态外科手术)。"""
    s = hq_slot(player, cfg) + slot_off
    return st._replace(
        alive=st.alive.at[s].set(True),
        etype=st.etype.at[s].set(TYPE_INFANTRY),
        pos=st.pos.at[s].set(jnp.asarray(rc, jnp.int32)),
        hp=st.hp.at[s].set(cfg.infantry_hp if hp is None else hp),
    )


def test_adjacent_mutual_damage_and_mutual_kill():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # 两个敌对步兵面对面放在中路空地
    a_rc, b_rc = (12, 5), (12, 6)
    st = spawn_inf(state, cfg, 0, 10, a_rc)
    st = spawn_inf(st, cfg, 1, 10, b_rc)
    sa, sb = hq_slot(0, cfg) + 10, hq_slot(1, cfg) + 10

    key = jax.random.PRNGKey(0)
    st1 = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    # 同时结算:双方同损
    assert int(st1.hp[sa]) == cfg.infantry_hp - cfg.infantry_atk
    assert int(st1.hp[sb]) == cfg.infantry_hp - cfg.infantry_atk

    # 血量恰好一击互杀 → 允许同归于尽
    st = spawn_inf(state, cfg, 0, 10, a_rc, hp=cfg.infantry_atk)
    st = spawn_inf(st, cfg, 1, 10, b_rc, hp=cfg.infantry_atk)
    st1 = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert not bool(st1.alive[sa]) and not bool(st1.alive[sb])


def test_hq_destroyed_ends_game_and_freezes():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    hq1 = hq_slot(1, cfg)
    # 敌 HQ 压到 1 血,派一个步兵贴脸。
    # 站位选 HQ 东侧 (r, c+1):敌方初始工人都在 HQ 西侧(玩家 0 出生位的旋转像),
    # 若站西侧,目标偏好「先打单位」会让步兵去砍工人而不是 HQ。
    st = state._replace(hp=state.hp.at[hq1].set(1))
    st = spawn_inf(st, cfg, 0, 10, (int(m.hq_pos[1][0]), int(m.hq_pos[1][1]) + 1))

    key = jax.random.PRNGKey(0)
    st1 = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert bool(st1.done) and int(st1.winner) == 0

    # 终局冻结:再 step 一切不变(含 tick)
    st2 = step_fn(st1, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert int(st2.tick) == int(st1.tick)
    assert jnp.array_equal(st2.hp, st1.hp)


def test_timeout_draw():
    cfg = Config(episode_len=5)
    state, _, step_fn, m = new_world(cfg)
    key = jax.random.PRNGKey(0)
    st = state
    for _ in range(6):
        key, sub = jax.random.split(key)
        st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), sub)
    assert bool(st.done) and int(st.winner) == 2
