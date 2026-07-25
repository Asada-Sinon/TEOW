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
        hp=st.hp.at[s].set(cfg.inf_hp_by_level[1] if hp is None else hp),
    )


def test_adjacent_mutual_damage_and_mutual_kill():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # 两个敌对步兵面对面放在中路空地
    a_rc, b_rc = (31, 30), (31, 31)   # 六边形中路空地(v1.5)
    st = spawn_inf(state, cfg, 0, 10, a_rc)
    st = spawn_inf(st, cfg, 1, 10, b_rc)
    sa, sb = hq_slot(0, cfg) + 10, hq_slot(1, cfg) + 10

    key = jax.random.PRNGKey(0)
    st1 = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    # 同时结算:双方同损
    assert int(st1.hp[sa]) == cfg.inf_hp_by_level[1] - cfg.inf_atk_by_level[1]
    assert int(st1.hp[sb]) == cfg.inf_hp_by_level[1] - cfg.inf_atk_by_level[1]

    # 血量恰好一击互杀 → 允许同归于尽
    st = spawn_inf(state, cfg, 0, 10, a_rc, hp=cfg.inf_atk_by_level[1])
    st = spawn_inf(st, cfg, 1, 10, b_rc, hp=cfg.inf_atk_by_level[1])
    st1 = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert not bool(st1.alive[sa]) and not bool(st1.alive[sb])


def test_hq_destroyed_ends_game_and_freezes():
    """v1.5 四人:摧毁一家=淘汰不终局;打掉最后一个对手才分胜负。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    key = jax.random.PRNGKey(0)
    # 先手术淘汰玩家 2/3(只剩 0 vs 1),再让步兵补刀 HQ1
    hq1 = hq_slot(1, cfg)
    st = state._replace(
        hp=state.hp.at[hq_slot(2, cfg)].set(0).at[hq_slot(3, cfg)].set(0))
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert not bool(st.done), "淘汰两家后仍有两家存活,不该终局"
    # 敌 HQ 压到 1 血,步兵站远离敌工人一侧贴脸(目标偏好先打单位)
    st = st._replace(hp=st.hp.at[hq1].set(1))
    # p1 工人在 HQ 东侧列(出生像),站西侧一格避开「先打单位」偏好
    st = spawn_inf(st, cfg, 0, 10, (int(m.hq_pos[1][0]), int(m.hq_pos[1][1]) - 1))

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
    assert bool(st.done) and int(st.winner) == cfg.n_players  # P=和局
