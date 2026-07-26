"""v1.6 空中域:飞跃栅栏/不出六边形/空地零碰撞/对空表(近战×,弓手√,迫击炮×)。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.config import (
    TYPE_AIRSHIP,
    TYPE_ARCHER,
    TYPE_FENCE_IRON,
    TYPE_INFANTRY,
    TYPE_MORTAR,
    Config,
)
from teow.state import ORDER_MOVE
from teow.stats import physical_damage
from teow.step import new_world


def _move(st, s, rc):
    return st._replace(order=st.order.at[s].set(ORDER_MOVE),
                       target_cell=st.target_cell.at[s].set(
                           jnp.asarray(rc, jnp.float32)))


def test_airship_flies_over_fence_wall():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # 竖起一排铁栅栏(col 31,rows 28-34),飞艇从西往东 MOVE 穿墙
    st = state
    for i, r in enumerate(range(28, 35)):
        st, _ = spawn(st, cfg, 1, 30 + i, TYPE_FENCE_IRON, cfg.fence_iron_hp,
                      (float(r), 31.0))
    st, ship = spawn(st, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp_by_level[1], (31.0, 27.0))
    st, inf = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 27.0))
    st = _move(st, ship, (31.0, 35.0))
    st = _move(st, inf, (31.0, 35.0))
    for t in range(20):
        st = one_tick(st, cfg, step_fn, seed=t)
    assert float(st.pos[ship, 1]) > 33.0, "飞艇必须穿过栅栏墙"
    assert float(st.pos[inf, 1]) < 30.6, "地面单位必须被栅栏挡住"


def test_air_stays_inside_hex():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, ship = spawn(state, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp_by_level[1],
                     (31.0, 55.0))
    st = _move(st, ship, (31.0, 63.0))   # 目标在六边形外
    for t in range(20):
        st = one_tick(st, cfg, step_fn, seed=t)
    import numpy as np
    cell = np.round(np.asarray(st.pos[ship])).astype(int)
    assert m.passable[cell[0], cell[1]], "空军不得飞出六边形可行区"


def test_anti_air_table():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # 敌飞艇悬停;近战步兵贴脸打不到,弓手打得到,迫击炮环内不开火
    st, ship = spawn(state, cfg, 1, 20, TYPE_AIRSHIP, cfg.airship_hp_by_level[1],
                     (31.0, 31.0))
    st, inf = spawn(st, cfg, 0, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 32.0))
    st1 = one_tick(st, cfg, step_fn)
    assert int(st1.hp[ship]) == cfg.airship_hp_by_level[1], "近战不可对空"
    st, arc = spawn(st, cfg, 0, 21, TYPE_ARCHER, cfg.archer_hp_by_level[1],
                    (31.0, 28.0))
    st2 = one_tick(st, cfg, step_fn, seed=2)
    expect = int(physical_damage(jnp.asarray(cfg.archer_atk_by_level[1]),
                                 jnp.asarray(cfg.airship_armor)))
    assert cfg.airship_hp_by_level[1] - int(st2.hp[ship]) == expect, "弓手必须可对空"

    # 迫击炮环内只有空军:不开火;弹落点也炸不到空军(M-1)
    state2, _, _, _ = new_world(cfg)
    st3, mor = spawn(state2, cfg, 0, 30, TYPE_MORTAR, cfg.mortar_hp, (31.0, 26.0))
    st3, ship2 = spawn(st3, cfg, 1, 30, TYPE_AIRSHIP, cfg.airship_hp_by_level[1],
                       (31.0, 31.0))
    st3 = one_tick(st3, cfg, step_fn, seed=3)
    assert int(st3.shell_timer[mor]) == 0, "迫击炮不可对空(不该开火)"


def test_air_ground_no_collision():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # 空地同格悬停:互不推挤(不同高度)
    st, ship = spawn(state, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp_by_level[1],
                     (31.0, 31.0))
    st, wk = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                   (31.0, 31.0))
    p0s, p0w = st.pos[ship], st.pos[wk]
    st = one_tick(st, cfg, step_fn)
    assert float(jnp.linalg.norm(st.pos[ship] - p0s)) < 1e-6
    assert float(jnp.linalg.norm(st.pos[wk] - p0w)) < 1e-6
