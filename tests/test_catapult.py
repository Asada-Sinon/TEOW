"""v1.6 投石车:射程<迫击炮/弹道延迟/落点衰减/不可对空/可被近战打/训练门。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn
from test_barracks import setup_barracks
from test_camp import RICH

from teow.actions import a_train_v16, legality_mask
from teow.config import (
    TYPE_AIRSHIP,
    TYPE_CATAPULT,
    TYPE_INFANTRY,
    Config,
)
from teow.state import owner_of_slots
from teow.stats import physical_damage
from teow.step import new_world


def test_range_shorter_than_mortar_and_flight():
    cfg = Config()
    assert cfg.catapult_range < cfg.mortar_range, "规格:不能太远程"
    state, _, step_fn, m = new_world(cfg)
    st, cat = spawn(state, cfg, 0, 20, TYPE_CATAPULT, cfg.catapult_hp_by_level[1],
                    (31.0, 27.0))
    st, inf = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.0))
    hp0 = cfg.inf_hp_by_level[1]
    # 开火拍:放弹未伤
    st = one_tick(st, cfg, step_fn)
    assert int(st.shell_timer[cat]) == cfg.catapult_flight_time
    assert int(st.hp[inf]) == hp0
    # 飞行 flight-1 拍 + 落地拍
    for t in range(cfg.catapult_flight_time - 1):
        st = one_tick(st, cfg, step_fn, seed=t)
    assert int(st.hp[inf]) == hp0
    st = one_tick(st, cfg, step_fn, seed=9)
    expect = int(physical_damage(
        jnp.asarray(int(jnp.ceil(cfg.catapult_atk_by_level[1] * 1.0))),
        jnp.asarray(cfg.infantry_armor)))
    assert hp0 - int(st.hp[inf]) == expect, "落点中心满额(站桩目标)"


def test_cannot_hit_air_but_melee_hits_it():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, cat = spawn(state, cfg, 0, 20, TYPE_CATAPULT, cfg.catapult_hp_by_level[1],
                    (31.0, 28.0))
    st, ship = spawn(st, cfg, 1, 20, TYPE_AIRSHIP, cfg.airship_hp_by_level[1], (31.0, 31.0))
    st = one_tick(st, cfg, step_fn)
    assert int(st.shell_timer[cat]) == 0, "投石车不可对空"
    # 敌近战贴脸:投石车是地面单位,可被打
    st, inf = spawn(st, cfg, 1, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 27.0))
    st = one_tick(st, cfg, step_fn, seed=2)
    taken = cfg.catapult_hp_by_level[1] - int(st.hp[cat])
    assert taken == int(physical_damage(jnp.asarray(cfg.inf_atk_by_level[1]),
                                        jnp.asarray(cfg.catapult_armor)))


def test_train_gate_barracks6():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, bar = setup_barracks(cfg, state, step_fn)
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[bar, a_train_v16(TYPE_CATAPULT, cfg)]), "兵营1级不可训"
    # 手术把兵营提到 6 级(等级门本身;上限链由升级路径另测)
    st6 = st._replace(level=st.level.at[bar].set(6))
    legal = legality_mask(st6, cfg, m, owner)
    assert bool(legal[bar, a_train_v16(TYPE_CATAPULT, cfg)])
    assert bool(legal[bar, a_train_v16(TYPE_AIRSHIP, cfg)])
    from teow.config import TYPE_DRAGON
    assert not bool(legal[bar, a_train_v16(TYPE_DRAGON, cfg)]), "龙需兵营7"
