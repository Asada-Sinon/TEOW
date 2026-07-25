"""v1.4 多哨塔:数量上限挂基地等级(1级0/2-3级1/4-5级2,在建计数)。"""

import jax.numpy as jnp
from test_camp import RICH, build_camp
from test_economy import W0, drive

from teow.actions import a_build_tower, a_upgrade, legality_mask
from teow.config import TYPE_TOWER, Config
from teow.state import hq_slot, owner_of_slots
from teow.step import new_world


def test_tower_cap_by_hq_level():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)

    # 基地 1 级:cap 0,建塔非法
    legal = legality_mask(state, cfg, m, owner)
    assert not bool(legal[W0, a_build_tower(cfg)])

    # 基地 2 级:cap 1——第一座合法;落地(在建即计数)后第二座非法
    st, _ = build_camp(cfg, state, step_fn)
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[W0, a_build_tower(cfg)])
    st = drive(st, step_fn, {0: [(W0, a_build_tower(cfg))]}, 2)
    assert int(jnp.sum(st.alive & (st.etype == TYPE_TOWER))) == 1  # 在建 alive
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_tower(cfg)]), "cap 1 时在建塔必须计数"

    # 升基地到 4(2→3→4):cap 2——第二座合法,第三座非法
    st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[2] + 1)
    st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[3] + 1)
    assert int(st.level[hq]) == 4
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[W0, a_build_tower(cfg)])
    st = drive(st, step_fn, {0: [(W0, a_build_tower(cfg))]}, 2)
    assert int(jnp.sum(st.alive & (st.etype == TYPE_TOWER))) == 2
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_tower(cfg)]), "基地 4 级 cap=2"
