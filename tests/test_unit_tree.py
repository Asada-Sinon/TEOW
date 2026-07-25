"""v1.4 兵种树:训练等级门(HQ 系/兵营系)、兵营升级链、出生血量查线表。"""

import jax.numpy as jnp
from test_barracks import setup_barracks
from test_camp import RICH
from test_economy import drive

from teow.actions import a_train_unit, a_upgrade, legality_mask
from teow.config import (
    TYPE_ARCHER,
    TYPE_HEAVY,
    TYPE_STRONGMAN,
    TYPE_WAGON,
    Config,
)
from teow.state import hq_slot, owner_of_slots
from teow.step import new_world


def test_barracks_upgrade_cost_time_and_cap_chain():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, bar = setup_barracks(cfg, state, step_fn)
    hq = hq_slot(0, cfg)
    assert int(st.level[hq]) == 2 and int(st.level[bar]) == 1

    # 升兵营 1→2:扣费+耗时,完成补满血差
    res0 = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(bar, a_upgrade(cfg))]}, 1)
    assert st.resources[0].tolist() == [
        res0[0] - cfg.barracks_up_cost_ore[1],
        res0[1] - cfg.barracks_up_cost_water[1]]
    st = drive(st, step_fn, {}, cfg.barracks_up_time[1] + 1, seed=2)
    assert int(st.level[bar]) == 2
    assert int(st.hp[bar]) == cfg.barracks_hp_by_level[2]
    # 上限链:兵营 2 = 基地 2,再升非法
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[bar, a_upgrade(cfg)])


def test_barracks_train_gates_by_level():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, bar = setup_barracks(cfg, state, step_fn)

    # 兵营 1 级:弓箭手(需2)非法
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[bar, a_train_unit(TYPE_ARCHER, cfg)])
    # 升到 2:弓箭手合法,重甲(需3)仍非法
    st = drive(st, step_fn, {0: [(bar, a_upgrade(cfg))]},
               cfg.barracks_up_time[1] + 1)
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[bar, a_train_unit(TYPE_ARCHER, cfg)])
    assert not bool(legal[bar, a_train_unit(TYPE_HEAVY, cfg)])
    # 训一个弓箭手:落地血量=弓箭手线 1 级表
    st = drive(st, step_fn, {0: [(bar, a_train_unit(TYPE_ARCHER, cfg))]},
               cfg.archer_time + 2)
    arc = int(jnp.argmax((st.etype == TYPE_ARCHER) & st.alive))
    assert int(st.etype[arc]) == TYPE_ARCHER
    assert int(st.hp[arc]) == cfg.archer_hp_by_level[1]


def test_hq_train_gates_strongman_wagon():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)

    # 基地 1 级:大力士/马车非法
    legal = legality_mask(state, cfg, m, owner)
    assert not bool(legal[hq, a_train_unit(TYPE_STRONGMAN, cfg)])
    assert not bool(legal[hq, a_train_unit(TYPE_WAGON, cfg)])
    # 升到 3:大力士合法,马车(需5)仍非法
    st = state
    for lv in (1, 2):
        st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]},
                   cfg.base_up_time[lv] + 1)
    assert int(st.level[hq]) == 3
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[hq, a_train_unit(TYPE_STRONGMAN, cfg)])
    assert not bool(legal[hq, a_train_unit(TYPE_WAGON, cfg)])
    # 训大力士:即扣费,落地固定基线血量
    res0 = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(hq, a_train_unit(TYPE_STRONGMAN, cfg))]}, 1)
    assert st.resources[0].tolist() == [res0[0] - cfg.strongman_cost_ore,
                                        res0[1] - cfg.strongman_cost_water]
    st = drive(st, step_fn, {}, cfg.strongman_time + 1, seed=4)
    sm = int(jnp.argmax((st.etype == TYPE_STRONGMAN) & st.alive))
    assert int(st.hp[sm]) == cfg.strongman_hp
    # W0 建造/采集门对大力士同样开放(is_harvester)
    from teow.actions import a_build
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[sm, a_build(0)])
