"""v1.4 采集单位线:三角关系(采速 大力士>马车>工人,载荷 马车>大力士>工人)、
不能攻击/驻守/驻旗/插旗、能建造、名额口径类型无关。"""

import jax.numpy as jnp
from test_armor import spawn
from test_economy import W0, drive

from teow.actions import (
    A_ATTACK,
    a_build,
    a_garrison_hq,
    a_harvest,
    a_plant_flag,
    legality_mask,
)
from teow.config import (
    RES_ORE,
    TYPE_LCAV,
    TYPE_STRONGMAN,
    TYPE_WAGON,
    TYPE_WORKER,
    Config,
)
from teow.economy import assigned_counts
from teow.state import ORDER_HARVEST, owner_of_slots
from teow.step import new_world


def test_config_triangle_relations():
    """规格关系断言(数值可调,关系不可破):采速 大力士>马车>工人;
    载荷 马车>大力士>工人;移速 大力士=工人、马车=轻骑兵。"""
    cfg = Config()
    assert cfg.strongman_mine_time < cfg.wagon_mine_time < cfg.worker_mine_time
    assert cfg.wagon_carry > cfg.strongman_carry > cfg.worker_carry
    assert cfg.speed_by_type[TYPE_STRONGMAN] == cfg.speed_by_type[TYPE_WORKER]
    assert cfg.speed_by_type[TYPE_WAGON] == cfg.speed_by_type[TYPE_LCAV]


def _mine_world():
    """公共前置:W0 建成 0 号矿。"""
    cfg = Config(start_ore=500, start_water=300)
    state, _, step_fn, m = new_world(cfg)
    st = drive(state, step_fn, {0: [(W0, a_build(0))]}, 200)
    assert int(st.node_owner[0]) == 0
    return cfg, st, step_fn, m


def test_strongman_and_wagon_harvest_cycle():
    cfg, st, step_fn, m = _mine_world()
    # 手术台放一个大力士与一辆马车在矿旁,各派采集
    st, sm = spawn(st, cfg, 0, 20, TYPE_STRONGMAN, cfg.strongman_hp, (3.0, 7.0))
    st, wg = spawn(st, cfg, 0, 21, TYPE_WAGON, cfg.wagon_hp, (3.0, 9.0))
    ore0 = int(st.resources[0, RES_ORE])
    st = drive(st, step_fn, {0: [(sm, a_harvest(0, cfg)),
                                 (wg, a_harvest(0, cfg))]}, 300, seed=7)
    gained = int(st.resources[0, RES_ORE]) - ore0
    # 一趟入账 = carry[类型] + bonus[矿级1](=0):两类载荷的非负整数组合
    c_sm, c_wg = cfg.strongman_carry, cfg.wagon_carry
    combos = {a * c_sm + b * c_wg for a in range(16) for b in range(16)}
    assert gained > 0 and gained in combos, f"入账 {gained} 不是载荷组合"
    assert gained >= c_sm + c_wg, "300 tick 两单位至少各跑一趟"


def test_harvesters_cannot_fight_or_garrison_but_can_build():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, sm = spawn(state, cfg, 0, 20, TYPE_STRONGMAN, cfg.strongman_hp,
                   (10.0, 10.0))
    st, wg = spawn(st, cfg, 0, 21, TYPE_WAGON, cfg.wagon_hp, (10.0, 12.0))
    legal = legality_mask(st, cfg, m, owner)
    for s in (W0, sm, wg):
        assert not bool(legal[s, A_ATTACK]), "采集单位不能 attack-move"
        assert not bool(legal[s, a_garrison_hq(cfg)]), "采集单位不能驻守"
        assert not bool(legal[s, a_plant_flag(cfg)]), "采集单位不能插旗"
        assert bool(legal[s, a_build(0)]), "采集单位必须能建造"


def test_assigned_counts_type_agnostic():
    cfg, st, step_fn, m = _mine_world()
    st, sm = spawn(st, cfg, 0, 20, TYPE_STRONGMAN, cfg.strongman_hp, (3.0, 7.0))
    st = drive(st, step_fn, {0: [(W0, a_harvest(0, cfg)),
                                 (sm, a_harvest(0, cfg))]}, 2, seed=1)
    both = (st.alive & (st.order == ORDER_HARVEST)
            & (st.target_node == 0))
    assert int(jnp.sum(both)) == 2
    assert int(assigned_counts(st, cfg)[0]) == 2, "名额必须计入所有采集单位类型"
