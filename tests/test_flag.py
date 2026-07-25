"""v1.3 军旗:插旗前提/上限/落格限制、驻守旗寻路、撤旗清理(critic B-1/M-2 回归)。"""

import jax.numpy as jnp
from test_barracks import setup_barracks
from test_camp import RICH
from test_economy import drive

from teow.actions import (
    A_ATTACK,
    a_garrison_flag,
    a_plant_flag,
    a_recall_flag,
    legality_mask,
)
from teow.config import TYPE_DOG, TYPE_INFANTRY, Config
from teow.state import ORDER_ATTACK, ORDER_GARRISON, ORDER_IDLE, hq_slot, owner_of_slots
from teow.step import new_world

_CACHE: dict = {}


def barracks_world():
    """共享前置(状态不可变,跨用例安全复用):玩家 0 拥有建成兵营。"""
    if "w" not in _CACHE:
        cfg = Config(**RICH)
        state, _, step_fn, m = new_world(cfg)
        st, bar = setup_barracks(cfg, state, step_fn)
        _CACHE["w"] = (cfg, st, step_fn, m, bar)
    return _CACHE["w"]


def add_unit(cfg, st, slot, etype, rc):
    """手术摆一个满血战斗单位。"""
    hp_tbl = cfg.dog_hp_by_level if etype == TYPE_DOG else cfg.inf_hp_by_level
    return st._replace(
        alive=st.alive.at[slot].set(True),
        etype=st.etype.at[slot].set(etype),
        pos=st.pos.at[slot].set(jnp.asarray(rc, jnp.float32)),
        hp=st.hp.at[slot].set(hp_tbl[1]),
    )


def flag_dist(st, slot, p, j):
    return float(jnp.linalg.norm(st.pos[slot] - st.flag_pos[p, j]))


def test_plant_requires_barracks():
    cfg = Config(**RICH)
    owner = owner_of_slots(cfg)
    inf = hq_slot(0, cfg) + 20
    # 无兵营:插旗非法
    state, _, _, m = new_world(cfg)
    st0 = add_unit(cfg, state, inf, TYPE_INFANTRY, (31.0, 31.0))
    assert not bool(legality_mask(st0, cfg, m, owner)[inf, a_plant_flag(cfg)])
    # 有建成兵营:合法
    _, st, _, m2, _ = barracks_world()
    st1 = add_unit(cfg, st, inf, TYPE_INFANTRY, (31.0, 31.0))
    assert bool(legality_mask(st1, cfg, m2, owner)[inf, a_plant_flag(cfg)])


def test_flag_cap_three():
    cfg, st, step_fn, m, _ = barracks_world()
    owner = owner_of_slots(cfg)
    dog = hq_slot(0, cfg) + 21
    st = add_unit(cfg, st, dog, TYPE_DOG, (31.0, 31.0))
    for t in range(cfg.max_flags):
        st = drive(st, step_fn, {0: [(dog, a_plant_flag(cfg))]}, 1, seed=t)
    assert st.flag_active[0].tolist() == [True] * cfg.max_flags
    # 第 4 面:掩码封死
    assert not bool(legality_mask(st, cfg, m, owner)[dog, a_plant_flag(cfg)])


def test_plant_on_node_cell_masked():
    cfg, st, _, m, _ = barracks_world()
    owner = owner_of_slots(cfg)
    inf = hq_slot(0, cfg) + 22
    node_rc = [float(x) for x in m.node_pos[4]]  # 公共矿点格(不可通行格)
    st = add_unit(cfg, st, inf, TYPE_INFANTRY, node_rc)
    assert not bool(legality_mask(st, cfg, m, owner)[inf, a_plant_flag(cfg)])


def test_recall_idles_garrisoned():
    cfg, st, step_fn, m, bar = barracks_world()
    dog = hq_slot(0, cfg) + 23
    st = add_unit(cfg, st, dog, TYPE_DOG, (31.0, 31.0))
    st = drive(st, step_fn, {0: [(dog, a_plant_flag(cfg))],
                             1: [(dog, a_garrison_flag(0, cfg))]}, 2)
    assert int(st.order[dog]) == ORDER_GARRISON
    assert int(st.garrison_id[dog]) == cfg.n_nodes + 1
    pos0 = st.pos[dog]
    # 兵营远程撤旗:狗原地转 IDLE,旗位清空
    st = drive(st, step_fn, {0: [(bar, a_recall_flag(0, cfg))]}, 1, seed=1)
    assert int(st.order[dog]) == ORDER_IDLE
    assert int(st.garrison_id[dog]) == -1
    assert not bool(st.flag_active[0, 0])
    assert st.flag_pos[0, 0].tolist() == [-1.0, -1.0]
    assert float(jnp.linalg.norm(st.pos[dog] - pos0)) < 1e-6  # 原地
    # 名额可复用:再插成功,回到 0 号旗位
    st = drive(st, step_fn, {0: [(dog, a_plant_flag(cfg))]}, 1, seed=2)
    assert bool(st.flag_active[0, 0])


def test_garrison_flag_pathing():
    cfg, st, step_fn, _, _ = barracks_world()
    dog = hq_slot(0, cfg) + 24
    st = add_unit(cfg, st, dog, TYPE_DOG, (31.0, 23.0))
    # 手术立一面 8 格外的旗(引擎语义只认 flag_active/flag_pos,不问来源)
    st = st._replace(
        flag_active=st.flag_active.at[0, 0].set(True),
        flag_pos=st.flag_pos.at[0, 0].set(jnp.asarray([31.0, 31.0])))
    st = drive(st, step_fn, {0: [(dog, a_garrison_flag(0, cfg))]}, 25)
    assert flag_dist(st, dog, 0, 0) <= cfg.garrison_hold_radius + 1e-3
    assert int(st.order[dog]) == ORDER_GARRISON


def test_garrison_flag_near_enemy_hq():
    # critic B-1 回归:旗贴着敌方 HQ,到达判定必须对 target_cell(旗位)而不是
    # 被 clip 错的 goal_center——否则单位在别处被误判到达
    cfg, st, step_fn, m, _ = barracks_world()
    dog = hq_slot(0, cfg) + 25
    hq1 = [int(x) for x in m.hq_pos[1]]
    flag_rc = [float(hq1[0] - 2), float(hq1[1])]  # 敌方 HQ 2 格内
    st = add_unit(cfg, st, dog, TYPE_DOG, (flag_rc[0] - 6.0, flag_rc[1]))
    st = st._replace(
        flag_active=st.flag_active.at[0, 1].set(True),
        flag_pos=st.flag_pos.at[0, 1].set(jnp.asarray(flag_rc)))
    st = drive(st, step_fn, {0: [(dog, a_garrison_flag(1, cfg))]}, 30)
    assert flag_dist(st, dog, 0, 1) <= cfg.garrison_hold_radius + 1e-3
    assert int(st.order[dog]) == ORDER_GARRISON


def test_recall_with_same_tick_attack():
    # critic M-2 回归:同 tick 撤旗 + 对驻守单位下 A_ATTACK——新指令生效
    # (不被撤旗清理打回 IDLE),garrison_id 残值一并清
    cfg, st, step_fn, _, bar = barracks_world()
    dog = hq_slot(0, cfg) + 26
    st = add_unit(cfg, st, dog, TYPE_DOG, (31.0, 31.0))
    st = drive(st, step_fn, {0: [(dog, a_plant_flag(cfg))],
                             1: [(dog, a_garrison_flag(0, cfg))]}, 2)
    assert int(st.order[dog]) == ORDER_GARRISON
    st = drive(st, step_fn, {0: [(bar, a_recall_flag(0, cfg)),
                                 (dog, A_ATTACK)]}, 1, seed=1)
    assert int(st.order[dog]) == ORDER_ATTACK
    assert int(st.garrison_id[dog]) == -1
    assert not bool(st.flag_active[0, 0])
