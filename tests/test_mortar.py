"""v1.4 迫击炮:解锁/限1/盲区/弹道延迟/锁点可走位躲弹/衰减/不溅建筑/冷却/炮死弹消。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn
from test_camp import RICH
from test_economy import W0, drive

from teow.actions import a_build_mortar, a_upgrade, legality_mask
from teow.config import (
    TYPE_INFANTRY,
    TYPE_MORTAR,
    Config,
)
from teow.state import hq_slot, owner_of_slots
from teow.stats import physical_damage
from teow.step import new_world


def spawn_mortar(st, cfg, player, slot_off, rc):
    """手术台:直接放一座建成迫击炮(btype=0,满血)。"""
    st, s = spawn(st, cfg, player, slot_off, TYPE_MORTAR, cfg.mortar_hp, rc)
    return st, s


def splash_dmg(cfg, dist, armor):
    """期望落点伤害:线性中心衰减 → 过护甲(与引擎同式)。"""
    fall = max(0.0, cfg.mortar_aoe_radius - dist) / cfg.mortar_aoe_radius
    base = int(jnp.ceil(cfg.mortar_atk * fall))
    if base <= 0:
        return 0
    return int(physical_damage(jnp.asarray(base), jnp.asarray(armor)))


def test_build_unlock_hq3_and_cap1():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)

    # 基地 2 级:迫击炮非法(解锁 3)
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_mortar(cfg)])
    # 升到 3:合法;落地(在建)后第二座非法(cap 1)
    st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[2] + 1)
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[W0, a_build_mortar(cfg)])
    st = drive(st, step_fn, {0: [(W0, a_build_mortar(cfg))]}, 2)
    assert int(jnp.sum(st.alive & (st.etype == TYPE_MORTAR))) == 1
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_mortar(cfg)])
    # 建成:满血、btype=0
    st = drive(st, step_fn, {}, cfg.mortar_build_time + 1, seed=2)
    mor = int(jnp.argmax(st.etype == TYPE_MORTAR))
    assert int(st.hp[mor]) == cfg.mortar_hp and int(st.btype[mor]) == 0


def test_blind_zone_and_flight_delay_and_falloff():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mor = spawn_mortar(state, cfg, 0, 10, (12.0, 4.0))
    # 盲区内敌兵(d=2.0 < min 2.5):永不被打
    st, near = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                     (12.0, 6.0))
    # 环内敌兵(d=5.0 ∈ [2.5, 7]):挨炮
    st, far = spawn(st, cfg, 1, 11, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 9.0))
    hp0 = cfg.inf_hp_by_level[1]
    # 开火拍:锁点放弹,尚无伤害
    st = one_tick(st, cfg, step_fn)
    mor_slot = hq_slot(0, cfg) + 10
    assert int(st.shell_timer[mor_slot]) == cfg.mortar_flight_time
    # 飞行期(flight_time-1 拍):无伤害
    for t in range(cfg.mortar_flight_time - 1):
        st = one_tick(st, cfg, step_fn, seed=t)
    assert int(st.hp[far]) == hp0, "弹未落地不该有伤害"
    assert int(st.hp[near]) == hp0
    # 落地拍:far(站着不动,弹点=其位置,d=0)吃满额溅射(过步兵甲);near 始终无伤
    st = one_tick(st, cfg, step_fn, seed=99)
    assert hp0 - int(st.hp[far]) == splash_dmg(cfg, 0.0, cfg.infantry_armor), \
        "落点中心伤害不符"
    assert int(st.hp[near]) == hp0, "盲区目标不该被打"
    # near 不在爆点 aoe 内(距爆点 3.0 > 1.5)本就无溅射,顺带验证


def test_dodge_by_moving_out_of_locked_point():
    """锁点弹道的设计意图:飞行期间走位离开落点 ≥ aoe 半径 → 零伤害。"""
    from teow.state import ORDER_MOVE
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mor = spawn_mortar(state, cfg, 0, 10, (12.0, 4.0))
    st, tgt = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 9.0))
    # 开火拍后立刻命令目标横向撤离(速度 0.5 × 7 拍 = 3.5 格 > aoe 1.5)
    st = one_tick(st, cfg, step_fn)          # 迫击炮开火,锁 (12,9)
    assert int(st.shell_timer[mor]) == cfg.mortar_flight_time
    st = st._replace(
        order=st.order.at[tgt].set(ORDER_MOVE),
        target_cell=st.target_cell.at[tgt].set(jnp.asarray([18.0, 9.0])))
    for t in range(cfg.mortar_flight_time + 1):
        st = one_tick(st, cfg, step_fn, seed=10 + t)
    assert int(st.hp[tgt]) == cfg.inf_hp_by_level[1], "走位出圈仍被炸:锁点语义破了"


def test_no_building_splash_and_cooldown_and_shell_dies():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mor = spawn_mortar(state, cfg, 0, 10, (12.0, 4.0))
    # 敌方建筑(哨塔)在环内:迫击炮不打建筑(选目标就滤掉)
    from teow.config import TYPE_TOWER
    st, twr = spawn(st, cfg, 1, 10, TYPE_TOWER, cfg.tower_hp_by_level[1],
                    (12.0, 9.5))  # 塔射程 4 够不到炮(5.5),炮也不该打塔
    st = one_tick(st, cfg, step_fn)
    assert int(st.shell_timer[mor]) == 0, "环内只有建筑不该开火"
    assert int(st.hp[twr]) == cfg.tower_hp_by_level[1]
    # 放一个敌兵触发开火;开火后冷却期内不再放第二发
    st, inf = spawn(st, cfg, 1, 11, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 9.0))
    st = one_tick(st, cfg, step_fn, seed=1)
    assert int(st.shell_timer[mor]) == cfg.mortar_flight_time
    assert int(st.atk_cd[mor]) == cfg.mortar_atk_period - 1
    # 弹在飞时把炮拆了:弹随炮消,目标无伤(炮死弹消,DECISIONS)
    st = st._replace(hp=st.hp.at[mor].set(0))
    for t in range(cfg.mortar_flight_time + 2):
        st = one_tick(st, cfg, step_fn, seed=20 + t)
    assert not bool(st.alive[mor])
    assert int(st.hp[inf]) == cfg.inf_hp_by_level[1], "炮死弹消:遗弹不该爆"
