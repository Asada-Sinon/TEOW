"""v1.6 地雷:触发即爆一次性/无友伤/不可被打/不挡路/空军不触发/cap5。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.actions import a_build_defense, legality_mask
from teow.config import (
    TYPE_AIRSHIP,
    TYPE_INFANTRY,
    TYPE_LANDMINE,
    Config,
)
from teow.state import ORDER_MOVE, hq_slot, owner_of_slots
from teow.stats import physical_damage
from teow.step import new_world


def test_trigger_explode_once_with_falloff_no_friendly_fire():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mine = spawn(state, cfg, 0, 20, TYPE_LANDMINE, cfg.landmine_hp,
                     (31.0, 31.0))
    # 敌兵进触发圈(d=0.75 < 1.0,二进制可精确表示——f32 下 0.8 会因表示
    # 误差把 ceil 顶高一档);己方兵也在爆炸圈内(无友伤断言)
    st, foe = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.75))
    # own 距雷 1.0(圈内)、距敌兵 1.75(近战圈外,不互殴污染断言)
    st, own = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 30.0))
    st1 = one_tick(st, cfg, step_fn)
    fall = (cfg.landmine_aoe_radius - 0.75) / cfg.landmine_aoe_radius
    base = int(jnp.ceil(cfg.landmine_atk * fall))
    expect = int(physical_damage(jnp.asarray(base),
                                 jnp.asarray(cfg.infantry_armor)))
    assert cfg.inf_hp_by_level[1] - int(st1.hp[foe]) == expect, "衰减爆伤不符"
    assert int(st1.hp[own]) == cfg.inf_hp_by_level[1], "地雷无友伤"
    assert not bool(st1.alive[mine]), "地雷一次性自毁"


def test_untargetable_and_nonblocking_and_air_immune():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mine = spawn(state, cfg, 0, 20, TYPE_LANDMINE, cfg.landmine_hp,
                     (31.0, 31.0))
    # 敌兵站在 触发圈外(1.2)近战圈内(≤1.5):不可打雷,雷不爆
    st, foe = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 32.2))
    st1 = one_tick(st, cfg, step_fn)
    assert bool(st1.alive[mine]) and int(st1.hp[mine]) == cfg.landmine_hp, \
        "地雷不可被攻击排除"
    # 己方单位从雷格走过:不触发、不被挡
    st2, walker = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                        (31.0, 29.0))
    st2 = st2._replace(order=st2.order.at[walker].set(ORDER_MOVE),
                       target_cell=st2.target_cell.at[walker].set(
                           jnp.asarray([31.0, 33.0], jnp.float32)))
    # 先把敌兵移走防它触发(手术挪远)
    st2 = st2._replace(pos=st2.pos.at[foe].set(jnp.asarray([25.0, 25.0])))
    for t in range(12):
        st2 = one_tick(st2, cfg, step_fn, seed=t)
    assert float(st2.pos[walker, 1]) > 32.0, "己方单位必须能直穿雷格(不挡路)"
    assert bool(st2.alive[mine]), "己方走过不触发"
    # 敌空军悬停雷上:不触发
    st3, ship = spawn(st, cfg, 1, 22, TYPE_AIRSHIP, cfg.airship_hp_by_level[1], (31.0, 31.0))
    st3 = st3._replace(pos=st3.pos.at[foe].set(jnp.asarray([25.0, 25.0])))
    st3 = one_tick(st3, cfg, step_fn, seed=9)
    assert bool(st3.alive[mine]), "空军不触发地雷"


def test_cap_five():
    cfg = Config(start_ore=3000, start_water=2000)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)
    W0 = hq + 1
    st = state._replace(level=state.level.at[hq].set(4))
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[W0, a_build_defense(TYPE_LANDMINE, cfg)])
    for i in range(5):
        st, _ = spawn(st, cfg, 0, 30 + i, TYPE_LANDMINE, cfg.landmine_hp,
                      (25.0 + i, 20.0))
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_defense(TYPE_LANDMINE, cfg)]), "限 5"
