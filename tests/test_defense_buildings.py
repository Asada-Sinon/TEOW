"""v1.6 防御建筑:法师塔(魔法穿甲/对空/中频)、激光炮(每 tick 魔法/对空)、
喷火器(自心圆平坦/只对地/在建不喷)、建造门与 cap。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.actions import a_build_defense, legality_mask
from teow.config import (
    TYPE_AIRSHIP,
    TYPE_FLAMER,
    TYPE_HEAVY,
    TYPE_INFANTRY,
    TYPE_LASER,
    TYPE_MAGETOWER,
    Config,
)
from teow.state import hq_slot, owner_of_slots
from teow.step import new_world


def test_magetower_magic_ignores_armor_and_hits_air():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mt = spawn(state, cfg, 0, 20, TYPE_MAGETOWER, cfg.magetower_hp,
                   (31.0, 28.0))
    st, hv = spawn(st, cfg, 1, 20, TYPE_HEAVY, cfg.heavy_hp_by_level[1],
                   (31.0, 31.0))
    st1 = one_tick(st, cfg, step_fn)
    assert cfg.heavy_hp_by_level[1] - int(st1.hp[hv]) == cfg.magetower_atk, \
        "法师塔魔法必须无视重甲"
    # 中频:开火后 period-1 拍冷却
    assert int(st1.atk_cd[mt]) == cfg.magetower_period - 1
    st2 = one_tick(st1, cfg, step_fn, seed=2)
    assert int(st2.hp[hv]) == int(st1.hp[hv]), "冷却期不开火"
    # 对空
    stA, mt2 = spawn(state, cfg, 0, 21, TYPE_MAGETOWER, cfg.magetower_hp,
                     (31.0, 34.0))
    stA, ship = spawn(stA, cfg, 1, 21, TYPE_AIRSHIP, cfg.airship_hp,
                      (31.0, 36.0))
    stA = one_tick(stA, cfg, step_fn, seed=3)
    assert cfg.airship_hp - int(stA.hp[ship]) == cfg.magetower_atk, "法师塔可对空"


def test_laser_continuous_magic_hits_air():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, lz = spawn(state, cfg, 0, 20, TYPE_LASER, cfg.laser_hp, (31.0, 28.0))
    st, ship = spawn(st, cfg, 1, 20, TYPE_AIRSHIP, cfg.airship_hp, (31.0, 32.0))
    st = one_tick(st, cfg, step_fn)
    st = one_tick(st, cfg, step_fn, seed=2)
    assert cfg.airship_hp - int(st.hp[ship]) == 2 * cfg.laser_atk, \
        "激光每 tick 结算(period 1)且魔法无视护甲"


def test_flamer_flat_aoe_ground_only_and_inbuild_silent():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, fl = spawn(state, cfg, 0, 20, TYPE_FLAMER, cfg.flamer_hp, (31.0, 31.0))
    st, a = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                  (31.0, 32.5))
    st, b = spawn(st, cfg, 1, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                  (30.0, 30.0))
    st, ship = spawn(st, cfg, 1, 22, TYPE_AIRSHIP, cfg.airship_hp, (31.0, 31.5))
    # own 在喷火圈内(1.58)但在两敌近战圈外(1.8/2.9),不互殴污染断言
    st, own = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (32.5, 31.5))
    from teow.stats import physical_damage
    st1 = one_tick(st, cfg, step_fn)
    expect = int(physical_damage(jnp.asarray(cfg.flamer_atk),
                                 jnp.asarray(cfg.infantry_armor)))
    assert cfg.inf_hp_by_level[1] - int(st1.hp[a]) == expect, "圈内敌 A 同伤"
    assert cfg.inf_hp_by_level[1] - int(st1.hp[b]) == expect, "圈内敌 B 同伤(平坦)"
    assert int(st1.hp[ship]) == cfg.airship_hp, "喷火器只对地"
    assert int(st1.hp[own]) == cfg.inf_hp_by_level[1], "无友伤"
    # 在建不喷:btype<0
    from teow.config import BTASK_BUILD_FLAMER
    st_ib = st._replace(btype=st.btype.at[fl].set(BTASK_BUILD_FLAMER),
                        btimer=st.btimer.at[fl].set(10))
    st2 = one_tick(st_ib, cfg, step_fn, seed=5)
    assert int(st2.hp[a]) == cfg.inf_hp_by_level[1], "在建喷火器不得开火"


def test_build_gates_and_caps():
    cfg = Config(start_ore=3000, start_water=2000)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)
    W0 = hq + 1
    # HQ2:全非法;逐级解锁 法师塔3/地雷4/喷火6/激光7(手术调基地级省时)
    for lv, unlocked in ((2, []), (3, [TYPE_MAGETOWER]),
                         (6, [TYPE_MAGETOWER, TYPE_FLAMER]),
                         (7, [TYPE_MAGETOWER, TYPE_FLAMER, TYPE_LASER])):
        st = state._replace(level=state.level.at[hq].set(lv))
        legal = legality_mask(st, cfg, m, owner)
        for t in (TYPE_MAGETOWER, TYPE_FLAMER, TYPE_LASER):
            assert bool(legal[W0, a_build_defense(t, cfg)]) == (t in unlocked), \
                f"基地{lv}级 类型{t} 解锁判定错"
    # cap 1:场上已有法师塔则掩死
    st = state._replace(level=state.level.at[hq].set(3))
    st, _ = spawn(st, cfg, 0, 30, TYPE_MAGETOWER, cfg.magetower_hp, (25.0, 20.0))
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_defense(TYPE_MAGETOWER, cfg)]), "限 1"
