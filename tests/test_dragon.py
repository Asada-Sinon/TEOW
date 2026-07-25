"""v1.6 龙骑兵:空中优先单体/喷火平坦AoE不伤建筑/共享CD/被弓手打近战打不到。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.config import (
    TYPE_AIRSHIP,
    TYPE_ARCHER,
    TYPE_DRAGON,
    TYPE_INFANTRY,
    TYPE_TOWER,
    Config,
)
from teow.stats import physical_damage
from teow.step import new_world


def test_air_priority_and_shared_cd():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, drg = spawn(state, cfg, 0, 20, TYPE_DRAGON, cfg.dragon_hp, (31.0, 31.0))
    # 敌空艇(射程内)+ 敌地面兵(喷吐圈内)同现:优先打空,地面无伤
    st, ship = spawn(st, cfg, 1, 20, TYPE_AIRSHIP, cfg.airship_hp, (31.0, 33.0))
    st, inf = spawn(st, cfg, 1, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 29.5))
    st1 = one_tick(st, cfg, step_fn)
    expect_air = int(physical_damage(jnp.asarray(cfg.dragon_air_atk),
                                     jnp.asarray(cfg.airship_armor)))
    assert cfg.airship_hp - int(st1.hp[ship]) == expect_air, "空中优先"
    assert int(st1.hp[inf]) == cfg.inf_hp_by_level[1], "打空拍不得同时喷火"
    assert int(st1.atk_cd[drg]) == cfg.dragon_period - 1, "共享 CD 置位"
    # CD 期内地面目标也不挨喷
    st2 = one_tick(st1, cfg, step_fn, seed=2)
    assert int(st2.hp[inf]) == cfg.inf_hp_by_level[1], "CD 期不得插队喷火"


def test_breath_flat_aoe_ground_only_no_buildings():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, drg = spawn(state, cfg, 0, 20, TYPE_DRAGON, cfg.dragon_hp, (31.0, 31.0))
    st, a = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                  (31.0, 33.0))
    st, b = spawn(st, cfg, 1, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                  (30.0, 30.0))
    st, twr = spawn(st, cfg, 1, 22, TYPE_TOWER, cfg.tower_hp_by_level[1],
                    (32.0, 31.0))
    st1 = one_tick(st, cfg, step_fn)
    expect = int(physical_damage(jnp.asarray(cfg.dragon_breath_atk),
                                 jnp.asarray(cfg.infantry_armor)))
    assert cfg.inf_hp_by_level[1] - int(st1.hp[a]) == expect, "圈内敌 A"
    assert cfg.inf_hp_by_level[1] - int(st1.hp[b]) == expect, "圈内敌 B 平坦同伤"
    # 塔挨的伤只可能来自它自身反击?塔打不到龙——断言塔满血且龙掉塔血为 0
    assert int(st1.hp[twr]) == cfg.tower_hp_by_level[1], "喷火不伤建筑"
    assert int(st1.atk_cd[drg]) == cfg.dragon_period - 1


def test_dragon_targetable_matrix():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, drg = spawn(state, cfg, 0, 20, TYPE_DRAGON, cfg.dragon_hp, (31.0, 31.0))
    # 敌近战贴脸打不到龙;敌弓手打得到
    st, inf = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.9))
    st, arc = spawn(st, cfg, 1, 21, TYPE_ARCHER, cfg.archer_hp_by_level[1],
                    (31.0, 28.0))
    st1 = one_tick(st, cfg, step_fn)
    expect = int(physical_damage(jnp.asarray(cfg.archer_atk_by_level[1]),
                                 jnp.asarray(cfg.dragon_armor)))
    taken = cfg.dragon_hp - int(st1.hp[drg])
    assert taken == expect, f"只有弓手的伤(近战不可对空);实收 {taken}"
