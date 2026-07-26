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
    st, drg = spawn(state, cfg, 0, 20, TYPE_DRAGON, cfg.dragon_hp_by_level[1], (31.0, 31.0))
    # 敌空艇(射程内)+ 敌地面兵(喷吐圈内)同现:优先打空,地面无伤
    st, ship = spawn(st, cfg, 1, 20, TYPE_AIRSHIP, cfg.airship_hp_by_level[1], (31.0, 33.0))
    st, inf = spawn(st, cfg, 1, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 29.5))
    st1 = one_tick(st, cfg, step_fn)
    expect_air = int(physical_damage(jnp.asarray(cfg.dragon_air_atk_by_level[1]),
                                     jnp.asarray(cfg.airship_armor)))
    assert cfg.airship_hp_by_level[1] - int(st1.hp[ship]) == expect_air, "空中优先"
    assert int(st1.hp[inf]) == cfg.inf_hp_by_level[1], "打空拍不得同时喷火"
    assert int(st1.atk_cd[drg]) == cfg.dragon_period - 1, "共享 CD 置位"
    # CD 期内地面目标也不挨喷
    st2 = one_tick(st1, cfg, step_fn, seed=2)
    assert int(st2.hp[inf]) == cfg.inf_hp_by_level[1], "CD 期不得插队喷火"


def test_breath_flat_aoe_units_full_buildings_discounted():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, drg = spawn(state, cfg, 0, 20, TYPE_DRAGON, cfg.dragon_hp_by_level[1], (31.0, 31.0))
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
    # v1.6 修订(用户复核):喷火伤建筑但打折;塔打不到龙,伤害只来自喷火
    bld_base = max(1, cfg.dragon_breath_atk * cfg.dragon_breath_bld_percent // 100)
    expect_b = int(physical_damage(jnp.asarray(bld_base),
                                   jnp.asarray(cfg.tower_armor)))
    assert cfg.tower_hp_by_level[1] - int(st1.hp[twr]) == expect_b, \
        "喷火对建筑=打折伤害"
    assert int(st1.atk_cd[drg]) == cfg.dragon_period - 1


def test_dragon_targetable_matrix():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, drg = spawn(state, cfg, 0, 20, TYPE_DRAGON, cfg.dragon_hp_by_level[1], (31.0, 31.0))
    # 敌近战贴脸打不到龙;敌弓手打得到
    st, inf = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.9))
    st, arc = spawn(st, cfg, 1, 21, TYPE_ARCHER, cfg.archer_hp_by_level[1],
                    (31.0, 28.0))
    st1 = one_tick(st, cfg, step_fn)
    expect = int(physical_damage(jnp.asarray(cfg.archer_atk_by_level[1]),
                                 jnp.asarray(cfg.dragon_armor)))
    taken = cfg.dragon_hp_by_level[1] - int(st1.hp[drg])
    assert taken == expect, f"只有弓手的伤(近战不可对空);实收 {taken}"


def test_v16_lines_capped_at_three():
    """v1.6 修订:投石车/飞艇/龙有升级线,上限 3(线级<3 可研,==3 掩死)。"""
    from teow.actions import a_research_line, legality_mask
    from teow.config import LINE_CATAPULT, LINE_DRAGON, N_LINES, TYPE_CAMP
    from teow.state import hq_slot, owner_of_slots

    cfg = Config(start_ore=5000, start_water=5000)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)
    # 手术台:基地 7、一座 7 级兵营、一座 7 级训练营(直接摆,免长途经济)
    from teow.config import TYPE_BARRACKS
    st, bar = spawn(state, cfg, 0, 30, TYPE_BARRACKS, cfg.barracks_hp_by_level[7],
                    (25.0, 20.0))
    st, camp = spawn(st, cfg, 0, 31, TYPE_CAMP, cfg.camp_hp_by_level[7],
                     (27.0, 20.0))
    st = st._replace(level=st.level.at[hq].set(7).at[bar].set(7)
                            .at[camp].set(7))
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[camp, a_research_line(LINE_CATAPULT, cfg)]), "线1<3 可研"
    assert bool(legal[camp, a_research_line(LINE_DRAGON, cfg)])
    # 线级顶到 3:掩死(营 7 级也没用)
    up = st.upgrades
    for li in (LINE_CATAPULT, LINE_DRAGON):
        up = up.at[0, li].set(3)
    st3 = st._replace(upgrades=up)
    legal = legality_mask(st3, cfg, m, owner)
    assert not bool(legal[camp, a_research_line(LINE_CATAPULT, cfg)]), "上限 3"
    assert not bool(legal[camp, a_research_line(LINE_DRAGON, cfg)]), "上限 3"
    # 旧八线不受影响(线2<7 可研)
    from teow.config import LINE_ARCHER
    st_a = st._replace(upgrades=st.upgrades.at[0, LINE_ARCHER].set(4))
    legal = legality_mask(st_a, cfg, m, owner)
    assert bool(legal[camp, a_research_line(LINE_ARCHER, cfg)])
    assert st.upgrades.shape[1] == N_LINES == 11
