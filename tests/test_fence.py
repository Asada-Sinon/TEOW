"""v1.5 栅栏:三档解锁门/低级仍可建/落地挡路/塔打不了/攻城车专克/槽满不扣费。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn
from test_economy import W0, drive

from teow.actions import a_build_fence, a_upgrade, legality_mask
from teow.config import (
    TYPE_FENCE_IRON,
    TYPE_FENCE_STONE,
    TYPE_FENCE_WOOD,
    TYPE_RAM,
    TYPE_TOWER,
    Config,
)
from teow.state import cell_of, hq_slot, owner_of_slots
from teow.stats import physical_damage
from teow.step import new_world

RICH = dict(start_ore=3000, start_water=2000)


def test_fence_tier_unlock_gates():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)

    # HQ1:全非法
    legal = legality_mask(state, cfg, m, owner)
    for ft in (TYPE_FENCE_WOOD, TYPE_FENCE_STONE, TYPE_FENCE_IRON):
        assert not bool(legal[W0, a_build_fence(ft, cfg)])
    # HQ2:木合法,石铁非法
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[W0, a_build_fence(TYPE_FENCE_WOOD, cfg)])
    assert not bool(legal[W0, a_build_fence(TYPE_FENCE_STONE, cfg)])
    # HQ3:石解锁;木仍可建(独立门)
    st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[2] + 1)
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[W0, a_build_fence(TYPE_FENCE_STONE, cfg)])
    assert bool(legal[W0, a_build_fence(TYPE_FENCE_WOOD, cfg)])
    assert not bool(legal[W0, a_build_fence(TYPE_FENCE_IRON, cfg)])


def test_fence_lands_blocks_and_is_attackable():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    hq = hq_slot(0, cfg)
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    res0 = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(W0, a_build_fence(TYPE_FENCE_WOOD, cfg))]}, 1)
    fence = int(jnp.argmax((st.etype == TYPE_FENCE_WOOD) & st.alive))
    assert bool(st.alive[fence])
    assert st.resources[0].tolist() == [res0[0] - cfg.fence_wood_cost_ore,
                                        res0[1] - cfg.fence_wood_cost_water]
    # 建成:满血,占格成为硬障碍
    st = drive(st, step_fn, {}, cfg.fence_wood_build_time + 1, seed=2)
    assert int(st.hp[fence]) == cfg.fence_wood_hp
    from teow.movement import building_cells
    fc = cell_of(st.pos[fence])
    assert bool(building_cells(st, cfg)[fc[0], fc[1]]), "栅栏必须占格挡路"


def test_tower_ignores_fence_ram_eats_it():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # 手术台:敌方哨塔旁一格放我方铁栅栏——塔只打单位,栅栏无伤
    st, fence = spawn(state, cfg, 0, 20, TYPE_FENCE_IRON, cfg.fence_iron_hp,
                      (31.0, 31.0))
    st, twr = spawn(st, cfg, 1, 20, TYPE_TOWER, cfg.tower_hp_by_level[1],
                    (31.0, 33.0))
    st = one_tick(st, cfg, step_fn)
    assert int(st.hp[fence]) == cfg.fence_iron_hp, "哨塔不该能打栅栏"
    # 攻城车贴脸:高攻过甲拆墙
    st, ram = spawn(st, cfg, 1, 21, TYPE_RAM, cfg.ram_hp_by_level[1],
                    (31.0, 30.0))
    st = one_tick(st, cfg, step_fn, seed=2)
    expect = int(physical_damage(jnp.asarray(cfg.ram_atk_by_level[1]),
                                 jnp.asarray(cfg.fence_iron_armor)))
    assert cfg.fence_iron_hp - int(st.hp[fence]) == expect


def test_fence_masked_when_slots_full():
    """critic MINOR-3:实体槽满时建栅栏必须被掩码挡掉(不下单不扣费)。"""
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq = hq_slot(0, cfg)
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    # 手术台:把玩家 0 的行块全部填活(占满槽)
    blk = slice(0, cfg.e_max)
    st = st._replace(alive=st.alive.at[blk].set(True))
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[W0, a_build_fence(TYPE_FENCE_WOOD, cfg)]), \
        "槽满必须掩码挡掉建栅栏"
