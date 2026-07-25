"""v1.4 远程单位:弓箭手射程即时命中(物理过甲)、法师魔法穿甲、attack-move 停在自身射程。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.config import (
    TYPE_ARCHER,
    TYPE_HEAVY,
    TYPE_INFANTRY,
    TYPE_MAGE,
    Config,
)
from teow.state import ORDER_ATTACK
from teow.stats import physical_damage
from teow.step import new_world


def test_archer_hits_at_range_physical():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, arc = spawn(state, cfg, 0, 10, TYPE_ARCHER, cfg.archer_hp_by_level[1],
                    (12.0, 4.0))
    # 距 3.0 ≤ 3.5:打得到;近战单位够不到弓箭手
    st, hv = spawn(st, cfg, 1, 10, TYPE_HEAVY, cfg.heavy_hp_by_level[1],
                   (12.0, 7.0))
    st = one_tick(st, cfg, step_fn)
    expect = int(physical_damage(jnp.asarray(cfg.archer_atk_by_level[1]),
                                 jnp.asarray(cfg.heavy_armor)))
    assert cfg.heavy_hp_by_level[1] - int(st.hp[hv]) == expect
    assert int(st.hp[arc]) == cfg.archer_hp_by_level[1]
    # 弓箭手物理被重甲减到最低 1(名义 5 × 40% = 2)
    assert expect < cfg.archer_atk_by_level[1]


def test_mage_vs_archer_on_heavy_armor():
    """同把重甲当靶:法师(魔法)全额,弓箭手(物理)被减——魔法克重甲的规格意图。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, hv = spawn(state, cfg, 1, 10, TYPE_HEAVY, cfg.heavy_hp_by_level[1],
                   (12.0, 10.0))
    st, mg = spawn(st, cfg, 0, 10, TYPE_MAGE, cfg.mage_hp_by_level[1],
                   (12.0, 7.5))   # 距 2.5 ≤ 3.0
    st = one_tick(st, cfg, step_fn)
    mage_dmg = cfg.heavy_hp_by_level[1] - int(st.hp[hv])
    assert mage_dmg == cfg.mage_atk_by_level[1], "魔法必须无视护甲"
    phys_dmg = int(physical_damage(jnp.asarray(cfg.archer_atk_by_level[1]),
                                   jnp.asarray(cfg.heavy_armor)))
    assert mage_dmg > phys_dmg


def test_attack_move_stops_at_own_range():
    """弓箭手 attack-move:射程内已有可打目标 → 原地停步输出,不再向敌方 HQ
    推进(plan D14;旧口径用 melee_range 判停,弓手会边走边射直到贴脸)。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, arc = spawn(state, cfg, 0, 10, TYPE_ARCHER, cfg.archer_hp_by_level[1],
                    (12.0, 4.0))
    # 敌兵在弓手射程内(d=3.0 ≤ 3.5)但远超近战圈
    st, inf = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 7.0))
    st = st._replace(order=st.order.at[arc].set(ORDER_ATTACK),
                     target_cell=st.target_cell.at[arc].set(
                         jnp.asarray([20.0, 20.0], jnp.float32)))
    p0 = st.pos[arc]
    for t in range(6):
        st = one_tick(st, cfg, step_fn, seed=t)
    moved = float(jnp.linalg.norm(st.pos[arc] - p0))
    assert moved < 0.2, f"射程内有目标仍在行军(挪了 {moved:.2f} 格)"
    d = float(jnp.linalg.norm(st.pos[arc] - st.pos[inf]))
    assert d > cfg.melee_range, f"弓箭手贴脸了(d={d:.2f})"
    # 且目标在挨打(每拍一箭)
    assert int(st.hp[inf]) < cfg.inf_hp_by_level[1]
