"""v1.4 护甲系统:百分比减伤公式/高甲物理减伤/魔法穿甲/采集单位零输出。

期望值一律经 stats.physical_damage 计算,不手写 ceil 字面量(plan D2)。
"""

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP
from teow.config import (
    TYPE_HEAVY,
    TYPE_INFANTRY,
    TYPE_MAGE,
    TYPE_WORKER,
    Config,
)
from teow.state import hq_slot
from teow.stats import physical_damage
from teow.step import new_world


def spawn(st, cfg, player, slot_off, etype, hp, rc):
    s = hq_slot(player, cfg) + slot_off
    return st._replace(
        alive=st.alive.at[s].set(True),
        etype=st.etype.at[s].set(etype),
        pos=st.pos.at[s].set(jnp.asarray(rc, jnp.float32)),
        hp=st.hp.at[s].set(hp),
        target_cell=st.target_cell.at[s].set(jnp.asarray(rc, jnp.float32)),
    ), s


def one_tick(st, cfg, step_fn, seed=0):
    return step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32),
                   jax.random.PRNGKey(seed))


def test_physical_damage_formula():
    # 零甲恒等;减伤向上取整;至少 1 点
    assert int(physical_damage(jnp.asarray(10), jnp.asarray(0))) == 10
    assert int(physical_damage(jnp.asarray(4), jnp.asarray(10))) == (4 * 90 + 99) // 100
    assert int(physical_damage(jnp.asarray(1), jnp.asarray(99))) == 1
    assert int(physical_damage(jnp.asarray(100), jnp.asarray(100))) == 1


def test_high_armor_reduces_physical():
    """重盔甲战士(高甲)挨步兵物理刀:实际伤害 = physical_damage(步兵攻,重甲甲)。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, inf = spawn(state, cfg, 0, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (10.0, 10.0))
    st, hv = spawn(st, cfg, 1, 10, TYPE_HEAVY, cfg.heavy_hp_by_level[1],
                   (10.0, 11.0))
    st = one_tick(st, cfg, step_fn)
    dmg_to_heavy = cfg.heavy_hp_by_level[1] - int(st.hp[hv])
    dmg_to_inf = cfg.inf_hp_by_level[1] - int(st.hp[inf])
    assert dmg_to_heavy == int(physical_damage(
        jnp.asarray(cfg.inf_atk_by_level[1]), jnp.asarray(cfg.heavy_armor)))
    # 反向:重甲的物理刀过步兵甲
    assert dmg_to_inf == int(physical_damage(
        jnp.asarray(cfg.heavy_atk_by_level[1]), jnp.asarray(cfg.infantry_armor)))
    # 高甲确实在减伤(名义 4 → 实际更小)
    assert dmg_to_heavy < cfg.inf_atk_by_level[1]


def test_magic_bypasses_armor():
    """法师魔法无视护甲:重甲挨法师=全额名义攻击(设计意图:魔法克重甲)。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, mg = spawn(state, cfg, 0, 10, TYPE_MAGE, cfg.mage_hp_by_level[1],
                   (10.0, 10.0))
    st, hv = spawn(st, cfg, 1, 10, TYPE_HEAVY, cfg.heavy_hp_by_level[1],
                   (10.0, 12.0))  # 距 2.0:法师射程内、近战射程外
    st = one_tick(st, cfg, step_fn)
    assert cfg.heavy_hp_by_level[1] - int(st.hp[hv]) == cfg.mage_atk_by_level[1]
    # 重甲够不着法师(近战 1.5 < 2.0),法师无伤
    assert int(st.hp[mg]) == cfg.mage_hp_by_level[1]


def test_harvesters_deal_no_damage():
    """v1.4 规格:采集单位不能攻击——工人贴脸敌军,敌军毫发无损。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, wk = spawn(state, cfg, 0, 10, TYPE_WORKER, cfg.worker_hp, (10.0, 10.0))
    st, inf = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (10.0, 11.0))
    st = one_tick(st, cfg, step_fn)
    assert int(st.hp[inf]) == cfg.inf_hp_by_level[1], "工人不该能造成伤害"
    # 步兵照打工人(零甲全额)
    assert cfg.worker_hp - int(st.hp[wk]) == int(physical_damage(
        jnp.asarray(cfg.inf_atk_by_level[1]), jnp.asarray(0)))
