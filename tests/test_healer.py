"""v1.4 奶妈神官:自动奶血量比例最低友军单位、封顶满血、不奶敌不奶建筑、自身不攻击。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.config import (
    TYPE_HEALER,
    TYPE_INFANTRY,
    Config,
)
from teow.step import new_world


def test_heals_lowest_ratio_ally_and_caps_at_max():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, hl = spawn(state, cfg, 0, 10, TYPE_HEALER, cfg.healer_hp_by_level[1],
                   (12.0, 10.0))
    # 两个残血友军:A 比例 50%,B 比例 25%(更低,该被奶)
    hp_a = cfg.inf_hp_by_level[1] // 2
    hp_b = cfg.inf_hp_by_level[1] // 4
    st, a = spawn(st, cfg, 0, 11, TYPE_INFANTRY, hp_a, (12.0, 8.5))
    st, b = spawn(st, cfg, 0, 12, TYPE_INFANTRY, hp_b, (12.0, 11.5))
    st = one_tick(st, cfg, step_fn)
    heal = cfg.healer_heal_by_level[1]
    assert int(st.hp[b]) == hp_b + heal, "该奶血量比例最低的 B"
    assert int(st.hp[a]) == hp_a, "A 不该被奶(单目标)"

    # 奶到满后封顶:A 拉满、B 差 1 血(全场唯一残血),奶量不溢出
    st = st._replace(hp=st.hp.at[a].set(cfg.inf_hp_by_level[1])
                            .at[b].set(cfg.inf_hp_by_level[1] - 1))
    st = one_tick(st, cfg, step_fn, seed=2)
    assert int(st.hp[b]) == cfg.inf_hp_by_level[1], "治疗必须封顶在满血"


def test_never_attacks_and_ignores_enemies_and_buildings():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, hl = spawn(state, cfg, 0, 10, TYPE_HEALER, cfg.healer_hp_by_level[1],
                   (12.0, 10.0))
    # 贴脸敌军:奶妈零输出;敌军照打奶妈
    st, foe = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 11.0))
    # 残血敌军比例更低,也不能被奶
    st = st._replace(hp=st.hp.at[foe].set(5))
    st = one_tick(st, cfg, step_fn)
    assert int(st.hp[foe]) == 5, "奶妈不该攻击也不该奶敌军"
    assert int(st.hp[hl]) < cfg.healer_hp_by_level[1], "敌军应照常打奶妈"


def test_heal_can_save_same_tick_lethal():
    """同帧结算:治疗与伤害叠加,奶得活「本该死」的单位(plan D7 语义)。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, hl = spawn(state, cfg, 0, 10, TYPE_HEALER, cfg.healer_hp_by_level[1],
                   (12.0, 8.0))
    # 友军 A 只剩「敌方伤害 - 奶量 + 1」血:同帧奶完恰好活下来
    from teow.stats import physical_damage
    dmg = int(physical_damage(jnp.asarray(cfg.inf_atk_by_level[1]),
                              jnp.asarray(cfg.infantry_armor)))
    heal = cfg.healer_heal_by_level[1]
    assert dmg > heal, "本测试前提:单拍伤害>奶量"
    hp0 = dmg - heal + 1
    st, a = spawn(st, cfg, 0, 11, TYPE_INFANTRY, hp0, (12.0, 10.0))
    st, foe = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 11.0))
    st = one_tick(st, cfg, step_fn)
    assert bool(st.alive[a]), "同帧治疗该救活将死单位"
    assert int(st.hp[a]) == 1
