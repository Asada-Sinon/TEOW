"""v1.4 攻城车:只打建筑(对单位零输出且无视其存在压向建筑)、对建筑高伤。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.config import (
    TYPE_INFANTRY,
    TYPE_RAM,
    TYPE_TOWER,
    Config,
)
from teow.state import ORDER_ATTACK
from teow.stats import physical_damage
from teow.step import new_world


def test_ram_ignores_units_hits_buildings():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, ram = spawn(state, cfg, 0, 10, TYPE_RAM, cfg.ram_hp_by_level[1],
                    (12.0, 10.0))
    # 贴脸敌兵:攻城车零输出(敌兵照打攻城车,被高甲减伤)
    st, inf = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (12.0, 11.0))
    # 贴脸敌塔:攻城车全力输出
    st, twr = spawn(st, cfg, 1, 11, TYPE_TOWER, cfg.tower_hp_by_level[1],
                    (12.0, 9.0))
    st = one_tick(st, cfg, step_fn)
    assert int(st.hp[inf]) == cfg.inf_hp_by_level[1], "攻城车不能打单位"
    expect = int(physical_damage(jnp.asarray(cfg.ram_atk_by_level[1]),
                                 jnp.asarray(cfg.tower_armor)))
    assert cfg.tower_hp_by_level[1] - int(st.hp[twr]) == expect
    # 敌兵+敌塔都在打攻城车,两笔均被高甲(40%)削减
    taken = cfg.ram_hp_by_level[1] - int(st.hp[ram])
    expect_in = int(physical_damage(jnp.asarray(cfg.inf_atk_by_level[1]),
                                    jnp.asarray(cfg.ram_armor)))
    expect_tw = int(physical_damage(jnp.asarray(cfg.tower_atk_by_level[1]),
                                    jnp.asarray(cfg.ram_armor)))
    assert taken == expect_in + expect_tw


def test_ram_attack_move_walks_past_unit_screen():
    """plan D14:攻城车 attack-move 对打不动的单位视而不见,继续压向敌方建筑,
    不在敌兵旁死锁。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    # v1.5:ATTACK 寻路目标=最近存活敌 HQ(p0 在 (33,17) 附近最近的是
    # p1 HQ);敌兵横在路上不挡格,只测「不因它停步」
    tgt_hq = jnp.asarray(m.hq_pos[1], jnp.float32)
    st, ram = spawn(state, cfg, 0, 10, TYPE_RAM, cfg.ram_hp_by_level[1],
                    (33.0, 17.0))
    st, inf = spawn(st, cfg, 1, 10, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (35.0, 16.0))
    st = st._replace(order=st.order.at[ram].set(ORDER_ATTACK),
                     target_cell=st.target_cell.at[ram].set(tgt_hq))
    d0 = float(jnp.linalg.norm(st.pos[ram] - tgt_hq))
    for t in range(40):
        st = one_tick(st, cfg, step_fn, seed=t)
        if not bool(st.alive[ram]):
            break
    d1 = float(jnp.linalg.norm(st.pos[ram] - tgt_hq))
    # 40 拍 × 0.3 格 = 12 格路程;哪怕被敌兵磨血,也必须显著推进(> 4 格)
    assert (not bool(st.alive[ram])) or d0 - d1 > 4.0, \
        f"攻城车在敌兵旁卡死(推进 {d0 - d1:.1f} 格)"
