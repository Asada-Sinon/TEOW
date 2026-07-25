"""v1.2 哨塔:射程内自动开火/射程外不打/只打单位/升级增伤补血/在建不开火。"""

import jax
import jax.numpy as jnp
from test_barracks import setup_barracks
from test_camp import RICH

from teow.actions import A_NOOP, a_build_tower, a_upgrade
from teow.config import TYPE_INFANTRY, TYPE_TOWER, Config
from teow.state import hq_slot
from teow.step import new_world


def build_tower(cfg, state, step_fn):
    from test_economy import W0, drive
    st, _ = setup_barracks(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_tower(cfg))]},
               cfg.tower_build_time + 1)
    tw = int(jnp.argmax((st.etype == TYPE_TOWER) & st.alive))
    assert int(st.etype[tw]) == TYPE_TOWER and int(st.level[tw]) == 1
    assert int(st.hp[tw]) == cfg.tower_hp_by_level[1]  # 未挨打建成=满血
    return st, tw


def put_enemy_inf(st, cfg, rc):
    s = hq_slot(1, cfg) + 20
    return st._replace(
        alive=st.alive.at[s].set(True),
        etype=st.etype.at[s].set(TYPE_INFANTRY),
        pos=st.pos.at[s].set(jnp.asarray(rc, jnp.float32)),
        hp=st.hp.at[s].set(cfg.inf_hp_by_level[1]),
    ), s


def test_tower_fires_in_range_only_at_units():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, tw = build_tower(cfg, state, step_fn)
    # 把塔挪到无菌区(远离双方工人活动带),排除路过单位补刀的污染
    st = st._replace(pos=st.pos.at[tw].set(jnp.asarray([6.0, 15.0])))
    tp = [float(x) for x in st.pos[tw]]
    key = jax.random.PRNGKey(0)

    # 射程内(3 格):被塔打
    st1, s = put_enemy_inf(st, cfg, (tp[0] + 3.0, tp[1]))
    st1 = step_fn(st1, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert int(st1.hp[s]) < cfg.inf_hp_by_level[1], "射程内敌步兵未被哨塔攻击"

    # 射程外(5 格):不挨打
    st2, s2 = put_enemy_inf(st, cfg, (tp[0] + 5.0, tp[1]))
    st2 = step_fn(st2, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert int(st2.hp[s2]) == cfg.inf_hp_by_level[1], "射程外不该被攻击"


def test_tower_upgrade_boosts_hp_and_atk():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, tw = build_tower(cfg, state, step_fn)
    from test_economy import drive
    st = drive(st, step_fn, {0: [(tw, a_upgrade(cfg))]}, cfg.tower_up_time[1] + 1)
    assert int(st.level[tw]) == 2
    # 升级补血量上限差额(未挨打=新等级满血)
    assert int(st.hp[tw]) == cfg.tower_hp_by_level[2]
    # 攻击按新等级查表:射程内敌人掉血 = tower_atk[2](塔挪无菌区防污染)
    st = st._replace(pos=st.pos.at[tw].set(jnp.asarray([6.0, 15.0])))
    tp = [float(x) for x in st.pos[tw]]
    st1, s = put_enemy_inf(st, cfg, (tp[0] + 2.0, tp[1]))
    key = jax.random.PRNGKey(1)
    st1 = step_fn(st1, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert (cfg.inf_hp_by_level[1] - int(st1.hp[s])) == cfg.tower_atk_by_level[2]
