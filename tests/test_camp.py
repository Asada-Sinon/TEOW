"""v1.1 技能训练营:建造/成长/研发/上限链/被拆语义。"""

import jax
import jax.numpy as jnp
from test_economy import W0, drive

from teow.actions import A_NOOP, a_build_camp, a_research, a_upgrade, legality_mask
from teow.config import (
    LINE_INFANTRY,
    LINE_WORKER,
    TYPE_CAMP,
    TYPE_INFANTRY,
    Config,
)
from teow.state import hq_slot, owner_of_slots
from teow.step import new_world

RICH = dict(start_ore=2000, start_water=1200)


def build_camp(cfg, state, step_fn):
    """公共前置:升基地到 2,再让 1 号工人起营;返回 (state, camp_slot)。"""
    hq = hq_slot(0, cfg)
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    assert int(st.level[hq]) == 2
    st1 = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, 1)
    camp = int(jnp.argmax(st1.etype == TYPE_CAMP))
    assert bool(st1.alive[camp]) and int(st1.level[camp]) == 1  # 在建
    return st, camp


def test_camp_unlock_build_and_completion():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)

    # 基地 1 级:建营非法
    legal = legality_mask(state, cfg, m, owner)
    assert not bool(legal[W0, a_build_camp(cfg)])

    st, camp = build_camp(cfg, state, step_fn)
    res0 = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, 1)
    # 扣费即时
    assert st.resources[0].tolist() == [res0[0] - cfg.camp_cost_ore,
                                        res0[1] - cfg.camp_cost_water]
    # 建成:恰在 build_time 拍后 level=2,未挨打则满血
    st = drive(st, step_fn, {}, cfg.camp_build_time, seed=2)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))
    assert int(st.level[camp]) == 2
    assert int(st.hp[camp]) == cfg.camp_hp_by_level[2]
    assert int(st.btype[camp]) == 0


def test_research_applies_globally_and_to_existing_units():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, cfg.camp_build_time + 1)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))
    assert int(st.level[camp]) == 2

    # 研步兵线:完成后线级 2;研发中同线动作非法
    owner = owner_of_slots(cfg)
    st1 = drive(st, step_fn, {0: [(camp, a_research(LINE_INFANTRY, cfg))]}, 1)
    assert int(st1.btype[camp]) < 0
    legal = legality_mask(st1, cfg, m, owner)
    assert not bool(legal[camp, a_research(LINE_INFANTRY, cfg)])

    t = cfg.inf_res_time[1]
    st2 = drive(st, step_fn, {0: [(camp, a_research(LINE_INFANTRY, cfg))]}, t + 1)
    assert int(st2.upgrades[0, LINE_INFANTRY]) == 2
    assert int(st2.upgrades[1, LINE_INFANTRY]) == 1  # 对手不受影响

    # 存量工人不受步兵线影响;研工人线后存量工人 hp 补差额
    whp_before = int(st2.hp[W0])
    assert whp_before == cfg.worker_hp_by_level[1]
    t = cfg.worker_res_time[1]
    st3 = drive(st2, step_fn, {0: [(camp, a_research(LINE_WORKER, cfg))]}, t + 1)
    assert int(st3.upgrades[0, LINE_WORKER]) == 2
    assert (int(st3.hp[W0]) - whp_before
            == cfg.worker_hp_by_level[2] - cfg.worker_hp_by_level[1])

    # 新训步兵吃到新表血量;攻击查表在 combat(此处只验生成血量)
    from teow.actions import a_train_infantry
    hq = hq_slot(0, cfg)
    st4 = drive(st2, step_fn, {0: [(hq, a_train_infantry(cfg))]},
                cfg.infantry_time + 1)
    inf = int(jnp.argmax((st4.etype == TYPE_INFANTRY) & st4.alive))
    assert int(st4.hp[inf]) == cfg.inf_hp_by_level[2]


def test_line_capped_by_camp_level():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, cfg.camp_build_time + 1)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))

    # 线升到 2(=营级)后,再研非法,直到营升 3(需基地 3)
    st = drive(st, step_fn, {0: [(camp, a_research(LINE_INFANTRY, cfg))]},
               cfg.inf_res_time[1] + 1)
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[camp, a_research(LINE_INFANTRY, cfg)])
    # 营也不能升(基地才 2 级):上限链第二环
    assert not bool(legal[camp, a_upgrade(cfg)])


def test_camp_destroyed_mid_research_keeps_bought_levels():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, cfg.camp_build_time + 1)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))
    st = drive(st, step_fn, {0: [(camp, a_research(LINE_INFANTRY, cfg))]},
               cfg.inf_res_time[1] + 1)
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2

    # 二次研发中把营拆了:研发中断不退款,已购等级保留
    st = drive(st, step_fn, {0: [(camp, a_upgrade(cfg))]}, 0)  # no-op 占位
    res_before = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(camp, a_research(LINE_WORKER, cfg))]}, 1)
    paid = st.resources[0].tolist()
    assert paid[0] == res_before[0] - cfg.worker_res_cost_ore[1]
    st = st._replace(hp=st.hp.at[camp].set(0))
    key = jax.random.PRNGKey(11)
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert not bool(st.alive[camp])
    assert st.resources[0].tolist() == paid                    # 不退款
    st = drive(st, step_fn, {}, cfg.worker_res_time[1] + 2, seed=5)
    assert int(st.upgrades[0, LINE_WORKER]) == 1               # 研发没完成
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2             # 已购保留
