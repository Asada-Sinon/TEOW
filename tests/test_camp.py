"""技能训练营:建造/成长/研发(v1.4 八线制)/解锁门/上限链/被拆语义。"""

import jax
import jax.numpy as jnp
from test_economy import W0, drive

from teow.actions import (
    A_NOOP,
    a_build_barracks,
    a_build_camp,
    a_research_line,
    a_upgrade,
    legality_mask,
)
from teow.config import (
    LINE_ARCHER,
    LINE_DOG,
    LINE_INFANTRY,
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

    # 先训一个步兵(1 级表血量),再研步兵线:研发中同线动作非法
    from teow.actions import a_train_infantry
    hq = hq_slot(0, cfg)
    st = drive(st, step_fn, {0: [(hq, a_train_infantry(cfg))]},
               cfg.infantry_time + 1)
    inf = int(jnp.argmax((st.etype == TYPE_INFANTRY) & st.alive))
    assert int(st.hp[inf]) == cfg.inf_hp_by_level[1]

    owner = owner_of_slots(cfg)
    st1 = drive(st, step_fn, {0: [(camp, a_research_line(LINE_INFANTRY, cfg))]}, 1)
    assert int(st1.btype[camp]) < 0
    legal = legality_mask(st1, cfg, m, owner)
    assert not bool(legal[camp, a_research_line(LINE_INFANTRY, cfg)])

    t = cfg.line_res_time[1]
    st2 = drive(st, step_fn, {0: [(camp, a_research_line(LINE_INFANTRY, cfg))]},
                t + 1)
    assert int(st2.upgrades[0, LINE_INFANTRY]) == 2
    assert int(st2.upgrades[1, LINE_INFANTRY]) == 1  # 对手不受影响
    # 存量步兵补血差额;存量工人(无线)不受影响
    assert (int(st2.hp[inf])
            == cfg.inf_hp_by_level[1]
            + cfg.inf_hp_by_level[2] - cfg.inf_hp_by_level[1])
    assert int(st2.hp[W0]) == cfg.worker_hp

    # 新训步兵吃到新表血量
    st4 = drive(st2, step_fn, {0: [(hq, a_train_infantry(cfg))]},
                cfg.infantry_time + 1)
    n_inf = jnp.sum((st4.etype == TYPE_INFANTRY) & st4.alive)
    assert int(n_inf) == 2
    slots = jnp.nonzero((st4.etype == TYPE_INFANTRY) & st4.alive)[0]
    new_inf = int(slots[-1])
    assert int(st4.hp[new_inf]) == cfg.inf_hp_by_level[2]


def test_research_unlock_gate_by_producer():
    """v1.4:线解锁门=兵种可训练。狗线要有建成兵营;弓箭手线要兵营 2 级。"""
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, cfg.camp_build_time + 1)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))

    # 无兵营:狗线/弓箭手线非法,步兵线合法
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[camp, a_research_line(LINE_INFANTRY, cfg)])
    assert not bool(legal[camp, a_research_line(LINE_DOG, cfg)])
    assert not bool(legal[camp, a_research_line(LINE_ARCHER, cfg)])

    # 建成兵营(1 级):狗线解锁,弓箭手线(需兵营 2)仍非法
    st = drive(st, step_fn, {0: [(W0, a_build_barracks(cfg))]},
               cfg.barracks_build_time + 1)
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[camp, a_research_line(LINE_DOG, cfg)])
    assert not bool(legal[camp, a_research_line(LINE_ARCHER, cfg)])


def test_line_capped_by_camp_level():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, cfg.camp_build_time + 1)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))

    # 线升到 2(=营级)后,再研非法,直到营升 3(需基地 3)
    st = drive(st, step_fn, {0: [(camp, a_research_line(LINE_INFANTRY, cfg))]},
               cfg.line_res_time[1] + 1)
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[camp, a_research_line(LINE_INFANTRY, cfg)])
    # 营也不能升(基地才 2 级):上限链第二环
    assert not bool(legal[camp, a_upgrade(cfg)])


def test_camp_destroyed_mid_research_keeps_bought_levels():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]}, cfg.camp_build_time + 1)
    camp = int(jnp.argmax(st.etype == TYPE_CAMP))
    st = drive(st, step_fn, {0: [(camp, a_research_line(LINE_INFANTRY, cfg))]},
               cfg.line_res_time[1] + 1)
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2

    # 建成兵营解锁狗线;狗线研发中把营拆了:中断不退款,已购(步兵线 2)保留
    st = drive(st, step_fn, {0: [(W0, a_build_barracks(cfg))]},
               cfg.barracks_build_time + 1)
    res_before = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(camp, a_research_line(LINE_DOG, cfg))]}, 1)
    paid = st.resources[0].tolist()
    assert paid[0] == res_before[0] - cfg.line_res_cost_ore[1]
    st = st._replace(hp=st.hp.at[camp].set(0))
    key = jax.random.PRNGKey(11)
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert not bool(st.alive[camp])
    assert st.resources[0].tolist() == paid                    # 不退款
    st = drive(st, step_fn, {}, cfg.line_res_time[1] + 2, seed=5)
    assert int(st.upgrades[0, LINE_DOG]) == 1                  # 研发没完成
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2             # 已购保留


def test_same_tick_double_research_dedup():
    """audit v1.1 P0-1 回归:同玩家两营同 tick 研同线,只批一笔(单倍扣费+1级)。"""
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, _ = build_camp(cfg, state, step_fn)
    # 起两座营(paid pass 每 tick 每玩家只批一座,分两拍下单)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))],
                             1: [(W0 + 1, a_build_camp(cfg))]},
               cfg.camp_build_time + 2)
    camps = [int(i) for i in jnp.nonzero((st.etype == TYPE_CAMP) & st.alive)[0]]
    assert len(camps) == 2 and all(int(st.level[c]) == 2 for c in camps)

    res0 = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(camps[0], a_research_line(LINE_INFANTRY, cfg)),
                                 (camps[1], a_research_line(LINE_INFANTRY, cfg))]},
               cfg.line_res_time[1] + 1)
    # 单倍扣费、只升一级
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2
    assert res0[0] - int(st.resources[0][0]) == cfg.line_res_cost_ore[1]
