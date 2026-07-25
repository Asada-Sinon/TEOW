"""v1.2 兵营与狗子:解锁/训狗/速度/双生产者同 tick 完工(critic B-1)。"""

import jax.numpy as jnp
from test_camp import RICH, build_camp
from test_economy import W0, drive

from teow.actions import (
    a_build_barracks,
    a_build_camp,
    a_train_dog,
    a_train_worker,
    legality_mask,
)
from teow.config import TYPE_BARRACKS, TYPE_DOG, TYPE_WORKER, Config
from teow.state import hq_slot, owner_of_slots
from teow.step import new_world


def setup_barracks(cfg, state, step_fn):
    """前置:升基地2 → 建营 → 建兵营;返回 (state, bar_slot)。"""
    st, _ = build_camp(cfg, state, step_fn)
    st = drive(st, step_fn, {0: [(W0, a_build_camp(cfg))]},
               cfg.camp_build_time + 1)
    st = drive(st, step_fn, {0: [(W0, a_build_barracks(cfg))]},
               cfg.barracks_build_time + 1)
    bar = int(jnp.argmax((st.etype == TYPE_BARRACKS) & st.alive))
    assert int(st.etype[bar]) == TYPE_BARRACKS
    return st, bar


def test_barracks_unlock_and_dog_training():
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)

    # 基地 1 级:兵营非法
    legal = legality_mask(state, cfg, m, owner)
    assert not bool(legal[W0, a_build_barracks(cfg)])

    st, bar = setup_barracks(cfg, state, step_fn)
    assert int(st.level[bar]) == 1          # 兵营建成是 1 级(营才是建成即 2)
    assert int(st.hp[bar]) == cfg.barracks_hp  # 未挨打建成=满血

    # 训狗:扣费在 paid pass,dog_time 后落地,吃步兵线的狗表血量
    res0 = st.resources[0].tolist()
    st = drive(st, step_fn, {0: [(bar, a_train_dog(cfg))]}, 1)
    assert st.resources[0].tolist() == [res0[0] - cfg.dog_cost_ore,
                                        res0[1] - cfg.dog_cost_water]
    st = drive(st, step_fn, {}, cfg.dog_time + 1, seed=3)
    dog = int(jnp.argmax((st.etype == TYPE_DOG) & st.alive))
    assert int(st.etype[dog]) == TYPE_DOG
    assert int(st.hp[dog]) == cfg.dog_hp_by_level[1]
    # 狗比步兵快(速度表)
    assert cfg.speed_by_type[TYPE_DOG] > cfg.speed_by_type[5]


def test_hq_and_barracks_complete_same_tick_no_overwrite():
    """critic B-1 回归:HQ 与兵营同 tick 完工,各自落地、无覆写无双扣。
    worker_time=40 与 dog_time=30:错拍下单让两者同 tick 完成。"""
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, bar = setup_barracks(cfg, state, step_fn)
    hq = hq_slot(0, cfg)

    n_workers0 = int(jnp.sum(st.alive & (st.etype == TYPE_WORKER)))
    n_dogs0 = int(jnp.sum(st.alive & (st.etype == TYPE_DOG)))
    # t=0 HQ 训工人(40 拍),t=10 兵营训狗(30 拍)→ 同在 t=40 完成
    st = drive(st, step_fn, {0: [(hq, a_train_worker(cfg))],
                             10: [(bar, a_train_dog(cfg))]}, 41, seed=5)
    n_workers1 = int(jnp.sum(st.alive & (st.etype == TYPE_WORKER)))
    n_dogs1 = int(jnp.sum(st.alive & (st.etype == TYPE_DOG)))
    assert n_workers1 == n_workers0 + 1, "工人未落地(被覆写?)"
    assert n_dogs1 == n_dogs0 + 1, "狗未落地(单生产者假设残留?)"
    assert int(st.btype[hq]) == 0 and int(st.btype[bar]) == 0


def test_dog_line_research_bumps_existing_dogs_only():
    """v1.4 八线制:狗子独立成线——狗线研发完成,存量狗补表差额;
    步兵线研发不再惠及狗(v1.2「狗吃步兵线」DECISIONS 已被规格推翻)。"""
    from teow.actions import a_research_line
    from teow.config import LINE_DOG, LINE_INFANTRY
    cfg = Config(**RICH)
    state, _, step_fn, m = new_world(cfg)
    st, bar = setup_barracks(cfg, state, step_fn)
    camp = int(jnp.argmax((st.etype == 6) & st.alive))  # TYPE_CAMP
    # 先训一只狗(1 级表血量)
    st = drive(st, step_fn, {0: [(bar, a_train_dog(cfg))]}, cfg.dog_time + 1)
    dog = int(jnp.argmax((st.etype == TYPE_DOG) & st.alive))
    assert int(st.hp[dog]) == cfg.dog_hp_by_level[1]
    # 研步兵线:狗不受影响
    st = drive(st, step_fn, {0: [(camp, a_research_line(LINE_INFANTRY, cfg))]},
               cfg.line_res_time[1] + 1)
    assert int(st.upgrades[0, LINE_INFANTRY]) == 2
    assert int(st.hp[dog]) == cfg.dog_hp_by_level[1], "步兵线不该惠及狗"
    # 研狗线:存量狗补 2 级-1 级差额
    st = drive(st, step_fn, {0: [(camp, a_research_line(LINE_DOG, cfg))]},
               cfg.line_res_time[1] + 1)
    assert int(st.upgrades[0, LINE_DOG]) == 2
    assert int(st.hp[dog]) == cfg.dog_hp_by_level[2], "存量狗未补血"
