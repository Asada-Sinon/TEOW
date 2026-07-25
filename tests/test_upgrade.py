"""v1.1 升级系统:时序、上限链、互斥、产量公式、多笔扣费不透支。"""

import jax
import jax.numpy as jnp
from test_economy import W0, drive  # tests/ 不是包,直接按模块名引(conftest 已加 path)

from teow.actions import A_NOOP, a_build, a_harvest, a_train_worker, a_upgrade, legality_mask
from teow.config import RES_ORE, RES_WATER, Config
from teow.state import hq_slot, owner_of_slots
from teow.step import make_scan, new_world


def test_base_upgrade_cost_timing_and_cap():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    hq = hq_slot(0, cfg)

    # 下升级单:立刻扣费(顺序对账 pass),T 拍后 level 2
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, 1)
    assert int(st.resources[0, RES_ORE]) == cfg.start_ore - cfg.base_up_cost_ore[1]
    assert int(st.resources[0, RES_WATER]) == cfg.start_water - cfg.base_up_cost_water[1]
    assert int(st.btimer[hq]) == cfg.base_up_time[1] and int(st.btype[hq]) < 0
    assert int(st.level[hq]) == 1  # 还没完成

    t = cfg.base_up_time[1]
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, t)
    assert int(st.level[hq]) == 1  # 差一拍
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, t + 1)
    assert int(st.level[hq]) == 2
    assert int(st.btype[hq]) == 0  # 完成必须清任务码(否则每 tick 重复+1)
    # 再跑 3 拍,等级不再变
    st2 = drive(st, step_fn, {}, 3, seed=9)
    assert int(st2.level[hq]) == 2


def test_upgrade_train_mutual_exclusion():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    hq = hq_slot(0, cfg)
    owner = owner_of_slots(cfg)

    # 升级中:训练非法
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, 1)
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[hq, a_train_worker(cfg)])
    assert not bool(legal[hq, a_upgrade(cfg)])
    # 训练中:升级非法(critic S-1:否则覆写在训单位不退款)
    st = drive(state, step_fn, {0: [(hq, a_train_worker(cfg))]}, 1)
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[hq, a_upgrade(cfg)])


def test_node_upgrade_cap_chain_and_yield():
    # 富开局:建矿(40)+升基地(100/50)+升矿(30/20)要一次付清,默认开局不够付
    cfg = Config(start_ore=500, start_water=300)
    state, _, step_fn, m = new_world(cfg)
    hq = hq_slot(0, cfg)
    owner = owner_of_slots(cfg)
    node = 0

    # 建矿
    st = drive(state, step_fn, {0: [(W0, a_build(node))]}, 200)
    ent = int(st.node_ent[node])
    assert ent >= 0

    # 基地 1 级:矿不可升(上限链)
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[ent, a_upgrade(cfg)])

    # 升基地到 2 → 矿可升;升矿到 2 → 一趟入账 = carry[1] + bonus[2]
    st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    assert int(st.level[hq]) == 2
    legal = legality_mask(st, cfg, m, owner)
    assert bool(legal[ent, a_upgrade(cfg)])

    st = drive(st, step_fn, {0: [(ent, a_upgrade(cfg))]}, cfg.node_up_time[1] + 1)
    assert int(st.level[ent]) == 2

    ore0 = int(st.resources[0, RES_ORE])
    st = drive(st, step_fn, {0: [(W0, a_harvest(node, cfg))]}, 260, seed=3)
    gained = int(st.resources[0, RES_ORE]) - ore0
    trip = cfg.worker_carry + cfg.node_yield_bonus[2]
    assert gained > 0 and gained % trip == 0, f"gained={gained} 应为 {trip} 的整数倍"


def test_random_random_resources_nonnegative():
    """critic B-1 回归:random 双方 300 tick,库存恒 >=0(多笔支出顺序对账)。"""
    cfg = Config()
    state, key, step_fn, m = new_world(cfg)
    from teow.controller import make_joint_controller
    joint = make_joint_controller("random", "random", cfg, m)
    scan = make_scan(step_fn, joint)

    # 分 30 段跑,每段检查一次(scan 内部状态不可见,分段即抽样检查)
    st = state
    for _ in range(30):
        st, key, _ = scan(st, key, 10)
        assert bool(jnp.all(st.resources >= 0)), st.resources.tolist()


def test_upgrade_interrupted_by_death_no_refund():
    cfg = Config(start_ore=500, start_water=300)
    state, _, step_fn, m = new_world(cfg)
    node = 0
    st = drive(state, step_fn, {0: [(W0, a_build(node))]}, 200)
    ent = int(st.node_ent[node])
    hq = hq_slot(0, cfg)
    st = drive(st, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)

    # 矿开升后立刻打死矿:升级中断、已扣资源不退、点位回收
    st = drive(st, step_fn, {0: [(ent, a_upgrade(cfg))]}, 1)
    res_after_pay = st.resources[0].tolist()
    assert int(st.btype[ent]) < 0
    st = st._replace(hp=st.hp.at[ent].set(0))
    key = jax.random.PRNGKey(7)
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), key)
    assert not bool(st.alive[ent]) and int(st.node_ent[node]) == -1
    assert int(st.btype[ent]) == 0 and int(st.level[ent]) == 1  # 死槽停泊
    assert st.resources[0].tolist() == res_after_pay  # 不退款
