"""经济链路:训练、建造、采集一体循环的时序与守恒。

驱动方式:python 循环喂动作数组给 jit step(测试里追求可读,不追求吞吐)。
"""

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP, a_build, a_harvest, a_train_worker, legality_mask
from teow.config import RES_ORE, TYPE_MINE, TYPE_WORKER, Config
from teow.state import ORDER_IDLE, hq_slot, owner_of_slots
from teow.step import new_world

W0 = 1  # 玩家 0 的 1 号工人槽(0 是 HQ)


def drive(state, step_fn, actions_by_tick, n_ticks, seed=0):
    """按 tick 喂动作(dict: tick -> [(slot, action)]),其余 NOOP。"""
    key = jax.random.PRNGKey(seed)
    cfg_n = state.alive.shape[0]
    for t in range(n_ticks):
        acts = jnp.full(cfg_n, A_NOOP, jnp.int32)
        for slot, a in actions_by_tick.get(t, []):
            acts = acts.at[slot].set(a)
        key, sub = jax.random.split(key)
        state = step_fn(state, acts, sub)
        if bool(state.done):
            break
    return state


def test_train_worker_cost_and_timing():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    hq = hq_slot(0, cfg)
    n_alive0 = int(jnp.sum(state.alive))

    # tick 0 下训练单 → 立刻扣费;worker_time 个 tick 后落地
    st = drive(state, step_fn, {0: [(hq, a_train_worker(cfg))]}, 1)
    assert int(st.resources[0, RES_ORE]) == cfg.start_ore - cfg.worker_cost_ore
    assert int(st.btimer[hq]) == cfg.worker_time
    assert int(jnp.sum(st.alive)) == n_alive0  # 还没落地

    st = drive(state, step_fn, {0: [(hq, a_train_worker(cfg))]}, cfg.worker_time)
    assert int(jnp.sum(st.alive)) == n_alive0  # 恰好还差一拍(第 T 拍开头落地)
    st = drive(state, step_fn, {0: [(hq, a_train_worker(cfg))]}, cfg.worker_time + 1)
    assert int(jnp.sum(st.alive)) == n_alive0 + 1
    new_slot = hq + 1 + cfg.start_workers
    assert int(st.etype[new_slot]) == TYPE_WORKER
    assert int(st.hp[new_slot]) == cfg.worker_hp


def test_build_mine_then_harvest_cycle():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    node = 0  # 玩家 0 近家矿点

    # 派 1 号工人去建矿,等它走到 + 建完(路程 << 100 tick)
    horizon = 200
    st = drive(state, step_fn, {0: [(W0, a_build(node))]}, horizon)
    assert int(st.node_owner[node]) == 0
    ent = int(st.node_ent[node])
    assert ent >= 0 and int(st.etype[ent]) == TYPE_MINE
    assert int(st.resources[0, RES_ORE]) == cfg.start_ore - cfg.mine_cost_ore
    assert int(st.order[W0]) == ORDER_IDLE  # 完工后工人放空

    # 再派同一工人采集:跑足够久,矿石应按整载荷入账
    ore_before = int(st.resources[0, RES_ORE])
    st2 = drive(st, step_fn, {0: [(W0, a_harvest(node, cfg))]}, 400, seed=1)
    gained = int(st2.resources[0, RES_ORE]) - ore_before
    assert gained > 0, "采集循环没有产出"
    assert gained % cfg.carry_cap == 0, "入账必须是整载荷(卸货点入账)"
    # 循环应当多次往返
    assert gained >= 2 * cfg.carry_cap


def test_node_capacity_enforced():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    node = 0
    # 建矿 + 全部 4 个工人都派去采集,再多训 2 个也派去
    plan = {0: [(W0, a_build(node))]}
    st = drive(state, step_fn, plan, 200)
    acts = {0: [(W0 + i, a_harvest(node, cfg)) for i in range(cfg.start_workers)]}
    st = drive(st, step_fn, acts, 300, seed=2)
    # 驻内人数任何时刻不超容量:跑若干 tick 逐拍检查
    key = jax.random.PRNGKey(3)
    for _ in range(60):
        key, sub = jax.random.split(key)
        st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), sub)
        inside_n = int(jnp.sum(st.inside))
        assert inside_n <= cfg.node_capacity


def test_legality_basics():
    cfg = Config()
    state, _, _, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    legal = legality_mask(state, cfg, m, owner)
    hq = hq_slot(0, cfg)
    # HQ 不能移动/建造;工人不能训练;死槽只有 NOOP
    assert not bool(jnp.any(legal[hq, 3:7]))
    assert bool(legal[hq, a_train_worker(cfg)])
    assert not bool(legal[W0, a_train_worker(cfg)])
    dead = hq + 1 + cfg.start_workers  # 空槽
    assert bool(legal[dead, A_NOOP]) and int(jnp.sum(legal[dead])) == 1
    # 未建矿前不可采集;无主点可建
    assert not bool(legal[W0, a_harvest(0, cfg)])
    assert bool(legal[W0, a_build(0)])
