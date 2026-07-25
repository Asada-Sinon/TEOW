"""经济链路:训练、建造、采集一体循环的时序与守恒。

驱动方式:python 循环喂动作数组给 jit step(测试里追求可读,不追求吞吐)。
"""

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP, a_build, a_harvest, a_train_worker, legality_mask
from teow.config import RES_ORE, TYPE_MINE, TYPE_WORKER, Config
from teow.economy import assigned_counts
from teow.state import ORDER_HARVEST, ORDER_IDLE, hq_slot, owner_of_slots
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
    assert gained % cfg.worker_carry == 0, "入账必须是整载荷(卸货点入账)"
    # 循环应当多次往返
    assert gained >= 2 * cfg.worker_carry


def _built_mine_world(cfg):
    """公共起手:W0 把 0 号矿建成(1 级),4 个初始工人全部空闲可派。"""
    state, _, step_fn, m = new_world(cfg)
    st = drive(state, step_fn, {0: [(W0, a_build(0))]}, 200)
    assert int(st.node_ent[0]) >= 0
    return st, step_fn, m


def test_harvest_cap_never_exceeded():
    cfg = Config()
    st, step_fn, m = _built_mine_world(cfg)
    node = 0
    cap = cfg.harvest_slots_by_level[int(st.level[int(st.node_ent[node])])]
    acts = {0: [(W0 + i, a_harvest(node, cfg)) for i in range(cfg.start_workers)]}
    st = drive(st, step_fn, acts, 300, seed=2)
    # 指派数与驻内人数任何时刻都不超该点等级名额:跑若干 tick 逐拍检查
    key = jax.random.PRNGKey(3)
    for _ in range(60):
        key, sub = jax.random.split(key)
        st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), sub)
        assert int(jnp.sum(st.inside)) <= cap
        assert int(assigned_counts(st, cfg)[node]) <= cap


def test_harvest_slots_cap():
    cfg = Config()
    st, step_fn, m = _built_mine_world(cfg)
    node = 0
    # 4 工人同 tick 对同一 1 级矿(名额 3)下 HARV:同 tick 超发仲裁按槽号
    # 保前 3 个,第 4 个被拒且保持原指令(IDLE)
    acts = {0: [(W0 + i, a_harvest(node, cfg)) for i in range(4)]}
    st = drive(st, step_fn, acts, 1, seed=2)
    assert int(assigned_counts(st, cfg)[node]) == cfg.harvest_slots_by_level[1]
    assert int(st.order[W0 + 2]) == ORDER_HARVEST
    assert int(st.order[W0 + 3]) == ORDER_IDLE


def test_harvest_slots_no_rotation_exploit():
    cfg = Config()
    st, step_fn, m = _built_mine_world(cfg)
    node = 0
    acts = {0: [(W0 + i, a_harvest(node, cfg)) for i in range(3)]}
    st = drive(st, step_fn, acts, 1, seed=2)
    owner = owner_of_slots(cfg)
    # 名额是指派即占用:即使有工人出矿进入运输段(inside=False),
    # 第 4 工人的 HARV 掩码也始终是 False(轮转卡名额的老 bug 不复存在)
    key = jax.random.PRNGKey(4)
    seen_transport = False
    for _ in range(200):
        key, sub = jax.random.split(key)
        st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), sub)
        legal = legality_mask(st, cfg, m, owner)
        assert not bool(legal[W0 + 3, a_harvest(node, cfg)])
        in_transit = bool(jnp.any(st.alive & (st.order == ORDER_HARVEST)
                                  & ~st.inside & (st.cargo > 0)))
        seen_transport = seen_transport or in_transit
    assert seen_transport  # 确认真的覆盖到了出矿运输时刻


def test_harvest_slots_upgrade_expands():
    cfg = Config()
    st, step_fn, m = _built_mine_world(cfg)
    node = 0
    acts = {0: [(W0 + i, a_harvest(node, cfg)) for i in range(3)]}
    st = drive(st, step_fn, acts, 1, seed=2)
    owner = owner_of_slots(cfg)
    ent = int(st.node_ent[node])
    assert not bool(legality_mask(st, cfg, m, owner)[W0 + 3, a_harvest(node, cfg)])
    # 手术把矿提到 3 级(名额 3→4;升级链路本身由 test_upgrade 覆盖)
    st = st._replace(level=st.level.at[ent].set(3))
    assert bool(legality_mask(st, cfg, m, owner)[W0 + 3, a_harvest(node, cfg)])
    st = drive(st, step_fn, {0: [(W0 + 3, a_harvest(node, cfg))]}, 1, seed=5)
    assert int(assigned_counts(st, cfg)[node]) == cfg.harvest_slots_by_level[3]


def test_harvest_slot_released_on_death():
    cfg = Config()
    st, step_fn, m = _built_mine_world(cfg)
    node = 0
    acts = {0: [(W0 + i, a_harvest(node, cfg)) for i in range(3)]}
    st = drive(st, step_fn, acts, 1, seed=2)
    owner = owner_of_slots(cfg)
    assert not bool(legality_mask(st, cfg, m, owner)[W0 + 3, a_harvest(node, cfg)])
    # 手术杀掉一个已指派工人(hp 归零),走一拍让 cleanup_deaths 结算
    st = st._replace(hp=st.hp.at[W0 + 1].set(0))
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), jax.random.PRNGKey(6))
    assert not bool(st.alive[W0 + 1])
    assert bool(legality_mask(st, cfg, m, owner)[W0 + 3, a_harvest(node, cfg)])
    st = drive(st, step_fn, {0: [(W0 + 3, a_harvest(node, cfg))]}, 1, seed=7)
    assert int(assigned_counts(st, cfg)[node]) == cfg.harvest_slots_by_level[1]


def test_harvest_reassign_rejected_keeps_cap():
    # v1.3 终审 P1-1 回归:同 tick「w1 从 A 点改派 k 点被仲裁拒绝 + 新人涌入 A」
    # 不得把 A 顶到 cap+1——HARVEST 改派的旧名额必须「新指派成功才释放」。
    cfg = Config(start_ore=500, start_water=500)
    st, step_fn, m = _built_mine_world(cfg)
    hq = hq_slot(0, cfg)
    # 再建 1 号点(近家水泵),W0 建完转 IDLE
    st = drive(st, step_fn, {0: [(W0, a_build(1))]}, 200, seed=8)
    assert int(st.node_ent[1]) >= 0
    # 串行训 3 个新工人(落槽位置不做假设,训完对比 alive 集合取新增)
    before = {i for i in range(cfg.n_total)
              if bool(st.alive[i]) and int(st.etype[i]) == TYPE_WORKER}
    for i in range(3):
        st = drive(st, step_fn, {0: [(hq, a_train_worker(cfg))]},
                   cfg.worker_time + 2, seed=10 + i)
    w_new = sorted({i for i in range(cfg.n_total)
                    if bool(st.alive[i]) and int(st.etype[i]) == TYPE_WORKER}
                   - before)
    assert len(w_new) == 3
    # 铺垫:w1(=W0+3)独占 A=0 点(1/3);W0+1、W0+2 占 k=1 点(2/3)
    st = drive(st, step_fn, {0: [(W0 + 3, a_harvest(0, cfg)),
                                 (W0 + 1, a_harvest(1, cfg)),
                                 (W0 + 2, a_harvest(1, cfg))]}, 1, seed=20)
    # 同 tick:W0(槽号低)与 w1 抢 k 的最后 1 个名额(w1 rank 高被拒回 A),
    # 3 个新工人同时涌入 A
    acts = {0: [(W0, a_harvest(1, cfg)), (W0 + 3, a_harvest(1, cfg))]
            + [(w, a_harvest(0, cfg)) for w in w_new]}
    st = drive(st, step_fn, acts, 1, seed=21)
    counts = assigned_counts(st, cfg)
    cap = cfg.harvest_slots_by_level[1]
    # 核心不变量:两点任何一侧都不得超名额
    assert int(counts[0]) <= cap and int(counts[1]) <= cap
    # w1 改派被拒:保持对 A 的原指派;A 点因 w1 保守持有,3 新人只进 2 个
    assert int(st.order[W0 + 3]) == ORDER_HARVEST
    assert int(st.target_node[W0 + 3]) == 0
    assert int(counts[0]) == cap
    assert int(st.order[w_new[2]]) == ORDER_IDLE


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


def test_building_never_placed_on_inside_worker_entry_cell():
    """v1.4 审计回归:矿内工人不占格,但其入口格(pos 保留)不得被自由格建筑
    落位——否则出矿弹回即被永久活埋(困在硬障碍格,场梯度归零,实测卡满
    1800 tick 并吊死该点采集名额)。"""
    import jax.numpy as jnp

    from teow.actions import a_build_camp
    from teow.config import TYPE_CAMP, TYPE_WORKER, Config
    from teow.state import PH_MINING, cell_of, hq_slot
    from teow.step import new_world

    cfg = Config(start_ore=2000, start_water=1200)
    state, _, step_fn, m = new_world(cfg)
    hq = hq_slot(0, cfg)
    # 升基地到 2(解锁营)
    from teow.actions import a_upgrade
    st = drive(state, step_fn, {0: [(hq, a_upgrade(cfg))]}, cfg.base_up_time[1] + 1)
    # 手术台:把 2 号工人塞进「矿内」状态,入口格 E=(10,10);
    # 建造者 1 号工人站 (9,9),E 是其 _SPAWN_DIRS 第一候选 (+1,+1)
    E = jnp.asarray([10.0, 10.0], jnp.float32)
    w_in, w_b = hq + 2, hq + 1
    st = st._replace(
        pos=st.pos.at[w_in].set(E).at[w_b].set(jnp.asarray([9.0, 9.0])),
        inside=st.inside.at[w_in].set(True),
        order=st.order.at[w_in].set(ORDER_HARVEST),
        phase=st.phase.at[w_in].set(PH_MINING),
        target_node=st.target_node.at[w_in].set(0),
        mine_timer=st.mine_timer.at[w_in].set(50),
    )
    st = drive(st, step_fn, {0: [(w_b, a_build_camp(cfg))]}, 1)
    camp = int(jnp.argmax((st.etype == TYPE_CAMP) & st.alive))
    assert bool(st.alive[camp]) and int(st.etype[camp]) == TYPE_CAMP
    ccell = cell_of(st.pos[camp])
    assert not bool(jnp.all(ccell == jnp.asarray([10, 10]))), \
        "营落在矿内工人的入口格上:出矿即活埋"
    assert int(st.etype[w_in]) == TYPE_WORKER
