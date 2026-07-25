"""v1.2 连续移动:镜像对撞不僵持、单位不进建筑格、采集节奏回归上界。"""

import jax
import jax.numpy as jnp
from test_economy import drive

from teow.actions import A_NOOP, a_build, a_harvest
from teow.config import RES_ORE, TYPE_INFANTRY, Config
from teow.state import ORDER_ATTACK, cell_of, hq_slot
from teow.step import new_world


def spawn_inf(st, cfg, player, slot_off, rc):
    s = hq_slot(player, cfg) + slot_off
    return st._replace(
        alive=st.alive.at[s].set(True),
        etype=st.etype.at[s].set(TYPE_INFANTRY),
        pos=st.pos.at[s].set(jnp.asarray(rc, jnp.float32)),
        hp=st.hp.at[s].set(cfg.inf_hp_by_level[1]),
        order=st.order.at[s].set(ORDER_ATTACK),
        target_cell=st.target_cell.at[s].set(
            jnp.asarray(rc, jnp.float32)),
    )


def test_mirror_head_on_collision_resolves():
    """180° 对称正对互推是精确共线退化(critic FYI-1):两个敌对步兵在同一行
    **相向 MOVE 对穿**,切向 epsilon 必须让它们错开(途中进入射程自动互殴),
    而不是原地对顶僵持到超时。
    v1.4 注:旧版用 attack-move 且靠敌方工人的 1 点攻击产生伤痕来判「交上手」;
    工人攻击已按规格移除,且 attack-move 的场梯度路径本就互不相交——改为
    MOVE 指令构造真正的共线对撞,直接断言「越过对方或互殴出结果」。"""
    from teow.state import ORDER_MOVE
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st = spawn_inf(state, cfg, 0, 10, (12.0, 4.0))
    st = spawn_inf(st, cfg, 1, 10, (12.0, 19.0))
    sa, sb = hq_slot(0, cfg) + 10, hq_slot(1, cfg) + 10
    # 相向 MOVE:目标=对方起点,路径精确共线
    st = st._replace(
        order=st.order.at[sa].set(ORDER_MOVE).at[sb].set(ORDER_MOVE),
        target_cell=st.target_cell.at[sa].set(jnp.asarray([12.0, 19.0]))
                                  .at[sb].set(jnp.asarray([12.0, 4.0])))

    key = jax.random.PRNGKey(0)
    for _ in range(120):
        key, sub = jax.random.split(key)
        st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), sub)
        if not (bool(st.alive[sa]) and bool(st.alive[sb])):
            break
    # 120 tick 内必须有结果:有人死/有人挂彩/两者已互相越过——
    # 僵持则两者满血、列坐标仍分列两侧且都没到目标
    stuck = (bool(st.alive[sa]) and bool(st.alive[sb])
             and int(st.hp[sa]) == cfg.inf_hp_by_level[1]
             and int(st.hp[sb]) == cfg.inf_hp_by_level[1]
             and float(st.pos[sa, 1]) < float(st.pos[sb, 1]))
    assert not stuck, "正对相遇 120 tick 没有交上手也没有互越:共线僵持未被打破"


def test_units_never_inside_building_cells():
    """互推殿后硬约束:任意 tick,在场单位所在格不得是建筑/静态障碍格。"""
    cfg = Config()
    state, key, step_fn, m = new_world(cfg)
    from teow.controller import make_joint_controller
    joint = jax.jit(make_joint_controller("scripted", "scripted", cfg, m))
    passable = jnp.asarray(m.passable)
    st = state
    for t in range(400):
        key, ka, ks = jax.random.split(key, 3)
        st = step_fn(st, joint(st, ka), ks)
        if t % 20 == 0:
            from teow.movement import building_cells
            hard = ~passable | building_cells(st, cfg)
            spd = jnp.asarray(cfg.speed_by_type)[
                jnp.clip(st.etype.astype(jnp.int32), 0, 31)]
            on = st.alive & ~st.inside & (spd > 0)
            cl = cell_of(st.pos)
            bad = on & hard[cl[:, 0], cl[:, 1]]
            assert not bool(jnp.any(bad)), f"t={t} 单位站进硬障碍格"


def test_harvest_trip_time_regression():
    """单工人一趟采集(走5格+采20tick+返5格)应在 60 tick 内——
    梯度寻路若退化成游走(如自阴影 bug)会到 140+。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st = drive(state, step_fn, {0: [(1, a_build(0))]}, 200)
    last = int(st.resources[0, RES_ORE])
    deps = []
    key = jax.random.PRNGKey(0)
    acts0 = jnp.full(cfg.n_total, A_NOOP, jnp.int32).at[1].set(a_harvest(0, cfg))
    for t in range(200):
        key, sub = jax.random.split(key)
        st = step_fn(st, acts0 if t == 0 else jnp.full(cfg.n_total, A_NOOP, jnp.int32), sub)
        cur = int(st.resources[0, RES_ORE])
        if cur > last:
            deps.append(t)
        last = cur
    assert len(deps) >= 2, f"200 tick 内不足两趟: {deps}"
    gaps = [b - a for a, b in zip(deps, deps[1:], strict=False)]
    assert max(gaps) <= 60, f"一趟超过 60 tick(寻路退化?): {gaps}"
