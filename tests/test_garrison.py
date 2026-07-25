"""v1.3 驻守:走到锚点圈内站住/被推离回岗/目标失效转 IDLE/驻守中照常战斗。"""

import jax
import jax.numpy as jnp
from test_economy import drive

from teow.actions import A_NOOP, a_garrison_node
from teow.config import TYPE_DOG, TYPE_INFANTRY, TYPE_MINE, Config
from teow.state import ORDER_GARRISON, ORDER_IDLE, hq_slot
from teow.step import new_world


def setup_mine_and_inf(cfg, state, m, inf_rc):
    """手术摆位:玩家 0 在 node 0 拥有建成矿 + 一个步兵站在 inf_rc。
    返回 (state, mine_slot, inf_slot)。"""
    mine_s = hq_slot(0, cfg) + 10
    inf_s = hq_slot(0, cfg) + 11
    node_rc = jnp.asarray(m.node_pos[0], jnp.float32)
    st = state._replace(
        alive=state.alive.at[mine_s].set(True).at[inf_s].set(True),
        etype=(state.etype.at[mine_s].set(TYPE_MINE)
               .at[inf_s].set(TYPE_INFANTRY)),
        pos=(state.pos.at[mine_s].set(node_rc)
             .at[inf_s].set(jnp.asarray(inf_rc, jnp.float32))),
        hp=(state.hp.at[mine_s].set(cfg.node_struct_hp)
            .at[inf_s].set(cfg.inf_hp_by_level[1])),
        node_id=state.node_id.at[mine_s].set(0),
        node_owner=state.node_owner.at[0].set(0),
        node_ent=state.node_ent.at[0].set(mine_s),
    )
    return st, mine_s, inf_s


def anchor_dist(st, m, inf_s):
    return float(jnp.linalg.norm(
        st.pos[inf_s] - jnp.asarray(m.node_pos[0], jnp.float32)))


def garrisoned_at_node0(cfg):
    """公共前置:步兵从 6 格外驻守 node 0,走到圈内。"""
    state, _, step_fn, m = new_world(cfg)
    st, mine_s, inf_s = setup_mine_and_inf(cfg, state, m, (2.0, 14.0))
    st = drive(st, step_fn, {0: [(inf_s, a_garrison_node(0, cfg))]}, 60)
    return st, step_fn, m, mine_s, inf_s


def test_garrison_walks_and_holds():
    cfg = Config()
    st, _, m, _, inf_s = garrisoned_at_node0(cfg)
    assert anchor_dist(st, m, inf_s) <= cfg.garrison_hold_radius + 1e-3
    assert int(st.order[inf_s]) == ORDER_GARRISON
    assert int(st.garrison_id[inf_s]) == 1  # 0=HQ,1..Nn=点 k=id-1


def test_garrison_returns_after_push():
    cfg = Config()
    st, step_fn, m, _, inf_s = garrisoned_at_node0(cfg)
    # 手术把驻守单位挪开 3 格(模拟被推离),应自动回岗
    st = st._replace(
        pos=st.pos.at[inf_s].set(st.pos[inf_s] + jnp.asarray([3.0, 0.0])))
    assert anchor_dist(st, m, inf_s) > cfg.garrison_hold_radius
    st = drive(st, step_fn, {}, 12, seed=1)
    assert anchor_dist(st, m, inf_s) <= cfg.garrison_hold_radius + 1e-3
    assert int(st.order[inf_s]) == ORDER_GARRISON


def test_garrison_cleared_on_node_lost():
    cfg = Config()
    st, step_fn, m, mine_s, inf_s = garrisoned_at_node0(cfg)
    # 拆掉目标矿:hp 归零,下一 tick cleanup 翻掩码并清驻守
    st = st._replace(hp=st.hp.at[mine_s].set(0))
    st = drive(st, step_fn, {}, 1, seed=2)
    assert not bool(st.alive[mine_s])
    assert int(st.order[inf_s]) == ORDER_IDLE
    assert int(st.garrison_id[inf_s]) == -1


def test_garrison_fights_back():
    cfg = Config()
    st, step_fn, m, _, inf_s = garrisoned_at_node0(cfg)
    # 敌狗走进近战圈:驻守步兵不动窝,战斗按位置自然生效
    dog_s = hq_slot(1, cfg) + 20
    st = st._replace(
        alive=st.alive.at[dog_s].set(True),
        etype=st.etype.at[dog_s].set(TYPE_DOG),
        pos=st.pos.at[dog_s].set(st.pos[inf_s] + jnp.asarray([1.0, 0.0])),
        hp=st.hp.at[dog_s].set(cfg.dog_hp_by_level[1]),
    )
    pos0 = st.pos[inf_s]
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32),
                 jax.random.PRNGKey(3))
    assert int(st.hp[dog_s]) == cfg.dog_hp_by_level[1] - cfg.inf_atk_by_level[1]
    assert int(st.order[inf_s]) == ORDER_GARRISON
    assert float(jnp.linalg.norm(st.pos[inf_s] - pos0)) < 1e-6  # 不动窝
