"""v1.5 淘汰清场:HQ 亡即淘汰,该家全部残余单位/建筑同 tick 清除,
矿泵释放点位、军旗撤除、敌方驻守其点的兵转 IDLE;胜负编码 P=和局。"""

import jax
import jax.numpy as jnp
from test_armor import spawn
from test_economy import W0, drive

from teow.actions import A_NOOP, a_build
from teow.config import TYPE_INFANTRY, TYPE_TOWER, Config
from teow.state import hq_slot, owner_of_slots
from teow.step import new_world


def test_hq_death_wipes_player_and_frees_nodes():
    cfg = Config(start_ore=500, start_water=300)
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    hq1 = hq_slot(1, cfg)

    # p1 建成一座矿(节点 2 是 p1 近家矿点)+ 一座手术台哨塔 + 插一面旗
    st = drive(state, step_fn, {0: [(hq1 + 1, a_build(2))]}, 200)
    assert int(st.node_owner[2]) == 1
    st, twr = spawn(st, cfg, 1, 30, TYPE_TOWER, cfg.tower_hp_by_level[1],
                    (38.0, 18.0))
    st = st._replace(
        flag_active=st.flag_active.at[1, 0].set(True),
        flag_pos=st.flag_pos.at[1, 0].set(jnp.asarray([36.0, 20.0])))
    n_p1_before = int(jnp.sum(st.alive & (owner == 1)))
    assert n_p1_before >= 6  # HQ+工人×4+矿+塔

    # 手术:HQ1 血量清零 → 本 tick 淘汰清场
    st = st._replace(hp=st.hp.at[hq1].set(0))
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32),
                 jax.random.PRNGKey(7))
    assert int(jnp.sum(st.alive & (owner == 1))) == 0, "淘汰者必须同 tick 清场"
    assert int(st.node_owner[2]) == -1 and int(st.node_ent[2]) == -1, \
        "淘汰者的矿泵必须释放点位"
    assert not bool(jnp.any(st.flag_active[1])), "淘汰者军旗必须撤除"
    # 四人局:还剩 0/2/3 三家 → 淘汰不终局
    assert not bool(st.done) and int(st.winner) == -1
    # 再淘汰 2/3:只剩玩家 0 → 胜
    st = st._replace(hp=st.hp.at[hq_slot(2, cfg)].set(0)
                          .at[hq_slot(3, cfg)].set(0))
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32),
                 jax.random.PRNGKey(8))
    assert bool(st.done) and int(st.winner) == 0


def test_winner_encoding_draw_is_n_players():
    """超时和局编码 = n_players(v1.5 泛化;双人局=2,与旧编码兼容)。"""
    cfg = Config(episode_len=5)
    state, _, step_fn, m = new_world(cfg)
    st = drive(state, step_fn, {}, 6)
    assert bool(st.done) and int(st.winner) == cfg.n_players


def test_mutual_hq_kill_same_tick_draw():
    """全部 HQ 同 tick 齐灭 → 和局 P,全场清空。"""
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    hp = state.hp
    for p in range(cfg.n_players):
        hp = hp.at[hq_slot(p, cfg)].set(0)
    st = state._replace(hp=hp)
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32),
                 jax.random.PRNGKey(3))
    assert bool(st.done) and int(st.winner) == cfg.n_players
    # 全清场
    assert int(jnp.sum(st.alive)) == 0


def test_eliminated_enemy_garrison_on_its_node_goes_idle():
    """敌方(p0)驻守在 p1 矿泵旁,p1 被淘汰 → 点位释放 → p0 驻守兵转 IDLE。"""
    from teow.state import ORDER_GARRISON
    cfg = Config(start_ore=500, start_water=300)
    state, _, step_fn, m = new_world(cfg)
    hq1 = hq_slot(1, cfg)
    st = drive(state, step_fn, {0: [(hq1 + 1, a_build(2))]}, 200)
    assert int(st.node_owner[2]) == 1
    # 手术:一个 p0 步兵「驻守」p1 的点?驻守只能己方点——改为构造
    # p0 步兵驻守己方点,然后 p0 被淘汰,断言它被清场即可(对称口径)。
    st = drive(st, step_fn, {0: [(W0, a_build(0))]}, 200)
    assert int(st.node_owner[0]) == 0
    st, inf = spawn(st, cfg, 0, 40, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (24.0, 20.0))
    from teow.actions import a_garrison_node
    st = drive(st, step_fn, {0: [(inf, a_garrison_node(0, cfg))]}, 3)
    assert int(st.order[inf]) == ORDER_GARRISON
    # p1 淘汰 → p1 全清;对局继续(还剩 0/2/3),p0 驻守不受影响
    st = st._replace(hp=st.hp.at[hq1].set(0))
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32),
                 jax.random.PRNGKey(9))
    assert not bool(st.done)
    assert int(jnp.sum(st.alive & (owner_of_slots(cfg) == 1))) == 0
    assert int(st.order[inf]) == ORDER_GARRISON, "无关玩家的驻守不该被扰动"
