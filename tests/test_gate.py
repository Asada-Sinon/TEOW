"""异界之门 sudden-death(v1.8):门开出阵营隔离怪、HP 无上限线性/攻击封顶、
慢速近战、玩家亡则其怪离场、必分胜负、决定论。"""

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP
from teow.config import TYPE_INFANTRY, Config
from teow.gate import gate_tick, monster_combat_tick
from teow.map import build_map
from teow.state import hq_slot, init_state, owner_of_slots
from teow.step import make_scan, new_world


def _put_unit(st, slot, etype, hp, pos):
    return st._replace(
        alive=st.alive.at[slot].set(True),
        etype=st.etype.at[slot].set(etype),
        hp=st.hp.at[slot].set(hp),
        pos=st.pos.at[slot].set(jnp.asarray(pos, jnp.float32)))


def _put_monster(st, p, m, hp, atk, pos):
    return st._replace(
        monster_alive=st.monster_alive.at[p, m].set(True),
        monster_hp=st.monster_hp.at[p, m].set(hp),
        monster_atk=st.monster_atk.at[p, m].set(atk),
        monster_pos=st.monster_pos.at[p, m].set(jnp.asarray(pos, jnp.float32)))


def _idle(cfg):
    return lambda s, k: jnp.full(cfg.n_total, A_NOOP, jnp.int32)


def test_gate_spawns_after_open_only():
    """门未开无怪;门开(tick>=gate_open_tick)首波向每个存活玩家出 wave_count 只。"""
    cfg = Config(gate_open_tick=2, monster_wave_count=2, monster_spawn_interval=100)
    state, key, step_fn, m = new_world(cfg)
    noop = jnp.full(cfg.n_total, A_NOOP, jnp.int32)
    st = state
    for _ in range(2):                       # gate_tick 见 tick=0,1 < 2:不 spawn
        key, sub = jax.random.split(key)
        st = step_fn(st, noop, sub)
    assert int(st.monster_alive.sum()) == 0, "门未开不得有怪"
    key, sub = jax.random.split(key)
    st = step_fn(st, noop, sub)               # 第 3 步:gate_tick 见 tick=2 → 首波
    assert int(st.monster_alive.sum()) == cfg.monster_wave_count * cfg.n_players


def test_faction_isolation():
    """行 p 的怪只打 p 的单位、也只有 p 能打它;跨阵营零交互(直接调 combat pass)。"""
    cfg = Config()
    m = build_map(cfg)
    st = init_state(cfg, m)
    owner = owner_of_slots(cfg)
    pos = (30.0, 30.0)
    s0, s1 = hq_slot(0, cfg) + 5, hq_slot(1, cfg) + 5
    inf_hp = int(cfg.inf_hp_by_level[1])
    st = _put_unit(st, s0, TYPE_INFANTRY, inf_hp, pos)   # p0 战斗单位
    st = _put_unit(st, s1, TYPE_INFANTRY, inf_hp, pos)   # p1 战斗单位(同点)
    st = _put_monster(st, 0, 0, 100, 10, pos)            # 打玩家 0 的怪(row 0)
    out = monster_combat_tick(st, cfg, owner)
    assert int(out.hp[s0]) < inf_hp, "怪应伤及其目标玩家(p0)的单位"
    assert int(out.hp[s1]) == inf_hp, "阵营隔离:p0 的怪不得伤 p1 单位"
    # 怪只被 p0 单位打了一份 inf_atk;若 p1 也能打会掉两份 → 恰等一份即证隔离
    assert int(out.monster_hp[0, 0]) == 100 - int(cfg.inf_atk_by_level[1])


def test_monster_hp_linear_atk_capped():
    """HP 无上限随 overtime 线性增;攻击力封顶(直接调 gate_tick 验生成公式)。"""
    cfg = Config(gate_open_tick=0, monster_spawn_interval=1, monster_wave_count=1,
                 monster_hp_base=50, monster_hp_slope=1.0,
                 monster_atk_base=5, monster_atk_slope=1.0, monster_atk_cap=20)
    m = build_map(cfg)
    base = init_state(cfg, m)
    s0 = gate_tick(base._replace(tick=jnp.asarray(0, jnp.int32)), cfg, m)
    s100 = gate_tick(base._replace(tick=jnp.asarray(100, jnp.int32)), cfg, m)
    assert int(s0.monster_hp.max()) == 50
    assert int(s100.monster_hp.max()) == 150, "HP 无上限:50 + 1.0*100"
    assert int(s0.monster_atk.max()) == 5
    assert int(s100.monster_atk.max()) == 20, "攻击封顶 min(20, 5+100)"


def test_dead_player_monsters_despawn():
    """玩家被淘汰(HQ 亡)→ 针对它的怪同 tick 离场;存活玩家的怪不受影响。"""
    cfg = Config(gate_open_tick=99999, monster_speed=0.0)  # 不 spawn、不移动
    state, key, step_fn, m = new_world(cfg)
    st = state
    for mm in range(3):
        st = _put_monster(st, 0, mm, 100, 5, (30.0, 30.0))
        st = _put_monster(st, 1, mm, 100, 5, (34.0, 34.0))
    st = st._replace(hp=st.hp.at[hq_slot(1, cfg)].set(0))   # 杀 p1 HQ
    st = step_fn(st, jnp.full(cfg.n_total, A_NOOP, jnp.int32), jax.random.PRNGKey(1))
    assert int(st.monster_alive[1].sum()) == 0, "被淘汰玩家的怪必须离场"
    assert int(st.monster_alive[0].sum()) == 3, "存活玩家的怪不受影响"


def test_gate_forces_unique_winner_before_hardcap():
    """空转局(全 NOOP)本会超时——异界之门放强快怪,应在硬帽前逼出唯一胜者,绝不和局。"""
    cfg = Config(gate_open_tick=3, monster_spawn_interval=3, monster_wave_count=4,
                 monster_hp_base=300, monster_hp_slope=5.0,
                 monster_atk_base=200, monster_atk_cap=200,
                 monster_speed=0.8, episode_len=400)
    state, key, step_fn, m = new_world(cfg)
    scan = make_scan(step_fn, _idle(cfg))
    st, _, _ = scan(state, key, cfg.episode_len)
    assert bool(st.done), "异界之门必须逼出终局"
    assert 0 <= int(st.winner) < cfg.n_players, "必分唯一胜者,绝不和局(winner!=P)"
    assert int(st.tick) < cfg.episode_len, "应由异界之门在防御硬帽前分出胜负"


def test_gate_determinism():
    """带异界之门的空转局跑进 overtime,同 seed 两次逐位一致(含怪物子表)。"""
    cfg = Config(gate_open_tick=3, monster_spawn_interval=3, monster_wave_count=2,
                 monster_atk_base=40, monster_speed=0.6, episode_len=200)

    def run():
        state, key, step_fn, m = new_world(cfg)
        st, _, _ = make_scan(step_fn, _idle(cfg))(state, key, cfg.episode_len)
        return st

    a, b = run(), run()
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert jnp.array_equal(x, y), "异界之门局必须逐位一致"
