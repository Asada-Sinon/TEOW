"""v1.6 飞艇:登艇/容量7/威胁禁区/舱内离场/空降即战/击落全灭/回艇锁。"""

import jax.numpy as jnp
from test_armor import one_tick, spawn

from teow.actions import A_NOOP, a_board, a_drop_all, legality_mask
from teow.config import (
    TYPE_AIRSHIP,
    TYPE_INFANTRY,
    TYPE_TOWER,
    TYPE_WORKER,
    Config,
)
from teow.state import owner_of_slots
from teow.step import new_world


def _act(cfg, pairs):
    a = jnp.full(cfg.n_total, A_NOOP, jnp.int32)
    for s, v in pairs:
        a = a.at[s].set(v)
    return a


def _step(st, cfg, step_fn, pairs, seed=0):
    import jax
    return step_fn(st, _act(cfg, pairs), jax.random.PRNGKey(seed))


def test_board_capacity_and_harvester_ban():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, ship = spawn(state, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp,
                     (31.0, 31.0))
    troops = []
    for i in range(8):
        st, s = spawn(st, cfg, 0, 21 + i, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                      (31.0 + 0.1 * i, 31.5))
        troops.append(s)
    st, wk = spawn(st, cfg, 0, 30, TYPE_WORKER, cfg.worker_hp, (31.0, 30.5))
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[wk, a_board(cfg)]), "采集单位不可登艇"
    assert bool(legal[troops[0], a_board(cfg)])
    # 8 兵同 tick 登 7 容量艇:恰 7 上,1 落地
    st = _step(st, cfg, step_fn, [(s, a_board(cfg)) for s in troops])
    aboard = [int(st.aboard[s]) for s in troops]
    assert sum(1 for a in aboard if a == ship) == cfg.airship_capacity
    assert sum(1 for a in aboard if a == -1) == 1, "超发必须裁到容量"
    # 舱内=离场:敌兵贴脸打不到乘员,乘员不还手。
    # 先把没挤上艇的落地兵挪远(它会合法自动互殴,污染断言)
    grounded = [s for s in troops if int(st.aboard[s]) == -1][0]
    st = st._replace(pos=st.pos.at[grounded].set(jnp.asarray([20.0, 20.0])))
    st, foe = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.2))
    st1 = one_tick(st, cfg, step_fn, seed=3)
    hp_ok = all(int(st1.hp[s]) == cfg.inf_hp_by_level[1]
                for s in troops if int(st1.aboard[s]) >= 0)
    assert hp_ok, "舱内乘员不可被打"
    assert int(st1.hp[foe]) == cfg.inf_hp_by_level[1], "舱内乘员不得攻击"


def test_drop_all_and_fight_and_shootdown():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    st, ship = spawn(state, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp,
                     (31.0, 31.0))
    st, inf = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.5))
    st = _step(st, cfg, step_fn, [(inf, a_board(cfg))])
    assert int(st.aboard[inf]) == ship
    # 空降:落在艇位,当拍恢复在场
    st = _step(st, cfg, step_fn, [(ship, a_drop_all(cfg))], seed=2)
    assert int(st.aboard[inf]) == -1
    d = float(jnp.linalg.norm(st.pos[inf] - st.pos[ship]))
    assert d < 1.0, "空降落点=艇当前位置附近"
    # 击落全灭:重新登艇后打掉艇
    st = _step(st, cfg, step_fn, [(inf, a_board(cfg))], seed=3)
    assert int(st.aboard[inf]) == ship
    st = st._replace(hp=st.hp.at[ship].set(0))
    st = one_tick(st, cfg, step_fn, seed=4)
    assert not bool(st.alive[ship]) and not bool(st.alive[inf]), \
        "艇亡乘员全灭(规格)"


def test_threat_zone_blocks_boarding():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, ship = spawn(state, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp,
                     (31.0, 31.0))
    st, inf = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.5))
    # 敌哨塔射程 4.0 覆盖登艇者 → 禁止
    st, twr = spawn(st, cfg, 1, 20, TYPE_TOWER, cfg.tower_hp_by_level[1],
                    (31.0, 35.0))
    legal = legality_mask(st, cfg, m, owner)
    assert not bool(legal[inf, a_board(cfg)]), "敌方攻击范围内禁止上艇"
    # 塔挪远(圈外):放行
    st2 = st._replace(pos=st.pos.at[twr].set(jnp.asarray([31.0, 40.0])))
    legal = legality_mask(st2, cfg, m, owner)
    assert bool(legal[inf, a_board(cfg)]), "威胁圈外即时上艇"


def test_reboard_lockout_after_firing():
    cfg = Config()
    state, _, step_fn, m = new_world(cfg)
    owner = owner_of_slots(cfg)
    st, ship = spawn(state, cfg, 0, 20, TYPE_AIRSHIP, cfg.airship_hp,
                     (31.0, 31.0))
    st, inf = spawn(st, cfg, 0, 21, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 31.5))
    # 手术:近期开过火(锁未清零)→ 禁登;移动不受限(MOVE 掩码仍开)
    st_locked = st._replace(reboard_lock=st.reboard_lock.at[inf].set(10))
    legal = legality_mask(st_locked, cfg, m, owner)
    assert not bool(legal[inf, a_board(cfg)]), "开火后锁定期内禁回艇"
    from teow.actions import A_MOVE0
    assert bool(legal[inf, A_MOVE0]), "锁定只限回艇,不限移动(规格)"
    # 开火事件确实置锁:与敌兵互殴一拍
    st, foe = spawn(st, cfg, 1, 20, TYPE_INFANTRY, cfg.inf_hp_by_level[1],
                    (31.0, 32.5))
    st1 = one_tick(st, cfg, step_fn, seed=5)
    assert int(st1.reboard_lock[inf]) == cfg.reboard_lockout, "开火必须置回艇锁"
    # 锁递减
    st2 = st1._replace(pos=st1.pos.at[foe].set(jnp.asarray([20.0, 20.0])))
    st2 = one_tick(st2, cfg, step_fn, seed=6)
    assert int(st2.reboard_lock[inf]) == cfg.reboard_lockout - 1
