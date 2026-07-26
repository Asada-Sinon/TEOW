"""异界之门 sudden-death 子系统(v1.8):门开后向每个存活玩家出**阵营隔离**怪物。

设计(用户 2026-07-26 在线拍板;DECISIONS D0-D3):
- **阵营隔离**:怪物住独立子表 `monster_*[P,Mmax]`,行 p 的怪只与 owner==p 的实体交互
  (打 p 的怪只打 p、也只有 p 能打它);跨阵营恒零——天然成立,不动 owner-by-row 主表。
- **gate_tick**(step 顺序里 movement 与 combat 之间):到 `gate_open_tick` 后每
  `spawn_interval` 拍向每个**存活**玩家中心出一波怪(填空槽);HP 无上限随 overtime 线性、
  攻击封顶、**强度生成时定死**(后出更强、已在场不变);再让所有存活怪沿其目标玩家 HQ
  距离场 argmin-邻格慢速下降(不参与单位互推,只受 impassable 经场天然绕障)。
- **monster_combat_tick**(combat 与 cleanup 之间,晚正常近战一个子阶段——见 step.py
  头注):同一 pre-pass 快照算两侧:怪打最近的 p 存活实体(单位/建筑/HQ,吃护甲物理),
  p 的可攻击实体打其射程内最近的怪(怪护甲 0,满伤);同帧结算允许互杀。
- 全程无 RNG(决定论不破);清怪在 cleanup_deaths(玩家亡则其怪离场,复用 hq_dead 掩码)。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, hq_slot
from .stats import atk_of, etype_idx, max_hp_of, physical_damage, type_tables

# 8 邻方向(慢速 descent 的邻格候选)
_DIRS = jnp.asarray([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                     [0, 1], [1, -1], [1, 0], [1, 1]], jnp.int32)


def gate_tick(state: WorldState, cfg: Config, mapdata) -> WorldState:
    """生成波次 + 怪物慢速 descent(gate 未开时为无 alive 怪的廉价 no-op)。"""
    st = state
    P, M = cfg.n_players, cfg.monster_cap
    h, w = cfg.grid_h, cfg.grid_w
    tick = st.tick                                  # gate_tick 在 _end_tick 前,tick=本拍值
    overtime = jnp.maximum(0, tick - cfg.gate_open_tick)
    gate_on = tick >= cfg.gate_open_tick
    spawn_now = gate_on & (
        ((tick - cfg.gate_open_tick) % cfg.monster_spawn_interval) == 0)
    hq_alive = jnp.stack([st.alive[hq_slot(p, cfg)] for p in range(P)])   # [P]

    # ---- 生成一波:填每个存活玩家行的前 wave_count 个空槽 ----
    empty = ~st.monster_alive                                            # [P,M]
    rank = jnp.cumsum(empty.astype(jnp.int32), axis=1) - 1               # 空槽行内序号
    fill = (empty & (rank < cfg.monster_wave_count) & spawn_now
            & hq_alive[:, None])                                         # [P,M]
    idx = jnp.arange(M)
    off = jnp.stack([((idx % 5) - 2) * 0.35, ((idx // 5) % 5 - 2) * 0.35], -1)  # [M,2]
    spawn_pos = jnp.stack([jnp.full(M, h / 2.0), jnp.full(M, w / 2.0)], -1) + off
    spawn_pos = jnp.broadcast_to(spawn_pos, (P, M, 2))
    hp_spawn = (cfg.monster_hp_base
                + jnp.round(cfg.monster_hp_slope * overtime)).astype(jnp.int32)
    atk_spawn = jnp.minimum(
        cfg.monster_atk_cap,
        cfg.monster_atk_base + jnp.round(cfg.monster_atk_slope * overtime)
    ).astype(jnp.int32)
    m_alive = st.monster_alive | fill
    m_pos = jnp.where(fill[..., None], spawn_pos, st.monster_pos)
    m_hp = jnp.where(fill, hp_spawn, st.monster_hp).astype(jnp.int32)
    m_atk = jnp.where(fill, atk_spawn, st.monster_atk).astype(jnp.int32)

    # ---- 慢速 descent:沿目标玩家 HQ 距离场,取 8 邻里场值最小者迈一步 ----
    fields_hq = jnp.asarray(mapdata.dist_fields)[cfg.n_nodes:cfg.n_nodes + P]  # [P,H,W]
    cells = jnp.round(m_pos).astype(jnp.int32)                          # [P,M,2]
    ncell = cells[:, :, None, :] + _DIRS[None, None]                    # [P,M,8,2]
    nr = jnp.clip(ncell[..., 0], 0, h - 1)
    nc = jnp.clip(ncell[..., 1], 0, w - 1)
    pidx = jnp.arange(P)[:, None, None]                                 # [P,1,1]
    nfield = fields_hq[pidx, nr, nc]  # [P,M,8] BIG_DIST 哨兵自然避障
    best = jnp.argmin(nfield, axis=-1)                                  # [P,M]
    tr = jnp.take_along_axis(nr, best[..., None], -1)[..., 0]
    tc = jnp.take_along_axis(nc, best[..., None], -1)[..., 0]
    tgt = jnp.stack([tr, tc], -1).astype(jnp.float32)                   # 目标邻格中心
    d = tgt - m_pos
    dist = jnp.linalg.norm(d, axis=-1, keepdims=True)
    stepv = jnp.where(dist > 1e-6, d / jnp.maximum(dist, 1e-6) * cfg.monster_speed, 0.0)
    m_pos = jnp.where((m_alive & gate_on)[..., None], m_pos + stepv, m_pos)

    return st._replace(monster_alive=m_alive, monster_pos=m_pos,
                       monster_hp=m_hp, monster_atk=m_atk)


def monster_combat_tick(state: WorldState, cfg: Config, owner: jax.Array) -> WorldState:
    """怪物近战 pass(阵营隔离):行 p 的怪 ↔ owner==p 的存活实体,同一快照算两侧。"""
    st = state
    P, M, e = cfg.n_players, cfg.monster_cap, cfg.e_max
    tt = type_tables(cfg)
    et = etype_idx(st)
    on_board = st.alive & ~st.inside & (st.aboard < 0)                  # [N] 在场(可被打/可打)
    is_unit = tt["speed"][et] > 0
    atk = atk_of(st, cfg, owner)                                        # [N]
    rng = tt["rng"][et]
    ready = jnp.where(is_unit, True, st.btype == 0) & (st.atk_cd == 0)  # 建筑须建成、cd 归零
    e_attacker = on_board & (atk > 0) & (rng > 0) & ready               # [N] 能打怪的实体

    def pe(x):                                                          # [N,...] -> [P,e,...]
        return x.reshape(P, e, *x.shape[1:])
    pos_e, onb_e = pe(st.pos), pe(on_board)
    att_e, atk_e, rng_e = pe(e_attacker), pe(atk), pe(rng)
    armor_e = pe(tt["armor"][et])

    m_alive, m_pos, m_hp, m_atk = (st.monster_alive, st.monster_pos,
                                   st.monster_hp, st.monster_atk)
    dist = jnp.linalg.norm(pos_e[:, :, None, :] - m_pos[:, None, :, :], axis=-1)  # [P,e,M]

    # ---- 怪 → 实体:每只怪打 melee_range 内最近的 p 存活实体(物理,吃护甲)----
    m2e_valid = (m_alive[:, None, :] & onb_e[:, :, None]
                 & (dist <= cfg.monster_melee_range))                   # [P,e,M]
    tgt_e = jnp.argmin(jnp.where(m2e_valid, dist, 1e9), axis=1)         # [P,M] 目标实体 idx
    m_has = jnp.any(m2e_valid, axis=1)                                  # [P,M]
    tgt_armor = jnp.take_along_axis(armor_e, tgt_e, axis=1)             # [P,M]
    m_dmg = jnp.where(m_has, physical_damage(m_atk, tgt_armor), 0).astype(jnp.int32)
    ent_inc = jax.vmap(lambda a, i, v: a.at[i].add(v))(
        jnp.zeros((P, e), jnp.int32), tgt_e, m_dmg)                     # [P,e]

    # ---- 实体 → 怪:每个能打的实体打其射程内最近的怪(怪护甲 0,满伤)----
    e2m_valid = (att_e[:, :, None] & m_alive[:, None, :]
                 & (dist <= rng_e[:, :, None]))                         # [P,e,M]
    tgt_m = jnp.argmin(jnp.where(e2m_valid, dist, 1e9), axis=2)         # [P,e] 目标怪 idx
    e_has = jnp.any(e2m_valid, axis=2)
    e_dmg = jnp.where(e_has, atk_e, 0).astype(jnp.int32)
    mon_inc = jax.vmap(lambda a, i, v: a.at[i].add(v))(
        jnp.zeros((P, M), jnp.int32), tgt_m, e_dmg)                     # [P,M]

    # ---- 同帧应用(允许互杀)----
    maxhp = max_hp_of(st, cfg, owner)
    hp = jnp.clip(st.hp - ent_inc.reshape(-1), 0, jnp.maximum(maxhp, st.hp))
    m_hp2 = jnp.maximum(m_hp - mon_inc, 0)
    m_alive2 = m_alive & (m_hp2 > 0)
    # 对怪开火者同样进入攻击冷却(审计 P2-2 修:否则 period>1 建筑/远程对怪每 tick 连发,
    # 系统性抬高龟缩/攻城/空军抗怪力、扭曲 v1.9 平衡评测;period=1 近战者 cd 恒 0 不变)。
    # 说明:能对怪开火者其正常近战 cd 必为 0(否则被 ready 门挡),故此处置 cd 与 combat 不冲突
    period = tt["period"][et]
    atk_cd = jnp.where(e_has.reshape(-1), (period - 1).astype(jnp.int16), st.atk_cd)
    return st._replace(hp=hp, atk_cd=atk_cd, monster_hp=m_hp2, monster_alive=m_alive2)
