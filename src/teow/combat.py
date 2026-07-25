"""战斗与死亡清理:相邻自动互砍,同 tick 同时结算(允许同归于尽)。

照抄 SMAX 链条(调研报告 §3):masked argmin 选目标(全无效必须门控,否则
argmin 返 0 造成幽灵攻击)→ `.at[].add` 确定性累加伤害 → 先结算全部伤害再翻
alive 掩码。目标偏好:先打单位再打建筑(score = 距离 + 10*是建筑),同分按槽号
——攻击方自己选谁挨打不构成玩家间不公平。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    LINE_INFANTRY,
    TYPE_CAMP,
    TYPE_HQ,
    TYPE_INFANTRY,
    TYPE_MINE,
    TYPE_PUMP,
    TYPE_WORKER,
    Config,
)
from .state import ORDER_BUILD, ORDER_HARVEST, ORDER_IDLE, PH_TO_NODE, WorldState


def combat_tick(state: WorldState, cfg: Config, owner: jax.Array) -> WorldState:
    st = state
    on_board = st.alive & ~st.inside
    is_unit = (st.etype == TYPE_WORKER) | (st.etype == TYPE_INFANTRY)
    attacker = on_board & is_unit
    targetable = on_board  # 建筑(HQ/矿/泵)都可被打;矿内工人离场不可被打

    # v1.2:欧氏射程(melee_range≈旧 Chebyshev≤1 含对角)
    eu = jnp.linalg.norm(st.pos[:, None, :] - st.pos[None, :, :], axis=-1)
    valid = (attacker[:, None] & targetable[None, :]
             & (owner[:, None] != owner[None, :]) & (eu <= cfg.melee_range))
    is_building = ((st.etype == TYPE_HQ) | (st.etype == TYPE_MINE)
                   | (st.etype == TYPE_PUMP) | (st.etype == TYPE_CAMP))
    score = eu + 10.0 * is_building[None, :].astype(jnp.float32)
    score = jnp.where(valid, score, 1e9)
    tgt = jnp.argmin(score, axis=1)
    has_tgt = jnp.any(valid, axis=1)               # 门控:全无效时 argmin 返 0

    # 步兵攻击按其玩家的步兵线等级查表(v1.1 全局线);工人攻击无升级线,恒标量
    il = st.upgrades[owner.astype(jnp.int32), LINE_INFANTRY]
    inf_atk = jnp.asarray(cfg.inf_atk_by_level)[il]
    atk = jnp.where(st.etype == TYPE_INFANTRY, inf_atk, cfg.worker_atk)
    dmg = jnp.where(has_tgt & attacker, atk, 0).astype(jnp.int32)
    incoming = jnp.zeros(cfg.n_total, jnp.int32).at[tgt].add(dmg)
    hp = jnp.maximum(st.hp - incoming, 0)
    return st._replace(hp=hp)


def cleanup_deaths(state: WorldState, cfg: Config, owner: jax.Array) -> WorldState:
    """翻 alive 位 + 连锁清理:
    - 矿/泵被摧毁 → 资源点回到无主可建;驻内工人原格弹出、不受伤、转 IDLE
      (弹出格若被占会出现短暂叠格,占用图按「格上有人」算,后续移动自然散开)。
    - 施工工人死亡 → 工地取消、点回无主,已扣资源不退(docs/DECISIONS.md)。
    - 采集指令指向的结构没了 → 工人转 IDLE(带着的货保留,可再派)。
    - 死槽「停泊」:计时/指令/载荷清零,pos 保留(无害)。"""
    st = state
    alive2 = st.alive & (st.hp > 0)
    newly_dead = st.alive & ~alive2

    # 结构死亡 → 点位重置
    dead_struct = newly_dead & ((st.etype == TYPE_MINE) | (st.etype == TYPE_PUMP))
    nid = jnp.clip(st.node_id.astype(jnp.int32), 0, cfg.n_nodes - 1)
    node_gone = (jnp.zeros(cfg.n_nodes, bool).at[nid].max(dead_struct))
    node_owner = jnp.where(node_gone, -1, st.node_owner).astype(jnp.int8)
    node_ent = jnp.where(node_gone, -1, st.node_ent).astype(jnp.int16)

    # 驻内工人弹出
    tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    pop = st.inside & node_gone[tn]
    inside = st.inside & ~pop
    order = jnp.where(pop, ORDER_IDLE, st.order)
    phase = jnp.where(pop, PH_TO_NODE, st.phase)
    mine_timer = jnp.where(pop, 0, st.mine_timer)

    # 施工工人死亡 → 工地取消(builder 槽号 gather 后查存活)
    nb = jnp.clip(st.node_builder.astype(jnp.int32), 0, cfg.n_total - 1)
    builder_dead = (st.node_builder >= 0) & (st.node_build_timer > 0) & ~alive2[nb]
    node_owner = jnp.where(builder_dead, -1, node_owner).astype(jnp.int8)
    node_build_timer = jnp.where(builder_dead, 0,
                                 st.node_build_timer).astype(jnp.int16)
    node_builder = jnp.where(builder_dead, -1, st.node_builder).astype(jnp.int16)

    # 采集目标失效 → 转 IDLE(在结构死亡同 tick 生效)
    harv = alive2 & (order == ORDER_HARVEST) & (st.target_node >= 0)
    tgt_bad = (node_ent[tn] == -1) | (node_owner[tn] != owner)
    inv = harv & tgt_bad & ~inside
    order = jnp.where(inv, ORDER_IDLE, order)

    # 建造目标失效 → 转 IDLE:目标点已被人占/在施工,且施工者不是自己。
    # 不清理会产生「僵尸建造工」:带着永不完成的 BUILD 指令永久站在矿入口
    # 当路障,堵死采集循环(实测导致矿石收入归零)。
    slots = jnp.arange(cfg.n_total)
    build_o = alive2 & (order == ORDER_BUILD) & (st.target_node >= 0)
    site_busy = (node_owner[tn] != -1) | (node_build_timer[tn] > 0)
    not_my_site = node_builder[tn] != slots
    zombie = build_o & site_busy & not_my_site
    order = jnp.where(zombie, ORDER_IDLE, order)

    # 死槽停泊
    park = ~alive2
    return st._replace(
        alive=alive2,
        inside=jnp.where(park, False, inside),
        order=jnp.where(park, ORDER_IDLE, order).astype(jnp.int8),
        phase=jnp.where(park, PH_TO_NODE, phase).astype(jnp.int8),
        cargo=jnp.where(park, 0, st.cargo).astype(jnp.int16),
        cargo_type=jnp.where(park, 0, st.cargo_type).astype(jnp.int8),
        mine_timer=jnp.where(park, 0, mine_timer).astype(jnp.int16),
        btype=jnp.where(park, 0, st.btype).astype(jnp.int8),
        btimer=jnp.where(park, 0, st.btimer).astype(jnp.int16),
        # level 必须停泊回 1:槽位复用时新实体不得继承前任建筑的等级
        level=jnp.where(park, 1, st.level).astype(jnp.int8),
        target_node=jnp.where(park, -1, st.target_node).astype(jnp.int8),
        node_id=jnp.where(park, -1, st.node_id).astype(jnp.int8),
        node_owner=node_owner,
        node_ent=node_ent,
        node_build_timer=node_build_timer,
        node_builder=node_builder,
    )
