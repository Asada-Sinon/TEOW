"""战斗与死亡清理:表驱动目标选择,同 tick 同时结算(允许同归于尽)。

v1.4 改造(plan D7):攻击能力全部查 per-type 表(射程/最小射程/可打单位/
可打建筑/攻击间隔/护甲/魔法),masked argmin 选目标(全无效必须门控,否则
argmin 返 0 造成幽灵攻击)→ `.at[].add` 确定性累加伤害 → 治疗同帧叠加 →
先结算全部伤害再翻 alive 掩码。目标偏好:先打单位再打建筑(score = 距离 +
10*是建筑),同分按槽号。

护甲(用户定案):物理伤害按百分比减伤(公式唯一定义处 stats.physical_damage),
魔法完全无视护甲。建筑类攻击者(塔/迫击炮)保留逐实体 btype==0 建成门
(plan-critic M-2:在建/升级中不开火,can-hit 表只管类型维)。

迫击炮(plan D5):开火=锁定目标当前位置,弹 shell_timer 递减,归零在锁定点
范围爆炸(中心线性衰减),伤害并入同一 incoming 累加器与近战同帧结算;
盲区(d < min_range)打不了;只伤敌方地面单位,不溅建筑、无友伤(DECISIONS)。

治疗(plan D7):奶妈自动奶射程内己方**单位**(不奶建筑/矿内)中血量比例
最低者(整数缩放比值,argmin 槽号平手),hp = clip(hp - incoming + heal, 0,
max_hp)——同帧结算,奶得活将死者。全程无 RNG。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    TYPE_HEALER,
    TYPE_MINE,
    TYPE_MORTAR,
    TYPE_PUMP,
    Config,
)
from .state import (
    ORDER_BUILD,
    ORDER_GARRISON,
    ORDER_HARVEST,
    ORDER_IDLE,
    PH_TO_NODE,
    WorldState,
)
from .stats import (
    atk_of,
    etype_idx,
    max_hp_of,
    physical_damage,
    type_tables,
)


def combat_tick(state: WorldState, cfg: Config, owner: jax.Array) -> WorldState:
    st = state
    tt = type_tables(cfg)
    et = etype_idx(st)
    on_board = st.alive & ~st.inside
    spd = tt["speed"][et]
    is_unit = spd > 0
    is_bld = st.alive & ~is_unit

    rng = tt["rng"][et]
    min_rng = tt["min_rng"][et]
    hit_u = tt["hit_u"][et]
    hit_b = tt["hit_b"][et]
    atk = atk_of(st, cfg, owner)
    # 建筑类攻击者必须建成且空闲(btype==0;在建/升级中不开火——critic M-2);
    # 冷却归零才可开火(普通攻击者 period=1,cd 恒 0,行为与 v1.3 逐位一致)
    ready = jnp.where(is_unit, True, st.btype == 0) & (st.atk_cd == 0)
    attacker = on_board & (atk > 0) & (rng > 0) & ready
    targetable = on_board  # 矿内工人离场不可被打

    eu = jnp.linalg.norm(st.pos[:, None, :] - st.pos[None, :, :], axis=-1)
    valid = (attacker[:, None] & targetable[None, :]
             & (owner[:, None] != owner[None, :])
             & (eu <= rng[:, None]) & (eu >= min_rng[:, None]))
    valid = valid & ((is_unit[None, :] & hit_u[:, None])
                     | (~is_unit[None, :] & hit_b[:, None]))
    score = eu + 10.0 * is_bld[None, :].astype(jnp.float32)
    score = jnp.where(valid, score, 1e9)
    tgt = jnp.argmin(score, axis=1)
    has_tgt = jnp.any(valid, axis=1)               # 门控:全无效时 argmin 返 0

    # ---- 即时攻击者(单体):护甲修正后 scatter(迫击炮不走这条,它只放弹)----
    is_mortar = st.etype == TYPE_MORTAR
    fire_now = has_tgt & attacker & ~is_mortar
    tgt_armor = tt["armor"][et[tgt]]
    dmg_full = jnp.where(tt["magic"][et], atk, physical_damage(atk, tgt_armor))
    dmg = jnp.where(fire_now, dmg_full, 0).astype(jnp.int32)
    incoming = jnp.zeros(cfg.n_total, jnp.int32).at[tgt].add(dmg)

    # ---- 迫击炮:飞行递减 → 落地爆炸 → 空闲且环内有目标则锁定开火 ----
    # 顺序固定(step.py 头注):先结算旧弹再放新弹,弹恰飞 flight_time 拍。
    shell_t = jnp.where(st.shell_timer > 0, st.shell_timer - 1, st.shell_timer)
    land = st.alive & is_mortar & (st.shell_timer > 0) & (shell_t == 0)
    # 爆炸:锁定点为圆心,线性中心衰减,只伤敌方在场单位(DECISIONS:不溅建筑、
    # 无友伤);多炮同时落地 add 叠加
    d_shell = jnp.linalg.norm(st.pos[None, :, :] - st.shell_target[:, None, :],
                              axis=-1)                       # [炮 i, 目标 j]
    aoe_r = tt["aoe"][et]
    fall = jnp.maximum(0.0, aoe_r[:, None] - d_shell) / jnp.maximum(
        aoe_r[:, None], 1e-6)
    base = jnp.ceil(atk[:, None].astype(jnp.float32) * fall).astype(jnp.int32)
    victim = (land[:, None] & on_board[None, :] & is_unit[None, :]
              & (owner[:, None] != owner[None, :]) & (base > 0))
    splash = jnp.where(victim, physical_damage(base, tt["armor"][et][None, :]), 0)
    incoming = incoming + jnp.sum(splash, axis=0)
    # 开火:建成空闲、无在途弹、冷却归零、环内有敌方单位(has_tgt 已含全部门)
    m_fire = has_tgt & attacker & is_mortar & (shell_t == 0)
    shell_t = jnp.where(m_fire, jnp.asarray(cfg.mortar_flight_time, jnp.int16),
                        shell_t).astype(jnp.int16)
    shell_target = jnp.where(m_fire[:, None], st.pos[tgt], st.shell_target)

    # ---- 冷却:开火者置 period-1,其余递减到 0 ----
    fired = fire_now | m_fire
    atk_cd = jnp.where(fired, tt["period"][et] - 1,
                       jnp.maximum(st.atk_cd - 1, 0)).astype(jnp.int16)

    # ---- 治疗(奶妈):射程内己方单位血量比例最低者;同帧与伤害合并结算 ----
    healer_lv = jnp.clip(
        st.upgrades[owner.astype(jnp.int32),
                    jnp.asarray(cfg.line_of_type)[TYPE_HEALER]], 0, 7)
    heal_amt = jnp.asarray(cfg.healer_heal_by_level, jnp.int32)[healer_lv]
    is_healer = st.etype == TYPE_HEALER
    maxhp = max_hp_of(st, cfg, owner)
    ratio = (st.hp * 1000) // jnp.maximum(maxhp, 1)          # 整数比例,千分位
    healable = (on_board & is_unit & (ratio < 1000))[None, :]
    h_valid = ((is_healer & on_board)[:, None] & healable
               & (owner[:, None] == owner[None, :])
               & (eu <= tt["rng"][et][:, None]))
    h_score = jnp.where(h_valid, ratio[None, :].astype(jnp.float32), 1e9)
    h_tgt = jnp.argmin(h_score, axis=1)
    has_h = jnp.any(h_valid, axis=1)
    healing = (jnp.zeros(cfg.n_total, jnp.int32)
               .at[h_tgt].add(jnp.where(has_h, heal_amt, 0)))

    hp = jnp.clip(st.hp - incoming + healing, 0, jnp.maximum(maxhp, st.hp))
    return st._replace(hp=hp, atk_cd=atk_cd, shell_timer=shell_t,
                       shell_target=shell_target)


def cleanup_deaths(state: WorldState, cfg: Config, owner: jax.Array) -> WorldState:
    """翻 alive 位 + 连锁清理:
    - 矿/泵被摧毁 → 资源点回到无主可建;驻内工人原格弹出、不受伤、转 IDLE
      (弹出格若被占会出现短暂叠格,占用图按「格上有人」算,后续移动自然散开)。
    - 施工工人死亡 → 工地取消、点回无主,已扣资源不退(docs/DECISIONS.md)。
    - 采集指令指向的结构没了 → 工人转 IDLE(带着的货保留,可再派)。
    - 淘汰清场(v1.5):HQ 亡的玩家其**全部**残余单位/建筑同 tick 清除
      (规格:HQ 被摧毁即淘汰、残余立即清场),军旗一并撤除;其矿泵经
      dead_struct 链自动释放点位,敌方驻守其点的兵经 gar_inv 链自动转 IDLE。
    - 死槽「停泊」:计时/指令/载荷/冷却/在途弹清零,pos 保留(无害)。"""
    from .state import hq_slot
    st = state
    alive2 = st.alive & (st.hp > 0)
    hq_dead = jnp.stack([~alive2[hq_slot(p, cfg)]
                         for p in range(cfg.n_players)])       # [P]
    alive2 = alive2 & ~hq_dead[owner.astype(jnp.int32)]
    newly_dead = st.alive & ~alive2
    flag_active = st.flag_active & ~hq_dead[:, None]

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

    # 驻守目标失效(v1.3):目标 node 被拆/易主 → 转 IDLE、清 garrison_id
    # (HQ 目标无失效分支:HQ 亡即终局;gid 门控在 order==GARRISON 上)
    gk = jnp.clip(st.garrison_id.astype(jnp.int32) - 1, 0, cfg.n_nodes - 1)
    gar_node = (alive2 & (order == ORDER_GARRISON) & (st.garrison_id >= 1)
                & (st.garrison_id <= cfg.n_nodes))
    gar_inv = gar_node & ((node_ent[gk] == -1) | (node_owner[gk] != owner))
    order = jnp.where(gar_inv, ORDER_IDLE, order)
    garrison_id = jnp.where(gar_inv, -1, st.garrison_id)

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
        garrison_id=jnp.where(park, -1, garrison_id).astype(jnp.int8),
        node_id=jnp.where(park, -1, st.node_id).astype(jnp.int8),
        atk_cd=jnp.where(park, 0, st.atk_cd).astype(jnp.int16),
        shell_timer=jnp.where(park, 0, st.shell_timer).astype(jnp.int16),
        flag_active=flag_active,
        node_owner=node_owner,
        node_ent=node_ent,
        node_build_timer=node_build_timer,
        node_builder=node_builder,
    )
