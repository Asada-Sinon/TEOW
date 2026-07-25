"""连续移动(v1.2):场梯度寻路 + 圆形互推 + 建筑格硬约束。

管线(顺序固定,硬约束殿后):
  1. 方向:场型指令(HARVEST/BUILD/ATTACK)对动态距离场做**双线性梯度采样**
     (采样前 clip 掉 BIG_DIST 哨兵——否则建筑邻域梯度被 BIG 支配,单位倒退
     不贴墙,critic S-3);MOVE 直指目标点。
  2. candidate = pos + dir * speed[etype](格/tick,Config 表)。
  3. 建筑/边界:目标格是硬障碍则分轴滑动(先试仅动 r,再试仅动 c,都不行原地)。
  4. 单位圆互推一轮(SMAX 模式):重叠量沿连线推开;**精确共线时按槽号奇偶加
     确定性切向分量**(180° 对称开局的正对互推会退化为原地对顶,critic FYI-1)。
  5. 建筑格硬 clamp 殿后:互推若把单位挤进硬障碍格,退回推前位置的分轴滑动结果。

动态场:静止单位从 v1.0 的硬障碍改为大 cell_cost 软障碍(连续单位可贴身绕过);
建筑格与静态不可通行格仍是硬障碍;**移动单位不计入场代价**(见 movement_tick 内注释)。
v1.0 的抢格仲裁/候选格/方向序取反全部退役(无格可抢,方向连续天然对称)。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .map import BIG_DIST, MapData
from .state import (
    ORDER_ATTACK,
    ORDER_BUILD,
    ORDER_GARRISON,
    ORDER_HARVEST,
    ORDER_IDLE,
    ORDER_MOVE,
    PH_MINING,
    PH_TO_HQ,
    WorldState,
    cell_of,
)


def _relax_fields(goal_seeds: jax.Array, blocked: jax.Array,
                  cell_cost: jax.Array, k_iters: int) -> jax.Array:
    """min-plus 松弛距离场(与 v1.1 相同;blocked=硬障碍,cell_cost=进格代价)。"""
    big = jnp.int32(BIG_DIST)
    d = jnp.where(goal_seeds, 0, big)

    def body(d, _):
        pad = ((0, 0), (0, 0))
        up = jnp.pad(d[:, 1:, :], (pad[0], (0, 1), pad[1]), constant_values=big)
        dn = jnp.pad(d[:, :-1, :], (pad[0], (1, 0), pad[1]), constant_values=big)
        lf = jnp.pad(d[:, :, 1:], (pad[0], pad[1], (0, 1)), constant_values=big)
        rt = jnp.pad(d[:, :, :-1], (pad[0], pad[1], (1, 0)), constant_values=big)
        nd = jnp.minimum(jnp.minimum(up, dn), jnp.minimum(lf, rt)) + cell_cost[None]
        nd = jnp.minimum(d, nd)
        nd = jnp.where(blocked[None], big, nd)
        nd = jnp.where(goal_seeds, 0, nd)
        return nd, None

    d, _ = jax.lax.scan(body, d, None, length=k_iters)
    return d


def _bilinear(field: jax.Array, pos: jax.Array) -> jax.Array:
    """[H,W] 场在连续点集 [N,2] 上的双线性插值。pos 超界按边缘 clamp。"""
    h, w = field.shape
    r = jnp.clip(pos[:, 0], 0.0, h - 1.001)
    c = jnp.clip(pos[:, 1], 0.0, w - 1.001)
    r0 = jnp.floor(r).astype(jnp.int32)
    c0 = jnp.floor(c).astype(jnp.int32)
    fr = r - r0
    fc = c - c0
    f00 = field[r0, c0]
    f01 = field[r0, c0 + 1]
    f10 = field[r0 + 1, c0]
    f11 = field[r0 + 1, c0 + 1]
    return ((1 - fr) * (1 - fc) * f00 + (1 - fr) * fc * f01
            + fr * (1 - fc) * f10 + fr * fc * f11)


def building_cells(state: WorldState, cfg: Config) -> jax.Array:
    """bool [H,W]:在场建筑占用格(建筑=硬障碍)。
    统一判据:speed_by_type>0 即单位,=0 即建筑(新类型进表自动生效)。"""
    spd = jnp.asarray(cfg.speed_by_type, jnp.float32)[
        jnp.clip(state.etype.astype(jnp.int32), 0, 15)]
    b = state.alive & (spd == 0)
    cell = cell_of(state.pos)
    return (jnp.zeros((cfg.grid_h, cfg.grid_w), bool)
            .at[cell[:, 0], cell[:, 1]].max(b))


def movement_tick(state: WorldState, cfg: Config, mapdata: MapData,
                  owner: jax.Array, key: jax.Array) -> WorldState:
    del key  # 连续移动全程确定性,不再需要抢格随机优先级(保留参数稳住签名)
    h, w = cfg.grid_h, cfg.grid_w
    st = state
    passable = jnp.asarray(mapdata.passable)
    node_pos = jnp.asarray(mapdata.node_pos, jnp.float32)
    hq_pos = jnp.asarray(mapdata.hq_pos, jnp.float32)

    speed = jnp.asarray(cfg.speed_by_type, jnp.float32)[
        jnp.clip(st.etype.astype(jnp.int32), 0, 15)]
    is_unit = speed > 0
    on_board = st.alive & ~st.inside
    tn = jnp.clip(st.target_node.astype(jnp.int32), 0, cfg.n_nodes - 1)
    own_i = owner.astype(jnp.int32)

    # ---- 目标点与目标场 ----
    goal = jnp.where(st.order == ORDER_ATTACK, cfg.n_nodes + (1 - own_i), tn)
    goal = jnp.where((st.order == ORDER_HARVEST) & (st.phase == PH_TO_HQ),
                     cfg.n_nodes + own_i, goal)
    # 驻守(v1.3):锚点 0=己方 HQ 场,1..Nn=点 k 场(现有 Nn+2 张场直接覆盖);
    # 场只用来取方向——到达/步长/回岗一律对 target_cell 求欧氏距离
    # (goal_center 对越界 goal 会 clip 错场,critic B-1)
    gid = jnp.clip(st.garrison_id.astype(jnp.int32), 0, cfg.n_nodes)
    is_gar = st.order == ORDER_GARRISON
    goal = jnp.where(is_gar,
                     jnp.where(gid == 0, cfg.n_nodes + own_i, gid - 1), goal)
    use_field = ((st.order == ORDER_HARVEST) | (st.order == ORDER_BUILD)
                 | (st.order == ORDER_ATTACK) | is_gar)
    goal_center = jnp.where(
        (goal < cfg.n_nodes)[:, None], node_pos[jnp.clip(goal, 0, cfg.n_nodes - 1)],
        hq_pos[jnp.clip(goal - cfg.n_nodes, 0, 1)])

    # ---- 到达判定(欧氏)----
    eu_goal = jnp.linalg.norm(st.pos - goal_center, axis=-1)
    eu_move = jnp.linalg.norm(st.pos - st.target_cell, axis=-1)
    arrived = jnp.where(use_field, eu_goal <= cfg.reach_radius, eu_move <= 0.4)
    # 驻守:锚点圈内即「到达」;被互推挤出圈,下 tick 自动回岗
    arrived = jnp.where(is_gar, eu_move <= cfg.garrison_hold_radius, arrived)
    # attack-move:射程内有敌就地开打
    dmat = jnp.linalg.norm(st.pos[:, None, :] - st.pos[None, :, :], axis=-1)
    enemy_near = jnp.any(
        (dmat <= cfg.melee_range) & (owner[None, :] != owner[:, None])
        & (st.alive & ~st.inside)[None, :], axis=-1)
    arrived = arrived | ((st.order == ORDER_ATTACK) & enemy_near)

    moving_order = ((st.order == ORDER_MOVE) | (st.order == ORDER_ATTACK)
                    | (st.order == ORDER_BUILD) | is_gar
                    | ((st.order == ORDER_HARVEST) & (st.phase != PH_MINING)))
    wants = on_board & is_unit & moving_order & ~arrived

    # ---- 动态场(硬障碍:静态+建筑格;软障碍:静止/移动单位)----
    bcells = building_cells(st, cfg)
    cell = cell_of(st.pos)
    moving_unit = on_board & is_unit & moving_order & ~arrived
    stat_unit = on_board & is_unit & ~moving_unit
    occ_stat = (jnp.zeros((h, w), bool).at[cell[:, 0], cell[:, 1]].max(stat_unit))
    blocked = ~passable | bcells
    # 只有静止单位计入场代价。**移动单位不计**:它自己的罚分会落在自己脚下的
    # 格上,双线性梯度在原地采样时被这个尖峰支配,单位每 tick「逃离自己的影子」
    # ——实测退化成布朗游走,一趟采集 35→140 tick。v1.1 需要 congestion 分道,
    # 连续模式下对向流由圆形互推物理解决,不再需要场层面的分道。
    cell_cost = 1 + cfg.stationary_cost * occ_stat.astype(jnp.int32)
    dyn = _relax_fields(jnp.asarray(mapdata.goal_seeds), blocked, cell_cost, h + w)
    # 双线性采样前 clip 哨兵(critic S-3)
    dyn_c = jnp.minimum(dyn, 4 * (h + w)).astype(jnp.float32)

    # ---- 方向:场梯度(中心差分,步长 0.5)或直指;vmap 逐实体在各自场上采样 ----
    def sample_one(fidx, p):
        return _bilinear(dyn_c[fidx], p[None, :])[0]

    grad_r = (jax.vmap(sample_one)(goal, st.pos + jnp.asarray([0.5, 0.0]))
              - jax.vmap(sample_one)(goal, st.pos - jnp.asarray([0.5, 0.0])))
    grad_c = (jax.vmap(sample_one)(goal, st.pos + jnp.asarray([0.0, 0.5]))
              - jax.vmap(sample_one)(goal, st.pos - jnp.asarray([0.0, 0.5])))
    gvec = -jnp.stack([grad_r, grad_c], axis=-1)            # 下降方向
    gnorm = jnp.linalg.norm(gvec, axis=-1, keepdims=True)
    dir_field = jnp.where(gnorm > 1e-6, gvec / jnp.maximum(gnorm, 1e-6), 0.0)
    dvec = st.target_cell - st.pos
    dnorm = jnp.linalg.norm(dvec, axis=-1, keepdims=True)
    dir_move = jnp.where(dnorm > 1e-6, dvec / jnp.maximum(dnorm, 1e-6), 0.0)
    direction = jnp.where(use_field[:, None], dir_field, dir_move)

    # 步长截断:驻守对 target_cell(锚点)截,不吃 goal_center
    step_len = jnp.minimum(
        speed, jnp.where(use_field & ~is_gar, eu_goal, eu_move))
    delta = direction * (step_len * wants)[:, None]
    cand = st.pos + delta

    # ---- 建筑/边界:分轴滑动 ----
    def blocked_at(p):
        cc = cell_of(p)
        cc = jnp.clip(cc, 0, jnp.asarray([h - 1, w - 1]))
        inb = ((p[:, 0] >= -0.4) & (p[:, 0] <= h - 0.6)
               & (p[:, 1] >= -0.4) & (p[:, 1] <= w - 0.6))
        return blocked[cc[:, 0], cc[:, 1]] | ~inb

    cand_r = jnp.stack([cand[:, 0], st.pos[:, 1]], axis=-1)  # 仅动 r
    cand_c = jnp.stack([st.pos[:, 0], cand[:, 1]], axis=-1)  # 仅动 c
    ok_full = ~blocked_at(cand)
    ok_r = ~blocked_at(cand_r)
    ok_c = ~blocked_at(cand_c)
    new_pos = jnp.where(ok_full[:, None], cand,
                        jnp.where(ok_r[:, None], cand_r,
                                  jnp.where(ok_c[:, None], cand_c, st.pos)))

    # ---- 单位圆互推一轮(在场单位;含静止者被挤开)----
    active = on_board & is_unit
    diff = new_pos[:, None, :] - new_pos[None, :, :]        # [N,N,2] i-j
    dist = jnp.linalg.norm(diff, axis=-1)
    pair = (active[:, None] & active[None, :]
            & ~jnp.eye(cfg.n_total, dtype=bool))
    overlap = jnp.maximum(0.0, 2 * cfg.unit_radius - dist) * pair
    # 共线退化兜底:距离≈0 时用槽号定向的确定性单位向量;并给推力加按槽号
    # 奇偶定向的切向分量,破 180° 对称正对僵持(critic FYI-1)
    slots = jnp.arange(cfg.n_total)
    ang = slots.astype(jnp.float32) * 2.399963  # 黄金角,槽间方向彼此错开
    eps_vec = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)  # [N,2]
    near0 = dist < 1e-5
    safe_diff = jnp.where(near0[:, :, None],
                          eps_vec[:, None, :] - eps_vec[None, :, :], diff)
    safe_n = jnp.linalg.norm(safe_diff, axis=-1, keepdims=True)
    push_dir = safe_diff / jnp.maximum(safe_n, 1e-6)
    tangent = jnp.stack([-push_dir[:, :, 1], push_dir[:, :, 0]], axis=-1)
    parity = (1.0 - 2.0 * (slots % 2).astype(jnp.float32))  # +1/-1
    push = jnp.sum(
        (0.5 * overlap)[:, :, None]
        * (push_dir + 0.15 * tangent * parity[:, None, None]), axis=1)
    pushed = new_pos + push

    # ---- 硬约束殿后:互推挤进硬障碍格则回退到推前位置 ----
    final = jnp.where(blocked_at(pushed)[:, None], new_pos, pushed)
    final = jnp.where((on_board & is_unit)[:, None], final, st.pos)

    # MOVE 到达 → 转 IDLE
    l2_after = jnp.linalg.norm(final - st.target_cell, axis=-1)
    move_done = (st.order == ORDER_MOVE) & (l2_after <= 0.4)
    order = jnp.where(move_done, ORDER_IDLE, st.order).astype(jnp.int8)

    return st._replace(pos=final.astype(jnp.float32), order=order)
