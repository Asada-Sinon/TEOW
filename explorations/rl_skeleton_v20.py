"""v2.0 RL 骨架:能空跑一步、验证正确的 JAX 自研 PPO(**不做真正训练**)。

本文件是 issue.md `## v2.0` 的最终交付物之一:把 obs 编码 / 奖励 / 策略价值网 /
批量 rollout / GAE / PPO 损失 / 一次 update step 全部用纯 JAX 落地,证明「RL 机器」能
在**真实批量 rollout 数据**上跑**恰好一次**更新。规格逐字对应
`docs/plans/20260727-v2-rl-approach/research.md` §2/§3/§4/§5/§7/§8。

性质标注:通篇 `[AI-DRAFT]`——设计推断,未经用户核验、更未经训练验证。凡设计文档留白
(§8 开放问题)处,本骨架挑一个简单具体默认值并注 `# [AI-DRAFT] v2.1 定`。

边界(issue.md v2.0 明文):**本版不开始训练**。故 main() 只跑「rollout 一次 → 算损失
→ 恰好一次 optax/手写 Adam update」并断言形状/有限,**绝不写多步训练循环、不落 checkpoint**。

依赖:pyproject 只有 jax/numpy/chex(无 optax/flax/equinox),故 MLP 与 Adam 全手写,
自成一体、可 `JAX_PLATFORMS=cpu` 直接空跑。数值参数唯一走 RLConfig(python-research #2:
禁止字面量);随机种子显式设置(python-research #1)。派生量复用 stats.py(不建第二真源)。

跑法:
  JAX_PLATFORMS=cpu .venv/bin/python explorations/rl_skeleton_v20.py
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from teow.actions import legality_mask, n_actions  # noqa: E402
from teow.config import (  # noqa: E402
    N_TYPES,
    TYPE_CAMP,
    TYPE_HEALER,
    TYPE_HQ,
    TYPE_MINE,
    TYPE_PUMP,
    TYPE_STRONGMAN,
    TYPE_WAGON,
    TYPE_WORKER,
    Config,
)
from teow.controller import make_controller, merge_actions  # noqa: E402
from teow.map import build_map  # noqa: E402
from teow.state import (  # noqa: E402
    ORDER_ATTACK,
    ORDER_BUILD,
    ORDER_GARRISON,
    ORDER_HARVEST,
    ORDER_IDLE,
    ORDER_MOVE,
    WorldState,
    hq_slot,
    init_state,
    owner_of_slots,
)
from teow.stats import (  # noqa: E402
    atk_of,
    atk_table,
    etype_idx,
    max_hp_of,
    type_tables,
)
from teow.step import build_step  # noqa: E402

# 掩码非法 logit 的近似 -inf:保持梯度有限(huang2022ppodetails 的实践;设计 §7.2
# 验收判据②要求「非法位 = -inf 且 NOOP 恒有限」,-1e9 在 softmax/熵里等效且不产 NaN)。
NEG_INF = -1e9


# =====================================================================================
# 1) RLConfig —— 所有超参走它(huang2022ppodetails 的「实现细节」做成显式开关)
# =====================================================================================
@dataclasses.dataclass(frozen=True)
class RLConfig:
    """PPO/obs/net 的唯一超参真源(禁止代码里出现数值字面量)。字段分四组:
    obs 维度与归一化 / 网络宽度 / PPO 旋钮 / rollout 规模。"""

    seed: int = 0

    # ---- obs 归一化尺度(§2;派生自 cfg 的量在 _derived_norms 里算,这里只放纯 RL 侧) ----
    res_scale: float = 500.0     # 资源 tanh 归一尺度 [AI-DRAFT] v2.1 定
    pot_scale: float = 300.0     # 势函数 Φ 的价值差 tanh 尺度 [AI-DRAFT] v2.1 定
    mon_hp_scale: float = 2000.0  # 怪总血 tanh 尺度 [AI-DRAFT] v2.1 定
    norm_obs: bool = False       # obs 逐特征白化开关(骨架先关;§7.2 旋钮,v2.1 定)

    # ---- 网络宽度(tiny;§2.4 方案 A + 全局上下文注入,§5 IPPO 单价值) ----
    type_embed_dim: int = 8      # etype 学习 embedding 维(§2.2:优于 29 维 one-hot)
    hidden: int = 64             # 逐实体 trunk / head 宽度 H(§2.4「H≈64」)
    global_hidden: int = 32      # 全局向量编码宽度 Hg
    orthogonal_init: bool = True  # 正交初始化 + tanh(huang2022ppodetails)

    # ---- PPO 旋钮(§7.2 不踩坑清单,全做成显式开关) ----
    clip_eps: float = 0.2        # surrogate clip ε
    gamma: float = 0.999         # 折扣(长时程需高;数千 tick,§7.2)
    gae_lambda: float = 0.95     # GAE λ
    vf_coef: float = 0.5         # 价值损失权重
    ent_coef: float = 0.01       # 熵奖励(掩码动作空间下按合法集算熵)
    lr: float = 3e-4             # 学习率
    max_grad_norm: float = 0.5   # 全局梯度裁剪
    norm_adv: bool = True        # 优势归一化(minibatch 级)
    norm_reward: bool = False    # reward 缩放(骨架先关;§7.2 旋钮)
    clip_vloss: bool = True      # value function clipping(huang2022)
    anneal_lr: bool = True       # 学习率线性退火(单步 smoke 里 frac≈1,占位实现)
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-5       # huang2022 用 1e-5(非默认 1e-8)

    # ---- 奖励(§4;shaping 系数退火由外部传标量) ----
    shaping_beta: float = 0.1    # PBRS 系数 β 初值(退火到 0;§4.2)[AI-DRAFT] v2.1 定
    anneal_shaping: bool = True  # β 退火开关(§4.2:有限预算下 Φ 不完美需退火)

    # ---- rollout 规模(§7.2;B≈64 是引擎甜点 [source: 20260726-v18-bench],
    #      但 smoke 只验 vmap/形状,不验吞吐,故取极小 B/T 保证编译+跑得快) ----
    num_envs: int = 8            # B:并行游戏数(smoke 值;训练甜点 ≈64)
    num_steps: int = 128         # T:每条 rollout 的 tick 数(smoke 值;短)
    update_epochs: int = 4       # on-policy 数据复用轮数(smoke 只跑 1 次,此为文档旋钮)
    num_minibatches: int = 4     # minibatch 数(同上,smoke 用整批)


# =====================================================================================
# 2) obs 编码 —— WorldState → (逐实体 token[N,F], alive 掩码, 全局向量[G], 动作掩码[N,A])
#    以行动方 player 为中心(egocentric);派生量复用 stats.py(§2)
# =====================================================================================
class Obs(NamedTuple):
    """定形 pytree(vmap 安全:所有字段静态 shape)。tokens/global 是策略输入,
    action_mask 用于掩非法 logit,loss_mask 圈定进 PPO 损失的实体(§3:只算己方
    alive & actable),entity_mask=alive 用于池化。"""

    tokens: jax.Array        # f32 [N, F]
    etype_idx: jax.Array     # int32 [N]   交给网络查 type embedding
    entity_mask: jax.Array   # bool [N]    = alive(池化用)
    loss_mask: jax.Array     # bool [N]    己方 alive & 可行动(logp/熵只算这些)
    global_vec: jax.Array    # f32 [G]
    action_mask: jax.Array   # bool [N, A]


def _derived_norms(cfg: Config) -> dict:
    """从 cfg 派生的 obs 归一化尺度(host 端算一次,作为常量闭包进 jit)。
    复用 stats.py 的表(atk_table 等),不另建真源。"""
    atk_max = float(np.asarray(atk_table(cfg)).max())
    rng_max = float(max(cfg.atk_range_by_type))
    carry_max = float(max(cfg.carry_by_type))
    time_max = float(max(cfg.train_time_by_type))
    return dict(
        atk=max(atk_max, 1.0),
        rng=max(rng_max, 1.0),
        carry=max(carry_max, 1.0),
        time=max(time_max, 1.0),
        base_max=float(cfg.base_max_level),
        h=float(cfg.grid_h),
        w=float(cfg.grid_w),
        mon_cap=float(cfg.monster_cap),
    )


def type_cost_table(cfg: Config) -> jax.Array:
    """f32 [N_TYPES]:每类型「投入资源价值」= ore+water 造价(§4.2 统一货币)。
    单位取 train_cost 表(cfg 真源),建筑逐类补造价字段;HQ=0(非投入,其价值由
    终局名次奖励承载,避免稀疏信号泄进 shaping)。"""
    ore = np.asarray(cfg.train_cost_ore_by_type, np.float32)
    wat = np.asarray(cfg.train_cost_water_by_type, np.float32)
    cost = ore + wat  # 覆盖全部单位;建筑在下方补
    bld = {
        TYPE_MINE: (cfg.mine_cost_ore, cfg.mine_cost_water),
        TYPE_PUMP: (cfg.pump_cost_ore, cfg.pump_cost_water),
        TYPE_CAMP: (cfg.camp_cost_ore, cfg.camp_cost_water),
    }
    # 兵营/塔/迫击炮/栅栏/v1.6 防御建筑造价逐类补(全部 cfg 字段,无字面量)
    from teow.config import (
        TYPE_BARRACKS,
        TYPE_FENCE_IRON,
        TYPE_FENCE_STONE,
        TYPE_FENCE_WOOD,
        TYPE_FLAMER,
        TYPE_LANDMINE,
        TYPE_LASER,
        TYPE_MAGETOWER,
        TYPE_MORTAR,
        TYPE_TOWER,
    )
    bld.update({
        TYPE_BARRACKS: (cfg.barracks_cost_ore, cfg.barracks_cost_water),
        TYPE_TOWER: (cfg.tower_cost_ore, cfg.tower_cost_water),
        TYPE_MORTAR: (cfg.mortar_cost_ore, cfg.mortar_cost_water),
        TYPE_FENCE_WOOD: (cfg.fence_wood_cost_ore, cfg.fence_wood_cost_water),
        TYPE_FENCE_STONE: (cfg.fence_stone_cost_ore, cfg.fence_stone_cost_water),
        TYPE_FENCE_IRON: (cfg.fence_iron_cost_ore, cfg.fence_iron_cost_water),
        TYPE_MAGETOWER: (cfg.magetower_cost_ore, cfg.magetower_cost_water),
        TYPE_LANDMINE: (cfg.landmine_cost_ore, cfg.landmine_cost_water),
        TYPE_FLAMER: (cfg.flamer_cost_ore, cfg.flamer_cost_water),
        TYPE_LASER: (cfg.laser_cost_ore, cfg.laser_cost_water),
    })
    for t, (o, w) in bld.items():
        cost[t] = float(o) + float(w)
    return jnp.asarray(cost, jnp.float32)


def encode_obs(state: WorldState, cfg: Config, mapdata, player: int,
               norms: dict, cost_tbl: jax.Array) -> Obs:
    """§2 落地。egocentric to `player`。全部定形、无 data-dependent shape(§7.2 判据①)。"""
    owner = owner_of_slots(cfg)                       # int8 [N]
    p = player
    P, e = cfg.n_players, cfg.e_max
    n = cfg.n_total
    et = etype_idx(state)                             # int32 [N] clip 安全下标
    tt = type_tables(cfg)
    own = owner == p
    alive = state.alive
    is_own = own & alive
    is_enemy = (~own) & alive
    is_harv = ((state.etype == TYPE_WORKER) | (state.etype == TYPE_STRONGMAN)
               | (state.etype == TYPE_WAGON))
    is_combat = tt["combat"][et]
    is_air = tt["air"][et]
    is_building = (tt["speed"][et] == 0) & alive
    is_hq = state.etype == TYPE_HQ
    is_res_struct = (state.etype == TYPE_MINE) | (state.etype == TYPE_PUMP)
    is_healer = state.etype == TYPE_HEALER

    mhp = jnp.maximum(max_hp_of(state, cfg, owner), 1)
    hp_frac = jnp.clip(state.hp / mhp, 0.0, 1.0)
    atk = atk_of(state, cfg, owner).astype(jnp.float32)
    rng = tt["rng"][et]
    armor = tt["armor"][et].astype(jnp.float32)

    hqpos = jnp.asarray(mapdata.hq_pos, jnp.float32)[p]   # [2]
    rel = (state.pos - hqpos) / norms["h"]                # 相对己方 HQ(§2.2 egocentric)
    absr = state.pos / jnp.asarray([norms["h"], norms["w"]])
    dist_hq = jnp.linalg.norm(state.pos - hqpos, axis=-1) / norms["h"]

    def f32(x):
        return x.astype(jnp.float32)

    order = state.order
    cols = [
        f32(alive),
        f32(is_own), f32(is_enemy),
        jnp.zeros(n, jnp.float32),          # is_monster 保留位(=0;§2.2 起步摘要进 global,
        #                                     v1 无怪 token;v2.1 拼 Mmax 条怪 token 时置 1)
        # [AI-DRAFT] v2.1 定:怪物 token 是否入 entity_tokens(§8-1)
        f32(is_harv), f32(is_combat), f32(is_building), f32(is_air), f32(is_hq),
        et.astype(jnp.float32) / N_TYPES,   # 原始类型 id 归一(辅 embedding)
        rel[:, 0], rel[:, 1], absr[:, 0], absr[:, 1], dist_hq,
        hp_frac,
        f32(state.level) / norms["base_max"],
        f32(order == ORDER_IDLE), f32(order == ORDER_HARVEST),
        f32(order == ORDER_BUILD), f32(order == ORDER_MOVE),
        f32(order == ORDER_ATTACK), f32(order == ORDER_GARRISON),
        f32(state.inside), f32(state.aboard >= 0), f32(state.atk_cd > 0),
        f32(state.cargo) / norms["carry"],
        f32(state.phase) / 2.0,
        f32(state.btimer > 0), f32(state.btimer) / norms["time"],
        atk / norms["atk"], rng / norms["rng"], armor / 100.0,
        f32(is_res_struct), f32(is_healer),
        f32(state.mine_timer > 0),
    ]
    tokens = jnp.stack(cols, axis=-1)                 # [N, F]
    tokens = tokens * f32(alive)[:, None]             # 死槽整条置 0(掩码兜底,保持干净)

    # ---- 全局向量 G(§2.3;egocentric:各 player 维度按 own-first 重排) ----
    order_p = (jnp.arange(P) + p) % P                 # [p, p+1, ..., p-1]
    res = state.resources.astype(jnp.float32)         # [P,2]
    res_feat = jnp.tanh(res[order_p] / norms["res"]).reshape(-1)   # [2P]
    up = f32(state.upgrades[p]) / norms["base_max"]   # [N_LINES]
    base_lv = f32(state.level[hq_slot(p, cfg)]) / norms["base_max"]
    tickf = f32(state.tick) / cfg.episode_len
    gate_open = f32(state.tick >= cfg.gate_open_tick)
    overtime = jnp.clip(
        (f32(state.tick) - cfg.gate_open_tick)
        / (cfg.episode_len - cfg.gate_open_tick), 0.0, 1.0)
    # 己方怪压力摘要(§2.2:overtime 前怪为空,摘要足够)
    m_alive = state.monster_alive[p]                  # [Mmax]
    m_n = f32(jnp.sum(m_alive)) / norms["mon_cap"]
    m_hp = jnp.tanh(jnp.sum(f32(state.monster_hp[p]) * m_alive) / norms["mon_hp"])
    m_d = jnp.linalg.norm(state.monster_pos[p] - hqpos, axis=-1)
    m_d = jnp.where(m_alive, m_d, jnp.inf)
    m_near = jnp.clip(jnp.min(m_d) / norms["h"], 0.0, 1.0)   # 无怪 → min(inf)→clip 1
    hq_alive_P = jnp.stack([state.alive[hq_slot(q, cfg)] for q in range(P)])
    hq_alive_feat = f32(hq_alive_P[order_p])          # [P] 谁还在局里(FFA 选敌/名次)
    alive_by_p = f32(alive.reshape(P, e).sum(1))[order_p] / e     # [P]
    combat_by_p = f32((alive & is_combat).reshape(P, e).sum(1))[order_p] / e  # [P]
    own_nodes = f32(jnp.sum(state.node_owner == p)) / cfg.n_nodes
    val = _invested_value(state, cfg, owner, cost_tbl)    # [P]
    val_feat = jnp.tanh(val[order_p] / norms["pot"])      # [P]
    global_vec = jnp.concatenate([
        res_feat, up, jnp.asarray([base_lv, tickf, gate_open, overtime,
                                   m_n, m_hp, m_near]),
        hq_alive_feat, alive_by_p, combat_by_p,
        jnp.asarray([own_nodes]), val_feat,
    ])                                                # [G]

    action_mask = legality_mask(state, cfg, mapdata, owner)   # bool [N, A]
    actable = alive & ~state.inside & (state.aboard < 0)
    loss_mask = is_own & actable                      # §3:只对己方可行动实体计损失
    return Obs(tokens=tokens, etype_idx=et, entity_mask=alive,
               loss_mask=loss_mask, global_vec=global_vec, action_mask=action_mask)


# =====================================================================================
# 3) 奖励 —— 稀疏 FFA 名次终局 + 退火 PBRS 势函数塑形(§4)
# =====================================================================================
def _invested_value(state: WorldState, cfg: Config, owner: jax.Array,
                    cost_tbl: jax.Array) -> jax.Array:
    """f32 [P]:各家投入价值 = 库存(ore+water) + Σ 存活实体 cost·(hp/max_hp)(§4.2)。"""
    P, e = cfg.n_players, cfg.e_max
    et = etype_idx(state)
    mhp = jnp.maximum(max_hp_of(state, cfg, owner), 1)
    hp_frac = jnp.clip(state.hp / mhp, 0.0, 1.0)
    val_i = cost_tbl[et] * hp_frac * state.alive.astype(jnp.float32)
    ent_val = val_i.reshape(P, e).sum(1)              # [P] 折血存量投入
    stock = state.resources.sum(1).astype(jnp.float32)   # [P] 库存 ore+water
    return ent_val + stock


def potential(state: WorldState, cfg: Config, owner: jax.Array,
              cost_tbl: jax.Array, pot_scale: float) -> jax.Array:
    """f32 [P]:Φ_p = tanh((Value_p − max_{q≠p} Value_q)/scale)(§4.2)。
    aggregate 取 max(压制当前领先者,FFA 合理;§4.2 倾向 max,确切形式是 §8-3 开放问题)。
    # [AI-DRAFT] v2.1 定:aggregate = max / mean / 逐对手(§8-3)。"""
    P = cfg.n_players
    val = _invested_value(state, cfg, owner, cost_tbl)   # [P]
    eye = jnp.eye(P, dtype=bool)
    others = jnp.where(eye, -jnp.inf, val[None, :])      # [P,P] 屏蔽自己
    opp_max = jnp.max(others, axis=1)                    # [P] 最强对手
    return jnp.tanh((val - opp_max) / pot_scale)


def terminal_rank_reward(elim_tick: jax.Array, cfg: Config) -> jax.Array:
    """f32 [P]:FFA 名次终局奖励 r = 1 − 2·(rank−1)/(P−1)(§4.1)。
    名次由淘汰 tick 定:活得越久名次越高(胜者永不淘汰=elim_tick 最大)。
    平手按 player id 破(argmax 式确定性,保证 rank 是 1..P 的排列)。"""
    P = cfg.n_players
    et = elim_tick.astype(jnp.int32)
    idx = jnp.arange(P)
    better = ((et[None, :] > et[:, None])                       # q 活得更久
              | ((et[None, :] == et[:, None]) & (idx[None, :] < idx[:, None])))
    rank = 1 + jnp.sum(better, axis=1)                          # [P] ∈ 1..P
    return 1.0 - 2.0 * (rank.astype(jnp.float32) - 1.0) / (P - 1)


# =====================================================================================
# 4) 共享参数 actor-critic 小网(§2.4 方案 A + 全局上下文注入;§5 IPPO 单价值)
#    纯函数 + 显式 params pytree(无 flax/equinox);正交初始化 + tanh(huang2022)
# =====================================================================================
def init_params(f_dim: int, g_dim: int, a_dim: int, rlcfg: RLConfig,
                key: jax.Array) -> dict:
    """按 obs 实测维度 (F,G,A) 建 tiny actor-critic 参数(不 hardcode 任何维度)。"""
    d, h, hg = rlcfg.type_embed_dim, rlcfg.hidden, rlcfg.global_hidden
    ks = jax.random.split(key, 9)
    s2 = math.sqrt(2.0)

    def w(k, shape, scale):
        if rlcfg.orthogonal_init:
            return jax.nn.initializers.orthogonal(scale)(k, shape)
        return jax.random.normal(k, shape) * 0.1

    fin = f_dim + d                                   # trunk 输入 = token ⊕ type embedding
    a_in = h + h + hg                                 # actor 输入 = hidden ⊕ (pooled ⊕ global)
    c_in = h + hg                                     # critic 输入 = pooled ⊕ global
    return dict(
        type_embed=jax.random.normal(ks[0], (N_TYPES, d)) * 0.1,
        W1=w(ks[1], (fin, h), s2), b1=jnp.zeros(h),
        W2=w(ks[2], (h, h), s2), b2=jnp.zeros(h),
        Wg=w(ks[3], (g_dim, hg), s2), bg=jnp.zeros(hg),
        Wa1=w(ks[4], (a_in, h), s2), ba1=jnp.zeros(h),
        Wa2=w(ks[5], (h, a_dim), 0.01), ba2=jnp.zeros(a_dim),   # 策略输出层小尺度(huang2022)
        Wc1=w(ks[6], (c_in, h), s2), bc1=jnp.zeros(h),
        Wc2=w(ks[7], (h, 1), 1.0), bc2=jnp.zeros(1),            # 价值输出层尺度 1
    )


def actor_critic(params: dict, obs: Obs) -> tuple[jax.Array, jax.Array]:
    """单条 obs → (logits[N,A] 已掩非法, value 标量)。外部用 vmap 批量。"""
    te = params["type_embed"][obs.etype_idx]          # [N, d]
    x = jnp.concatenate([obs.tokens, te], axis=-1)    # [N, F+d]
    hh = jnp.tanh(x @ params["W1"] + params["b1"])
    hh = jnp.tanh(hh @ params["W2"] + params["b2"])   # [N, H] 逐实体隐藏
    mask = obs.entity_mask.astype(jnp.float32)[:, None]
    pooled = jnp.sum(hh * mask, axis=0) / jnp.maximum(jnp.sum(mask), 1.0)  # [H] 态势
    g = jnp.tanh(obs.global_vec @ params["Wg"] + params["bg"])            # [Hg]
    ctx = jnp.concatenate([pooled, g])                # [H+Hg] 全局上下文
    ctx_b = jnp.broadcast_to(ctx, (hh.shape[0], ctx.shape[0]))
    a = jnp.concatenate([hh, ctx_b], axis=-1)         # [N, H+H+Hg]
    a = jnp.tanh(a @ params["Wa1"] + params["ba1"])
    logits = a @ params["Wa2"] + params["ba2"]        # [N, A]
    logits = jnp.where(obs.action_mask, logits, NEG_INF)  # 掩非法(§3 invalid-action masking)
    c = jnp.tanh(ctx @ params["Wc1"] + params["bc1"])
    value = (c @ params["Wc2"] + params["bc2"])[0]    # 标量 V_p
    return logits, value


def _joint_logp_entropy(logits: jax.Array, action: jax.Array,
                        loss_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """逐实体因子化策略:联合 logp = Σ_{可行动己方实体} logp_i(action_i),熵同理(§3 损失掩码)。"""
    logsm = jax.nn.log_softmax(logits, axis=-1)       # [N, A]
    lp = jnp.take_along_axis(logsm, action[:, None], axis=-1)[:, 0]   # [N]
    m = loss_mask.astype(jnp.float32)
    joint_logp = jnp.sum(lp * m)
    probs = jnp.exp(logsm)
    ent_i = -jnp.sum(probs * logsm, axis=-1)          # [N] 掩码动作集的熵
    joint_ent = jnp.sum(ent_i * m)
    return joint_logp, joint_ent


# =====================================================================================
# 5) 批量 rollout —— 扩 eval.matchup_runner:vmap B 局 × lax.scan T tick,收集轨迹(§7.1-5)
# =====================================================================================
def collect_rollout(cfg: Config, mapdata, params: dict, rlcfg: RLConfig,
                    key: jax.Array, opp_names: tuple[str, ...]):
    """vmap B 局独立游戏,每局 lax.scan T tick:policy 占 player 0,脚本占其余 3 座。
    返回 (traj, last_value):traj 各字段 [B, T, ...];last_value [B] 供 GAE bootstrap。"""
    owner = owner_of_slots(cfg)
    step_fn = build_step(cfg, mapdata)
    P = cfg.n_players
    player = 0                                         # [AI-DRAFT] policy 固定占 0 座(§6.1 起步)
    if len(opp_names) != P - 1:
        raise ValueError(f"需要 {P - 1} 个对手脚本名,收到 {len(opp_names)}")
    opp_ctrls = [make_controller(nm, q, cfg, mapdata)
                 for q, nm in zip(range(1, P), opp_names, strict=True)]
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = (
        rlcfg.res_scale, rlcfg.pot_scale, rlcfg.mon_hp_scale)
    cost_tbl = type_cost_table(cfg)
    state0 = init_state(cfg, mapdata)
    big_t = jnp.int32(cfg.episode_len + 1)
    beta = rlcfg.shaping_beta                          # 退火系数:骨架传初值(§4.2 退火由外部标量)

    def one(k0):
        elim0 = jnp.full(P, big_t, jnp.int32)

        def body(carry, _):
            state, key, elim = carry
            key, k_pol, k_opp, k_step = jax.random.split(key, 4)
            obs = encode_obs(state, cfg, mapdata, player, norms, cost_tbl)
            logits, value = actor_critic(params, obs)
            # 逐实体从掩码 logits 采样(线程 key;非法位 -1e9 → 永不采到非法)
            action = jax.random.categorical(k_pol, logits, axis=-1).astype(jnp.int32)
            logp, _ = _joint_logp_entropy(logits, action, obs.loss_mask)
            # 对手脚本出招,按 owner 合并成 joint(policy 只贡献 player 0 的槽)
            opp_keys = jax.random.split(k_opp, P - 1)
            opp_acts = [c(state, key=kk) for c, kk in zip(opp_ctrls, opp_keys, strict=True)]
            joint = merge_actions(owner, [action] + opp_acts)
            nstate = step_fn(state, joint, k_step)

            # 奖励:β·(γ·Φ(s')−Φ(s)) 逐 tick(终局冻结后置 0)+ 终局名次(仅 done 跃迁)
            prev_done = state.done
            phi_s = potential(state, cfg, owner, cost_tbl, rlcfg.pot_scale)
            phi_n = potential(nstate, cfg, owner, cost_tbl, rlcfg.pot_scale)
            shaping = beta * (rlcfg.gamma * phi_n - phi_s) * (~prev_done)
            hq_alive_now = jnp.stack([nstate.alive[hq_slot(q, cfg)] for q in range(P)])
            elim = jnp.minimum(elim, jnp.where(hq_alive_now, big_t, nstate.tick))
            done_trans = nstate.done & (~prev_done)
            r_all = shaping + terminal_rank_reward(elim, cfg) * done_trans
            reward = r_all[player]

            out = dict(
                tokens=obs.tokens, etype_idx=obs.etype_idx,
                entity_mask=obs.entity_mask, loss_mask=obs.loss_mask,
                global_vec=obs.global_vec, action_mask=obs.action_mask,
                action=action, logp=logp, value=value,
                reward=reward, done=nstate.done)
            return (nstate, key, elim), out

        (final_state, _, _), traj = jax.lax.scan(
            body, (state0, k0, elim0), None, length=rlcfg.num_steps)
        obs_T = encode_obs(final_state, cfg, mapdata, player, norms, cost_tbl)
        _, last_value = actor_critic(params, obs_T)
        return traj, last_value

    subkeys = jax.random.split(key, rlcfg.num_envs)    # B 条独立游戏
    traj, last_value = jax.vmap(one)(subkeys)          # 各字段 [B, T, ...]
    return traj, last_value


# =====================================================================================
# 6) GAE(λ) + PPO 损失 + 一次 update step(§7.1-6)
# =====================================================================================
def compute_gae(reward, value, done, last_value, gamma, lam):
    """单条 [T] 轨迹的 GAE-λ;返回 (advantages[T], returns[T])。外部 vmap over B。"""
    def body(carry, x):
        gae, next_value = carry
        rew, val, dn = x
        delta = rew + gamma * next_value * (1.0 - dn) - val
        gae = delta + gamma * lam * (1.0 - dn) * gae
        return (gae, val), gae

    _, adv = jax.lax.scan(
        body, (jnp.float32(0.0), last_value),
        (reward, value, done.astype(jnp.float32)), reverse=True)
    return adv, adv + value


def ppo_loss(params, batch, rlcfg: RLConfig):
    """clipped surrogate + (clipped) value loss + entropy bonus(huang2022ppodetails)。
    batch 各字段 [M, ...](M = B*T 展平);返回 (loss, aux)。"""
    obs = Obs(tokens=batch["tokens"], etype_idx=batch["etype_idx"],
              entity_mask=batch["entity_mask"], loss_mask=batch["loss_mask"],
              global_vec=batch["global_vec"], action_mask=batch["action_mask"])
    logits, value = jax.vmap(lambda o: actor_critic(params, o))(obs)   # [M,N,A], [M]
    new_logp, ent = jax.vmap(_joint_logp_entropy)(
        logits, batch["action"], batch["loss_mask"])                   # [M], [M]

    adv = batch["adv"]
    if rlcfg.norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = jnp.exp(new_logp - batch["logp"])
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - rlcfg.clip_eps, 1.0 + rlcfg.clip_eps)
    pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

    ret, old_v = batch["ret"], batch["value"]
    v_unclipped = (value - ret) ** 2
    if rlcfg.clip_vloss:
        v_clipped = old_v + jnp.clip(value - old_v, -rlcfg.clip_eps, rlcfg.clip_eps)
        v_loss = 0.5 * jnp.mean(jnp.maximum(v_unclipped, (v_clipped - ret) ** 2))
    else:
        v_loss = 0.5 * jnp.mean(v_unclipped)

    ent_loss = jnp.mean(ent)
    loss = pg_loss + rlcfg.vf_coef * v_loss - rlcfg.ent_coef * ent_loss
    approx_kl = jnp.mean(batch["logp"] - new_logp)
    aux = dict(pg=pg_loss, vf=v_loss, ent=ent_loss, approx_kl=approx_kl,
               ratio_mean=jnp.mean(ratio))
    return loss, aux


# ---- 手写 Adam(无 optax;huang2022 的 eps=1e-5)----
def adam_init(params):
    z = jax.tree.map(jnp.zeros_like, params)
    return dict(m=z, v=jax.tree.map(jnp.zeros_like, params), t=0)


def _global_norm(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def adam_step(params, grads, opt, rlcfg: RLConfig, lr: float):
    """全局梯度裁剪 + Adam 一步。返回 (params, opt, grad_norm)。"""
    gnorm = _global_norm(grads)
    scale = jnp.minimum(1.0, rlcfg.max_grad_norm / (gnorm + 1e-8))
    grads = jax.tree.map(lambda g: g * scale, grads)
    t = opt["t"] + 1
    b1, b2, eps = rlcfg.adam_b1, rlcfg.adam_b2, rlcfg.adam_eps
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
    mc = 1 - b1 ** t
    vc = 1 - b2 ** t
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / mc) / (jnp.sqrt(v / vc) + eps),
        params, m, v)
    return params, dict(m=m, v=v, t=t), gnorm


# =====================================================================================
# 7) main() / smoke —— rollout 一次 → GAE → PPO 损失 → **恰好一次** update；断言 + 打印
#    绝不写训练循环(issue.md v2.0「不开始训练」)
# =====================================================================================
def _flatten_time_batch(x):
    """[B, T, ...] → [B*T, ...](展平成 minibatch;骨架用整批做「一个 minibatch」)。"""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def main():
    print("=" * 78)
    print("TEOW v2.0 RL 骨架 smoke:空跑一步 PPO(不训练)")
    print("backend:", jax.default_backend())
    rlcfg = RLConfig()
    cfg = Config()
    mapdata = build_map(cfg)
    key = jax.random.PRNGKey(rlcfg.seed)
    k_init, k_roll = jax.random.split(key)

    A = n_actions(cfg)                                 # 116(default);永远用函数,不 hardcode
    # 用 init_state 上跑一次 encode_obs 测 F/G(不 hardcode 维度)
    s0 = init_state(cfg, mapdata)
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = (
        rlcfg.res_scale, rlcfg.pot_scale, rlcfg.mon_hp_scale)
    cost_tbl = type_cost_table(cfg)
    obs0 = encode_obs(s0, cfg, mapdata, 0, norms, cost_tbl)
    F, G = obs0.tokens.shape[1], obs0.global_vec.shape[0]
    N = cfg.n_total
    print(f"维度: N={N}  A={A}  F(entity token)={F}  G(global)={G}  "
          f"P={cfg.n_players}  Nn={cfg.n_nodes}  Mmax={cfg.monster_cap}")
    assert obs0.action_mask.shape == (N, A)
    assert bool(obs0.action_mask[:, 0].all()), "NOOP(id 0)必须恒合法"
    assert jnp.isfinite(obs0.tokens).all() and jnp.isfinite(obs0.global_vec).all()

    params = init_params(F, G, A, rlcfg, k_init)
    n_param = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    print(f"参数量: {n_param:,}  (tiny;H={rlcfg.hidden}, "
          f"type_embed={rlcfg.type_embed_dim}, Hg={rlcfg.global_hidden})")

    # ---- rollout 一次(真实批量引擎数据)----
    opp = ("scripted", "scripted", "scripted")         # 其余 3 座固定脚本(§6.1 起步)
    print(f"\nrollout: B={rlcfg.num_envs} 局 × T={rlcfg.num_steps} tick,"
          f"policy@座0 vs {opp}  (编译中,CPU 慢…)")
    traj, last_value = collect_rollout(cfg, mapdata, params, rlcfg, k_roll, opp)
    jax.block_until_ready(last_value)

    B, T = rlcfg.num_envs, rlcfg.num_steps
    shapes = {k: tuple(v.shape) for k, v in traj.items()}
    print("轨迹张量形状:")
    for kk in ("tokens", "global_vec", "action_mask", "action",
               "logp", "value", "reward", "done"):
        print(f"  {kk:12s} {shapes[kk]}")
    assert shapes["tokens"] == (B, T, N, F)
    assert shapes["action_mask"] == (B, T, N, A)
    assert shapes["action"] == (B, T, N)
    assert shapes["reward"] == (B, T) and shapes["value"] == (B, T)
    assert last_value.shape == (B,)
    for kk, v in traj.items():
        assert bool(jnp.isfinite(v).all()) if v.dtype != jnp.bool_ else True, \
            f"{kk} 有非有限值"
    print(f"  reward: mean={float(traj['reward'].mean()):+.4e}  "
          f"std={float(traj['reward'].std()):.4e}  "
          f"done 触发数={int(traj['done'].sum())}(T={T}<gate={cfg.gate_open_tick},"
          f"通常不终局,终局名次奖励路径见下方独立自检)")

    # ---- GAE(λ) over B ----
    adv, ret = jax.vmap(compute_gae, in_axes=(0, 0, 0, 0, None, None))(
        traj["reward"], traj["value"], traj["done"], last_value,
        rlcfg.gamma, rlcfg.gae_lambda)
    assert adv.shape == (B, T)
    assert bool(jnp.isfinite(adv).all()) and bool(jnp.isfinite(ret).all())
    print(f"GAE: adv mean={float(adv.mean()):+.4e} std={float(adv.std()):.4e}  "
          f"ret mean={float(ret.mean()):+.4e}")

    # ---- 展平成一个 minibatch,算 PPO 损失 + 恰好一次 Adam update ----
    batch = dict(
        tokens=_flatten_time_batch(traj["tokens"]),
        etype_idx=_flatten_time_batch(traj["etype_idx"]),
        entity_mask=_flatten_time_batch(traj["entity_mask"]),
        loss_mask=_flatten_time_batch(traj["loss_mask"]),
        global_vec=_flatten_time_batch(traj["global_vec"]),
        action_mask=_flatten_time_batch(traj["action_mask"]),
        action=_flatten_time_batch(traj["action"]),
        logp=_flatten_time_batch(traj["logp"]),
        value=_flatten_time_batch(traj["value"]),
        adv=_flatten_time_batch(adv),
        ret=_flatten_time_batch(ret),
    )
    (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
        params, batch, rlcfg)
    lr = rlcfg.lr * (1.0 if rlcfg.anneal_lr else 1.0)   # 单步:frac=1(退火占位)
    opt = adam_init(params)
    new_params, opt, gnorm = adam_step(params, grads, opt, rlcfg, lr)
    jax.block_until_ready(jax.tree.leaves(new_params))

    print("\nPPO 损失(一次 minibatch,M =", B * T, "):")
    print(f"  total     = {float(loss):+.6f}")
    print(f"  pg_loss   = {float(aux['pg']):+.6f}")
    print(f"  vf_loss   = {float(aux['vf']):+.6f}")
    print(f"  entropy   = {float(aux['ent']):.6f}")
    print(f"  approx_kl = {float(aux['approx_kl']):+.6f}  "
          f"ratio_mean = {float(aux['ratio_mean']):.6f}")
    print(f"  grad_norm(裁剪前) = {float(gnorm):.6f}")

    # ---- 断言:损失/梯度有限、参数确有更新、形状不变(§7.2 验收判据④⑤)----
    assert jnp.isfinite(loss), "loss 非有限"
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads)), "梯度非有限"
    assert bool(jnp.isfinite(gnorm)), "grad_norm 非有限"
    delta = _global_norm(jax.tree.map(lambda a, b: a - b, new_params, params))
    assert float(delta) > 0.0, "一次 update 后参数未变化"
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(new_params), strict=True):
        assert a.shape == b.shape, "update 后参数形状改变"
    print(f"  参数更新量 |Δθ| = {float(delta):.6e}  (>0:确实更新;形状不变 ✓)")

    # ---- 独立自检:终局名次奖励路径(rollout 太短通常不终局,单独喂合成淘汰序验证)----
    #  假设淘汰 tick:座2 最早亡、座1 次之、座3 再次、座0 从未亡(胜)
    synth_elim = jnp.asarray([cfg.episode_len + 1, 200, 100, 300], jnp.int32)
    tr = terminal_rank_reward(synth_elim, cfg)
    tr_r = [round(float(x), 3) for x in tr]
    print(f"\n终局名次奖励自检: elim_tick={synth_elim.tolist()} → r={tr_r}")
    assert abs(float(tr[0]) - 1.0) < 1e-6, "从未淘汰者(座0)应得 +1"
    assert abs(float(tr[2]) + 1.0) < 1e-6, "最早淘汰者(座2)应得 −1"
    assert bool(jnp.isfinite(tr).all())

    print("\n" + "=" * 78)
    print("SMOKE PASS ✓  —— 空跑一步成功(rollout→GAE→PPO 损失→一次 Adam update)")
    print("           **未做真正训练**(无多步循环、无 checkpoint),对齐 issue.md v2.0")
    print("=" * 78)


if __name__ == "__main__":
    main()
