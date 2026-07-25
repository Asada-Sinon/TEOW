"""世界状态:一张定容实体表 + 资源点表,全是静态形状数组。

单表设计(调研报告 §7):玩家 p 占行块 [p*e_max,(p+1)*e_max)(owner 由行号
决定,是常量,不进 state);每家 HQ 固定在自己行块的 0 号槽(v1.5 起 P 家泛化)。
死 = 翻 alive 位,生 = scatter 进本半区第一个空槽,绝不 resize。
死槽「停泊」在无害值:pos 合法、hp 0、timer 0,算式照算,最后用掩码挑。

WorldState 是 NamedTuple → JAX 自动视为 pytree。静态地图(MapData)不在这里。
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import N_LINES, TYPE_HQ, TYPE_WORKER, Config
from .map import MapData

# 常驻指令(state.order):控制器不下新指令(no-op)时按它继续行动
ORDER_IDLE = 0
ORDER_HARVEST = 1   # 采集循环(配合 phase)
ORDER_BUILD = 2     # 去 target_node 建矿/泵
ORDER_MOVE = 3      # 走到 target_cell 后转 IDLE
ORDER_ATTACK = 4    # attack-move 向敌方 HQ
ORDER_GARRISON = 5  # 驻守(v1.3):走到锚点站住,被推离自动回岗,永不自转 IDLE

# 采集循环相位(order==HARVEST 时有效)
PH_TO_NODE = 0
PH_MINING = 1       # 在矿内(inside=True)
PH_TO_HQ = 2        # 满载返程


class WorldState(NamedTuple):
    # ---- 实体表 [N = n_players*e_max] ----
    alive: jax.Array        # bool  [N]
    etype: jax.Array        # int8  [N]  TYPE_*
    pos: jax.Array          # f32   [N,2] (row,col) 连续坐标(v1.2)。坐标约定:
    #                          格 (r,c) 覆盖 [r-0.5, r+0.5),**格心=整数坐标**,
    #                          归格唯一经 cell_of()。建筑写格心;inside 保留入口点。
    hp: jax.Array           # int32 [N]
    order: jax.Array        # int8  [N]  ORDER_*
    phase: jax.Array        # int8  [N]  PH_*
    target_node: jax.Array  # int8  [N]  资源点 id;-1 无
    target_cell: jax.Array  # f32   [N,2] MOVE/ATTACK 的目的点
    cargo: jax.Array        # int16 [N]  工人载荷量
    cargo_type: jax.Array   # int8  [N]  载荷资源类型(出矿时定格;不能从 target_node
    #                                     推——途中矿被拆/被改派会把矿石错记成水)
    mine_timer: jax.Array   # int16 [N]  矿内剩余开采 tick
    inside: jax.Array       # bool  [N]  在矿内(离场:不占格/不可被打/不攻击)
    btype: jax.Array        # int8  [N]  生产建筑当前在造的单位类型;0 无
    btimer: jax.Array       # int16 [N]  生产剩余 tick
    node_id: jax.Array      # int8  [N]  矿/泵实体所在资源点 id;-1 非采集建筑
    level: jax.Array        # int8  [N]  建筑等级(HQ 的 level 即基地等级;矿/泵/营
    #                                     用它;单位不用——单位强度走 upgrades 线)。
    #                                     矿被拆重建回 1 级(实体随槽重生)。
    garrison_id: jax.Array  # int8  [N]  驻守锚点 id;-1 无。0=己方 HQ,
    #                                     1..Nn=资源点 k=id-1(Nn+1..Nn+3=旗 j,v1.3
    #                                     Phase 4)。消费方必须门控 order==GARRISON。
    atk_cd: jax.Array       # int16 [N]  攻击冷却(v1.4;atk_period>1 的类型开火后
    #                                     置 period-1,每 tick 递减;普通攻击者恒 0)
    shell_timer: jax.Array  # int16 [N]  在途炮弹剩余飞行 tick(v1.4 迫击炮;0=无弹)
    shell_target: jax.Array  # f32  [N,2] 炮弹锁定落点(开火拍的目标位置)
    # ---- 军旗表 [P, max_flags](v1.3;旗不是实体:无血量、不可拆)----
    flag_pos: jax.Array     # f32  [P,F,2] 旗所在格心;未激活 -1
    flag_active: jax.Array  # bool [P,F]
    # ---- 资源点表 [Nn] ----
    node_owner: jax.Array        # int8  [Nn] -1 无主
    node_ent: jax.Array          # int16 [Nn] 结构实体槽号;-1 未建
    node_build_timer: jax.Array  # int16 [Nn] 建造剩余 tick;0 无施工
    node_builder: jax.Array      # int16 [Nn] 施工工人槽号;-1 无
    # ---- 全局 ----
    resources: jax.Array    # int32 [P,2]  [player][RES_ORE/RES_WATER]
    upgrades: jax.Array     # int8  [P,N_LINES]  [player][线],初始 1(v1.4 八线,
    #                                       按兵种);全局生效(存量+未来单位),
    #                                       营被拆仍保留
    tick: jax.Array         # int32 标量
    done: jax.Array         # bool 标量
    winner: jax.Array       # int8 标量  -1 未分;0..P-1 胜者;P 和局(v1.5 泛化)


def cell_of(pos: jax.Array) -> jax.Array:
    """连续坐标 → 所在格 (round;格心=整数)。所有格索引唯一经此(坐标约定
    单定义处,critic S-2)。"""
    return jnp.round(pos).astype(jnp.int32)


def owner_of_slots(cfg: Config) -> jax.Array:
    """int8 [N]:每个槽位的归属(常量,调用方闭包使用,不进 state)。"""
    return jnp.repeat(jnp.arange(cfg.n_players, dtype=jnp.int8), cfg.e_max)


def hq_slot(player: int, cfg: Config) -> int:
    return player * cfg.e_max


def init_state(cfg: Config, mapdata: MapData) -> WorldState:
    """初始局面:每家 1 座 HQ + start_workers 个工人,起始资源入库。
    确定性(不吃 key):随机性只存在于 step 内的仲裁,由外部传入的 key 驱动。"""
    n, nn = cfg.n_total, cfg.n_nodes

    alive = np.zeros(n, bool)
    etype = np.zeros(n, np.int8)
    pos = np.zeros((n, 2), np.float32)
    hp = np.zeros(n, np.int32)

    for p in range(cfg.n_players):
        base = hq_slot(p, cfg)
        alive[base] = True
        etype[base] = TYPE_HQ
        pos[base] = mapdata.hq_pos[p]
        hp[base] = cfg.hq_hp
        for i in range(cfg.start_workers):
            s = base + 1 + i
            alive[s] = True
            etype[s] = TYPE_WORKER
            pos[s] = mapdata.spawn_pos[p, i]
            hp[s] = cfg.worker_hp

    return WorldState(
        alive=jnp.asarray(alive),
        etype=jnp.asarray(etype),
        pos=jnp.asarray(pos),
        hp=jnp.asarray(hp),
        order=jnp.zeros(n, jnp.int8),
        phase=jnp.zeros(n, jnp.int8),
        target_node=jnp.full(n, -1, jnp.int8),
        target_cell=jnp.asarray(pos),
        cargo=jnp.zeros(n, jnp.int16),
        cargo_type=jnp.zeros(n, jnp.int8),
        mine_timer=jnp.zeros(n, jnp.int16),
        inside=jnp.zeros(n, bool),
        btype=jnp.zeros(n, jnp.int8),
        btimer=jnp.zeros(n, jnp.int16),
        node_id=jnp.full(n, -1, jnp.int8),
        level=jnp.ones(n, jnp.int8),
        garrison_id=jnp.full(n, -1, jnp.int8),
        atk_cd=jnp.zeros(n, jnp.int16),
        shell_timer=jnp.zeros(n, jnp.int16),
        shell_target=jnp.zeros((n, 2), jnp.float32),
        flag_pos=jnp.full((cfg.n_players, cfg.max_flags, 2), -1.0, jnp.float32),
        flag_active=jnp.zeros((cfg.n_players, cfg.max_flags), bool),
        node_owner=jnp.full(nn, -1, jnp.int8),
        node_ent=jnp.full(nn, -1, jnp.int16),
        node_build_timer=jnp.zeros(nn, jnp.int16),
        node_builder=jnp.full(nn, -1, jnp.int16),
        resources=jnp.asarray(
            [[cfg.start_ore, cfg.start_water]] * cfg.n_players, jnp.int32),
        upgrades=jnp.ones((cfg.n_players, N_LINES), jnp.int8),
        tick=jnp.asarray(0, jnp.int32),
        done=jnp.asarray(False),
        winner=jnp.asarray(-1, jnp.int8),
    )
