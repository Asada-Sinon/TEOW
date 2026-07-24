"""TEOW 引擎配置:唯一的数值参数真源。

frozen dataclass,被 build_step 闭包进 jit——改任何字段都会触发重编译,这是刻意的:
数值参数只能从这里进入引擎,代码里出现平衡数字字面量即 bug(CLAUDE.md 项目特定)。
派生量一律 @property,不做字段(与 alicization/underworld/config.py 同一纪律)。

平衡初值的依据与不确定度见 docs/DECISIONS.md 与 docs/changelog/(平衡区)。
"""

from __future__ import annotations

import dataclasses

# 实体类型编码(state.etype)。v1.1+ 的新类型(兵营/哨塔/狗子)往后追加,不重排。
TYPE_EMPTY = 0
TYPE_HQ = 1
TYPE_MINE = 2  # 建在矿点上的采集建筑
TYPE_PUMP = 3  # 建在水点上的采集建筑
TYPE_WORKER = 4
TYPE_INFANTRY = 5

# 资源类型编码(state.resources 的第二维;与资源点 node_type 一致)
RES_ORE = 0
RES_WATER = 1


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 0

    # ---- 世界 ----
    grid_h: int = 24
    grid_w: int = 24
    episode_len: int = 3000  # 超时判和局(winner=2)

    # ---- 容量(静态形状;e_max 满 = 天然人口上限,是特性不是 bug)----
    e_max: int = 64          # 每玩家实体槽数(单位+建筑共用一张表)
    n_nodes: int = 6         # 资源点数:每家附近 1矿+1水,中部公共 1矿+1水
    node_capacity: int = 4   # 每个矿/泵同时进驻的工人上限

    # ---- 采集一体循环 ----
    mine_time: int = 20      # 工人在矿内开采一趟所需 tick
    carry_cap: int = 10      # 一趟载荷(卸货时全额入账)
    move_cooldown: int = 2   # 每移动一格的冷却 tick(所有单位同速,v1.2 狗子再分化)

    # ---- 起始条件 ----
    start_ore: int = 100
    start_water: int = 50
    start_workers: int = 4

    # ---- 成本与耗时(ore, water, tick)----
    worker_cost_ore: int = 20
    worker_cost_water: int = 0
    worker_time: int = 40
    infantry_cost_ore: int = 30
    infantry_cost_water: int = 10
    infantry_time: int = 60
    mine_cost_ore: int = 40
    mine_cost_water: int = 0
    mine_time_build: int = 60
    pump_cost_ore: int = 20
    pump_cost_water: int = 30
    pump_time_build: int = 60

    # ---- 血量与攻击(伤害/tick,近战 Chebyshev<=1)----
    worker_hp: int = 20
    worker_atk: int = 1
    infantry_hp: int = 40
    infantry_atk: int = 4
    hq_hp: int = 400
    node_struct_hp: int = 100  # 矿与泵共用

    # ---- 脚本 AI(controller.scripted;不属于引擎规则,放这里是为了同一份
    #      resolved config 能完整复现一场对局)----
    ai_worker_target: int = 8    # 工人数低于此值时 HQ 优先补工人
    ai_attack_threshold: int = 6 # 步兵攒到此数全军压向敌方 HQ

    # ---- 派生量(不是字段)----
    @property
    def n_total(self) -> int:
        """实体表总行数:前 e_max 行归玩家 0,后 e_max 行归玩家 1。"""
        return 2 * self.e_max

    @property
    def n_cells(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def n_goals(self) -> int:
        """距离场数量:每个资源点一张 + 每个 HQ 一张。"""
        return self.n_nodes + 2
