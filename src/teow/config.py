"""TEOW 引擎配置:唯一的数值参数真源。

frozen dataclass,被 build_step 闭包进 jit——改任何字段都会触发重编译,这是刻意的:
数值参数只能从这里进入引擎,代码里出现平衡数字字面量即 bug(CLAUDE.md 项目特定)。
派生量一律 @property,不做字段(与 alicization/underworld/config.py 同一纪律)。

平衡初值的依据与不确定度见 docs/DECISIONS.md 与 docs/changelog/(平衡区)。
"""

from __future__ import annotations

import dataclasses

# 实体类型编码(state.etype)。v1.2+ 的新类型(兵营/哨塔/狗子)往后追加,不重排。
TYPE_EMPTY = 0
TYPE_HQ = 1
TYPE_MINE = 2  # 建在矿点上的采集建筑
TYPE_PUMP = 3  # 建在水点上的采集建筑
TYPE_WORKER = 4
TYPE_INFANTRY = 5
TYPE_CAMP = 6      # 技能训练营(v1.1;基地2级解锁,建成即2级)
TYPE_BARRACKS = 7  # 兵营(v1.2;基地2级解锁,出狗子)
TYPE_DOG = 8       # 狗子(v1.2;快/脆/低攻,吃步兵捆绑线)

# 资源类型编码(state.resources 的第二维;与资源点 node_type 一致)
RES_ORE = 0
RES_WATER = 1

# 升级线编码(state.upgrades 的第二维)
LINE_INFANTRY = 0  # 步兵捆绑线:每级血+攻一起升
LINE_WORKER = 1    # 工人经济线:载荷/开采速度/血量,无攻击

# btype 任务码:正数=在训单位类型(v1.0 语义);负数=特殊任务。
# 解码必须集中在 economy 单处;完成分支必须 btype←0(否则下 tick 重复触发)。
BTASK_UPGRADE = -1        # 建筑自升级(HQ/矿/泵/营)
BTASK_RESEARCH_INF = -2   # 训练营:研发步兵线
BTASK_RESEARCH_WORKER = -3  # 训练营:研发工人线
BTASK_BUILD_CAMP = -4     # 在建训练营的专属标记(建成前 hp 线性成长)
BTASK_BUILD_BARRACKS = -5  # 在建兵营(v1.2;同营的成长语义,建成 level=1)


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
    # 开采耗时/载荷已并入工人经济线的 *_by_level 表(v1.1);一趟入账公式:
    # carry_cap[工人线级] + node_yield_bonus[矿泵级]。

    # ---- 连续移动(v1.2:单位 360°,建筑/资源点仍格子锚定)----
    # speed 按实体类型查表(格/tick;工/兵 0.5 与 v1.1 的 move_cooldown=2 等效,
    # 行为近似保持);建筑 0。表长 16 给 v1.2+ 新类型留位。
    speed_by_type: tuple = (0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.9, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_radius: float = 0.35   # 单位圆半径(<0.5,圆不出格,建筑推离用格判定)
    reach_radius: float = 1.2   # 入驻/卸货/开工的欧氏到达半径(≈旧 4 邻)
    melee_range: float = 1.5    # 近战射程(≈旧 Chebyshev≤1,含对角)
    stationary_cost: int = 24   # 动态场里静止单位所在格的软障碍附加代价
    #                             (v1.0 的硬障碍软化:连续单位可贴身挤过)
    # (v1.1 的 congestion_cost 已退役:连续模式下移动单位不计场代价——
    #  自己的罚分落在自己脚下会让梯度采样退化成游走;对向流由圆形互推解决)

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
    # 单位属性走升级线查表(v1.1);建筑血量 v1.1 仍为平值(升级只提产量/解锁,
    # 不加建筑血,记 changelog 已知取舍)
    worker_atk: int = 1        # 工人攻击无升级线(纯经济线),保持标量
    hq_hp: int = 400
    node_struct_hp: int = 100  # 矿与泵共用

    # ---- 等级体系(v1.1)----
    # 约定:所有 *_by_level 表长 8,直接用等级 1..7 下标(0 位是废位填 0);
    # 升级/研发的 cost/time 表按「当前等级」取(花费=从 L 升到 L+1),
    # 有效位 1..6(营 2..6),表长同 8。数值初值均 [AI-DRAFT],依据见 DECISIONS。
    base_max_level: int = 7
    # 解锁表(v1.2 扩成表结构,按 TYPE_* 下标):基地几级解锁该类型的建造。
    # 0 = 不受基地等级限制;表长 16 与 speed_by_type 对齐。
    # camp=2(TYPE_CAMP=6)、兵营=2(TYPE_BARRACKS=7)、哨塔=2(v1.2 Phase3)
    unlock_level_by_type: tuple = (0, 0, 0, 0, 0, 0, 2, 2, 0, 2,
                                   0, 0, 0, 0, 0, 0)

    # 基地升级(收益=解锁+矿泵/营等级上限,零单位加成——issue v1.1 设计决策)
    base_up_cost_ore: tuple = (0, 100, 150, 250, 400, 600, 900, 0)
    base_up_cost_water: tuple = (0, 50, 100, 150, 250, 400, 600, 0)
    base_up_time: tuple = (0, 150, 200, 250, 300, 350, 400, 0)

    # 矿/泵升级(产量走 node_yield_bonus;等级上限=基地等级)
    node_up_cost_ore: tuple = (0, 30, 50, 80, 120, 180, 250, 0)
    node_up_cost_water: tuple = (0, 20, 30, 50, 80, 120, 180, 0)
    node_up_time: tuple = (0, 80, 100, 120, 140, 160, 180, 0)
    node_yield_bonus: tuple = (0, 0, 2, 4, 6, 9, 12, 15)  # 一趟入账的加成,按矿泵等级

    # 技能训练营(建成即 2 级;等级上限=基地等级;被拆研发中断不退款、已购保留)
    camp_cost_ore: int = 60
    camp_cost_water: int = 40
    camp_build_time: int = 100
    camp_hp_by_level: tuple = (0, 0, 150, 180, 210, 240, 270, 300)
    camp_up_cost_ore: tuple = (0, 0, 80, 120, 180, 260, 360, 0)
    camp_up_cost_water: tuple = (0, 0, 50, 80, 120, 180, 260, 0)
    camp_up_time: tuple = (0, 0, 100, 120, 140, 160, 180, 0)

    # 步兵捆绑线(每级血+攻同升;线等级上限=营等级)
    inf_hp_by_level: tuple = (0, 40, 48, 56, 66, 78, 92, 108)
    inf_atk_by_level: tuple = (0, 4, 5, 6, 7, 8, 10, 12)
    inf_res_cost_ore: tuple = (0, 60, 90, 140, 210, 300, 420, 0)
    inf_res_cost_water: tuple = (0, 40, 60, 90, 140, 200, 280, 0)
    inf_res_time: tuple = (0, 120, 150, 180, 210, 240, 270, 0)

    # 兵营与狗子(v1.2;狗子吃步兵捆绑线,不单开线——DECISIONS)
    max_barracks: int = 2   # 每玩家兵营数量上限
    barracks_cost_ore: int = 80
    barracks_cost_water: int = 40
    barracks_build_time: int = 120
    barracks_hp: int = 200
    dog_cost_ore: int = 20
    dog_cost_water: int = 5
    dog_time: int = 30
    dog_hp_by_level: tuple = (0, 24, 29, 34, 40, 47, 55, 64)
    dog_atk_by_level: tuple = (0, 3, 4, 4, 5, 6, 7, 8)

    # 工人经济线(载荷/开采速度/血量,无攻击;线等级上限=营等级)
    worker_carry_by_level: tuple = (0, 10, 12, 14, 17, 20, 24, 28)
    worker_mine_time_by_level: tuple = (0, 20, 18, 16, 14, 12, 10, 9)
    worker_hp_by_level: tuple = (0, 20, 24, 28, 33, 38, 44, 50)
    worker_res_cost_ore: tuple = (0, 50, 80, 120, 180, 260, 360, 0)
    worker_res_cost_water: tuple = (0, 30, 50, 80, 120, 180, 260, 0)
    worker_res_time: tuple = (0, 100, 130, 160, 190, 220, 250, 0)

    # ---- 脚本 AI(controller.scripted;不属于引擎规则,放这里是为了同一份
    #      resolved config 能完整复现一场对局)----
    ai_worker_target: int = 8    # 工人数低于此值时 HQ 优先补工人
    ai_attack_threshold: int = 6 # 步兵攒到此数全军压向敌方 HQ
    ai_base_level_target: int = 3  # 脚本 AI 把基地升到几级为止(v1.1)
    ai_upgrade_reserve: int = 40   # 库存超出升级成本多少才肯升级/研发(留军费;
    #                                150 时实测对局在 ~730 tick 分胜负,脚本到死
    #                                都攒不齐,升级机制全程闲置)

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
