"""TEOW 引擎配置:唯一的数值参数真源。

frozen dataclass,被 build_step 闭包进 jit——改任何字段都会触发重编译,这是刻意的:
数值参数只能从这里进入引擎,代码里出现平衡数字字面量即 bug(CLAUDE.md 项目特定)。
派生量一律 @property,不做字段(与 alicization/underworld/config.py 同一纪律)。

平衡初值的依据与不确定度见 docs/DECISIONS.md 与 docs/changelog/(平衡区)。
v1.4 起 per-type 表长 32(16 已用尽);类型标量字段是真源,32-表一律 @property 组装。
"""

from __future__ import annotations

import dataclasses

# 实体类型编码(state.etype)。新类型往后追加,不重排。
TYPE_EMPTY = 0
TYPE_HQ = 1
TYPE_MINE = 2  # 建在矿点上的采集建筑
TYPE_PUMP = 3  # 建在水点上的采集建筑
TYPE_WORKER = 4
TYPE_INFANTRY = 5  # 近战步兵(v1.4 起正式定位;id 保号不重排)
TYPE_CAMP = 6      # 技能训练营(v1.1;基地2级解锁,建成即2级)
TYPE_BARRACKS = 7  # 兵营(v1.2;基地2级解锁;v1.4 起可升级,1-5 级爆兵)
TYPE_DOG = 8       # 狗子(v1.2;骑兵速度,战力低于近战步兵,骚扰位)
TYPE_TOWER = 9     # 哨塔(v1.2;基地2级解锁;v1.4 数量挂基地等级)
TYPE_STRONGMAN = 10  # 大力士工人(v1.4;HQ3,采集快载荷多,移速=工人)
TYPE_WAGON = 11      # 马车(v1.4;HQ5,载荷最大,采速介于工人与大力士,移速=轻骑)
TYPE_ARCHER = 12     # 弓箭手(v1.4;兵营2,远程物理即时命中)
TYPE_LCAV = 13       # 轻骑兵(v1.4;兵营3,快速近战)
TYPE_HEAVY = 14      # 重盔甲战士(v1.4;兵营3,中血高甲近战,魔法克制)
TYPE_MAGE = 15       # 法师(v1.4;兵营4,远程单体魔法,无视护甲)
TYPE_HEALER = 16     # 奶妈神官(v1.4;兵营4,不攻击,自动奶血量比例最低友军单位)
TYPE_RAM = 17        # 攻城车(v1.4;兵营5,近战,只能打建筑)
TYPE_MORTAR = 18     # 迫击炮(v1.4;防御建筑,HQ3 解锁,限1,弹道+盲区+范围衰减)
TYPE_FENCE_WOOD = 19   # 木栅栏(v1.5;HQ2,占格挡路可被打,无攻击无上限)
TYPE_FENCE_STONE = 20  # 石栅栏(v1.5;HQ3)
TYPE_FENCE_IRON = 21   # 铁栅栏(v1.5;HQ5;三档独立建筑,高级解锁后低级仍可造)
TYPE_MAGETOWER = 22    # 法师塔(v1.6;HQ3,限1,单体魔法,可对空)
TYPE_LANDMINE = 23     # 地雷(v1.6;HQ4,限5,触发即爆一次性;不可被打不挡路)
TYPE_FLAMER = 24       # 喷火器(v1.6;HQ6,限1,自心小半径持续物理,只对地)
TYPE_LASER = 25        # 激光炮(v1.6;HQ7,限1,单体高持续魔法,可对空)
TYPE_CATAPULT = 26     # 投石车(v1.6;兵营6,远程范围物理,射程<迫击炮,不可对空)
TYPE_AIRSHIP = 27      # 飞艇(v1.6;兵营6,空中运输载7,无攻击,飞跃地面障碍)
TYPE_DRAGON = 28       # 龙骑兵(v1.6;兵营7,空军:对地喷火AoE+对空单体,共享CD)
N_TYPES = 32  # per-type 表长(29..31 留给后续版本)

# 资源类型编码(state.resources 的第二维;与资源点 node_type 一致)
RES_ORE = 0
RES_WATER = 1

# 升级线编码(state.upgrades 的第二维;v1.4 起按兵种一条线,工人经济线取消)
LINE_INFANTRY = 0
LINE_DOG = 1
LINE_ARCHER = 2
LINE_LCAV = 3
LINE_HEAVY = 4
LINE_MAGE = 5
LINE_HEALER = 6
LINE_RAM = 7
LINE_CATAPULT = 8   # v1.6 修订:三兵种开线,上限 3 级(用户 2026-07-26 复核)
LINE_AIRSHIP = 9
LINE_DRAGON = 10
N_LINES = 11
N_LINES_LEGACY = 8  # v1.5 动作 id 公式冻结值(扩线不得位移旧 id,保号契约)
# 线 → 兵种(研发解锁门用:该兵种可训练则线可研)
TYPE_OF_LINE = (TYPE_INFANTRY, TYPE_DOG, TYPE_ARCHER, TYPE_LCAV,
                TYPE_HEAVY, TYPE_MAGE, TYPE_HEALER, TYPE_RAM,
                TYPE_CATAPULT, TYPE_AIRSHIP, TYPE_DRAGON)

# btype 任务码:正数=在训单位类型;负数=特殊任务。
# 解码必须集中在 economy 单处;完成分支必须 btype←0(否则下 tick 重复触发)。
BTASK_UPGRADE = -1        # 建筑自升级(HQ/矿/泵/营/兵营/塔)
# −2/−3 已退役(v1.3 的双线研发码;v1.4 研发码统一走 −(16+line))
BTASK_BUILD_CAMP = -4     # 在建训练营的专属标记(建成前 hp 线性成长)
BTASK_BUILD_BARRACKS = -5  # 在建兵营
BTASK_BUILD_TOWER = -6     # 在建哨塔
BTASK_BUILD_MORTAR = -7    # 在建迫击炮(v1.4 Phase 3)
BTASK_BUILD_FENCE_WOOD = -8   # 在建木栅栏(v1.5)
BTASK_BUILD_FENCE_STONE = -9  # 在建石栅栏(v1.5)
BTASK_BUILD_FENCE_IRON = -10  # 在建铁栅栏(v1.5)
BTASK_BUILD_MAGETOWER = -11   # 在建法师塔(v1.6)
BTASK_BUILD_LANDMINE = -12    # 在建地雷(v1.6)
BTASK_BUILD_FLAMER = -13      # 在建喷火器(v1.6)
BTASK_BUILD_LASER = -14       # 在建激光炮(v1.6)
# −15 留给后续版本
BTASK_RESEARCH_BASE = -16  # 研发线 l 的任务码 = -(16+l),l∈[0,8) → −16..−23


def btask_research(line: int) -> int:
    return BTASK_RESEARCH_BASE - line


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 0

    # ---- 世界 ----
    grid_h: int = 64         # v1.5:方格网格上雕六边形可行区(规格「约 64」)
    grid_w: int = 64
    episode_len: int = 6000  # 硬帽/scan 边界(v1.8:不再判和局;异界之门保证门开后必分
    #                          胜负,episode_len 沦为防御硬帽,正常应因怪物升级不可达)
    gate_open_tick: int = 4000  # v1.8 异界之门开门时刻(< episode_len,留 overtime 余量);
    #                             旧「超时」语义迁移至此,门开后 overtime 逼出唯一胜者

    # ---- 容量(静态形状;e_max 满 = 天然人口上限,是特性不是 bug)----
    n_players: int = 4       # 玩家数 P(v1.5:四人自由混战;编译期静态量。
    #                          胜负编码:-1 未分,0..P-1 胜者,P=和局)
    e_max: int = 64          # 每玩家实体槽数(单位+建筑共用一张表)
    n_nodes: int = 20        # 资源点数(v1.5):每家家门 1矿1水×4 + 六边形顶点公共 12
    # 采集名额(v1.3 指派即占用):按矿泵等级查「可同时指派(order==HARVEST)」
    # 的工人上限;名额在指派侧占用,不因工人出矿运输而释放。0 位是废位。
    harvest_slots_by_level: tuple = (0, 3, 3, 4, 4, 5, 5, 6)
    max_flags: int = 3       # 军旗(v1.3):每玩家上限,不随任何等级增加

    # ---- 异界之门 sudden-death 怪物(v1.8;数值 [AI-DRAFT],P4 校准)----
    # 门开(tick>=gate_open_tick)后每 spawn_interval 拍向每个存活玩家中心出一波怪。
    # 阵营隔离:打玩家 p 的怪只打 p、也只有 p 能打它。HP 无上限随 overtime 线性增、
    # 攻击封顶、慢速近战、强度生成时定死(后出更强、已在场不变)、各阵营压力一致。
    monster_cap: int = 64         # Mmax:每玩家怪容量(静态形状;批量显存乘子)
    monster_spawn_interval: int = 50   # 每多少拍出一波
    monster_wave_count: int = 2   # 每玩家每波怪数
    monster_hp_base: int = 50     # HP 基(spawn 时 = base + round(hp_slope*overtime),无上限)
    monster_hp_slope: float = 0.5  # HP 每 overtime 拍线性增量(P4 调此控收敛速度)
    monster_atk_base: int = 5     # 攻击基
    monster_atk_slope: float = 0.005  # 攻击每 overtime 拍增量
    monster_atk_cap: int = 20     # 攻击上限(封顶)
    monster_speed: float = 0.15   # 移速(格/tick;慢于任何单位,可被风筝)
    monster_melee_range: float = 1.5   # 近战判定半径(=melee_range)

    # ---- 连续移动(v1.2:单位 360°,建筑/资源点仍格子锚定)----
    unit_radius: float = 0.35   # 单位圆半径(<0.5,圆不出格,建筑推离用格判定)
    reach_radius: float = 1.2   # 入驻/卸货/开工的欧氏到达半径(≈旧 4 邻)
    garrison_hold_radius: float = 1.2  # 驻守(v1.3):离锚点超此距离才回岗移动
    melee_range: float = 1.5    # 近战射程(≈旧 Chebyshev≤1,含对角)
    stationary_cost: int = 24   # 动态场里静止单位所在格的软障碍附加代价

    # ---- 单位移速(格/tick;建筑 0。真源在此,speed_by_type 表由 property 组装)----
    worker_speed: float = 0.5
    infantry_speed: float = 0.5
    dog_speed: float = 0.9
    strongman_speed: float = 0.5   # =工人(规格)
    wagon_speed: float = 0.9       # =轻骑兵(规格)
    archer_speed: float = 0.5
    lcav_speed: float = 0.9
    heavy_speed: float = 0.4
    mage_speed: float = 0.45
    healer_speed: float = 0.45
    ram_speed: float = 0.3
    catapult_speed: float = 0.3
    airship_speed: float = 0.7   # 空中单位(v1.6):无视地面障碍直线飞
    dragon_speed: float = 0.8

    # ---- 起始条件 ----
    start_ore: int = 100
    start_water: int = 50
    start_workers: int = 4

    # ---- 单位成本与耗时(ore, water, tick;32-表由 property 组装)----
    worker_cost_ore: int = 20
    worker_cost_water: int = 0
    worker_time: int = 40
    infantry_cost_ore: int = 30
    infantry_cost_water: int = 10
    infantry_time: int = 60
    dog_cost_ore: int = 20
    dog_cost_water: int = 5
    dog_time: int = 30
    strongman_cost_ore: int = 40
    strongman_cost_water: int = 10
    strongman_time: int = 60
    wagon_cost_ore: int = 60
    wagon_cost_water: int = 30
    wagon_time: int = 80
    archer_cost_ore: int = 30
    archer_cost_water: int = 15
    archer_time: int = 50
    lcav_cost_ore: int = 45
    lcav_cost_water: int = 20
    lcav_time: int = 70
    heavy_cost_ore: int = 50
    heavy_cost_water: int = 25
    heavy_time: int = 80
    mage_cost_ore: int = 40
    mage_cost_water: int = 40
    mage_time: int = 90
    healer_cost_ore: int = 30
    healer_cost_water: int = 40
    healer_time: int = 90
    ram_cost_ore: int = 80
    ram_cost_water: int = 40
    ram_time: int = 120
    catapult_cost_ore: int = 90
    catapult_cost_water: int = 50
    catapult_time: int = 130
    airship_cost_ore: int = 120
    airship_cost_water: int = 80
    airship_time: int = 150
    dragon_cost_ore: int = 180
    dragon_cost_water: int = 140
    dragon_time: int = 220

    # ---- 建筑成本与耗时 ----
    mine_cost_ore: int = 40
    mine_cost_water: int = 0
    mine_time_build: int = 60
    pump_cost_ore: int = 20
    pump_cost_water: int = 30
    pump_time_build: int = 60

    # ---- 血量(建筑平值/表值;单位血量走线表,采集单位固定基线)----
    hq_hp: int = 400
    node_struct_hp: int = 100  # 矿与泵共用
    worker_hp: int = 20        # v1.4:工人经济线取消,回固定基线
    strongman_hp: int = 30
    wagon_hp: int = 40

    # ---- 采集参数(载荷/开采 tick;per-type,真源标量)----
    worker_carry: int = 10
    worker_mine_time: int = 20
    strongman_carry: int = 16      # 大力士:采集快、载荷多
    strongman_mine_time: int = 14
    wagon_carry: int = 24          # 马车:载荷最大,采速介于工人与大力士
    wagon_mine_time: int = 17

    # ---- 护甲与伤害类型(v1.4;护甲=物理减伤百分比,魔法完全无视护甲)----
    hq_armor: int = 10
    infantry_armor: int = 10
    tower_armor: int = 20
    wagon_armor: int = 10
    lcav_armor: int = 20
    heavy_armor: int = 60   # 重盔甲战士:高甲,由魔法克制(规格设计意图)
    ram_armor: int = 40
    mortar_armor: int = 20

    # ---- 射程(欧氏;近战单位用 melee_range,远程/防御建筑在此)----
    tower_range: float = 4.0
    archer_range: float = 3.5
    mage_range: float = 3.0
    healer_range: float = 3.0     # 治疗射程(奶妈无攻击)
    mortar_range: float = 7.0
    mortar_min_range: float = 2.5  # 盲区:近处打不了(规格)
    mortar_aoe_radius: float = 1.5
    mortar_atk: int = 30
    mortar_atk_period: int = 40    # 攻击间隔(tick;开火后冷却)
    mortar_flight_time: int = 8    # 弹道飞行 tick(< atk_period,单弹槽恒够)
    mortar_cost_ore: int = 80
    mortar_cost_water: int = 60
    mortar_build_time: int = 110
    mortar_hp: int = 150

    # ---- 栅栏(v1.5;三档独立建筑,占格挡路,无攻击,无数量上限,不可升级。
    #      解锁 木HQ2/石HQ3/铁HQ5=规格;数值 [AI-DRAFT])----
    fence_wood_cost_ore: int = 10
    fence_wood_cost_water: int = 0
    fence_wood_build_time: int = 20
    fence_wood_hp: int = 80
    fence_stone_cost_ore: int = 25
    fence_stone_cost_water: int = 5
    fence_stone_build_time: int = 35
    fence_stone_hp: int = 200
    fence_stone_armor: int = 20
    fence_iron_cost_ore: int = 40
    fence_iron_cost_water: int = 20
    fence_iron_build_time: int = 50
    fence_iron_hp: int = 400
    fence_iron_armor: int = 40

    # ---- v1.6 防御建筑群(除哨塔/地雷外每种限 1;数值 [AI-DRAFT])----
    magetower_cost_ore: int = 70
    magetower_cost_water: int = 50
    magetower_build_time: int = 100
    magetower_hp: int = 140
    magetower_range: float = 4.5
    magetower_atk: int = 20        # v1.7 用户定案 14→20(现值守不住等价近战超弱)
    magetower_period: int = 4      # v1.7 用户定案 5→4 提 dps 2.8→5.0
    # [source: 20260726-v17-tune-magetower-dps]
    landmine_cost_ore: int = 30
    landmine_cost_water: int = 10
    landmine_build_time: int = 25
    landmine_hp: int = 30          # 名义血量(不可被打;自爆用 incoming 加满)
    landmine_trigger_radius: float = 1.0
    landmine_aoe_radius: float = 2.0
    landmine_atk: int = 50
    flamer_cost_ore: int = 80
    flamer_cost_water: int = 50
    flamer_build_time: int = 100
    flamer_hp: int = 160
    flamer_armor: int = 20
    flamer_radius: float = 2.0     # 自心圆持续物理(period 1,平坦无衰减)
    flamer_atk: int = 3
    laser_cost_ore: int = 120
    laser_cost_water: int = 90
    laser_build_time: int = 130
    laser_hp: int = 150
    laser_armor: int = 20
    laser_range: float = 5.0
    laser_atk: int = 5             # 每 tick 魔法(period 1=「高持续」)
    # ---- v1.6 兵营 6/7 级兵种 ----
    catapult_hp_by_level: tuple = (0, 100, 115, 132, 132, 132, 132, 132)
    catapult_atk_by_level: tuple = (0, 24, 28, 33, 33, 33, 33, 33)
    catapult_armor: int = 30
    catapult_range: float = 5.0    # 明显短于迫击炮 7.0(规格「不能太远程」)
    catapult_aoe_radius: float = 1.2
    catapult_period: int = 30
    catapult_flight_time: int = 4
    airship_hp_by_level: tuple = (0, 140, 165, 195, 195, 195, 195, 195)
    airship_armor: int = 10
    airship_capacity: int = 7      # 载员上限(用户定案)
    reboard_lockout: int = 60      # 空降部队开火后多少 tick 内不可回艇
    dragon_hp_by_level: tuple = (0, 220, 255, 295, 295, 295, 295, 295)
    dragon_armor: int = 30
    dragon_air_atk_by_level: tuple = (0, 45, 52, 60, 60, 60, 60, 60)  # 对空单体
    dragon_air_range: float = 3.0
    dragon_breath_atk: int = 12    # 对地喷火(自心圆平坦持续)
    dragon_breath_bld_percent: int = 50  # 喷火对建筑打折(用户修订;[AI-DRAFT])
    dragon_breath_radius: float = 4.5   # v1.7 用户定案 2.5→4.5 大范围火海(对地AoE)
    # [source: 20260726-v17-tune]
    dragon_period: int = 10        # 两种攻击共享 CD

    # ---- 等级体系 ----
    # 约定:所有 *_by_level 表长 8,直接用等级 1..7 下标(0 位是废位填 0);
    # 升级/研发的 cost/time 表按「当前等级」取(花费=从 L 升到 L+1)。
    base_max_level: int = 7
    # 解锁表(按 TYPE_* 下标):基地几级解锁该类型的**建造**;0=不受限。
    # 单位的训练等级门槛走 train_level_by_type(生产建筑自身等级)。

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

    # 研发(v1.4:八线共用一套成本/耗时表,按当前线级取;per-line 差异化留 v1.7。
    # DECISIONS [AI-DRAFT])
    line_res_cost_ore: tuple = (0, 60, 90, 140, 210, 300, 420, 0)
    line_res_cost_water: tuple = (0, 40, 60, 90, 140, 200, 280, 0)
    line_res_time: tuple = (0, 120, 150, 180, 210, 240, 270, 0)

    # 兵营(v1.2;v1.4 起可升级 1→7,上限=基地等级,升级解锁高级兵种)
    max_barracks: int = 2   # 每玩家兵营数量上限
    barracks_cost_ore: int = 80
    barracks_cost_water: int = 40
    barracks_build_time: int = 120
    barracks_hp_by_level: tuple = (0, 200, 240, 280, 330, 380, 440, 500)
    barracks_up_cost_ore: tuple = (0, 60, 90, 130, 190, 270, 370, 0)
    barracks_up_cost_water: tuple = (0, 40, 60, 90, 130, 190, 270, 0)
    barracks_up_time: tuple = (0, 90, 110, 130, 150, 170, 190, 0)

    # 哨塔(等级上限=基地等级,升级提升血量与攻击;只攻单位不攻建筑)
    # v1.4 多塔:可建数量挂基地等级,每两级 +1(用户确认数值,非草案):
    # 1级0座、2-3级1座、4-5级2座、6-7级3座。下标=基地等级,0 位废位。
    tower_cap_by_hq_level: tuple = (0, 0, 1, 1, 2, 2, 3, 3)
    tower_cost_ore: int = 50
    tower_cost_water: int = 30
    tower_build_time: int = 90
    tower_hp_by_level: tuple = (0, 120, 150, 180, 220, 260, 300, 350)
    tower_atk_by_level: tuple = (0, 3, 8, 10, 13, 16, 20, 24)
    tower_up_cost_ore: tuple = (0, 40, 60, 90, 130, 180, 240, 0)
    tower_up_cost_water: tuple = (0, 25, 40, 60, 85, 120, 160, 0)
    tower_up_time: tuple = (0, 70, 85, 100, 115, 130, 145, 0)

    # ---- 战斗单位线表(每级血/攻;线等级上限=训练营等级)。数值 [AI-DRAFT] ----
    inf_hp_by_level: tuple = (0, 40, 48, 56, 66, 78, 92, 108)
    inf_atk_by_level: tuple = (0, 4, 5, 6, 7, 8, 10, 12)
    dog_hp_by_level: tuple = (0, 24, 29, 34, 40, 47, 55, 64)
    dog_atk_by_level: tuple = (0, 3, 4, 4, 5, 6, 7, 8)
    archer_hp_by_level: tuple = (0, 30, 36, 42, 50, 59, 69, 81)
    archer_atk_by_level: tuple = (0, 5, 6, 7, 8, 10, 12, 14)
    lcav_hp_by_level: tuple = (0, 45, 54, 63, 74, 87, 102, 120)
    lcav_atk_by_level: tuple = (0, 6, 7, 8, 10, 12, 14, 17)
    heavy_hp_by_level: tuple = (0, 60, 72, 84, 99, 116, 136, 160)
    heavy_atk_by_level: tuple = (0, 5, 6, 7, 8, 10, 12, 14)
    mage_hp_by_level: tuple = (0, 25, 30, 35, 41, 48, 56, 66)
    mage_atk_by_level: tuple = (0, 7, 8, 10, 12, 14, 17, 20)
    healer_hp_by_level: tuple = (0, 25, 30, 35, 41, 48, 56, 66)
    healer_heal_by_level: tuple = (0, 3, 4, 5, 6, 7, 8, 10)
    ram_hp_by_level: tuple = (0, 120, 140, 165, 195, 230, 270, 320)
    ram_atk_by_level: tuple = (0, 25, 30, 36, 43, 51, 60, 72)

    # ---- 脚本 AI(controller.scripted;不属于引擎规则,放这里是为了同一份
    #      resolved config 能完整复现一场对局)----
    ai_worker_target: int = 8    # 工人数低于此值时 HQ 优先补工人
    ai_attack_threshold: int = 6 # 战斗单位攒到此数全军压向敌方 HQ
    ai_base_level_target: int = 3  # 脚本 AI 把基地升到几级为止(默认 3 保持对局节奏;
    #                                v1.4 覆盖型审计局传 5-7 并配富开局解锁全兵种)
    ai_upgrade_reserve: int = 40   # 库存超出升级成本多少才肯升级/研发(留军费)

    # ---- 派生量(不是字段)----
    @property
    def n_total(self) -> int:
        """实体表总行数:玩家 p 占行块 [p*e_max, (p+1)*e_max)。"""
        return self.n_players * self.e_max

    @property
    def n_cells(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def n_goals(self) -> int:
        """距离场数量:每个资源点一张 + 每个 HQ 一张 + P 玩家 × max_flags 张
        军旗动态通道(v1.3,种子每 tick 生成,见 movement)。"""
        return self.n_nodes + self.n_players * (1 + self.max_flags)

    # ---- per-type 32-表(全部由真源标量/表组装;下标=TYPE_*)----
    def _t32(self, mapping: dict, default=0) -> tuple:
        return tuple(mapping.get(t, default) for t in range(N_TYPES))

    @property
    def speed_by_type(self) -> tuple:
        """格/tick;=0 即建筑(movement/combat 的唯一单位/建筑判据)。"""
        return self._t32({
            TYPE_WORKER: self.worker_speed, TYPE_INFANTRY: self.infantry_speed,
            TYPE_DOG: self.dog_speed, TYPE_STRONGMAN: self.strongman_speed,
            TYPE_WAGON: self.wagon_speed, TYPE_ARCHER: self.archer_speed,
            TYPE_LCAV: self.lcav_speed, TYPE_HEAVY: self.heavy_speed,
            TYPE_MAGE: self.mage_speed, TYPE_HEALER: self.healer_speed,
            TYPE_RAM: self.ram_speed, TYPE_CATAPULT: self.catapult_speed,
            TYPE_AIRSHIP: self.airship_speed,
            TYPE_DRAGON: self.dragon_speed}, default=0.0)

    @property
    def unlock_level_by_type(self) -> tuple:
        """基地几级解锁该**建筑**的建造;0=不受限(单位走 train_level_by_type)。"""
        return self._t32({TYPE_CAMP: 2, TYPE_BARRACKS: 2, TYPE_TOWER: 2,
                          TYPE_MORTAR: 3, TYPE_FENCE_WOOD: 2,
                          TYPE_FENCE_STONE: 3, TYPE_FENCE_IRON: 5,
                          TYPE_MAGETOWER: 3, TYPE_LANDMINE: 4,
                          TYPE_FLAMER: 6, TYPE_LASER: 7})

    @property
    def train_level_by_type(self) -> tuple:
        """训练该单位要求其生产建筑达到的等级(HQ 系看基地级,兵营系看兵营级);
        0=不可训练。"""
        return self._t32({
            TYPE_WORKER: 1, TYPE_INFANTRY: 1, TYPE_STRONGMAN: 3, TYPE_WAGON: 5,
            TYPE_DOG: 1, TYPE_ARCHER: 2, TYPE_LCAV: 3, TYPE_HEAVY: 3,
            TYPE_MAGE: 4, TYPE_HEALER: 4, TYPE_RAM: 5,
            TYPE_CATAPULT: 6, TYPE_AIRSHIP: 6, TYPE_DRAGON: 7})

    @property
    def train_cost_ore_by_type(self) -> tuple:
        return self._t32({
            TYPE_WORKER: self.worker_cost_ore, TYPE_INFANTRY: self.infantry_cost_ore,
            TYPE_DOG: self.dog_cost_ore, TYPE_STRONGMAN: self.strongman_cost_ore,
            TYPE_WAGON: self.wagon_cost_ore, TYPE_ARCHER: self.archer_cost_ore,
            TYPE_LCAV: self.lcav_cost_ore, TYPE_HEAVY: self.heavy_cost_ore,
            TYPE_MAGE: self.mage_cost_ore, TYPE_HEALER: self.healer_cost_ore,
            TYPE_RAM: self.ram_cost_ore, TYPE_CATAPULT: self.catapult_cost_ore,
            TYPE_AIRSHIP: self.airship_cost_ore,
            TYPE_DRAGON: self.dragon_cost_ore})

    @property
    def train_cost_water_by_type(self) -> tuple:
        return self._t32({
            TYPE_WORKER: self.worker_cost_water,
            TYPE_INFANTRY: self.infantry_cost_water,
            TYPE_DOG: self.dog_cost_water, TYPE_STRONGMAN: self.strongman_cost_water,
            TYPE_WAGON: self.wagon_cost_water, TYPE_ARCHER: self.archer_cost_water,
            TYPE_LCAV: self.lcav_cost_water, TYPE_HEAVY: self.heavy_cost_water,
            TYPE_MAGE: self.mage_cost_water, TYPE_HEALER: self.healer_cost_water,
            TYPE_RAM: self.ram_cost_water,
            TYPE_CATAPULT: self.catapult_cost_water,
            TYPE_AIRSHIP: self.airship_cost_water,
            TYPE_DRAGON: self.dragon_cost_water})

    @property
    def train_time_by_type(self) -> tuple:
        return self._t32({
            TYPE_WORKER: self.worker_time, TYPE_INFANTRY: self.infantry_time,
            TYPE_DOG: self.dog_time, TYPE_STRONGMAN: self.strongman_time,
            TYPE_WAGON: self.wagon_time, TYPE_ARCHER: self.archer_time,
            TYPE_LCAV: self.lcav_time, TYPE_HEAVY: self.heavy_time,
            TYPE_MAGE: self.mage_time, TYPE_HEALER: self.healer_time,
            TYPE_RAM: self.ram_time, TYPE_CATAPULT: self.catapult_time,
            TYPE_AIRSHIP: self.airship_time, TYPE_DRAGON: self.dragon_time})

    @property
    def carry_by_type(self) -> tuple:
        return self._t32({TYPE_WORKER: self.worker_carry,
                          TYPE_STRONGMAN: self.strongman_carry,
                          TYPE_WAGON: self.wagon_carry})

    @property
    def mine_time_by_type(self) -> tuple:
        return self._t32({TYPE_WORKER: self.worker_mine_time,
                          TYPE_STRONGMAN: self.strongman_mine_time,
                          TYPE_WAGON: self.wagon_mine_time})

    @property
    def armor_by_type(self) -> tuple:
        """物理减伤百分比(0-100);魔法伤害无视本表。"""
        return self._t32({
            TYPE_HQ: self.hq_armor, TYPE_INFANTRY: self.infantry_armor,
            TYPE_TOWER: self.tower_armor, TYPE_WAGON: self.wagon_armor,
            TYPE_LCAV: self.lcav_armor, TYPE_HEAVY: self.heavy_armor,
            TYPE_RAM: self.ram_armor, TYPE_MORTAR: self.mortar_armor,
            TYPE_FENCE_STONE: self.fence_stone_armor,
            TYPE_FENCE_IRON: self.fence_iron_armor,
            TYPE_FLAMER: self.flamer_armor, TYPE_LASER: self.laser_armor,
            TYPE_CATAPULT: self.catapult_armor,
            TYPE_AIRSHIP: self.airship_armor, TYPE_DRAGON: self.dragon_armor})

    @property
    def dmg_magic_by_type(self) -> tuple:
        """该类型攻击是否魔法(1=魔法,无视护甲)。"""
        return self._t32({TYPE_MAGE: 1, TYPE_MAGETOWER: 1, TYPE_LASER: 1})

    @property
    def atk_range_by_type(self) -> tuple:
        """攻击(奶妈=治疗)射程;0=无攻击能力。近战一律 melee_range。"""
        m = self.melee_range
        return self._t32({
            TYPE_INFANTRY: m, TYPE_DOG: m, TYPE_LCAV: m, TYPE_HEAVY: m,
            TYPE_RAM: m, TYPE_TOWER: self.tower_range,
            TYPE_ARCHER: self.archer_range, TYPE_MAGE: self.mage_range,
            TYPE_HEALER: self.healer_range, TYPE_MORTAR: self.mortar_range,
            TYPE_MAGETOWER: self.magetower_range, TYPE_LASER: self.laser_range,
            TYPE_CATAPULT: self.catapult_range,
            TYPE_DRAGON: self.dragon_air_range},
            default=0.0)

    @property
    def atk_min_range_by_type(self) -> tuple:
        """最小射程(盲区,严格小于此距离打不了);仅迫击炮非零。"""
        return self._t32({TYPE_MORTAR: self.mortar_min_range}, default=0.0)

    @property
    def can_hit_units_by_type(self) -> tuple:
        """可打**地面**单位(v1.6 起语义收窄;空中目标看 can_hit_air)。"""
        return self._t32({TYPE_INFANTRY: 1, TYPE_DOG: 1, TYPE_ARCHER: 1,
                          TYPE_LCAV: 1, TYPE_HEAVY: 1, TYPE_MAGE: 1,
                          TYPE_TOWER: 1, TYPE_MORTAR: 1, TYPE_MAGETOWER: 1,
                          TYPE_LASER: 1, TYPE_CATAPULT: 1})

    @property
    def can_hit_buildings_by_type(self) -> tuple:
        """攻城车只打建筑;塔/迫击炮只打单位(v1.2 规格沿袭)。"""
        return self._t32({TYPE_INFANTRY: 1, TYPE_DOG: 1, TYPE_ARCHER: 1,
                          TYPE_LCAV: 1, TYPE_HEAVY: 1, TYPE_MAGE: 1,
                          TYPE_RAM: 1})

    @property
    def atk_period_by_type(self) -> tuple:
        """攻击间隔 tick(1=每 tick);炮击类为长间隔。"""
        return self._t32({TYPE_MORTAR: self.mortar_atk_period,
                          TYPE_MAGETOWER: self.magetower_period,
                          TYPE_CATAPULT: self.catapult_period,
                          TYPE_DRAGON: self.dragon_period}, default=1)

    @property
    def aoe_radius_by_type(self) -> tuple:
        """弹着点范围伤害半径(0=单体);迫击炮/投石车(自心圆另见
        self_aoe_radius_by_type)。"""
        return self._t32({TYPE_MORTAR: self.mortar_aoe_radius,
                          TYPE_CATAPULT: self.catapult_aoe_radius}, default=0.0)

    @property
    def build_cap_by_type(self) -> tuple:
        """每玩家该类型建筑数量上限;0=本表不设限(哨塔另查 tower_cap_by_hq_level,
        兵营沿用 max_barracks)。迫击炮限 1(规格:除哨塔/地雷外每种防御建筑限 1;
        v1.6 防御建筑与地雷=5 直接进此表)。"""
        return self._t32({TYPE_MORTAR: 1, TYPE_MAGETOWER: 1, TYPE_FLAMER: 1,
                          TYPE_LASER: 1, TYPE_LANDMINE: 5})

    @property
    def is_air_by_type(self) -> tuple:
        """空中单位(v1.6):无视地面障碍直线飞,只被可对空者攻击。"""
        return self._t32({TYPE_AIRSHIP: 1, TYPE_DRAGON: 1})

    @property
    def can_hit_air_by_type(self) -> tuple:
        """可攻击空中单位(规格对空表):弓/法师/哨塔/法师塔/激光炮/龙。"""
        return self._t32({TYPE_ARCHER: 1, TYPE_MAGE: 1, TYPE_TOWER: 1,
                          TYPE_MAGETOWER: 1, TYPE_LASER: 1, TYPE_DRAGON: 1})

    @property
    def is_combat_by_type(self) -> tuple:
        """战斗单位判据(v1.6 拆分,plan D0/D7):八线兵种 ∪ {投石车,龙};
        飞艇不算(无攻击,只许 MOVE/STOP/空降)。"""
        m = {t: 1 for t in TYPE_OF_LINE}
        m[TYPE_CATAPULT] = 1
        m[TYPE_DRAGON] = 1
        return self._t32(m)

    @property
    def shell_flight_by_type(self) -> tuple:
        """弹道飞行 tick(>0 即弹道攻击者:锁点-飞行-落点 AoE);
        迫击炮/投石车。"""
        return self._t32({TYPE_MORTAR: self.mortar_flight_time,
                          TYPE_CATAPULT: self.catapult_flight_time})

    @property
    def self_aoe_radius_by_type(self) -> tuple:
        """自心圆持续 AoE 半径(平坦无衰减,period 节拍):喷火器/龙喷火,
        只伤敌方地面单位。"""
        return self._t32({TYPE_FLAMER: self.flamer_radius,
                          TYPE_DRAGON: self.dragon_breath_radius}, default=0.0)

    @property
    def line_cap_by_line(self) -> tuple:
        """各线等级硬上限(v1.6 修订:投石车/飞艇/龙线压到 3 防数值膨胀,
        其余 7;实际上限 = min(此值, 训练营等级))。"""
        return (7,) * N_LINES_LEGACY + (3,) * (N_LINES - N_LINES_LEGACY)

    @property
    def line_of_type(self) -> tuple:
        """TYPE → 升级线;-1=无线(采集单位/建筑)。同时充当 is_combat 判据。"""
        m = {t: line for line, t in enumerate(TYPE_OF_LINE)}
        return self._t32(m, default=-1)
