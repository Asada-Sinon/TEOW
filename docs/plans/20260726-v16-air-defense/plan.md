# TEOW v1.6 防御建筑群 + 空中域 — 实施计划

(Plan agent 产出,行号按 v1.5 HEAD 亲核;plan-critic 意见文末追记)

## 关键设计决策(全部记 DECISIONS [AI-DRAFT])

**D0 三新兵种不开升级线(N_LINES 保持 8)。** 硬约束:`a_build_fence` id =
`_v14_base+9+N_LINES+idx`——扩线会使 v1.5 栅栏动作 id 位移,违反保号契约、
v1.5 审计轨迹全失效;且 upgrades [P,8] 形状变更连带 npz/前端/测试。定性:
三者是攻城车级超级单位,规格未要求新线;v1.7 要开线时以 `N_LINES_LEGACY=8`
固化旧公式、在 v1.6 块后追加新研发 id。代价:需新增 `is_combat_by_type` 表
(D7)。

**D1 空中域。** `is_air_by_type`(飞艇/龙),`can_hit_air_by_type`(弓/法师/
哨塔/法师塔/激光炮/龙)。目标合法唯一公式(combat 与 movement 停步共用,
提炼 stats 单函数防漂移):
`valid(i→j) = is_air[j] ? can_hit_air[i] : (is_unit[j] ? hit_u[i] : hit_b[i])`。
空中移动:受静态六边形边界约束(界外死区无意义),**无视 building_cells**;
方向不走场、直线冲目标(六边形凸集,连线不出界);互推分组 air-air/ground-
ground,空地零碰撞;不进 occupancy_grid 与动态场软障碍。

**D2 飞艇容器。** state 新增 `aboard: int16[N]`(载具槽,-1 无;不复用
inside——那是采集相位机)与 `reboard_lock: int16[N]`(开火置 60,递减,
==0 才可上艇)。上艇=单动作 A_BOARD:reach 内最近己方有空位艇,即时;门=
地面战斗单位(不含采集单位)+ lock==0 + **威胁禁区**(任意敌方 atk_range>0
实体的射程圆内禁止)+ 容量 7(同 tick 超发按 rank 仲裁);登艇清 order/
target/garrison。空降=艇侧单动作 A_DROP_ALL,全员落艇位(暂时叠格互推散开)。
舱内乘员 pos 每 tick snap 到艇位;击落=乘员连锁死(cleanup gather 一层);
不提供自动走靠。

**D3 龙骑兵双攻。** 模式自动选、空中优先(射程内有敌空军→单体高物理打最近;
否则喷吐半径内有敌地面单位→喷火);喷火=以龙为圆心平坦 AoE,只伤敌方地面
**单位**不伤建筑(对齐喷火器措辞与迫击炮溅射先例;龙不能独自拆家,攻坚交给
飞艇空投——空军克军队、地面克建筑);两模式 fired 共享 atk_cd 单字段,
天然「不能同时」。表位 hit_u=0(地面伤害只走喷火 pass)。

**D4 地雷。** TYPE_LANDMINE 走自由格管线(短建造 25t),HQ4,cap 5;
**不挡路**(building_cells 显式排除——挡路=5 枚不可拆的廉价永久栅栏,超模
死锁;「绕道」的语义是绕开触发圈)、**不可被打**(targetable 排除)、可见、
保留在 occupancy_grid(防同格双实体)。触发:建成雷,敌方地面单位进
trigger_radius → 当拍以雷为圆心线性衰减爆炸(复用迫击炮 fall 公式),只伤
敌方地面单位、自毁(incoming 加自身 hp);多雷独立叠加;在建不触发。

**D5 喷火器/激光炮。** 喷火器:period=1 自心圆平坦 AoE(半径小无衰减),
rng=0 防双重结算,与龙喷火共用数据通路。激光炮:完全走通用单体路径
(rng/period=1/magic/对空),零新机制。法师塔同路径(period 5 中频)。
三者 hit_b=0(防御建筑只打单位先例)。

**D6 前端。** TYPE_NAMES 22..28 + 矢量;空军椭圆阴影+上移 0.25 格、z 最高;
server 帧加 aboard;building_types 由 speed 表自动收编。

**D7 is_combat 拆分。** `is_combat_by_type` = line≥0 ∪ {投石车,龙};飞艇
不算(无攻击,只给 MOVE/STOP/DROP);movable = is_harvester|is_combat|飞艇。
消费三处:actions is_inf、controller is_army、A_STOP/MOVE 放行集合。

## 分阶段

**P1 空中域基建+对空表+投石车**:7 个 TYPE 一次占号(22 法师塔/23 地雷/
24 喷火器/25 激光炮/26 投石车/27 飞艇/28 龙);is_air/can_hit_air/is_combat/
shell_flight_by_type(迫击炮 8、投石车 4,弹道机制表化,is_mortar 泛化为
flight>0);movement 空中全套;投石车训练接线(三新兵种合法门全接,等级门
自然挡);test_air_domain + test_catapult。
**P2 四座防御建筑**:BTASK -11..-14;structs/grow_specs 各 +4;自心圆 AoE
pass + 地雷触发 pass;building_cells 排雷;test_defense_buildings +
test_landmine。
**P3 飞艇容器**(唯一 state 形状变更):aboard/reboard_lock;A_BOARD/
A_DROP_ALL;on_board 口径统一提炼 stats.on_board_of(combat/movement/
occupancy/legality actable 五处,防口径分叉);test_airship。
**P4 龙骑兵**:双模式+停步特判;test_dragon。
**P5 脚本 AI+前端+README+收尾**:防御建筑建造链(互斥顺延);bar_types 追加
三兵种(caps 99/1/1);is_army 换表;登艇/空降 scripted 不用(手术验证);
test_scripted_v16(富开局 target 7);审计脚本追加(地雷一次性守恒/乘员随艇/
空军不入 building_cells);五件套。

## 数值草案(全部 [AI-DRAFT])
投石车 90/50/130t/hp100/甲30/速0.3/rng5.0/aoe1.2/atk24/period30/flight4;
法师塔 70/50/建100/hp140/rng4.5/atk14/period5/魔法/对空;
地雷 30/10/建25/trigger1.0/aoe2.0/atk50/cap5;
喷火器 80/50/建100/hp160/甲20/radius2.0/atk3/period1;
激光炮 120/90/建130/hp150/甲20/rng5.0/atk5/period1/魔法/对空;
飞艇 120/80/150t/hp140/甲10/速0.7/载7/reboard_lockout60;
龙 180/140/220t/hp220/甲30/空攻45/喷火12/rng3.0/breath2.5/period10;
train_level:投石车6/飞艇6/龙7。

## 规格解释八条(记 DECISIONS)
①三兵种不开线 ②龙喷火不伤建筑 ③投石车只伤单位/无盲区/短弹道 ④法师塔/
激光炮 hit_b=0 ⑤地雷不挡路/占落位格/在建不触发/可见 ⑥登艇限战斗单位/reach
内/无自动走靠/DROP_ALL 单动作/登艇清指令 ⑦威胁禁区=敌 atk_range>0 实体射程圆
(不含地雷触发圈) ⑧空军直线转向/受静态边界/空地零碰撞/不占格。
