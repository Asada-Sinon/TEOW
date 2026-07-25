# TEOW v1.4 实现计划(定稿:Plan agent 两轮产出 + plan-critic 审查合并)

规格依据:issue.md v1.4 节(48d5c95)。critic 裁定:无 BLOCKER;M-1(config 表
顺序倒挂)与 M-2(在建塔开火门)已按建议吸收进下文;三条 MINOR 已落实
(第三处 clip 拷贝入清单、采集单位动作门与其行为同 phase 落地、奶妈不奶建筑
记 DECISIONS)。

## 设计决策(全部经 critic 核对代码假设成立)

- **D1 类型与表**:TYPE_STRONGMAN=10, WAGON=11, ARCHER=12, LCAV=13, HEAVY=14,
  MAGE=15, HEALER=16, RAM=17, MORTAR=18;所有 per-type 表 16→32;clip [0,15]→
  [0,31] 共三处:movement.py:82、movement.py:99、tests/test_movement_continuous.py:65。
  TYPE_INFANTRY=5 保号,「近战步兵」仅文档/精灵改名。
- **D2 护甲**:`armor_by_type`(32,物理减伤%)、`dmg_magic_by_type`(32);
  唯一公式助手 `physical_damage(atk, armor) = max(1, (atk*(100-armor)+99)//100)`,
  魔法绕过;测试一律经同一助手算期望,不手写 ceil 字面量。
- **D3 Max-HP 派生不入 state**:新模块 stats.py 从 Config 组装静态 [32,8]
  `TYPE_HP_TABLE`/`TYPE_ATK_TABLE`;`max_hp_of` 有效等级=建筑 state.level、
  战斗单位 upgrades[owner, line_of_type]、采集单位恒 1。
- **D4 升级线** `upgrades[2,8]`:LINE_INFANTRY=0, DOG=1, ARCHER=2, LCAV=3,
  HEAVY=4, MAGE=5, HEALER=6, RAM=7;`line_of_type`(32,-1=非战斗)同时充当
  is_combat 判据(替换 actions.py:167 的 is_inf)。研发门=营建成空闲 & 线<营级
  & 付得起 & 同线无并研 & **兵种已解锁**(拥有 level≥train_level_by_type[t] 的
  建成生产建筑;步兵线恒解锁)。研发成本共用一套 line_res_cost_ore/water/time
  (记 DECISIONS [AI-DRAFT],per-line 差异化留 v1.7)。
- **D5 迫击炮弹逐实体字段**:shell_timer int16[N]、shell_target f32[N,2]、
  atk_cd int16[N](通用攻击冷却,`atk_period_by_type` 普通攻击者=1 行为不变)。
  约束 `mortar_flight_time < mortar_atk_period`(测试断言),单弹槽恒够。
  DECISIONS [AI-DRAFT]:炮死弹消;AoE 只伤敌方地面单位(不溅建筑、无友伤);
  v1.4 迫击炮不可升级。衰减:`base = ceil(atk*(R-d)/R)` 再过护甲。
- **D6 通用建筑数量 cap**:`tower_cap_by_hq_level=(0,0,1,1,2,2,3,3)`(用户确认)
  + `build_cap_by_type`(32,0=不限;迫击炮=1)。计数含在建(alive 即计),
  paid_orders_pass 每 tick 每玩家每种至多批一座 ⇒ 掩码时 <cap 则终态 ≤cap。
- **D7 战斗改造**:per-type 表(atk_range/atk_min_range/can_hit_units/
  can_hit_buildings/aoe_radius/atk_period)驱动统一 masked-argmin 单体 pass;
  is_building 判据统一为 speed==0。**M-2 修正:建筑类攻击者(塔/迫击炮)保留
  逐实体 btype==0 建成门,can-hit 矩阵只管类型维;「在建塔不开火」断言原样保留,
  不许随 test_tower 重写。** 治疗 pass:射程内己方在场单位 hp/max_hp 最低者
  (整数缩放比值+槽号平手,确定性),scatter-add heal_by_level,
  `hp = min(max_hp, hp - incoming + healing)` 同时结算(奶得活将死者)。
  奶妈不奶建筑(规格「友军」收窄,记 DECISIONS [AI-DRAFT])。全程无 RNG。
- **D8 生产轮数**:range(3) → range(1 + cfg.max_barracks)(economy.py:361)。
- **D9 付费路径沿现分野**:HQ 系(工人/步兵/大力士/马车)apply_orders 即扣
  (每家一座 HQ 安全);兵营系入 dog 的 cumsum 对账,由
  `train_cost/time_by_type` 32-表驱动(吸收现有 worker/infantry/dog 标量,
  changelog 记「结构迁移」非平衡)。
- **D10 动作 id 只追加**:旧 a_research 两 id(11/12+2Nn)永久掩 False 保号;
  尾部追加(B=18+3Nn+2F=48 起):B+0..B+7 八兵种训练(大力士/马车/弓箭手/
  轻骑/重甲/法师/奶妈/攻城车)、B+8 建迫击炮、B+9..B+16 八线研发。n_actions=65。
- **D11 BTASK**:BTASK_BUILD_MORTAR=-7;研发 -(16+line)(−16..−23,int8 安全);
  −2/−3 退役。
- **D14 attack-move 停步条件能力化**:movement.py:135-139 改「我的射程内存在
  我能打的目标」(stats 共享助手)——否则攻城车在打不动的单位旁死锁、
  弓手法师贴脸。

## Phase 1 — 引擎核心泛化(最大一刀,critic M-1:全部 config 表在此落齐)

文件:config.py、stats.py(新)、state.py、combat.py、economy.py、actions.py、
movement.py、controller.py(保绿最小)、tests。
- config:TYPE 10-18;全部 32-表落齐——speed、unlock_level、armor、dmg_magic、
  atk_range、atk_min_range、can_hit_units、can_hit_buildings、aoe_radius、
  atk_period、train_level_by_type、train_cost/water/time_by_type、
  harvest_carry_by_type、harvest_mine_time_by_type、line_of_type;八条线的
  hp/atk(healer 为 hp+heal)8-元组;line_res_* 共用研发表(初值沿用旧 inf_res_*);
  删 worker_atk / worker_*_by_level / worker_res_* / inf_res_*,加 worker_hp=20、
  strongman/wagon 参数。LINE_* 8 常量,BTASK 研发块 −16..−23,删 −2/−3。
  数值草案(全 [AI-DRAFT]):armor——步兵 10/轻骑 20/重甲 60/攻城车 40/马车 10/
  HQ 10/塔 20/迫击炮 20,其余 0;线表 1 级锚点 archer 30hp/5atk r3.5、
  lcav 45/6、heavy 60/5、mage 25/7 r3.0(魔法)、healer 25hp+heal 3 r3.0、
  ram 120hp/25atk(只打建筑);速度 strongman 0.5、wagon 0.9、archer 0.5、
  lcav 0.9、heavy 0.4、mage 0.45、healer 0.45、ram 0.3;成本(ore/water/time)
  strongman 40/10/60、wagon 60/30/80、archer 30/15/50、lcav 45/20/70、
  heavy 50/25/80、mage 40/40/90、healer 30/40/90、ram 80/40/120。
- stats.py:hp_table/atk_table [32,8]、max_hp_of、atk_of、physical_damage、
  is_building(speed==0)/is_combat(line≥0)/is_harvester、has_target_in_range。
- state:upgrades → [2,8];新增 atk_cd/shell_timer int16[N]、shell_target
  f32[N,2](P3 用,pytree 一次改齐,少折腾一轮测试);工人 hp 用 worker_hp。
- combat 重写:统一 pass + 护甲 + cd;**建筑攻击者保留 btype==0 门(M-2)**;
  工人攻击在此移除;cleanup 停泊新字段。
- economy:研发完成 8 线向量化补血差(hp_table gather,狗/步兵共享 workaround
  死亡);paid_orders_pass 研发去重+cumsum 泛化到 8 线;_unit_spawn_hp → stats;
  harvest 读 per-type 表。
- actions:a_research_line(l) 尾部追加,旧 id 掩死;研发合法门按 D4。
- movement:三处 clip → 31;D14。
- controller:LINE_WORKER 引用清除,camp 研发改「未及营级的已解锁线取最低」。
- 测试:新 test_armor.py(公式/高甲/魔法穿甲/工人零输出/在建塔不开火保留断言);
  重写 test_camp.py(新 id、8 线、解锁门、去重、被拆保留);test_barracks 狗线
  断言反转;test_combat_win/test_tower 伤害断言过 physical_damage 助手。

## Phase 2 — 多哨塔 + 通用 cap

- config:tower_cap_by_hq_level(用户确认值)、build_cap_by_type。
- actions 塔 legality 加计数门(count_of_type 助手);controller 塔分支
  ~has_tower → n<cap。
- 新测试 test_multi_tower.py:HQ1 掩死/HQ2-3 一座/HQ4 二座/在建计数。

## Phase 3 — 迫击炮

- config [AI-DRAFT]:cost 80/60、build 110、hp 150、atk 30、range 7.0、
  min 2.5、aoe 1.5、period 40、flight 8(断言 flight<period)、unlock 3、cap 1。
- actions a_build_mortar;paid structs 第 4 行——**先把三段复制粘贴的在建成长
  块重构成 struct 描述符循环再加第 4 个**;combat 开火/飞行/落地(共享 incoming,
  同 tick 同时结算);cleanup 停泊。
- 新测试 test_mortar.py:解锁/cap-1/盲区/飞行延迟/锁点走位躲弹/衰减单调/
  不溅建筑/无友伤/冷却间隔/炮死弹消。

## Phase 4 — 兵种树:兵营升级 + 8 训练动作 + 奶妈/攻城车行为

- config:barracks_hp_by_level(1级=旧标量,删标量)、barracks_up_cost_*/time。
- actions:HQ 双训练(即扣路径,门 hq_lv≥train_level);兵营六训练(cumsum
  路径,门 兵营.level≥train_level);兵营入 a_upgrade 掩码。
- economy:upgrade_cost/time_of 加兵营;升级完成补血差;兵营训练付费入 cumsum;
  生产轮数派生。
- combat:治疗 pass 接线;ram 经 can-hit 表生效;验证 D14 覆盖 ram/archer/mage。
- 新测试 test_unit_tree.py(等级门/出生血量/兵营升级链)、test_ranged.py、
  test_healer.py、test_ram.py。
- 风险:unit_costs 变形的两处调用点(actions.py:210,422);同 tick HQ+兵营
  完成不覆写(test_barracks:57 保绿)。

## Phase 5 — 采集单位线(动作门与行为同 phase,critic MINOR-2)

- actions:is_worker 门 → is_harvester(STOP/MOVE/BUILD/HARVEST,
  actions.py:174,184,195-202,244-260);驻守/插旗保持 is_combat(采集单位
  天然被排除)。
- 验证名额口径 type-agnostic(economy.py:59-66 门在 order 上,成立)。
- controller:工人分支放宽到 harvester;idle 让路清单加新类型。
- 新测试 test_harvesters.py:三角关系产量/耗时、三类采集单位对相邻敌零伤害
  (工人回归!)、驻守/插旗/ATTACK 掩死、能建造。

## Phase 6 — 脚本 AI 全扩展(审计覆盖 phase)

- ai_base_level_target → 5 [AI-DRAFT];HQ 出大力士(HQ3,≤2)与马车(HQ5,1);
  迫击炮建造分支(克隆塔分支,HQ≥3);兵营向 HQ 级升级;兵营按「缺额最小」
  训已解锁兵种(healer/ram 计数封顶);camp 研最低已解锁线;attack 阈值计
  全战斗类(line_of_type≥0);预留算式含新训练成本(controller.py:153-177,
  防研发饿死——历史 bug 同款)。
- 新测试 test_scripted_v14.py(事件旗:迫击炮建成且开过火、兵营≥4 级、
  archer/heavy/mage/healer 各≥1、某线≥2、采集单位全场零输出)。
- 风险:test_scripted_upgrades/v13 时间线敏感,预算一轮再平衡。

## Phase 7 — 前端 + README + 版本收尾

- sprites.js TYPE_NAMES + 9 矢量 fallback(命名与 fig/ 映射对齐:strongman/
  wagon/archer/cavalry/heavy/mage/healer/ram/mortar);render.js 删 MAXHP
  硬编码改读服务端 mx(顺手修 config 单真源违例);render.py 9 marker,
  建筑判定改 speed 约定;server.py 帧加 mx。
- README「科技树」节(中文按建筑组织)+ 护甲/迫击炮/采集线入玩家手册。
- 收尾五件套:全功能脚本对决 → engine-auditor → changelog(平衡区:worker_atk
  移除/狗线拆分/研发成本合并)→ DECISIONS(炮死弹消/敌方限定 AoE/不可升级/
  奶妈不奶建筑/共用研发表)→ tag v1.4 → push → handoff。

## 预期破坏测试(全清单经 critic 核对)

test_camp 全量(P1 重写)、test_barracks 狗线断言(P1)、test_combat_win/
test_tower 伤害字面量(P1 过助手;在建塔断言原样保留)、test_economy/
test_movement_continuous(worker 槽=旧 1 级值则绿,守住)、test_scripted_*
(P6 事件旗化)、test_upgrade 资源非负=全程金丝雀(红了=对账 bug)。

每 phase 门禁:JAX_PLATFORMS=cpu pytest -q + ruff 全绿;一 phase 一 commit。
