# Research: v1.3 采集名额制 + 双角公共点 + 驻守/军旗 + 哨塔平衡

调研方式:4 个并行 Explore subagent(采集循环 / 动作表与 n_nodes 影响面 /
移动寻路与驻守可行性 / 实验设施与前端),以下为回报汇总,断言均带 `文件:行号`。

## 1. 相关文件位置

### 采集循环与槽位
- `src/teow/economy.py:509-574` `harvest_tick`:入驻抢槽(520-536)、矿内计时
  (540-542)、出矿(548-559)、卸货入账(561-574)。
- `src/teow/economy.py:58-62` `inside_counts`:按 `state.inside` 计数,
  **唯一调用方**是 harvest_tick 的入驻仲裁(economy.py:523)。
- `src/teow/economy.py:533` 容量硬约束唯一检查点:
  `enter = cand & (counts[k] + rank < cfg.node_capacity)`。
- `src/teow/actions.py:159-160` HARV_k 合法条件:存活、不在矿内、是工人、
  点为己方已建;**不查容量**(actions.py:124-125 注明动态争用顺延不算非法)。
- `src/teow/controller.py:77-87` scripted 派工软门控:已用「驻内+在途」的
  指派口径计数(controller.py:78-79),与新语义天然一致。
- `state.target_node` 生命周期:赋值 `actions.py:280-281`;除死亡停泊
  (combat.py:135)外**从不清成 -1**,失效靠 order 门控软失效
  (combat.py:105-109 统一转 IDLE,只改 order 不清 target_node)。
  所有消费都门控在 `order==ORDER_HARVEST`(economy.py:515),残值无害。
- 经济线查表消费点:`worker_mine_time_by_level` 仅 economy.py:526,
  `worker_carry_by_level` 仅 economy.py:550。

### 地图与动作表
- `src/teow/map.py:68-143` `build_map`:资源点定义 90-95(近家 ore0=(2,8)/
  water0=(8,2) 缩放,公共矿 (11,12) 居中),`n_nodes != len(list)` 检查 96-97,
  唯一性检查 104-106,距离场 goals 134-139。旋转 `_rot` 46-48,缩放 80-83。
- `src/teow/actions.py:48-104` 动作 id 全部由 `cfg.n_nodes` 派生
  (`n_actions = 16 + 2*Nn`,actions.py:103-104);n_nodes 6→8 自动扩展,
  掩码逐点循环(155-160)与 apply 区间判定(263-264)均自适应。
  **无运行代码硬编码动作数**;过时注释:actions.py:7、actions.py:152、
  economy.py:4(写着 Nn=6,仅文档)。
- 新动作追加位置:`a_build_tower`(actions.py:98-100)之后按序追加,
  n_actions 常数随之改(actions.py:104);**不得插中间**打乱既有 id。
- `src/teow/controller.py:37-45` random 全靠 legality_mask 采样,零 id 依赖;
  scripted(48-202)全部走 `a_*()` 辅助函数,n_nodes=8 自动正确;
  新动作若要 scripted 用,需加决策分支(不加也不报错)。
- 会被 n_nodes=8 打破的测试:仅 `tests/test_map.py` 两个用例——
  `test_shapes_and_types`(20-21 写死 6 个类型)与 `test_rotational_symmetry`
  (30-33 硬编下标 4/5 中央公共点)。其余测试自适应或不触碰节点几何。

### 移动、指令状态机与驻守缺口
- ORDER 全集 `state.py:22-27`(IDLE/HARVEST/BUILD/MOVE/ATTACK);
  phase 常量 30-32。
- 寻路:**每 tick 用 min-plus 松弛全量重算距离场**,`_relax_fields`
  (movement.py:38-57),调用 movement.py:142;goal_seeds 静态闭包在 mapdata
  (map.py:137-139);静态 BFS 场 `MapData.dist_fields` 是遗留死代码,
  movement 不用。软障碍 `cell_cost = 1 + stationary_cost*occ_stat`
  (movement.py:141),移动单位刻意不计入(movement.py:137-140)。
- 梯度采样 `_bilinear`(movement.py:60-74)+ 中心差分(147-156)。
- ORDER_MOVE 直指 target_cell 走直线(movement.py:157-160),
  到达判定 `eu<=0.4` 两处硬编码(movement.py:117、213),到达转 IDLE
  (movement.py:211-214)。ORDER_ATTACK 走敌方 HQ 场(movement.py:105),
  途中遇敌=「射程内有敌判 arrived 停下」(movement.py:119-123)。
- 战斗完全按位置不看 order(combat.py:30-68):驻守(静止)单位自然参战;
  塔射程 combat.py:44、只打单位 combat.py:47、目标偏好 51-54。
- **驻守缺口**:互推(movement.py:183-209)会把静止单位推走,且 IDLE 无
  「回到锚点」语义(全库无 anchor 概念);直线+分轴滑动(movement.py:166-181)
  对凹形障碍会卡死(map.py:8-9 头注释正是为此改 flow-field)。
- 单 tick 结算顺序(step.py:63-72):production → special_tasks →
  construction → harvest → apply_orders → paid_orders_pass →
  start_constructions → movement → combat → cleanup → _end_tick。

### 状态表与军旗落点
- WorldState 字段全集 `state.py:35-69`;定容先例:资源点并行数组
  [Nn](state.py:59-62,-1 哨兵);二维 per-player 先例 `resources/upgrades`
  [2,…](state.py:64-65)。NamedTuple 自动 pytree,新字段自动进
  scan/jit/终局冻结(step.py:74-75)。
- 初始化 `init_state`(state.py:88-138),哨兵示例 state.py:118、128。
- 同 tick 多笔落格/占槽的仲裁先例:`economy.paid_orders_pass`
  (economy.py:98-237,自由格建筑 cumsum 对账 185-237)。

### 实验设施与前端
- `src/run.py`:play/replay/serve/bench 四子命令(183-218);provenance
  `write_provenance`(87-102,resolved_config.json/seed/backend/command/git);
  `--set FIELD=VALUE` config 覆盖(cmd_play 109-160);trajectory.npz
  由 `state_to_numpy = st._asdict()`(105-106)逐帧堆叠——**state 新增字段
  自动进 npz**。
- 塔攻击**无冷却概念**,每 tick 打满(combat.py:30-68);加攻击间隔需新
  int16[N] 字段(btimer 被升级占用有冲突,combat 调研与 actions.py:186-190)。
- 审计脚本模式:`explorations/audit_v12_conservation_invariants.py:11-33`
  从 run 目录反读 resolved_config 重建世界逐位重放;必须 JAX_PLATFORMS=cpu。
- scripted 无法 config 成「只造狗骚扰」:建造链硬编码(controller.py:48-202),
  ai_* 参数仅 4 个(config.py:170-177);骚扰场景需手工建局脚本
  (test_tower.py:35-71 的外科手术摆位模式)或新 controller
  (make_controller 注册,controller.py:205-214)。
- 前端:server.py `load_replay`(28-80)出 meta+frames,实体字段 54-79;
  web/sprites.js `drawSprite`(28-141)按 etype 分发,军旗非实体不走它,
  需独立 `drawFlag`;web/render.js `draw`(54-119),node 绘制模式 70-82
  可仿;matplotlib 在 render.py `_draw_frame`(29-97),资源点循环 46-49。
- 测试组织:一机制一文件(12 文件 33 用例);完整对局先例
  `test_determinism.py:10-29`(make_scan 300 tick rollout);
  跨文件复用 helper(test_tower.py:5-6 import setup_barracks/RICH)。

## 2. 数据如何流动

- **采集占用**:HARV_k 动作 → apply_orders 写 order/target_node/phase
  (actions.py:273-292)→ 下一 tick harvest_tick 按 inside_counts 仲裁入驻
  (economy.py:520-536)→ inside 翻位即「占槽」。占用信息分散在
  `order+target_node+inside` 三个字段,没有独立槽位表;
  名额制=把计数口径从 inside 改为 order+target_node,数据流不变。
- **移动意图**:order/phase → goal 与 use_field(movement.py:101-127)→
  距离场松弛(142)→ 梯度采样(147-156)→ 步进+分轴滑动(157-181)→
  圆形互推(183-209)→ 到达清指令(211-214)。驻守要在「goal 选择」和
  「到达不清指令」两处插入。
- **对局产物**:state 逐帧 `_asdict()` → trajectory.npz → render.py /
  server.py / 审计脚本三方消费;新增 state 字段(军旗)自动进 npz,
  但 server.py frames 序列化与两个渲染端要显式取用。

## 3. 关键设计判断(相当于根因假设,均已被调研证实/证伪)

1. **轮转卡 bug 的根因**:占用只在 `inside==True` 期间成立
   (economy.py:533 只查驻内计数),运输段名额空出,故 N>3 工人轮转可垄断
   一个点。→ 证实。修法:计数口径改「指派即占用」。
2. **驻守不能用「MOVE 到点+IDLE」凑**:a) MOVE 只走直线,凹障碍卡死
   (map.py:8-9);b) 到达即转 IDLE,被互推挤走后永不回岗
   (movement.py:183-214)。→ 证实,必须新增 ORDER_GARRISON:
   走距离场、到达不清指令、离锚点超过阈值就重新产生移动意图。
3. **军旗寻路无预计算场是否致命**:否。场本来就每 tick 全量重算
   (movement.py:142),把军旗做成动态 goal 通道(种子从 state 来)
   即可复用同一套松弛;代价是 n_goals 从 10(8 点+2HQ)再加 6
   (2 玩家×3 旗),松弛计算量 +60%,需 bench 确认可接受。[AI-DRAFT:
   计算量比例是按通道数线性估的,实测为准]
4. **塔超模的削弱杠杆**:造价/攻击力/血量是纯 config 改动;攻击间隔要
   新字段+combat 改动。实验先扫 config-only 杠杆,不够再上攻击间隔。

## 4. 既有模式与约束

- **定容纪律**:军旗字段必须静态形状(f32[2,3,2] + bool[2,3]),
  -1/False 哨兵,绝不 resize(state.py:59-65 先例)。
- **数值唯一真源 config.py**:名额表、旗上限、驻守半径全部进 Config,
  代码零字面量;node_capacity(config.py:56)语义被取代,应删除换新表,
  平衡改动记 changelog。
- **动作表只许尾部追加**(actions.py 布局约定),掩码与 apply 双侧都要加。
- **同 tick 多笔仲裁**:插旗若同 tick 多单位申请,须仿 paid_orders_pass
  的槽号 rank 模式(economy.py:185-237);采集指派同 tick 超发同理
  (economy.py:530-536 的 rank 模式可搬进 apply_orders)。
- **step 顺序契约**(step.py:3-13 头注释)改动要同步注释。
- **决定论口径**:审计逐位重放依赖 split(key,3) 与 run.py 一致
  (HANDOFF 坑);新状态字段自动进重放比对,无需特判。
- **测试/门禁一律 JAX_PLATFORMS=cpu**(MEMORY LEARN:env)。
- **规格约束(issue.md v1.3 节)**:驻守=站桩不巡逻;军旗敌方不可拆、
  免费即时、上限 3 不随等级;撤旗后驻守兵原地转 IDLE;编队不做引擎机制;
  README 中文两部分且数值与 config 同步。
- **待用户裁决的规格细化**:驻守目标「己方建筑」的动作编码——塔/兵营/营
  是自由格实体,逐实体寻址会让动作表爆炸;建议收敛为
  「HQ + 己方矿泵(8 点)+ 3 旗」共 12 个离散目标,营/兵营/塔的防守
  用「在旁边插旗」覆盖。此为对规格字面的收窄,须用户确认。
