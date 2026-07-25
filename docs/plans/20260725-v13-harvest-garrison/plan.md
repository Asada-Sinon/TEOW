# Plan: v1.3 采集名额制 + 双角公共点 + 驻守/军旗 + 哨塔平衡 + README

关联 research: ./research.md
规格依据: issue.md「v1.3」节(commit df348b9)

## 目标

一句话:扩张与守点成为真实博弈——单点采集有名额上限(指派即占用,堵轮转
卡 bug)、公共资源挪到双角、部队可驻守建筑/军旗,哨塔数值经实验重校,
并交付中文 README(项目介绍+玩家手册)。

## 不在范围内

- 兵营自升级内容(继续顺延,v1.2 DECISIONS 不变)。
- 驻守的巡逻/追击/警戒半径——驻守只站桩(规格明文)。
- 编队引擎机制(规格明文:控制器层自理)。
- 塔「攻击间隔」机制**默认不做**:Phase 6 实验若证明 config-only 杠杆
  (造价/攻击力/血量)不够,才追加为独立 phase 并先向用户报告。
- 军旗 PNG 美术与生图提示词包更新(矢量图形即可)。
- v2 RL 相关的一切。

## 待用户确认(已裁决 2026-07-25)

驻守目标动作编码收窄为「己方 HQ + 己方矿泵(按资源点 k)+ 己方 3 面旗」
共 12 个离散目标;营/兵营/塔的防守用「在旁边插旗再驻守旗」覆盖。
理由:营/兵营/塔是自由格实体,逐实体寻址动作表爆炸且与 RL 掩码语义冲突。
**用户已确认同意收窄**(issue.md v1.3 节已同步),Phase 3/4 解除封锁。

## Phase 1: 双角公共点(n_nodes 6→8)

- 改 `src/teow/config.py:55`:`n_nodes: int = 6` → `8`。
- 改 `src/teow/map.py:90-95`:删中央 `pub_ore=(11,12)`,新增左下公共对
  `publ_ore=scale((17,4))`、`publ_water=scale((20,7))`,右上为其旋转像
  (类型保持:`_rot(publ_ore)` 仍是矿、`_rot(publ_water)` 仍是水——
  180° 旋转下 (pos,type) 集合自映射,严格公平,且可顺手删掉
  map.py:10-13「公共点旋转换类型」的旧取舍注释)。
  `node_pos_list` 顺序:近家 4 点下标 0-3 不动,公共 4 点接 4-7。
  坐标初值可调,须过 map.py:104-106 唯一性检查与 77-78 网格下限检查。
- 改 `tests/test_map.py`:
  - `test_shapes_and_types`(13-21):类型断言改 8 项
    `[ORE,WATER,ORE,WATER,ORE,WATER,ORE,WATER]`。
  - `test_rotational_symmetry`(24-36):公共点对称对改为 (4,6)、(5,7),
    且类型相同(不再是换类型对)。
- 顺手改过时注释:`src/teow/actions.py:7`(n_actions 公式)、
  `actions.py:152`、`economy.py:4` 的「Nn=6」。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/pytest -x -q`
- 判据:33 用例全过(test_map.py 两用例按新布局重写后 pass,其余不动)。

## Phase 2: 采集名额制(指派即占用)

- 改 `src/teow/config.py`:
  - 删 `node_capacity`(config.py:56)。
  - 新增 `harvest_slots_by_level: tuple = (0, 3, 3, 4, 4, 5, 5, 6)`
    (矿泵 1 级 3 名额,3 级 4,5 级 5,7 级 6;规格初值,平衡区记 changelog)。
- 改 `src/teow/economy.py:58-62`:`inside_counts` 改名 `assigned_counts`,
  口径改为 `alive & (order==ORDER_HARVEST) & (target_node==k)`
  (残留 target_node 被 order 门控,安全性见 research §1;矿内工人 order
  仍是 HARVEST,自然计入)。
- 改 `src/teow/economy.py:520-536` 入驻段:删除容量再仲裁
  (counts/rank),`enter = cand`——名额已在指派侧保证 ≤cap,驻内数
  不可能超。
- 改 `src/teow/actions.py`:
  - `legality_mask`(159-160):HARV_k 追加
    `assigned_counts(...)[k] < cap_k`,其中
    `cap_k = harvest_slots_by_level[clip(level[node_ent[k]],0,7)]`;
    已指派到 k 的工人自己不被掩(允许重复下同一指令 no-op 化,
    实现:掩码条件对「本人已指派 k」豁免)。
  - `apply_orders`(263-292)加同 tick 超发仲裁:按点循环(Nn=8 编译期
    展开),同 tick 对点 k 的新指派按槽号 rank,
    `keep = cand & (assigned_excl + rank < cap_k)`,其中 assigned_excl
    排除本 tick 收到新指令的实体(从 A 点改派 k 点者同 tick 释放 A 名额);
    仲裁失败者保持原指令不变(动作等效 no-op,与 actions.py:124-125
    的顺延哲学一致)。
- 改 `src/teow/controller.py:77-87`:软门控 `cfg.node_capacity` 改查
  `harvest_slots_by_level`(口径已是指派计数,仅换上限来源)。
- 全库 grep `node_capacity` 清零(tests 里如有引用一并改)。
- 新增测试(加进 `tests/test_economy.py`):
  - `test_harvest_slots_cap`:富开局 4 工人对同一 1 级矿下 HARV,
    第 4 个被仲裁拒绝(order 保持原值),`assigned_counts==3`。
  - `test_harvest_slots_no_rotation_exploit`:3 工人指派后其中 1 个在
    运输段(inside=False),第 4 工人 HARV 掩码为 False(名额不因出矿释放)。
  - `test_harvest_slots_upgrade_expands`:矿升到 3 级后第 4 工人可指派。
  - `test_harvest_slot_released_on_death`:杀掉 1 个已指派工人,
    下 tick 第 4 工人可指派。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/pytest -x -q`
- 判据:新增 4 用例 pass,存量用例数量不减、全过。

## Phase 3: 驻守指令 ORDER_GARRISON(先建筑目标,依赖「待用户确认」)

- 改 `src/teow/state.py`:
  - `state.py:27` 后追加 `ORDER_GARRISON = 5`(驻守:走到锚点站住,
    被推离自动回岗,永不自转 IDLE)。
  - WorldState 新增字段 `garrison_id: jax.Array`(int8[N],-1 无;
    0=己方 HQ,1..Nn=资源点 k=garrison_id-1,Nn+1..Nn+3=旗 j;
    init_state 全 -1,state.py:118 模式)。
- 改 `src/teow/config.py`:新增 `garrison_hold_radius: float = 1.2`
  (离锚点超过此距离才产生回岗移动意图;与 reach_radius 同量级)。
- 改 `src/teow/actions.py`:
  - 追加 `a_garrison_hq(cfg)=16+2*Nn`、`a_garrison_node(k,cfg)=17+2*Nn+k`
    (k=0..Nn-1);`n_actions` 改 `17+3*n_nodes`(actions.py:104)。
  - `legality_mask`:两类驻守动作,条件 = actable & (is_inf|is_dog) &
    目标存在(HQ 恒真;node k 需 `node_owner[k]==owner & node_ent[k]>=0`)。
  - `apply_orders`:写 `order←ORDER_GARRISON`、`garrison_id`、
    `target_cell←锚点格`(HQ 位置 / node_pos[k])。
- 改 `src/teow/movement.py`:
  - goal 选择(101-127):GARRISON 且目标是 HQ/node 时 `use_field=True`,
    goal = 己方 HQ 场(n_nodes+own)或点 k 场(现有 8+2 张场直接覆盖)。
  - `moving_order`(125-128):GARRISON 加入,但仅当
    `eu(pos, target_cell) > cfg.garrison_hold_radius` 时 wants=True
    (到锚点圈内站住;被互推挤出圈,下 tick 自动回岗)。
  - **不**加入 211-214 的到达转 IDLE(MOVE 专属);GARRISON 永不自清。
  - **锚点距离一律用 `eu(pos, target_cell)`**,不用 goal_center
    (critic B-1:movement.py:110-112 的 goal_center 对超出 Nn+2 的 goal
    索引会被 clip 到玩家 1 HQ,arrived/step_len/静止分类三处全吃它;
    GARRISON 的 arrived、step_len 截断、moving/stationary 分类都改成
    对 target_cell 求欧氏距离,场只用来取方向)。
  - 纪律一句(critic m-3):`garrison_id` 的一切消费方必须门控在
    `order==ORDER_GARRISON` 上;同时 A_STOP 在 apply_orders 顺手清
    `garrison_id←-1`(actions.py:274 处),不留残值。
- 改 `src/teow/combat.py:105-119` 失效清理:追加「GARRISON 目标失效」
  分支——目标 node 被拆/易主(`node_ent==-1 | node_owner!=owner`)时
  order←IDLE、garrison_id←-1;死亡停泊分支(122-135)补 garrison_id←-1。
- 改 `src/teow/step.py:3-13` 头注释:结算顺序说明补 GARRISON 语义一句。
- 新增 `tests/test_garrison.py`:
  - `test_garrison_walks_and_holds`:步兵驻守己方矿点,若干 tick 后
    位于 hold_radius 内且 order 仍为 GARRISON。
  - `test_garrison_returns_after_push`:手术把驻守单位挪开 3 格,
    step 若干 tick 后回到圈内(验「回岗」)。
  - `test_garrison_cleared_on_node_lost`:拆掉目标矿,单位转 IDLE、
    garrison_id==-1。
  - `test_garrison_fights_back`:敌狗走进 melee_range,驻守步兵不动窝
    且敌狗掉血(战斗按位置自然生效,combat 无改动即应 pass)。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/pytest -x -q`
- 判据:新增 4 用例 pass;`test_determinism.py` 两用例仍 pass
  (新字段进 scan carry 无形状问题)。

## Phase 4: 军旗(插旗/撤旗/驻守旗)

- 改 `src/teow/state.py`:WorldState 新增
  `flag_pos: f32[2,3,2]`(init -1)、`flag_active: bool[2,3]`(init False)。
- 改 `src/teow/config.py`:新增 `max_flags: int = 3`(每玩家,不随等级)。
- 改 `src/teow/actions.py`:
  - 追加 `a_garrison_flag(j,cfg)=17+3*Nn+j`(j=0..2)、
    `a_plant_flag(cfg)=20+3*Nn`、`a_recall_flag(j,cfg)=21+3*Nn+j`;
    `n_actions` 改 `24+3*n_nodes`(Nn=8 时 48)。
  - `legality_mask`:
    - 插旗:actable & (is_inf|is_dog) & 己方**建成**兵营≥1(计数条件含
      `btype==0`,仿训狗的建成判定 actions.py:218;critic m-2:自由格
      建筑开工瞬间即 alive,在建不算「拥有兵营」)&
      `sum(flag_active[own])<cfg.max_flags` & 脚下格既非资源点格也非
      建筑占用格(passable & 无建筑,复用 economy.py 建筑占格判定口径)。
    - 驻守旗 j:actable & (is_inf|is_dog) & `flag_active[own,j]`。
    - 撤旗 j:是己方**建成**兵营(同上含 btype==0)& `flag_active[own,j]`。
  - `apply_orders`:
    - 插旗:同 tick 多申请按槽号 rank 仲裁
      (`已激活数 + rank < max_flags`,仿 economy.py:185-237 模式;
      免费无扣费,可留在 apply_orders 不进 paid_orders_pass),
      旗落 `cell_of(pos)`,写入最小空旗位。
    - 驻守旗:order←GARRISON、garrison_id←Nn+1+j、
      target_cell←flag_pos[own,j]。
    - 撤旗:`flag_active[own,j]←False`、`flag_pos[own,j]←-1`,并把
      `order==GARRISON & garrison_id==Nn+1+j` 的己方单位 order←IDLE、
      garrison_id←-1(规格:原地转 IDLE,引擎保证无悬空引用)。
      **求值顺序明写**(critic M-2):撤旗对单位的清理在个体 order 覆写
      **之后**求值——同 tick 既被撤旗又收到新指令(如 ATTACK)的单位,
      新指令生效、不被打回 IDLE;只清「覆写后仍为 GARRISON 指向该旗」者。
- 改 `src/teow/movement.py`:
  - 动态旗场:goal_seeds 从静态 [G,H,W] 扩为
    `concat(静态 G=Nn+2 张, 动态 6 张)`,动态种子每 tick 由
    `state.flag_pos/flag_active` scatter 生成(inactive → 全 BIG 场);
    goal 索引:旗 j of player p = `Nn+2 + p*3 + j`。
  - GARRISON 目标为旗时 use_field=True 走对应旗场;target_cell 同步用
    flag_pos(撤旗即失效,由 apply_orders 清理,movement 无需特判)。
- 改 `src/teow/config.py:189-192`:`n_goals` property 改
  `n_nodes + 2 + 6`(注明 6=2 玩家×3 旗动态通道)。
  同步检查 map.py:134-139 goal_seeds 构造只出静态部分,形状注释更新。
- 新增 `tests/test_flag.py`(复用 test_barracks.setup_barracks):
  - `test_plant_requires_barracks`:无兵营插旗掩码 False;有则 True。
  - `test_flag_cap_three`:已 3 面,第 4 次插旗掩码 False。
  - `test_plant_on_node_cell_masked`:站在资源点格上插旗掩码 False。
  - `test_recall_idles_garrisoned`:狗驻守旗 0,撤旗后狗原地 order==IDLE、
    旗位清空、名额可复用(再插成功)。
  - `test_garrison_flag_pathing`:狗驻守 6 格外的旗,若干 tick 后到圈内
    (验动态场真的把单位引过去)。
  - `test_garrison_flag_near_enemy_hq`(critic B-1 回归):旗插在敌方 HQ
    2 格内,驻守单位仍能走到旗圈内停稳(而非在别处被误判到达)。
  - `test_recall_with_same_tick_attack`(critic M-2 回归):同 tick 撤旗
    +对驻守单位下 A_ATTACK,断言该单位 order==ATTACK 且 garrison_id==-1。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/pytest -x -q`
- 判据:新增 7 用例 pass;bench 对比——**开工前先在当前 HEAD 跑**
  `JAX_PLATFORMS=cpu .venv/bin/python src/run.py bench --ticks 200` 落盘
  基线(critic m-1:不得引用 v1.0 时期的记忆数字),Phase 4 完成后同机
  同命令重跑,吞吐回退 <30%(动态场 +6 通道的代价;超了就把旗场松弛
  改成按需门控再测)。

## Phase 5: scripted AI 用上新机制(为收尾对决铺路)

- 改 `src/teow/controller.py:48-202` scripted 追加分支(全 jnp、无 Python
  分支依赖运行时值):
  - 扩张后驻守:拥有兵营且狗子数≥2 时,槽号最小的 2 只狗驻守
    「己方离家最远的已建矿泵点」(a_garrison_node)。
  - 插旗+驻守旗:狗子数≥3 且无激活旗时,第 3 只狗在其当前位置插旗
    (它会在驻守途中/圈内插,位置自然靠前线);此后新狗驻守旗 0
    (a_garrison_flag)。
  - 总攻撤旗:触发 `ai_attack_threshold` 总攻那一 tick,兵营同时下撤旗 0
    (a_recall_flag);全军照旧 A_ATTACK(驻守单位被新指令覆盖,
    撤旗清理与 ATTACK 覆写同 tick 共存,后写者 apply_orders 生效)。
- 新增测试 `tests/test_scripted_v13.py::test_scripted_uses_garrison_and_flag`:
  scripted vs scripted 跑 900 tick(make_scan 骨架,
  test_determinism.py:10-29 模式;critic m-4:600 tick 视野没人量过,
  v1.2 对局 ~730 tick 分胜负,取 900 留余量;若仍假红,先把视野放到
  episode_len 复查时间线再判),断言过程中出现过
  `order==ORDER_GARRISON` 的单位与 `flag_active.any()`。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/pytest -x -q`
- 判据:新用例 pass;全量绿。

## Phase 6: 哨塔平衡实验(config-only 杠杆,数据交用户定案)

- 新增 `explorations/exp_tower_balance_v13.py`(只读 src、产物写
  `experiments/20260725-tower-balance-<variant>/`,provenance 走
  run.py 的 write_provenance 同款字段:git hash + resolved config + seed):
  - 场景 A「狗 rush 微操局」:手工建局(test_tower.py:35-71 摆位模式)
    ——1 塔+3 工人 vs N∈{2,3,4,5} 狗,变体
    {现值, atk 6→4, atk 6→3, cost 50/30→80/50, hp 120→90} 组合扫描,
    记录:塔存活与否、狗存活数、工人伤亡、分出结果的 tick 数。
  - 场景 B「全局对局敏感性」:scripted vs scripted,每变体 8 seeds,
    变体在脚本内用 `dataclasses.replace(Config(), tower_atk_by_level=...)`
    构造(critic M-1:run.py 的 `--set` 解析不支持 tuple 字段,
    `from __future__ import annotations` 下 f.type 是字符串,传了会
    TypeError;不修 parse_overrides,实验脚本自行 replace + 自写
    provenance),记录胜负分布与终局 tick。
  - 输出汇总表(每变体一行)落 run 目录 + 结论草稿(标 [AI-DRAFT]
    [source: run_id])写入 research-log.md。
- **决策点**:把汇总表给用户过目,定终值;若用户选「攻击间隔」路线,
  另开 phase(新字段 atk_cooldown int16[N] + combat 门控 + 测试),
  本 plan 不含。
- 定案后:改 `src/teow/config.py` 对应字段,changelog 平衡区记
  旧值→新值 + [source: run_id]。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/python explorations/exp_tower_balance_v13.py`
- 判据:run 目录含 resolved config/seed/git hash 三件套;汇总表覆盖
  全部变体×场景;终值获用户明确认可后才写进 config.py。

## Phase 7: 双端渲染(军旗 + 8 资源点)

- `trajectory.npz` 自动含 flag_pos/flag_active(state_to_numpy 走
  `_asdict()`,run.py:105-106,无需改录制)。
- 改 `src/teow/render.py:_draw_frame`(29-97):资源点循环(46-49)后
  加旗绘制(active 旗画阵营色三角小旗,▲+杆)。
- 改 `src/teow/server.py:load_replay`(54-79):frames 加
  `flags: [[p, r, c], ...]`(仅 active);meta 无需改。
- 改 `web/render.js:draw`(54-119):node 段(70-82)后画旗
  (P_COLOR 染色);`web/sprites.js` 加独立 `drawFlag(ctx,owner,x,y,s)`
  (非实体不进 TYPE_NAMES/drawSprite 分发)。
- 验证:录一局
  `JAX_PLATFORMS=cpu .venv/bin/python src/run.py play --p0 scripted --p1 scripted --record --slug v13-frontend-demo`,
  然后 `.venv/bin/python - <<'EOF'`(explorations/ 临时脚本亦可)调
  `teow.server.load_replay(run_dir)` 断言任一帧含非空 `flags`。
- 判据:load_replay 数据契约含旗;matplotlib `replay` 手动抽查一帧
  可见旗(用户浏览器验收另列,见端到端)。

## Phase 8: README.md(中文)

- 新建 `/README.md`,两部分(规格明文):
  1. **项目介绍**:一段话定位(纯 JAX tick 制 RTS 引擎,v1 引擎/v2 RL
     路线)、目录结构一览、快速开始(pytest / run.py play / serve 三条
     命令,照抄 CLAUDE.md 命令区口径)。
  2. **玩家手册**:机制章(资源与采集名额、建筑与升级链、单位与战斗、
     驻守与军旗、胜负判定,每机制 3-5 句)+ 数值表(工人/步兵/狗/塔/
     建筑的成本、血量、攻击、速度等,从 config.py 摘抄,表头注明
     「数值真源 src/teow/config.py,本表对应 vX.Y」)。
- 验证:派 claim-verifier agent 核对 README 数值表逐项与 config.py 一致。
- 判据:claim-verifier 报告零不一致;README 中数值均能映射到 config 字段。

## 端到端验证

- 命令(依次):
  1. `JAX_PLATFORMS=cpu .venv/bin/pytest -x -q`
  2. `.venv/bin/ruff check src/ tests/`
  3. `JAX_PLATFORMS=cpu .venv/bin/python src/run.py play --p0 scripted --p1 scripted --seed 7 --record --slug v13-e2e`
  4. 扩 `explorations/audit_v12_conservation_invariants.py` 为 v1.3 版
     (新文件 `explorations/audit_v13_invariants.py`):守恒+决定论逐位
     重放照旧,新增不变量——每点指派数 ≤ 该点等级名额、
     `flag_active` 每玩家 ≤3、GARRISON 单位 garrison_id 有效、
     撤旗后无单位引用该旗;对第 3 步 run 目录跑。
- 判据:
  1. 全部用例 pass(预计 33+新增 16≈49,数量只增不减),0 failed。
  2. ruff 0 error。
  3. 对局正常终局(分胜负或和局),metrics.jsonl 完整,对局中出现过
     驻守与旗(scripted 分支保证)。
  4. 审计脚本全部不变量 0 违例、逐位重放一致。
- 版本收尾(engine-auditor 终审、changelog、tag、handoff)走
  `/version-close`,不在本 plan 内展开。

## 发现但未做

- **Phase 6 发现(未做)**:scripted vs scripted 对局对 seed 完全不敏感——场景 B
  每变体 8 seeds 的 winner/end_tick/tower_seen 逐 seed 相同,「多 seed」退化为
  单样本(推测:v1.2 起 movement 不吃 key,step 内两处随机仲裁在这些对局未产生
  可见分歧;未逐位核验)。影响端到端验证的「多 seed 看分布」设想与 v2 rollout
  多样性,后续引擎侧敏感性实验需用 random 控制器或扰动初始条件;数据见
  experiments/20260725-tower-balance-*/scenario_b.jsonl。另:Phase 6 只出数据,
  哨塔终值待用户定案,config.py 未动(plan 决策点本就如此,非遗漏)。

- **Phase 5 发现(未做)**:seed 0 scripted vs scripted 整局不对称——p0 全场
  0 军队(兵营 775 tick 才起、狗/步兵始终 0),p1 节奏正常(兵营 760、狗x3
  @1005、总攻 1035)并于 1116 tick 获胜;双方逻辑相同,不对称疑来自槽号仲裁下
  的经济分配(名额制+多重预留把 p0 压在练兵线以下)。非 Phase 5 引入:新分支
  只作用于狗/旗,对无狗玩家恒等。时间线量自
  `explorations/debug_v13_scripted_garrison.py`。端到端验证与收尾对决时需留意
  「一边倒」是否普遍(多 seed 看分布),必要时调 ai_* 参数——走 /exp,不在本
  phase 顺手改。
- **Phase 5 例外扩界(已做,记档供 validate 对账)**:涌现测试视野从 plan 字面
  的 900 tick 放到 `cfg.episode_len`(3000)——900 视野实测假红(v1.3 名额制
  把攒狗推迟到 ~1000 tick),这是 plan critic m-4 自带的后备口径,时间线已复查
  (驻守 966 / 插旗 1006 / 终局 1116)。

- **Phase 4 例外扩界(已做,记档供 validate 对账)**:①`tests/test_map.py`
  两处 `cfg.n_goals` 断言(dist_fields 形状、场遍历)在 n_goals 10→16 后必挂,
  依例外条款改为静态口径 `cfg.n_nodes + 2`(plan 未列此测试改动);
  ②「拥有建成兵营」判定用 `btype >= 0`(排除在建负值任务)而非 plan 字面的
  `btype == 0`:issue.md 规格「拥有兵营(1 级即可)」「撤旗挂在兵营上,免费
  即时」均无空闲要求,btype==0 会把正在训狗的兵营排除、并卡死 Phase 5 总攻
  tick 的撤旗;规格优先于 plan,已记 docs/DECISIONS.md;③撤旗清理在
  「order 仍为 GARRISON → IDLE」之外,把**指向被撤旗的 garrison_id 残值一律
  清 -1**(含同 tick 被改派 ATTACK 者)——这是 plan 自己的 critic M-2 回归
  用例断言 `garrison_id==-1` 所要求的口径,字面 plan 只写了清 GARRISON 者。

- **Phase 2 例外扩界(已做,记档供 validate 对账)**:名额制把 1 级点采集从
  4 人压到 3 人后,`test_scripted_upgrades` 挂掉(诊断:矿石被练兵吃在升本线
  以下打转,733 tick 分胜负但全场 0 升本)。依例外条款动了
  `src/teow/controller.py` 科技优先预留块一处:基地未到 ai_base_level_target
  时,练兵预留追加「升本成本+ai_upgrade_reserve」(与 Phase 1 的研发预留同根
  同款)。Phase 5 改 controller 时须复核相容性。
- `src/teow/actions.py` apply_orders 名额仲裁的已知角落(plan 设计自带,按
  plan 原样实现):工人从 A 点改派 k 点会同 tick 释放 A 名额,若它在 k 被仲裁
  拒绝(act 退回 NOOP、保留 HARVEST A),而同 tick 另一工人恰好拿走了被释放的
  A 名额,则 A 点瞬时超额 1。scripted 从不同 tick 改派(只派 IDLE 工人),仅
  random 控制器可触发;端到端审计的「每点指派数 ≤ 名额」不变量若在 random 局
  上跑需注意此口径。修法(若要修):改派的旧名额释放改为「新指派成功才释放」
  (两遍仲裁),但会让满员点之间的同 tick 对换互卡一拍。

- **Phase 1 例外扩界(已做,非未做,记档供 validate 对账)**:8 点地图让
  `test_scripted_upgrades` 行为回归挂掉(全场 0 研发),依 /impl 例外条款
  (不改无法完成本 phase)动了 `src/teow/controller.py` 三处 scripted 决策:
  ①扩张分层预算(每类资源首点裸成本、额外扩张须留 ai_upgrade_reserve);
  ②科技优先预留(研发排队时练兵先扣除低线研发成本再判可负担);
  ③训狗同吃该预留门控。Phase 5 改 controller 时须复核这三处与新分支的相容性。
- `controller.py:87` 的 harvestable 软门控仍引用 `cfg.node_capacity`,
  Phase 2 删该字段时必改(plan Phase 2 已列,这里再钉一次防漏)。
