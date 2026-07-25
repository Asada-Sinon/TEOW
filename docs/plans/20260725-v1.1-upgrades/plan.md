# v1.1 升级系统 — plan

规格依据:issue.md「## v1.1」节。设计取舍见同目录 research.md。
每个 phase = 一个功能级 commit(`v1.1 <功能>: …`),phase 完成判据必须贴真实输出。

## 目标

基地 1→7 级、矿泵升级(上限=基地级)、技能训练营(基地 2 级解锁、建成即 2 级、
步兵捆绑线+工人经济线、上限链 升级≤营≤基地)、基地升级零单位加成。

## 不在范围内

兵营/哨塔/狗子/前端(v1.2)、RL(v2)、迷雾(v3)、除营外的新建筑内容、
scripted 难度参数扩展、地形墙(1 宽走廊回归用例仍属 v1.2 前置)。

## Phase 1 数值表与状态字段

- config.py:每级数值表全部用 tuple 字段(基地升级 cost/time[6]、矿泵升级
  cost/time 与 extraction_rate[7]、训练营建造 cost/time 与升级 cost/time、
  步兵线 hp/atk[7] 与研发 cost/time、工人线 carry/mine_time/hp[7] 与研发
  cost/time、解锁表 camp_unlock_level=2);TYPE_CAMP=6;btype 负数任务码常量。
  v1.0 标量(inf_hp 等)改为表的 1 级项,消除双真源。
- state.py:`level: int8[N]`(初始 1)、`upgrades: int8[2,2]`(初始 1)。
- 验证:`JAX_PLATFORMS=cpu .venv/bin/pytest -q` 全绿(既有断言按新字段修);
  `ruff check` 绿。

## Phase 2 基地/矿泵升级

- **扣费仲裁 pass(critic B-1,本版关键)**:v1.0「即时扣费」安全前提是每家
  仅 1 座 HQ 同 tick 至多一笔支出;v1.1 的 A_UPGRADE 可同 tick 落在多个建筑上。
  升级/研发/建营的扣费**不走 apply_orders 向量化路径**,收拢进一个专门的
  仲裁 pass:同玩家按槽号顺序对账扣费(照 start_constructions 的模式,
  economy.py:141-166),付不起的自动 no-op,库存恒 ≥0。
- actions.py:新动作 `A_UPGRADE`(建筑通用自升级)。合法性:等级上限链
  (HQ<7、矿泵/营<基地级)**且 `btimer==0` 建筑空闲**(critic S-1:否则会
  覆写在训单位/在建营,蒸发不退款)。
- economy.py:负数 btype 任务码**解码集中单处**,完成分支**必须 btype←0**
  (critic S-3:否则 `btype<0 & btimer==0` 下一 tick 重复触发,level 每 tick+1
  直到溢出);完成时 level+=1。矿泵升级期间驻矿产出**不中断**(按当前级查表);
  HQ/营升级期间不能同时生产/研发(单槽队列天然保证)。
- **产量合成公式(critic S-2,写死并记 DECISIONS)**:
  一趟入账 = `carry_cap[工人线]` + `node_yield_bonus[矿泵级]`;
  开采耗时 = `mine_time[工人线]`。矿泵的「产量随级提升」全部经 yield_bonus
  表达,工人线管载荷与速度,两真源正交不重叠。
- 验证:tests/test_upgrade.py——升级时序恰=表值、上限链掩码(基地 1 时矿不可
  升)、双向互斥(升级中训练非法 **且** 训练中升级非法)、产量公式守恒对账、
  **random vs random 300 tick 后 `resources>=0` 恒成立**(B-1 回归)。

## Phase 3 技能训练营

- 建造:`A_BUILD_CAMP`(工人,基地≥2 解锁)。**扣费+落格+占槽都走 Phase 2 的
  仲裁 pass**(critic B-1:两个工人的「相邻第一空闲格」可能是同一格,落格按
  槽号顺序认领,晚者顺延);在工人相邻第一空闲格生成**在建营**实体
  (专属任务码 `btype=-4`,critic S-3),btimer 建造计时,完成 level=2、btype←0。
- **营血量表(critic S-4)**:`camp_hp[营级]` 进 Config;在建期 hp 由 btimer
  反推线性成长(起步 = 满血/10),无额外状态;被拆=打断且不退款。
- 目标偏好:combat 的 `is_building` 集合加入 TYPE_CAMP(critic FYI-1)。
- 研发:`A_RESEARCH_INF` / `A_RESEARCH_WORKER`(营动作;合法 ⇔ 线级<营级、
  付得起、营空闲);完成 `upgrades[p,line]+=1`,存量单位 hp += Δ上限。
- 属性接线:combat 的 atk/economy 的 carry、mine_time、生成单位 hp 全部按
  `upgrades[owner,line]` 查表;营被拆:研发中断不退款、已购等级保留
  (upgrades 本就在玩家轴上,天然满足)。
- 验证:tests/test_camp.py——建成即 2 级、研发生效对存量单位(hp 差量)、
  营拆毁中断研发但 upgrades 保留、上限链(营 2 级时线不可到 3)、
  在建营可被拆且不退款。

## Phase 4 scripted AI 用上新机制

- controller.py:资源富余优先级 升基地(≥阈值)→建营(基地 2 级且无营)→
  轮流研发双线→升矿泵;建营位置=派工人回 HQ 附近(复用现有 BUILD 派遣模式)。
- metrics.jsonl 在**本 phase**加 base_level/upgrades 字段(critic FYI-2:
  Phase 4 的验证要用,不留到 Phase 5)。
- 验证:scripted vs scripted 完整对局中出现基地≥2、营建成、双线≥2 级、
  矿泵升级的事件(metrics 断言);对局仍能分出胜负,无 300+ tick
  经济停摆(复用 explorations/audit_conservation_determinism.py 的停摆检测)。
- 接线扫尾:actions.py:177 的 `cargo >= cfg.carry_cap` 等散点改为按线查表
  (critic FYI-3;标量字段改名为表,让漏改处直接报错)。

## Phase 5 渲染与收尾准备

- render.py:建筑/单位标注等级(数字角标);metrics.jsonl 增加 base_level、
  upgrades 字段。
- 全量 pytest + ruff;录制一场展示局(--record)。

## Phase 6 /version-close v1.1

- 新鲜审计对局(含「营建在 HQ 旁」场景)→ engine-auditor(现已注册,直接派)
  → 分诊 → docs/changelog/v1.1.md → tag v1.1 → push → handoff。

## 端到端验证

`JAX_PLATFORMS=cpu .venv/bin/python src/run.py play --p0 scripted --p1 scripted
--seed 0 --record`:对局中肉眼可见(replay)双方升本、建营、兵变强(交战时长
变化)、矿泵产出加速;pytest 含新用例全绿;审计 P0 清零。

## 风险与对策

- 涌现死锁(在建营挡卸货环):scripted 建营位置避开 HQ 正邻格(斜角以外),
  收尾审计专项覆盖;
- btype 负数码散落:解码集中在 economy 单处(research.md 风险 2);
- 平衡失真(升级性价比):收尾对局观测,数值只记 changelog 平衡区,不追求
  本版调准。
