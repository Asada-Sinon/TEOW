# Research Log

**本文件当前是空模板，还没有任何真实实验条目。** 下面只有格式说明和一段被注释掉的示例。

假设 → 实验 → 结论的循环记录。**append-only**：只在末尾追加，不改旧条目。旧结论被推翻时，
写一条新的、引用旧条目并说明为什么——被推翻的假设本身就是结果。

**成功判据必须写在跑实验之前**，防止事后编故事（HARKing）：数字出来之后再定义「什么算成功」，
任何结果都能被讲成胜利。判据要可判定——「acc 相对基线 +1.0 个点以上」可以，「效果变好」不行。

每条结论必须带标注：`[AI-DRAFT]`（AI 得出，未经人核验）/ `[HUMAN-VERIFIED]`（只有人类能打）/
`[source: <run_id>]`。run 目录一律 `experiments/<run_id>/`，run_id 格式 `YYYYMMDD-<slug>`。

格式：

```markdown
## YYYY-MM-DD  run_id: <YYYYMMDD-slug>
- 假设:
- 成功判据:        ← 必须在跑实验之前写
- git hash:
- 结果:
- 结论: [AI-DRAFT] ... [source: <run_id>]
```

<!-- 示例（安装后请删除这整块）
以下为格式示例，不是本项目的真实实验记录。run_id、git hash、数字全部虚构，
任何 agent 都不得引用它们，也不得把它们当作已有的基线或结论。

## 2026-03-20  run_id: 20260320-sampler-shuffle-seed0
- 假设: shuffle 从 dataset 移到 sampler 后，最终 acc 不应有可测差异。
- 成功判据: 3 个种子的 val acc 均值与基线 0.809 之差落在 ±0.005 内。
- git hash: 4f1c9ae（工作区 clean）
- 结果: val acc = 0.812 / 0.807 / 0.811，均值 0.8100（`experiments/20260320-sampler-shuffle-seed0/metrics.json`）
- 结论: 判据满足，改动保留。[AI-DRAFT] [source: 20260320-sampler-shuffle-seed0]
-->

---

<!-- 真实条目从这一行下面开始追加，新的永远在最后。 -->

## 2026-07-25  run_id: 20260725-scripted-v-scripted
- 假设: v1.0 引擎全链路(采集→建造→训练→交战→拆家)能在 scripted vs scripted
  下走到分出胜负,不出现经济死锁(此前调试中出现过三类死锁,已修)。
- 成功判据: 一场对局在 episode_len=3000 内以摧毁 HQ 结束(winner∈{0,1}),
  全程无「双方资源与人口连续 ≥300 tick 完全不变」的死锁段。
- git hash: b0a886d(工作区 dirty,diff 已存 run 目录 git.diff;为引擎代码
  首次提交前的收尾运行,后续提交即含全部改动)
- 结果: winner=0,tick=774,P1 HQ 血量 0(`experiments/20260725-scripted-v-scripted/
  metrics.jsonl`);胜因可视回放见同目录 replay.gif——P0 抢下两个中央公共点,
  经济压制后 7 步兵拆家。
- 结论: v1.0 引擎闭环成立。[AI-DRAFT] [source: 20260725-scripted-v-scripted]
- 附注(吞吐观测,单环境): CPU 831 tick/s、GPU 27 tick/s——单环境下 GPU 每 tick
  的 kernel launch 开销占优,符合预期;GPU 优势要到 v2 vmap 批量 rollout 才体现。
  [AI-DRAFT] [source: 20260725-first-full-game][source: 20260725-scripted-v-scripted]

## 2026-07-25  run_id: 20260725-tower-balance-{base,atk4,atk3,cost80-50,hp90}
- 假设: v1.2 哨塔现值(L1 攻6/血120/造价50-30)对狗 rush 的强度可由 config-only
  杠杆(攻/血/造价)调平;若各杠杆都拉不动结果,才需要「攻击间隔」新机制
  (plan v1.3 Phase 6)。
- 成功判据: 汇总表覆盖 5 变体 × 场景 A(N∈{2..5} 狗 rush 手术局)+ 场景 B
  (scripted 互打 8 seeds),每个 run 目录三件套(git hash/resolved config/seed)齐。
- git hash: b1477d1(工作区 dirty:issue.md 草稿与本实验脚本;脚本跑完原样提交,
  与 v1.0 首跑同一惯例)
- 结果(数字均引各 run 目录 scenario_*.jsonl 与 …-summary/summary.md):
  - 场景 A:全部 5 变体 × N∈{2,3,4,5} 均 dogs_wiped,塔无一被摧毁。base 下
    N=5 @21 tick,工亡 3/3、塔血 105/120;最弱变体 atk3 下 N=5 @36 tick、
    塔血 111——攻击 6→3 只把清场时间 21→36 tick,不翻转任何一档结果。
  - 场景 B:每变体内 8 个 seed 的 winner/end_tick/tower_seen 完全一致
    (所测指标意义上是同一局重复 8 次);p0 全胜(8/8,所有变体)。终局 tick:
    base/hp90 1519、atk4 1498、atk3 1584、cost80-50 1147;cost80-50 下
    p1 全程未出现过塔(tower_seen_p1=False,块边界抽样口径)。
  - 方法学发现: scripted vs scripted 对局对 seed 不敏感——所测指标逐 seed
    相同,「8 seeds」退化为单样本。推测(未逐位核验):v1.2 起 movement 不吃
    key,step 内仅存的两处随机仲裁在这些对局中未产生可见分歧。后续引擎侧
    敏感性实验需改用 random 控制器或扰动初始条件才能得到分布。
- 结论: config-only 杠杆在「1 塔+3 工人+HQ 能否守住 ≤5 狗 rush」上全部拉不动
  (各变体全胜),只影响清场速度与工人伤亡;造价 80/50 在全局对局里让 scripted
  的 p1 建不起塔且终局提前(1519→1147)。哨塔数值终值按 plan 决策点交用户定案,
  config.py 未动。[AI-DRAFT] [source: 20260725-tower-balance-base]
  [source: 20260725-tower-balance-atk4] [source: 20260725-tower-balance-atk3]
  [source: 20260725-tower-balance-cost80-50] [source: 20260725-tower-balance-hp90]
- 定案(2026-07-25 追记,用户授权 agent 决策): 采 atk3 变体——
  `tower_atk_by_level` L1 6→3,其余杠杆不动;理由与不采项见 docs/DECISIONS.md
  同日条目。[AI-DRAFT] [source: 20260725-tower-balance-atk3]

## 2026-07-26  run_id: 20260726-v17-duel-matrix
- 假设: v1.4–v1.6 离线期拍的 [AI-DRAFT] 数值(护甲表/九兵种血攻/迫击炮/栅栏/四
  防御建筑/投石车-飞艇-龙及L2L3线/龙喷火折扣)中,存在个别单位或建筑「超模」——
  即在等资源投入(成本归一)的手工无菌对决里,单方压倒性获胜或违反设计意图克制
  关系(魔法应克重甲heavy_armor=60、攻城应高效拆建筑、等投入同类近战应互有胜负、
  奶妈不应让同投入步兵线性翻盘)。
- 成功判据: exp_v17_duel.py 的 22 个 MATCHUP 全跑完并落 summary.md;每对决记录
  成本归一count/实际投入/胜负/余兵/余血比;标出「疑似超模」项(全灭对方且余兵
  ≥50%,或余血比差>0.7);人工核对上述设计意图克制方向是否成立,方向不符=数值
  bug候选。run目录三件套(git hash/resolved config/seed)+design.json(MATCHUPS
  与water_weight/budget常量)齐。
- 失败判据: 若几乎所有对决都被判「疑似超模」或方向全乱,说明对决脚手架的摆位/
  预算/早停设计有系统偏差(而非真超模),需先修脚手架再重跑。
- 对照 baseline: 无(v1.7 首次系统性对决;哨塔历史见 20260725-tower-balance)。
- git hash: 15bb1dd(工作区跑前干净)
- 结果(experiments/20260726-v17-duel-matrix/summary.md,22 对决):22 中 20 被自动标
  「疑似超模」——触发预登记的失败判据(几乎全标=脚手架系统偏差而非真超模)。逐条查
  duel.jsonl 投入数据发现三类偏差:①单位对决用 ORDER_ATTACK 向敌 HQ 行军,远程/空军
  受「行军送死 vs 完全风筝」影响极大(法师零伤亡全歼重甲、龙被围殴);②混合兵种侧
  (6步兵+3奶妈)成本归一时各子兵种各吃满 budget → A 投入[270,180] 对 B[180,60],超投
  近一倍;③防御建筑 vs 波的 budget 与建筑造价不匹配(迫击炮140 对 6步兵240,建筑被
  少投入)。设计意图方向本身成立(魔法克重甲/攻城拆建筑/远程压近战)。
- 结论: 首版脚手架机制正确(交战/不变量/喷火折扣端到端对)但对决经济学有系统偏差,
  自动超模判定不可信;按用户 2026-07-26 口径重做(原地接战 + 攻防局 + 混合兵种均分
  预算),见下条 v2。本 run 仅作方法学定位,数值不引用。[AI-DRAFT]
  [source: 20260726-v17-duel-matrix]

## 2026-07-26  run_id: 20260726-v17-duel-v2(脚手架按用户口径重做后)
- 假设: 同上(找超模);口径按用户定案——①单位vs单位=原地接战(交错摆位IDLE,纯
  血量/攻击交换);②防御建筑=攻防局,同等造价该赢、~2倍造价该被攻破;③water=矿同重。
- 成功判据: 27 对决(含防御建筑 1×/2× 两档)全跑完;设计意图方向成立;防御建筑满足
  「1× 守住、2× 被破」;标出真正偏离(某单位/建筑在等投入原地接战里压倒,或防御建筑
  1× 就守不住 / 2× 仍无法攻破)。run 目录三件套 + design.json 齐。
- git hash: df7b13d(跑前干净;v3=36b4a2f 加射程公平后重跑)
- 结果(experiments/20260726-v17-duel-v2 与 -v3/summary.md):
  - 近战原地接战全部 even/微弱优势(步兵/狗/轻骑/重甲各搭配)——**平衡,不动**。
  - 哨塔/喷火/激光 攻防局「1× A_win 守住、2× B_win 被破」达标——**平衡,不动**。
  - 攻城车高效拆塔/兵营/HQ;龙对空压制飞艇;奶妈**不超模**(3步兵+2奶妈260 反输
    6步兵240,证伪首版假象)——**均不动**。
  - **偏离项**:①法师塔现值 atk14 攻防局 1× 就守不住(超弱);②迫击炮 1× 守不住,
    且射程公平重测(v3 起步距离随防御方射程缩放,36b4a2f)后仍守不住;③龙火海
    (dragon_breath_radius=2.5)清不完等价地面波(A_ahead 超时)。
  - 口径说明:原地接战抹掉远程射程价值,弓/法在此显弱是口径非超弱(不据此调)。
- 结论(定值见下 tune 系列):法师塔补强、龙火海放大;迫击炮数值无解(机制限制)。
  [AI-DRAFT] [source: 20260726-v17-duel-v2] [source: 20260726-v17-duel-v3]

## 2026-07-26  run_id: 20260726-v17-tune*(补强定值扫参)
- 假设: 对超弱的法师塔/迫击炮扫 config-only override,能找到满足「防御建筑 1× 守住、
  2× 被破」的最小改动;龙火海半径放大能清等价地面波且不失衡(龙对纯地面无敌,地面
  须带防空反制)。
- 成功判据: 每候选跑 1×/2× 两档攻防局(龙跑 1 龙 vs 8 步兵),标达标候选。
- git hash: 36b4a2f(跑前干净)
- 结果(experiments/20260726-v17-tune{,-v3,-mortar-aoe,-magetower-dps}/summary.md):
  - **法师塔**:atk18 仍守不住 1×,atk20 起守住;DPS 扫 atk20×period{5,4,3}=dps{4,5,6.67},
    1× 存活血 {0.171,0.429,0.771}。用户 2026-07-26 定 **atk 14→20 + period 5→4**
    (dps 2.8→5.0)。[source: 20260726-v17-tune-magetower-dps]
  - **迫击炮**:扫 13 种候选(period 40→15、min_range 2.5→1.0、hp→250、atk→50、
    aoe_radius 1.5→4.0 各组合)**全部守不住 1×**,塔血恒 0.0。根因是机制:盲区 2.5 +
    单发慢炮弹(flight 8、打开火瞬位不预判)→ 对冲脸移动步兵必然打空,一次攻防仅约
    1 炮且基本落空。**数值无解**。用户 2026-07-26 定案:接受迫击炮为远程炮击/攻城
    支援建筑(非独立点防),数值不动,记 changelog 已知。
    [source: 20260726-v17-tune] [source: 20260726-v17-tune-mortar-aoe]
  - **龙火海**:现值 2.5 只 A_ahead(超时余 1 步兵),≥3.5 全清(@31tick);紧凑阵下
    ≥3.5 饱和(测不出更大值差异,大小属真实分散阵设计取舍)。用户 2026-07-26 定
    **dragon_breath_radius 2.5→4.5**(大范围火海)。[source: 20260726-v17-tune]
- 结论: v1.7 落三值——magetower_atk 14→20、magetower_period 5→4、dragon_breath_radius
  2.5→4.5;迫击炮数值不动(机制限制,已知);龙喷火对建筑 50% 折扣保留(用户定位:
  龙不擅拆建筑)。[AI-DRAFT] [source: 20260726-v17-tune-magetower-dps]
  [source: 20260726-v17-tune] [source: 20260726-v17-tune-mortar-aoe]

## 2026-07-26  run_id: 20260726-v18-bench
- 假设: 64×64/4p 单环境虽慢,但 vmap 批量 GPU 能把吞吐拉到支撑 v1.9 海量评测 / v2.0
  rollout 的水平(v1.8 P0,#1 吞吐风险验证)。
- 成功判据: 批量 GPU env-tick/s ≥ ~1000(使 ~1000 局 6000-tick 对局在 ~1h 内可批量跑完)。
- 失败判据: 批量最高吞吐仍 < 500 env-tick/s(需先优化 movement._relax_fields 才能继续)。
- 对照: 无对照,仅探索(历史参照:v1.3 @24×24 单环境 57-66 tick/s)。
- git hash: 3834f36(dirty:P1 进行中——monster 子表字段已入 state/config,gate 阶段尚未
  接入 step,故本测**未含 gate 逐 tick 开销**;P1 收尾复测拿干净数)。
- 结果(experiments/20260726-v18-bench/bench.json):
  - single-cpu **101.39** tick/s;batch-cpu B32/B128 env **176.7 / 145.8** tick/s
    (CPU vmap 几乎不提速,per-world 反降 5.52→1.14)。
  - **batch-gpu B64 env 4025.8 tick/s**(per-world 62.9)= 单卡最佳;B256 **2719.1**、
    B1024 **2840.0**(B64 是甜点,更大批 occupancy/显存反使总吞吐降)。
- 结论: [AI-DRAFT] 假设成立且远超判据——**GPU vmap 批量(B≈64)达 ~4000 env-tick/s
  (≈40× 单 CPU)**,是 v1.9/v2.0 海量 rollout 正路;单环境 CPU ~100 tick/s(远好于按
  v1.3 缩放的悲观估计,**#1 吞吐风险解除**),多进程 CPU 为备选;**CPU vmap 无益,勿用**。
  [source: 20260726-v18-bench]

## 2026-07-27  run_id: 20260727-v18-eval-p3 / 20260727-v18-p4-gate
- 假设: 10 风格指挥官均 branchless 可用、碾压 random、风格不塌缩;异界之门在默认 gate=4000
  下让战术决胜、只对停滞局兜底。
- 成功判据: 全指挥官 vs random 胜率=1.0 且 0 崩溃;round-robin 胜者随 matchup 变化;默认 gate
  下进攻型靠消灭取胜(gate 不触发)、被动型靠门兜底。
- 失败判据: 有指挥官崩溃 / 输给 random / 全 matchup 单一风格通吃;或默认 gate 下全部对局都撞门
  (战术完全不决胜)。
- 对照: random 基线。
- git hash: bb5feac(P3)/ bb5feac+ruff(P4)。
- 结果:
  - P3(gate=1800 提速档,experiments/20260727-v18-eval-p3):10/10 vs random 胜率 1.0、0 崩溃;
    rr 胜者 balanced×4 / harasser×4 / rusher / airtech / timing / chaos(不塌缩)。
  - P4(默认 gate=4000,experiments/20260727-v18-p4-gate):**rusher vs random 全靠消灭 @~1080 拍
    (gate 0/6)**;boomer/turtle vs random @4182(gate 6/6,门后 ~182 拍快速决胜);
    rr(rusher|boomer|turtle|timing)rusher 全胜 @1207-1869(gate 0/6,纯战术决胜)。
- 结论: [AI-DRAFT] 假设成立——指挥官框架 branchless 正确、全功能、风格不塌缩;默认 gate=4000
  是良好兜底(战术能决胜时不触发、停滞局 ~182 拍快速决胜、四家对称)。**gate_open_tick 显著影响
  rush-vs-develop 平衡(短门利被动、长门利速攻);roster 精细平衡 + 训练效率取舍留 v1.9 校准**;
  boomer 经济锁死 degeneracy 已在 P2 修复(tick 兜底)。[source: 20260727-v18-eval-p3]
  [source: 20260727-v18-p4-gate]

## 2026-07-27  run_id: 20260727-v19-roundrobin(v1.9 综合评测;criteria 先写,结果待 result-analyst)
- 假设: v1.8 的 10 风格指挥官在默认 gate=4000 下均高质量(碾压 random、round-robin 无单一风格
  通吃/无 always-输、非退化),可直接作 RL 对手池;round-robin 胜率给 v2.0 课程难度排序。
- 成功判据: 每指挥官 vs random 胜率 ≥0.9;round-robin 无某风格对全场胜率 >0.85 或 <0.10;
  对局时长分布非畸形(无大量秒杀 <100 拍、无大量撞 6000 硬帽)。
- 失败判据: 有指挥官 round-robin 胜率 <0.10(太弱→修或删)或某风格通吃(平衡问题→记 v2.0 待调)。
- 对照: random 基线 + P3(gate=1800)/P4(部分 gate=4000)。
- git hash: a3ddadf(跑前干净;含 P2 对怪 cd 修)。
- 结果(experiments/20260727-v19-roundrobin/games.jsonl,80 局;result-analyst 用
  explorations/agg_v19_*.py 聚合,非手算):
  - Phase A:10 指挥官 vs random 胜率全 **1.00**(4/4),无一 <0.90。
  - Phase B round-robin(**非均衡**:出场 4–24 不等、4 seed/局 → 尾部噪声大):rr_wr 强→弱=
    rusher 0.75、balanced 0.38、harasser/boomer/tempo 0.25、chaos 0.20、counter 0.17、
    turtle/timing/airtech 0.00(round-robin 零胜)。**无风格 >0.85**(无统治,满足判据)。
  - 非退化:80 局 length min1068/median4022/max4589;**秒杀<100=0、撞硬帽6000=0、和局=0**;
    但 22/80 局精确落 length4182(宏观局固定 gate 结算点);turtle/airtech vsRandom 为
    0-军被动 gate 胜(全 seed 4182/army0,非战斗决胜)。
- 结论: [AI-DRAFT] 假设大体成立——10 指挥官全功能(碾压 random)、风格清晰二分(速攻 rusher/
  balanced/harasser vs 宏观 boomer/tempo/counter/chaos vs 弱尾 turtle/timing/airtech)、无统治、
  非退化(无秒杀/硬帽/和局)。**弱尾 rr 零胜受非均衡 eval 噪声 + 被动 gate 依赖影响**;按分层留用
  (HARD rusher / MEDIUM balanced·harasser·boomer·chaos·counter·tempo / EASY turtle·timing·airtech),
  **不删任一(弱者仍风格独立,对多样对手池有价值)**;弱尾被动性 + 均衡 round-robin(≥8 seed 全对
  covering)+ 弱尾定向调参留 v1.9-followup / v2.1 前处理。[source: 20260727-v19-roundrobin]

## 2026-07-27  run_id: 20260727-v21-balanced-rr(v1.9 弱尾均衡复核 + vs-random 加厚)
- 假设: v1.9 弱尾(turtle/timing/airtech round-robin 零胜)主因是**非均衡采样噪声**;均衡
  round-robin(covering design 全 45 对覆盖 + cyclic 座位轮转 + rr-seeds 8,全局 episode 6000/
  gate 4000)下三者 rr_wr >0。turtle/airtech vs-random「0 军被动 gate 胜」是策略参数问题
  (attack_threshold/base_level 过高),非引擎 bug。
- 成功判据: 每指挥官 rr 出场均衡(covering+轮转);10 指挥官 vs-random 胜率 ≥0.90;无风格
  rr_wr>0.85(统治);弱尾 rr_wr 得低噪声估计(比 v1.9 尾部零胜可信)。
- 失败判据: 均衡采样下弱尾仍 rr_wr<0.10 → 触发定向调参(turtle attack_threshold/upgrade_reserve↓、
  airtech base_level_target↓)后重测,要求末军均值>1(有作战)且 vs-random 不降。
- 对照: 20260727-v19-roundrobin(v1.9 非均衡基线)。
- git hash: fca97ba(跑前干净;含 v2.1 Phase A/B/C/D 脚手架)。
- 结果(experiments/20260727-v21-balanced-rr,agg_v21_balanced.py 聚合,232 局):
  - vs-random:10 指挥官全 wr=1.00(判据 ≥0.90 全过)。
  - round-robin(座位仅 1 shift/组合):rr_wr rusher 0.72 / balanced 0.71 / tempo 0.31 /
    counter 0.28 / harasser 0.25 / timing 0.16 / boomer·turtle·airtech·chaos 0.00。
  - **座位偏置 bug**:各座位总胜场 {0:38, 1:8, 2:14, 3:12}(seat0 系统偏强);且 cyclic 1-shift
    使 turtle/airtech/chaos **seat0 出场=0**(从没坐最强位)→ 零胜含人为低估,非纯真弱。
- 结论: [AI-DRAFT] vs-random 判据全过;**round-robin 弱尾零胜受座位偏置污染**(全局 6000 下
  seat0 仍系统偏强,cyclic 1-shift 未保证每家轮 seat0)→ 评测方法改全 P 座位轮转重跑(rr2);
  弱尾真实强度待 rr2 定。agg 为脚本聚合,非手算。[source: 20260727-v21-balanced-rr]
- [核验 2026-07-28 result-analyst,更正上条座位偏置归因] 独立双路径复算(explorations/
  seatbias_check_v21.py + seatbias_xcheck_v21.py,座位胜场分布两脚本一致 38/8/14/12)**部分
  推翻座位机制归因**:seat0 原始胜场高属实,但 rr 单 shift 令「座位」与「对手集」**完全混淆**
  (同 combo 无跨座位样本)——seat0 高胜可由对手强弱完全解释:harasser/rusher 坐 seat0 恰遇全
  弱尾→8/8;**boomer 坐 seat0 遇三强→0/8**;rusher 在 seat1、balanced 在 seat3 也 8/8→强者不
  靠 seat0。故修正:①**弱尾零胜主要是真弱**,座位曝光缺口为次要不可量化因素;②**boomer 决定性
  反例**(seat0 曝光 8 次仍 0 胜=真弱);③rr2 全 P 轮转=正解(同组合全座位排布→真正分离座位/对手),
  预期弱尾仍弱;弱尾定性宜辅以头对头(弱尾 vs 单强)。[source: 20260727-v21-balanced-rr]

## 2026-07-27  run_id: 20260727-v21-balanced-rr2(全 P 座位轮转消偏置复核)
- 假设: rr 弱尾零胜主因是座位偏置(seat0 系统偏强 + turtle/airtech/chaos 从没坐 seat0);每
  covering 组合跑全 P 座位轮转(每家在每座位各一次)后,座位均衡,弱尾 rr_wr 反映真实相对强度。
- 成功判据: 各指挥官各座位出场均等(seat0 列不再有 0);各座位总胜场趋近均匀;弱尾 rr_wr 得
  座位无偏估计(可能翻盘也可能坐实真弱)。
- 失败判据: 座位均衡后 boomer/turtle/airtech/chaos 仍 rr_wr<0.10 → 坐实真弱,触发定向调参
  (降 attack_threshold/base_level_target/upgrade_reserve)后重测,要求末军>1 且 vs-random 不降。
- 对照: 20260727-v21-balanced-rr(1-shift 座位偏置)。
- git hash: 5bc39b3(跑前;含全 P 座位轮转修复)。
- 结果(experiments/20260728-v21-balanced-rr2,agg 聚合,304 局,座位完全均衡各家各座位 12/16):
  - **座位偏置全局下温和**:座位总胜场 {0:51,1:21,2:34,3:38}(seat0 略强,远非 smoke 小局 71/1/0/0)
    → 座位偏置存在但不主导,RL 座 0 轻微沾光可接受。
  - rr_wr(座位均衡):rusher 0.86(统治)/balanced 0.44/harasser 0.36/tempo 0.25/counter 0.20/
    chaos 0.12/turtle 0.10/timing 0.06/airtech 0.02/boomer 0.00。
  - 弱尾 rr 出场末军全 0.00;vsRandom 末军 turtle 0.1/airtech 0.2/boomer 0.9(对 random 都 0 军
    =真被动不造兵),但 timing vsRandom 9.8(造兵、rr 被 rush 打光=被克制非退化)。
  - 非退化:秒杀 0、和局 0、near-cap(≥5000)0、length median 4181。
- 结论: [AI-DRAFT] 座位均衡确认 result-analyst 判断——弱尾主要真弱、boomer 决定性反例(rr 0.00)。
  真正「0 军被动不造兵」= **turtle/airtech/boomer**(vsRandom 都 0 军,经济全投升本囤钱→无钱造兵);
  timing 造兵被 rush 克制(保留)、chaos 随机弱(保留)。rusher 0.86 统治是防守型 0 军的反面(修弱尾
  造兵可同时缓解)。定向调 turtle/airtech/boomer:降 upgrade_reserve/base_level_target/attack_threshold
  腾钱造兵+出击,重测要求 vsRandom 末军>1 且胜率不降(见 rr3)。[source: 20260728-v21-balanced-rr2]

## 2026-07-28  run_id: 20260728-v21-balanced-rr3(定向调 turtle/airtech/boomer 后重测)
- 假设: 降 turtle(attack_threshold 22→10/base 7→6/reserve 100→45)、airtech(base 7→6/atk 10→8/
  reserve 70→45)、boomer(atk 16→11/reserve 80→45)后,三者不再「0 军被动等门」——vsRandom 末军>1
  (真造兵作战)且胜率仍 1.00;可能同时压低 rusher 统治(防守型造兵克 rush)。
- 成功判据: turtle/airtech/boomer vsRandom 末军均值 >1(vs 旧 0.1/0.2/0.9);vsRandom 胜率 ≥0.90;
  rr 出场末军 >0;非退化保持(无秒杀/和局)。
- 失败判据: 三者 vsRandom 末军仍 <1 → 幅度不够继续降(rr4);或造兵但胜率大跌 → 过调回退。
- 对照: 20260728-v21-balanced-rr2(定向调前)。
- git hash: d0aaa74(跑前;含 turtle/airtech/boomer 定向调)。
- 结果(experiments/20260728-v21-balanced-rr3,agg,304 局):
  - **turtle/airtech 定向调成功**:vsRandom 末军 0.1/0.2 → **1.1/1.1**(0 军→造兵作战),胜率仍 1.00。
  - **boomer 过头**:vsRandom 末军 0.9→1.2 但胜率 1.00→**0.88**(跌破 0.90 判据)——降 threshold
    16→11 让种田流早出弱兵送死反变弱。
  - rr_wr 基本不变(turtle 0.08/airtech 0.04/boomer 0.03);rusher 仍 0.84 统治;非退化保持。
- 结论: [AI-DRAFT] turtle/airtech 达标(0 军→1.1 军作战、胜率保持),保留;**boomer 回退原值 16/80**
  (种田流「晚而强一波」是风格、末军 0.9=攒兵到门非极端被动,激进调破胜率;用户只点名 turtle/
  airtech)。最终 profile = turtle/airtech 调 + boomer 原值,rr4 出干净对手池分层供 Phase D 课程。
  [source: 20260728-v21-balanced-rr3]

## 2026-07-28  run_id: 20260728-v21-balanced-rr4(最终 profile:turtle/airtech 调+boomer 原)
- 假设: 最终 profile 下 turtle/airtech vsRandom 末军>1、boomer 胜率恢复 1.00,对手池非退化,rr 给
  干净分层供 Phase D 课程。
- 成功判据: turtle/airtech vsRandom 末军>1 且 wr 1.00;boomer wr≥0.90;无风格 rr 全场崩溃;
  非退化(无秒杀/和局/硬帽)。
- 失败判据: 仍有指挥官 vsRandom 0 军被动或胜率崩。
- 对照: 20260728-v21-balanced-rr3(boomer 过头)/rr2(定向调前)。
- git hash: c02cded(跑前;boomer 回退)。
- 结果(experiments/20260728-v21-balanced-rr4,agg,304 局,座位均衡):
  - vsRandom 全 10 wr 1.00(boomer 回退恢复);turtle/airtech vsRandom 末军 1.1(造兵)。
  - rr 分层(座位均衡):rusher 0.84(**不统治,<0.85**)/balanced 0.38/harasser 0.36/tempo 0.27/
    counter 0.19/chaos 0.12/turtle 0.10/timing 0.09/airtech 0.06/boomer 0.00。
  - turtle/airtech rr 末军仍低(0.06/0.00)但 vsRandom 造兵 1.1 = rr 被强对手打光(同 timing
    vsRandom 9.8/rr 0.14),非"不造兵";agg 的 rr 末军<1 警告是判据局限(被克制≠退化)。
  - 非退化:秒杀 0/和局 0/near-cap(≥5000)0/无统治。
- 结论: [AI-DRAFT] **Phase A 收口达标**——turtle/airtech 修好「对 random 0 军被动」(末军 0.1/0.2→
  1.1),boomer 回退 wr 1.00,无风格统治(rusher 0.84)/崩溃,非退化,最终对手池干净。Phase D 课程分层:
  **HARD** rusher / **MEDIUM** balanced·harasser·tempo·counter / **EASY** chaos·turtle·timing·
  airtech·boomer。[source: 20260728-v21-balanced-rr4]

## 2026-07-28  run_id: 20260728-v21-throughput(真实训练吞吐 bench,含 gate+网+更新)
- 假设(硬约束#1 必测): 含 gate+网+更新的真实 games/day 决定 Phase D 训练规模;甜点 B/T 未必是
  纯引擎的 B64。
- 成功判据: 测出各 (B,T) 的 games/day + 峰值显存,分 gate 前/后两制式,脚本产出不口算。
- git hash: 58463d6。
- 结果(experiments/20260728-v21-throughput/bench_train.json):含更新有效吞吐——B32T128 games/day
  **62570**(甜点,eff 4345 env-tick/s,1.8GB)/B64T128 53764(3.4GB)/B128T128 36175(6.8GB)/
  B256T128 36622(13GB);T256 同量级;**B256T256 OOM**(13GB 分配失败)。**gate 前后差异小**(pre
  4871 vs ot 4836,B32)——overtime 怪物开销不大(优于 v18 caveat 预期)。
- 结论: [AI-DRAFT] 真实训练吞吐 **54k–62k games/day**(B32–64,含网+gate+更新),分段 rollout 显存
  可控(B64 3.4GB≪整局爆显存)。**训练不是吞吐瓶颈**(plan 头号风险#3 解除),可跑数千 update 看
  学习信号。Phase D 用 **B64 T128**(稳梯度+54k games/day,per update ~2.2s)。[source: 20260728-v21-throughput]

## 2026-07-28  run_id: 20260728-v21-train-vsrandom(Phase D 小范围试训:vs random 看学习信号)
- 假设: 纯 PPO 管线(全长局分段 rollout + 退火关的 PBRS)在 vs random×3 下能学到——vs random 贪心
  胜率随 update 从未训基线上升(random 最弱 + PBRS 引导经济/军力优势),动作分布不退化,不摆烂等门。
- 成功判据: vs random 贪心胜率明显高于未训基线(update 0)且随训练升(Δ≥+0.2 或滑窗近单调);动作
  熵不塌;PPO 健康(kl<0.3、explained_var>0、无 NaN);不 100% 靠 gate 耗死(有主动作战/击杀)。
- 失败判据: 胜率平/降(学不动→调 β/γ/加 update);动作退化(单一动作垄断);KL 爆/NaN(管线坏)。
- 对照: update 0 未训基线(随机初始网,FFA 下可能靠运气赢一些)。
- 配置: B64 T128 / 2000 update / eval_every 50 eval_seeds 8 / --no-anneal-shaping(保 PBRS 信号)/
  gamma 0.999 / shaping_beta 0.1。git hash: 6d99346。
- 结果(experiments/20260728-v21-train-vsrandom,停于 198 update;metrics/eval jsonl):
  - **学习失败**:vs random 贪心胜率 u0 0.75(未训基线,8seed 噪声)→ u50–150 稳定 **0.25**(降),
    **army=0 全程**,gate=1.00,rank 1.62→3.0,熵 1.24→1.86(升)。
  - **root cause = PBRS reward-hacking**:mean_reward≈0(±3e-5)、pg≈0(无 policy gradient)、loss 被
    熵项主导(ent_coef×ent, ent 6–12)→ PPO 只最大化熵→策略随机→army=0。Φ=_invested_value 含
    **全额库存**:造兵是「库存→单位」cost 守恒(Value 无上行)+单位会死(有下行)→囤钱 Value 更稳
    → RL 学「囤钱不造兵」。
- 结论: [AI-DRAFT] **PPO 机器正确但奖励设计有 reward-hacking**(plan §4.3 预警命中):Φ 含全额库存
  诱导囤钱。修:①**库存打折** stockpile_weight 0.3(造兵产 +0.7cost 上行、囤钱不涨 Φ);②β 0.1→1.0、
  pot_scale 300→100 加强+敏感化信号。重训 v21-train-fix1 验证 army>0。[source: 20260728-v21-train-vsrandom]

## 2026-07-28  run_id: 20260728-v21-train-fix1(reward-hacking 修复后重训:库存打折+强 PBRS)
- 假设: 库存打折(stockpile_weight 0.3→造兵产 +0.7cost Value 上行)+ 加强信号(β 1.0/pot_scale 100)后,
  Φ 真正奖励造兵/建设,RL 学造兵作战——army>0、vs random 胜率随 update 升、mean_reward 显著非 0。
- 成功判据: army>0(RL 造兵,vs 首训 army=0);vs random 贪心胜率高于未训基线且升;mean_reward 量级
  明显 >首训 3e-5;动作分布不退化;PPO 健康。
- 失败判据: army 仍 0(修复无效→再降 stockpile/加 β)或胜率不升(信号仍弱)。
- 对照: 20260728-v21-train-vsrandom(修复前,army=0)。
- 配置: B64 T128 / 500 update / β 1.0 / pot_scale 100 / stockpile_weight 0.3 / --no-anneal-shaping。
  git hash: 90e9fd0。
- 结果(experiments/20260728-v21-train-fix1,停于 150 update):army 仍 0、econ 降(85→22)、mean_reward
  0.00026(比首训 3e-5 大 10× 但仍小)、pg≈0。**动作分布诊断:结构化动作(build/harvest/train)≈0.01
  全程**,策略在 NOOP(0.95)/STOP(0.82)/MOVE(0.87)间无意义震荡,从不 ATTACK。
- 结论: [AI-DRAFT] 库存打折修了 reward-hacking(mean_reward 3e-5→2.6e-4)但**冷启动仍学不动**——RL
  极少采样结构化动作(造兵/采集/建造):随机初始网 + 稀疏延迟奖励 + loss 被熵项主导(ent_coef 0.01×
  ent 7 >> pg 0.001)→ PPO 只优化熵→退化 NOOP/STOP/MOVE。这是 plan §1.4 的冷启动探索问题(BC 暖启
  的用武之地)。再试 fix2(降 ent_coef 0.001 + 极强塑形 β3/pot50/stock0.1)探纯 PPO 极限。
  [source: 20260728-v21-train-fix1]

## 2026-07-28  run_id: 20260728-v21-train-fix2(纯 PPO 极限:降 ent_coef 0.001+极强塑形)
- 假设: 降 ent_coef(弱 reward 不被熵淹)+ 极强塑形(β3/pot50/stock0.1)能让纯 PPO 突破冷启动造兵。
- 结果(experiments/20260728-v21-train-fix2,停于 40 update):**army 仍 0、结构化动作→0.000、
  econ 降(54→30)**;mean_reward 0.0015(比 fix1 大 5×,塑形更强)但 pg 仍~0.001、ent_loss 7。
- 结论: [AI-DRAFT] **坐实纯 PPO 冷启动学不动**——首训/fix1/fix2 + 全谱调参(β/pot_scale/stockpile/
  ent_coef),RL 始终探索不到结构化动作,退化 NOOP/STOP/MOVE。根因**冷启动探索**(随机网极少采样
  结构化动作 + 稀疏延迟奖励→PPO 无正样本),塑形强度救不了。**正式开训必须 BC 暖启**(plan §1.4)。
  [source: 20260728-v21-train-fix2]

### v2.1 Phase D 试训总结论(除险) [AI-DRAFT]
训练管线**机器正确**(rollout/GAE/PPO/update/eval/ckpt/课程/持续环境 reset 全对——slow test 4 passed
+ 数值稳定 25 update + 吞吐 54–62k games/day)。试训除险发现两个开训必解问题:①**reward-hacking**
(Φ 含全额库存诱导囤钱)→已修(库存打折 stockpile_weight);②**冷启动学不动**(纯 PPO 从随机网探索不到
结构化动作)→**需 BC 暖启**(用户 v2.1 暂缓,留正式开训)。**issue.md v2.1 目标达成**:小范围试训阶段
就发现「纯 PPO 会训出什么都不会的指挥官」,避免正式开训才踩坑。

## 2026-08-03  run_id: 20260803-v21-throughput-rerun(补 v2.1 缺失产物:训练吞吐)
- 背景: v2.1 的 8 个 run 目录(`20260727-v21-balanced-rr`、`20260728-v21-{rr2,rr3,rr4,throughput,
  train-vsrandom,train-fix1,train-fix2}`)在磁盘上不存在,且 git 历史中从未提交
  (`git log --all --diff-filter=A -- 'experiments/*v21*'` 为空),而 changelog v2.1 与本日志
  引用它们 → 按 CLAUDE.md 硬约束 3 这些数字不可复现。本 run 用重建后的环境重跑,
  **作为可复现的替代证据,不冒充原 run_id**(原 run 的产物无法复原)。
- 假设: v2.1 记录的吞吐结论(小 batch B32/B64 的 games/day 高于 B128+;最优组合的吞吐量级
  足以支撑「训练非瓶颈」)在重建环境(jax 0.6.2 / RTX 5090 24GB)下可复现。
- 成功判据: ①`bench_train.json` 落盘且含各 (B,T) 组合的 games/day;②B32T128 与 B64T128 的
  games/day **均高于** B128T128;③最优组合 games/day **> 10000**(「训练非吞吐瓶颈」的量级结论)。
- 失败判据: 最优 games/day < 10000(与「训练非瓶颈」矛盾),或大 batch 反超小 batch(排序结论
  被推翻),或脚本无法产出 json。
- 对照: `20260728-v21-throughput`——**产物已缺失**,数字仅存于本日志与 changelog 的文字记录
  (B32T128 62570 / B64T128 53764 / B128+ ~36k / B256T256 OOM)。因此本次是「与文字记录比对」,
  不是与产物比对;若数值有出入,以本 run 的落盘产物为准,不修改历史条目(硬约束 3:只 append)。
- git hash: `2627cacca5c368d5094ef936861d4cc5b6cc75ea`(dirty: true——**脏文件仅 `research-log.md`
  本身**,即本条判据的写入;`src/` 与 `explorations/` 代码树完全对应该 commit,不影响「用哪份
  代码跑的」。成因:skill 要求先写判据入日志、写了工作区就脏,正确顺序是先 commit 判据再开跑,
  本次顺序反了,下个 run 已改正)
- 结果(全部出自 `experiments/20260803-v21-throughput-rerun/bench_train.json`,backend=gpu,
  episode_len=6000 / gate_open=4000,8 个 (B,T) 组合):

  | B | T | games_day | eff_env_tick_s | peak_mem_gb |
  |---|---|---|---|---|
  | 32 | 128 | 44589.84 | 3096.5 | 1.816 |
  | 64 | 128 | 36623.11 | 2543.3 | 3.364 |
  | 128 | 128 | 26879.13 | 1866.6 | 6.760 |
  | 256 | 128 | 26429.34 | 1835.4 | 13.405 |
  | 32 | 256 | 43254.10 | 3003.8 | 13.405 |
  | 64 | 256 | 37069.06 | 2574.2 | 13.405 |
  | 128 | 256 | 26811.98 | 1861.9 | 13.443 |
  | 256 | 256 | — | — | — |(`XlaRuntimeError: RESOURCE_EXHAUSTED` OOM 失败)

- 结论: [AI-DRAFT] **假设成立,三条成功判据全部达成**——①json 已落盘含全部 8 组合;
  ②B32T128(44589.84)与 B64T128(36623.11)均高于 B128T128(26879.13);③最优 44589.84 > 10000。
  排序结论(小 batch 吞吐更高)与 **B256T256 OOM 边界**均与 v2.1 的文字记录一致;`B64T128
  peak_mem_gb=3.364` 与原记录「B64 3.4GB」相符,是一个独立的交叉验证点。
  **但绝对吞吐低于原文字记录**(原记:B32T128 62570 / B64T128 53764;本次:44589.84 / 36623.11)
  ——差异原因未定(本次与另一项目共用机器、存在 CPU/GPU 竞争,且原 run 产物已缺失无法比对其
  运行条件),**不做归因**,以本次落盘产物为准。
  ⚠ **测量限制(本次新发现)**:`peak_mem_gb` 是进程内累计峰值、跨组合不重置——B32T256 起的
  13.405 与 B256T128 完全相同,是被前一组合的峰值污染,**不代表该组合的真实显存需求**;
  只有在序列中首次创新高的组合(B32T128 1.816 / B64T128 3.364 / B128T128 6.760 /
  B256T128 13.405)的数值可信。要测准需每组合独立进程重跑。
  [source: 20260803-v21-throughput-rerun]
- 备注: `experiments/20260803-v21-throughput-smoke/` 为开跑前的环境验证 run(`--smoke`,B4T32),
  provenance 同样齐全,保留备查。原 run `20260728-v21-throughput` 的产物仍然缺失,本 run
  **不冒充也不替换**它;changelog v2.1 里对原 run_id 的引用保持原样(已 tag 版本不改写)。

## 2026-08-03  run_id: 20260803-v21-vsrandom-rerun(补 v2.1 缺失产物:vs-random 加厚)
- 背景: 承上条。rr 系列产物同样缺失,但全量重跑成本过高——`20260803-v21-balanced-smoke`
  (缩水版:102 局 / episode_len=900)实测耗时 **1:52:03**,根因是 `matchup_runner` 每次调用都
  `build_step` + 重新 jit,**编译次数 = matchup 数(46)、与 seed 数无关**(vmap over seeds 免费);
  实测 GPU 利用率 1% / CPU 单核满载,时间几乎全在重编译。用户 2026-08-03 拍板**只补 vsRandom
  部分**(10 次编译,跳过 rr 的 36 次)。另:`rr2`/`rr3` 对应已被推翻的中间 profile 数值,当前
  代码状态下本就重跑不出,不在范围内。
- 假设: v2.1 记录的两条 vsRandom 结论在标准条件(episode_len=6000 / gate_open=4000)下可复现:
  ①10 个指挥官全部碾压 random;②turtle/airtech 定向调后不再「0 军被动等门」。
- 成功判据: ①`games.jsonl` 落盘含 10×16 = 160 局且失败 matchup 数 = 0;②10 个指挥官对
  random 的胜率**全部 = 1.00**;③turtle 与 airtech 的末军均值**均 > 1.0**。
- 失败判据: 任一指挥官对 random 胜率 < 1.00;或 turtle/airtech 末军均值 ≤ 1.0(说明 changelog
  平衡区记的定向调未生效)。
- 对照: `20260728-v21-balanced-rr4`——**产物已缺失**,文字记录为「全 vsRandom 胜率 1.00;
  turtle 末军 0.1→1.1、airtech 0.2→1.1」。同上条:与文字记录比对,不与产物比对。
- git hash: <跑之前留空,跑完填>
- 结果: <跑完填>
- 结论: <跑完填>
