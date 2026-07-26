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
