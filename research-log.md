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
