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
