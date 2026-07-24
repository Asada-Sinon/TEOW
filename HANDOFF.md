# HANDOFF

**本文件当前是空模板，还没有任何真实历史。** 下面只有格式说明和一段被注释掉的示例。

这不是文档，是上一个 agent 写给下一个 agent 的信。要短、要具体、只写下次用得上的。
不写背景介绍，不写「本项目旨在……」——那些在 CLAUDE.md 里。

规矩：
- 新会话结束时加一节，**最新的在最上面**。
- 只保留最近 3 节，更旧的直接删掉（历史在 git 里，不用囤在这）。
- `PENDING` 是下一个 agent 开工的第一件事，必须写成可执行的动作，不是「继续优化」。
- 教训不要写这里，写 `MEMORY.md`：HANDOFF 会过期，教训不会。
- 提到实验产物时路径一律 `experiments/<run_id>/`，run_id 格式 `YYYYMMDD-<slug>`。

格式：

```markdown
## Session YYYY-MM-DD
- 完成: ...
- PENDING: ...        ← 下次第一件事
- 坑: ...
```

<!-- 示例（安装后请删除这整块）
以下为格式示例，不是本项目的真实历史。这里出现的日期、文件名、run_id、数字全部虚构，
任何 agent 都不得把它们当作本项目的事实、进度或依据。

## Session 2026-03-14
- 完成: 把 dataloader 的 shuffle 挪到 sampler 层，`tests/test_loader.py` 全绿
  （`pytest -x -q tests/test_loader.py`，17 passed）。
- PENDING: `src/train.py:118` 只存了 config 路径，没落盘 resolved config，违反
  CLAUDE.md 硬约束第 3 条。下次第一件事：把展开后的 dict dump 成
  `experiments/20260314-shuffle-sampler-seed0/config.resolved.yaml`，再补跑一次验证。
- 坑: 直接 `python src/train.py` 用的是系统 python，缺 torch；必须用 CLAUDE.md
  命令区里那个解释器的绝对路径。
-->

---

<!-- 真实的 session 记录从这一行下面开始写，最新的一节永远插在紧挨本行的下面。 -->

## Session 2026-07-25(用户睡眠期自主推进)
- 完成: v1.0 全部落地并打 tag——工作流特化(versioning 规则/engine-auditor/
  /version-close)、issue.md 通透版、纯 JAX 引擎全链路、14 测试绿、两轮无上下文
  审计(P0-1 对向工人流死锁已修复并复审确认)、docs/changelog/v1.0.md。
  对局基准:scripted 互打 700-1200 tick 分胜负,random 双向皆负于 scripted
  (experiments/20260725-v1.0-audit2/ 等)。
- PENDING: ①用户醒后复核 docs/DECISIONS.md 全部 [AI-DRAFT] 条目(尤其三处
  「调研报告建议 vs 用户指示」裁决与 tag 由 agent 执行);②用户看一眼
  experiments/20260725-scripted-v-scripted/replay.gif 感受对局节奏,对
  config.py 数值提改动意见;③下一版本 v1.1(升级系统)走 /plan 读 issue.md
  第 63-71 行起手。
- 坑: 跑任何 python 一律 `.venv/bin/python`(py3.12+jax0.6.2 钉版);测试/门禁
  必须 `JAX_PLATFORMS=cpu`(GPU 每个 Config 变体重新 jit,5min vs 14s)。
