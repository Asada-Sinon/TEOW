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

## Session 2026-07-25(晚,v1.2 全程,v1 引擎收官)
- 完成: v1.2 打 tag 收官——连续 360° 移动(浮点坐标/场梯度寻路/圆形互推)、
  兵营+狗子、哨塔、浏览器观战(run.py serve + Canvas 矢量 + PNG 替换槽 +
  提示词包)。终审两轮(P0-1 存量狗补血已修复复审关闭),33 测试绿,
  守恒/决定论/连续移动专项不变量全零违例(experiments/20260725-v1.2-audit2/)。
- PENDING: ①**用户浏览器验收前端**:`.venv/bin/python src/run.py serve
  experiments/20260725-v1.2-frontend-demo` → http://127.0.0.1:8000/(headless
  只验到数据契约);②用户复核 DECISIONS 的 v1.2 条目(兵营升级顺延/melee
  1.5 vs reach 1.2/决定论口径收窄);③下一版本 v2 RL——先与用户对齐范围再
  /plan(旧 PPO 方案存 docs/plans/20260725-v1.2-continuous/ 之前的 v1 计划里,
  avail_actions 掩码与 controller 接口已为 RL 预留)。
- 坑: npz 惰性解压循环里用会平方级卡死(必须一次物化);server.py 开着
  future annotations,fastapi 类型必须模块级导入否则 WS 403;serve 前必须
  JAX_PLATFORMS=cpu(run.py 已内置);审计重放的 key 流必须 split(key,3)
  与 run.py 一致。

## Session 2026-07-25(下午,用户在线,v1.1 全程)
- 完成: v1.1 升级系统全部落地并打 tag——用户改需求(升本零单位加成,改为技能
  训练营双线研发,上限链 线≤营≤基地)→ plan+plan-critic(抓到扣费透支 BLOCKER)
  → 6 phase 实现 → 两轮无上下文审计(P0-1 跨营并研双倍扣费已修复复审关闭)
  → changelog v1.1。25 项测试绿。对局涌现「攀科技 vs 爆兵」双路线均有胜局
  (experiments/20260725-v1.1-audit2/ 等)。
- PENDING: ①v1.2(兵营/狗子/哨塔/浏览器前端)走 /plan 读 issue.md v1.2 节;
  实现前先做 changelog v1.1 已知问题里的三件工具债:builder 超时自愈、
  解锁表扩表结构、审计对账脚本补「同 tick 死亡交互」口径;②1 宽走廊回归用例
  (v1.0 遗留,v1.2 若加地形墙必须先补)。
- 坑: 同 tick 多笔支出必须走 paid_orders_pass 顺序对账,任何新付费动作不得在
  apply_orders 直接扣费;负数 btype 任务完成必须 btype←0;测试给富开局要用
  Config(start_ore=..., start_water=...) 不要改默认值。

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
