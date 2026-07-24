# docs/plans

每个任务一个目录：

```
docs/plans/<YYYYMMDD>-<slug>/
├── research.md   # 调研：读了哪些文件、现状如何、约束是什么、有哪几条路
└── plan.md       # 方案：分步骤、每步的验证方式、明确的不做项
```

例：`docs/plans/20260320-sampler-shuffle/`。slug 用小写连字符，短到能一眼认出。

## 为什么要落盘进 git

对话历史是易失的：compact 一次、换个会话、机器一重启，推理过程就没了，
下一个 agent 只能重新读一遍代码库再想一遍——同样的钱付第二次。

plan 文件是持久的、可移植的、只付费一次的。
写进 git 之后它能被 diff、被 review、被别的机器上的别的 agent 直接读走，
也能在实现跑偏时作为「当初说好要做什么」的凭据。

## 怎么用

1. `/plan` 先产出 `research.md`，停下来。
2. 你 review `research.md`，确认现状判断没错。
3. 再产出 `plan.md`，停下来。
4. 你 review `plan.md`，改到满意为止。
5. `/impl` 按 `plan.md` 执行，每步做完对照验证方式。

**你只需要认真 review 这两个文件。** 前面把方案审对，后面的实现基本就是照抄；
反过来在几百行 diff 里找方向性错误，代价高得多。

参考：https://cheesecakelabs.com/blog/plan-mode-claude-code/
