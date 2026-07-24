---
description: "版本循环纪律:commit 规范、changelog 格式契约、版本完成定义、issue.md 协议。无 paths,无条件加载——commit 规范需要在每个实现 session 在场,而 commit 走 Bash 不会触发 paths 匹配。"
---

# 版本循环纪律

本项目按版本推进(v1.0 → v1.1 → …),需求唯一入口是根目录 `issue.md`(按版本分节)。

## issue.md 协议(用户定义,优先级最高)

- 用户往文末「草稿箱」写口语化想法;Claude 看不懂就问,**完全吃透后**把它改写进
  对应版本的规格区并清空草稿(通透版,不保留初稿)。规格区由 Claude 维护,
  但**语义必须与用户草稿等价**——吃透≠自由发挥,歧义必须问过才能写。
- 实现只以规格区为依据;规格与代码冲突时以 issue.md 为准并停下来对齐。

## 版本完成定义(五件套,缺一不算收尾)

1. `pytest -x -q` 与 `ruff check src/ tests/` 全绿(以真实输出为证,不以转述为证)
2. `engine-auditor` 无上下文审核已跑,P0 清零(P1 用户在线则裁决,离线则按最合理
   解释处理并记 docs/DECISIONS.md;P2 记入 changelog)
3. `docs/changelog/vX.Y.md` 已写(格式见下,契约唯一定义处在本文件)
4. `git tag vX.Y` 已打并 push(用户已授权 agent 执行;这偏离通用模板的「tag 留人工」
   默认,依据见 docs/DECISIONS.md)
5. `/handoff` 已落盘并 commit

## commit 规范

- message 以版本号开头:`vX.Y <功能短语>: <一句话说清这个 commit 做了什么>`
  例:`v1.0 采矿: 工人满载后自动寻路返回大本营并入库`
- 一个功能点(≈ plan 的一个 phase)一个 commit;phase 验证通过即提交,不攒大 diff。
- 记忆文件(HANDOFF/MEMORY/current-focus/research-log)的更新单独 commit:
  `vX.Y chore: session 交接落盘`
- commit 与 push 已获用户永久授权直接执行;force-push / 改历史仍然禁止。

## changelog 格式契约(唯一定义处;/version-close 逐字照抄,不做等价改写)

路径 `docs/changelog/vX.Y.md`,四节固定,空节写「无」:

```markdown
# vX.Y — <一句话主题>(YYYY-MM-DD)

## 新增
- <功能一句话>(commit <hash 前 7 位>)

## 修复
- <bug 现象 → 修法>(commit <hash>)

## 平衡
- <数值改动:参数名 旧值 → 新值,为什么>

## 已知问题
- <engine-auditor 的 P2、/validate 的「仅记录」落这里;带 文件:行号>
```

- 每条尽量挂 commit hash——changelog 是「版本对外说明」,git log 是底账,两者靠 hash 互指。
- 平衡数值若来自模拟实验,须带 `[source: <run_id>]`(与 rules/notes.md 三标签制度一致)。
