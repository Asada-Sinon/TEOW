# TEOW「版本循环」工作流特化设计报告

日期:2026-07-25
性质:调研 + 可落地设计。本报告只调研和给草稿,不改任何配置文件;落地由后续 session 按 §9 的顺序执行。
断言标注:全文区分 **【官方】**(带 URL)/ **【模板设计文档】**(出自 workflow 仓库 docs/,该文档自己也区分了官方与判断)/ **【本报告判断】**(可以不同意并改掉)。

---

## 0. 结论速览

TEOW 的「按版本推进 + 每版本一个闭环」需求,与已装的 R-P-I-V 模板**高度同构**:一个版本 = 一次放大的 R-P-I-V 循环 + 一段模板没有的「版本收尾」。因此:

1. **7 个 skill、4 个 subagent、4 个 hook、3 条 rules 全部保留**,一个都不删(按需加载,闲置成本≈0);其中 5 个核心 skill(kick/plan/impl/validate/handoff)直接复用,零改动。
2. **新增 3 个文件**:`.claude/rules/versioning.md`(版本纪律,无 `paths:`,无条件加载,理由见 §4.1)、`.claude/agents/engine-auditor.md`(无上下文引擎审核员——issue.md 里 v1.0 明确要求的那个角色)、`.claude/skills/version-close/SKILL.md`(收尾流程壳)。完整草稿在 §4,可直接照抄。
3. **不做** `/issue-sync` skill(§4.4)、**不做** DECISIONS.md(§7.3)——两者的职责已被现有文件覆盖,加了违反模板「同一条信息只写一处」的防重叠原则。
4. **两处偏离模板默认值**,均已写明理由:把 `Bash(git commit:*)` 加进 allow(用户明确要求 agent 功能级频繁提交,§3.3);启用 `verify.sh` 完成门禁(pytest+ruff 全绿是用户定义的版本硬条件,§4.5)。`git tag` 保持人工——它是版本发布 gate。

---

## 1. 现状盘点:现有模板机制读透

### 1.1 TEOW 项目现状

- git 仓库已 init,仅 1 个 commit(`345d01d init: workflow scaffold (R-P-I-V)`),`src/`、`tests/` 尚不存在。
- `CLAUDE.md` 的测试/lint 命令还是 `# TODO(填测试命令)` 占位符——**这是第一件要修的事**,否则 `/kick` 核验、`/impl` 验证、verify.sh 全部悬空。
- `verify.sh` 未启用(只有 `.example`,模板刻意默认关闭)。
- `issue.md` 已有 v1.0–v1.2 三个版本的真实需求,v1.0 段末尾明确写了「需要你自己指定没有上下文的 agent 审核」——engine-auditor(§4.2)就是对这句话的机制化。
- `docs/plans/20260725-jax-rts-engine/` 已存在但为空(疑似另一 session 建的,本报告不动它)。

### 1.2 组件清单与版本循环中的角色

| 组件 | 干什么 | 版本循环中的角色 |
|---|---|---|
| `/kick` | session 开场:读 HANDOFF/MEMORY/focus + git 核验,输出三段 | **直接复用**,每个新 session 第一步 |
| `/plan` | Research(派 subagent 调研)→ plan.md(分 phase、四问精度)→ plan-critic 对抗审 | **直接复用**,做「版本 plan」:输入改为 issue.md 的 vX.Y 段 |
| `/impl` | 一轮一个 phase,跑 plan 里的验证命令并贴真实输出 | **直接复用**,phase 粒度 = 一个功能级 commit |
| `/validate` | 派 plan-critic 对照 plan 验收 diff,三档输出,带反过度工程黑名单 | **直接复用**,版本内验收(工程正确性) |
| `/handoff` | 更新 HANDOFF/MEMORY/research-log/current-focus,派 claim-verifier | **直接复用**,版本收尾最后一步 |
| `/exp` | 先写判据再跑,run 目录落六项 provenance,派 result-analyst 读日志 | **保留、重定位**:对局模拟/平衡实验/性能 benchmark/后期 RL 训练都走它(§5) |
| `/lit` | 文献入库,fork context,DOI 强制 | **保留、现阶段闲置**,RL 指挥官阶段启用(§5) |
| `plan-critic` | 对抗审 plan 与 diff,必须实读代码核对假设 | **直接复用**(被 /plan、/validate 派) |
| `claim-verifier` | 把文档量化断言映射回代码与产物,四分类判定 | **保留**,平衡数值/性能数字进 research-log 后有用 |
| `result-analyst` | 读实验日志出带出处的数字,禁止心算 | **保留**,对局日志统计、benchmark 对比走它 |
| `lit-reviewer` | 文献检索(唯一带 memory 的 agent) | **保留、现阶段闲置** |
| `session_context.py` | SessionStart(startup/resume/clear/compact)把 HANDOFF/focus/MEMORY/git 状态注入 context | **零改动**。整套跨 session 交接的地基 |
| `protect_paths.py` | PreToolUse 拦 Edit/Write:已有产物阻断、新建放行;`!` 规则无条件阻断 | **零改动**。对局日志/RL checkpoint 落 `experiments/` 后受它保护 |
| `format_lint.py` | PostToolUse 跑 ruff format/check,永不阻断 | **零改动**(项目装 ruff 后自动生效) |
| `verify_stop.py` | Stop 门禁:跑 `.claude/verify.sh`,不过 exit 2 打回 | **启用它**:cp example 并填 pytest+ruff(§4.5) |
| `rules/experiments.md` | 产物区只读纪律 + run 六项必备(paths 条件加载) | **零改动**,语义完全适用于对局日志 |
| `rules/python-research.md` | 种子显式、config 走文件、不改全局状态(paths: `**/*.py`) | **零改动**,对 JAX 引擎同样成立(PRNGKey 显式传递、数值参数进 config) |
| `rules/notes.md` | 三标签制度([AI-DRAFT]/[HUMAN-VERIFIED]/[source:]) | **零改动** |
| `settings.json` | 权限白名单 + 4 hook 挂载 | **小改**:allow 加 `Bash(git commit:*)`(§3.3) |

### 1.3 必须记住的四条机制事实(决定后面所有设计)

1. **写进文件的东西比留在对话里的活得久。** SessionStart hook 在 startup/resume/clear/compact 四个 source 上都会把 HANDOFF/focus/MEMORY 重新读盘注入(`session_context.py` 文件头注释;matcher 在 settings.json 里四个全写)。所以版本循环的一切关键状态(plan、changelog、审核结论)都必须落盘,不能依赖对话。
2. **skill 的 `allowed-tools` 不是硬限制**,只是「本轮免批准范围」——【模板设计文档,实测于 2.1.218】(DESIGN_RATIONALE §4.7)。所以 engine-auditor「只读」靠的是 subagent 的 `tools:` 字段 + 正文纪律 + 它拿不到主对话历史,而不是 skill 层。
3. **hook 只拦 Edit/Write,拦不住 Bash 的重定向和 rm**(DESIGN_RATIONALE §7.1 #2)。`rules/experiments.md` 是 Bash 路径上唯一防线,且带 `paths:` 的 rules 压缩后不自动重载(【官方】https://code.claude.com/docs/en/context-window ,由模板文档引用)——三条 rules 都写了「自查信号」作补偿。新增的 versioning.md 之所以不带 `paths:`,一半原因在此(§4.1)。
4. **不要装第二套约定**(ADOPTING.md 核心原则)。版本循环设计里凡是模板已有等价物的(plan 格式、验收流程、记忆文件分工、run 目录规范),一律沿用,只补真正缺的:版本纪律、无上下文引擎审核、收尾流程壳。

---

## 2. Claude Code 机制能力确认(官方依据)

以下由子 agent 查证官方文档(code.claude.com/docs)返回,已与模板实际用法交叉核对。**给出的骨架可直接照抄。**

### 2.1 自定义 Skill —— 【官方】https://code.claude.com/docs/en/skills.md

已确认的 frontmatter 字段(本项目用到的子集):`name`(默认目录名)、`description`(Claude 判断何时自动调用的依据)、`argument-hint`、`disable-model-invocation: true`(仅用户可 `/name` 触发)、`allowed-tools`(激活期免权限提示的工具)、`context: fork`(隔离子上下文运行)、`background`(配合 fork,`false` = 等结果;2.1.218+ fork 默认后台,所以 `/lit` 显式写了 `background: false`)、`model` / `effort` / `paths` / `hooks` 等亦官方支持。`$ARGUMENTS` 占位符官方支持(还有 `$0`/`$1` 位置参数);skill 正文不含 `$ARGUMENTS` 时参数会自动以 `ARGUMENTS: <value>` 追加。`/name` 调用 skill 是官方行为。

骨架(照抄):

````markdown
---
name: my-skill
description: "一句话:干什么 + 什么时候用(Claude 靠它决定是否自动调用)"
argument-hint: "[参数提示]"
disable-model-invocation: true   # 阶段转换类 skill 必须 true,只允许人触发
allowed-tools: Read, Write, Glob, Grep, Task, Bash
---

# 标题

任务:$ARGUMENTS

## 第 1 步 ...
````

### 2.2 自定义 Subagent —— 【官方】https://code.claude.com/docs/en/sub-agents.md

frontmatter:`name`(必填,小写+连字符)、`description`(必填,自动委派依据)、`tools`(**省略则继承全部** subagent 可用工具——所以只读 agent 必须显式写)、`model`(`sonnet`/`opus`/`haiku`/完整 ID/`inherit`,默认 `inherit`)、`memory`(user/project/local,可选)、`maxTurns`/`permissionMode`/`isolation` 等亦官方支持。**subagent 拿不到主对话历史**——这是「无上下文审核」的机制基础(官方 sub-agents 文档;模板四个 agent 的「你的处境」一节都以此开头)。

骨架(照抄):

````markdown
---
name: my-agent
description: "干什么 + 何时派给它(写具体触发语句,便于自动委派)"
tools: Read, Grep, Glob, Bash
model: inherit
---

# 你的处境
你是一个 subagent,拿不到主对话历史……

# 工作流程 / 返回格式 ...
````

### 2.3 Hooks —— 【官方】https://code.claude.com/docs/en/hooks.md

- 事件:`SessionStart`/`SessionEnd`(每 session)、`UserPromptSubmit`/`Stop`(每 turn)、`PreToolUse`/`PostToolUse`(每次工具调用)。
- matcher:字母数字为精确匹配(`Edit|Write`),含其他字符按正则;`"*"`/省略 = 全匹配。
- exit code:`0` 成功;`2` **阻断**并把 stderr 回给 Claude;其余非阻断。
- SessionStart:`source` 取值 `startup|resume|clear|compact|fork`;stdout/`additionalContext` **确实注入 context**——`session_context.py` 的全部依据。
- Stop:官方示例含 `last_assistant_message`。**注意:子 agent 本次在当前官方文档中未找到 `stop_hook_active` 字段**,而 `verify_stop.py` 第一步就短路检查它(防「hook 打回 → 又想停 → 又打回」死循环)。【本报告判断】该字段在官方 hooks 文档的历史版本中出现过、社区广泛使用;即使字段真不存在,`data.get("stop_hook_active")` 返回 None 也只是不短路,另有【模板设计文档】引用的官方行为兜底:Stop hook 连续阻塞 8 次会被强制覆盖(DESIGN_RATIONALE §7.1 #5)。**结论:保留现有代码,不动;** 但落地时值得做一次 10 秒实测(故意让 verify.sh 失败,看会不会循环)。

### 2.4 `.claude/rules/*.md` —— 【官方】https://code.claude.com/docs/en/memory.md

官方支持("path-scoped rules"):`paths:` frontmatter 是 glob 列表,**读/编辑匹配文件时**该规则才载入 context;**无 `paths:` 时无条件加载,与 CLAUDE.md 同级、launch 时载入**。目录递归扫描,支持子目录。这直接支撑 §4.1 的选择:versioning.md 不带 `paths:`。

---

## 3. 推荐的版本循环流程(核心设计)

### 3.1 文字流程图

一个版本 vX.Y 从需求到 tag 的完整闭环。标注:【人】/【主 agent】/【subagent:名字】;→ 后是落盘产物。

```
【人】往 issue.md 追加/修改 vX.Y 需求段(唯一需求入口;agent 不改此文件)
  │
  ▼ 新 session
【主 agent】/kick
  读 HANDOFF/MEMORY/current-focus + git 核验 → 输出「我在哪/遗留/下一步」,人选定「本 session 做 vX.Y 的哪部分」
  │
  ▼
【主 agent】/plan 实现 vX.Y(读 issue.md 的 vX.Y 段,只读这一段)
  ├─ 派【subagent:Explore(内置)】并行调研 src/ 现状(v1.0 时 src 为空,此步退化为技术选型调研)
  ├─ 写 docs/plans/<YYYYMMDD>-vX.Y-<slug>/research.md(相关文件/数据流/根因假设/既有约束)
  ├─ 写 同目录/plan.md(分 phase;每 phase 可运行可验证可提交;含「不在范围内」和端到端验证)
  │   ★ 版本特化:每个 phase 就是一个「功能级 commit」的粒度;phase 名即未来 commit message 的功能短语
  └─ 派【subagent:plan-critic】对抗审(实读代码核对假设)→ 逐条处理进 plan.md
  │
  ▼
【人】gate:按 5 问 review research.md + plan.md(唯一需要人认真看的东西)
  │
  ▼ /clear(plan 已落盘,对话可弃)
┌──────── 每个 phase 循环 ────────────────────────────────────────────┐
│ 【主 agent】/impl <plan 路径>  一轮只做一个 phase                      │
│   跑该 phase 的验证命令,贴真实输出,对照判据                            │
│   (PostToolUse: format_lint.py 每次编辑后自动 ruff)                   │
│ 【主 agent】git commit -m "vX.Y <phase 功能短语>: <一句话>"            │
│   ★ 偏离模板默认:commit 由 agent 执行(见 §3.3),tag 仍归人            │
│ (Stop: verify_stop.py 跑 verify.sh —— pytest+ruff 不绿停不下来)      │
│ 【人】扫一眼验证输出 → /clear → 下一个 phase                           │
└─────────────────────────────────────────────────────────────────────┘
  │ 所有 phase 完成
  ▼
【主 agent】/validate
  派【subagent:plan-critic】对照 plan.md 逐条验收 git diff(带反过度工程黑名单)
  → 必须修:回 /impl;建议修:人裁决;仅记录:进 plan.md「发现但未做」
  │
  ▼
【主 agent】/version-close vX.Y(新增 skill,§4.3)
  1. 门禁复核:pytest -x -q + ruff check 真实输出全绿;git 工作区干净
  2. 生成审计对局:跑脚本化对局 → experiments/<YYYYMMDD>-vX.Y-audit/(六项 provenance,受 protect_paths 保护)
  3. 派【subagent:engine-auditor】(★ 无上下文审核,§4.2)
     输入 = issue.md 的 vX.Y 段路径 + src/ 范围 + 对局日志 run 目录;输出 = P0/P1/P2 三档
  4. 分诊:P0(逻辑错误)必须修——回 /impl 或直接修后重跑 1-3;P1(规格偏差)人裁决;P2(平衡/瑕疵)进 changelog
  5. 写 docs/changelog/vX.Y.md(四节:新增/修复/平衡/已知问题;格式契约定义在 rules/versioning.md)
  6. 输出 tag 命令供人执行:git tag -a vX.Y -m "..."(agent 不自己打)
  │
  ▼
【人】核对 changelog → 执行 git tag vX.Y
  │
  ▼
【主 agent】/handoff
  HANDOFF.md 顶部加节 / MEMORY.md 追加本版本教训(仅真实纠正)/ current-focus.md 覆写为「下一版本」
  本版本改动可能推翻 research-log 既有结论时派【subagent:claim-verifier】
  → commit 记忆文件
  │
  ▼
【人】往 issue.md 写 v(X.Y+1) 需求 → 回到顶部
```

### 3.2 与 R-P-I-V 的映射(为什么几乎不用改模板)

| 版本循环步骤 | 模板对应物 | 差异 |
|---|---|---|
| 读 issue.md → 版本 plan | `/plan` | 仅输入源变化:`$ARGUMENTS` 写成「实现 issue.md 的 vX.Y 段」 |
| 分 phase 实现 + 验证 | `/impl` | phase = 功能级 commit 粒度(plan 阶段就按此拆) |
| pytest/ruff 全绿 | `verify_stop.py` + verify.sh | 从「可选」变「启用」 |
| 对照验收 | `/validate` | 零改动 |
| 无上下文引擎审核 | **缺** → 新增 engine-auditor | 模板已有同型 agent(plan-critic/claim-verifier)可抄结构 |
| changelog + tag | **缺** → versioning.md + /version-close | 纯新增 |
| 状态落盘 | `/handoff` | 零改动 |

【本报告判断】版本收尾没有塞进 `/validate` 或 `/handoff`,而是独立成 `/version-close`:`/validate` 的对象是「diff vs plan」(工程正确性),engine-auditor 的对象是「实现 vs issue 规格 + 引擎不变量」(领域正确性),两者审的东西、参照物、频次都不同(每 phase 可 validate,每版本才 close);`/handoff` 是 session 级动作,一个版本横跨多个 session,不能绑死。

### 3.3 两个关键取舍(偏离模板处,显式声明)

**① agent 自动 commit(偏离 DESIGN_RATIONALE §6.3「不自动 git commit」)。**
模板默认把 commit 留给人,理由是「回滚点该由知道自己想退回哪里的人放」。但用户对本项目的明确要求是「分阶段实现 + 功能级频繁 git 提交(message 以 vX.Y 开头)」——这是把 commit 从「人的 gate」改判为「循环的机械步骤」。【本报告判断】在本项目采纳用户要求,理由:(a) solo 项目,commit 不影响他人;(b) phase 粒度已在 plan 阶段被人审过,「一个 phase 一个 commit」的粒度实际上仍是人定的;(c) 频繁功能级 commit 恰好强化了模板自己论证的「git 是唯一可靠回滚点」(bash 改动 /rewind 救不了);(d) commit message 规范(vX.Y 前缀)写进 rules/versioning.md,格式可 grep 可审计。落地动作:`settings.json` 的 allow 加 `Bash(git commit:*)` 与 `Bash(git tag:*)` **不加**——tag 是版本发布宣告,保持每次弹确认,作为人工 gate 的机制兜底(这正是模板「把兜底放在 L1 权限层而非 skill 措辞层」的方法,DESIGN_RATIONALE §4.7)。

**② issue.md 的所有权与读法。**
【本报告判断】issue.md 归人独有:agent 只读、不写、不「整理格式」。需求歧义时问人,答案由人写回 issue.md 或由 agent 落进 plan.md 的假设区——不许 agent 替人改需求原文。plan 阶段只读当前 vX.Y 段(目前全文才 2.3KB,全读也无妨,但约定先立下,防它长大后每次全量入 context)。

---

## 4. 需要新增的文件:清单与完整草稿

### 4.1 `.claude/rules/versioning.md` —— 无 `paths:`,无条件加载

**为什么不带 `paths:`(这是本报告被要求回答的问题)。** 三个理由:
1. **触达时机错配。**【官方】带 `paths:` 的规则在「读/编辑匹配文件」时才加载(https://code.claude.com/docs/en/memory.md )。而本规则最需要在场的时刻是 **git commit 时**(message 前缀规范)——commit 走 Bash,不触发任何文件读写匹配;若设 `paths: ["docs/changelog/**"]`,规则只会在写 changelog 那一刻才出现,整个实现期的 commit 规范形同虚设。
2. **压缩存活性。**【模板设计文档,引官方 context-window 文档】带 `paths:` 的规则 compact 后不自动重载,只能靠「自查信号」这种建议性补偿;无 `paths:` 规则与 CLAUDE.md 同级加载,压缩后随记忆体系重新注入,更稳。版本纪律是「必须永远在场」类信息。
3. **体量撑得起常驻。** 草稿约 40 行(数百 token),远低于把它塞进 CLAUDE.md 会造成的稀释——CLAUDE.md 只放一行指针(§6),细则在此。这符合模板「CLAUDE.md 必须短,细节下沉」的分层(DESIGN_RATIONALE §4.1),同时避免了 skill 化(skill 只在被调用的 session 在场,而 commit 发生在每个 impl session)。

**完整草稿**(可直接写入 `.claude/rules/versioning.md`):

````markdown
---
description: "版本循环纪律:commit 规范、changelog 格式契约、版本完成定义。无 paths,无条件加载——commit 规范需要在每个实现 session 在场,而 commit 走 Bash 不会触发 paths 匹配。"
---

# 版本循环纪律

本项目按版本推进(v1.0 → v1.1 → …),需求唯一入口是根目录 `issue.md`(按版本分节)。
**issue.md 归用户:agent 只读,不写、不整理。** 需求歧义 → 问,不猜。

## 版本完成定义(五件套,缺一不算收尾)

1. `pytest -x -q` 与 `ruff check .` 全绿(以 verify.sh 真实输出为证,不以转述为证)
2. `engine-auditor` 无上下文审核已跑,P0 清零(P1 经用户裁决,P2 已记入 changelog)
3. `docs/changelog/vX.Y.md` 已写(格式见下,契约唯一定义处在本文件)
4. `git tag vX.Y` 已打(**由用户执行**,agent 只输出建议命令)
5. `/handoff` 已落盘并 commit

## commit 规范

- message 以版本号开头:`vX.Y <功能短语>: <一句话说清这个 commit 做了什么>`
  例:`v1.0 采矿: 工人满载后自动寻路返回大本营并入库`
- 一个功能点(≈ plan 的一个 phase)一个 commit;phase 验证通过即提交,不攒大 diff。
- 记忆文件(HANDOFF/MEMORY/current-focus/research-log)的更新单独 commit:
  `vX.Y chore: session 交接落盘`
- **tag 不由 agent 执行。** `git tag` 每次都会弹权限确认,那是刻意的 gate,不要绕。

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
````

### 4.2 `.claude/agents/engine-auditor.md` —— 无上下文引擎审核员

设计要点(结构照抄模板四个 agent 的成熟形态):
- **只读**:`tools: Read, Grep, Glob, Bash`,Bash 仅用于跑只读命令与确定性重放脚本;分析脚本写 `explorations/`(与 result-analyst 同一套写权限纪律)。subagent 的 `tools:` 字段是官方机制(§2.2),比 skill 的 allowed-tools 可靠。
- **不配 memory**,并在 HTML 注释里写明是刻意的——它的全部价值是 fresh eyes(§7.1)。
- **输入契约写死**:派它的 prompt 必须带三样(issue 规格段、src 范围、对局日志 run 目录),缺了它自己找并记入「本次假设」。
- **输出三档 P0/P1/P2** 对齐用户要的「问题清单分档」,并对齐 changelog 的「已知问题」节(P2 直接可搬)。
- **抗过度工程约束前置**——审核 agent 最大的失败模式是硬凑意见(/validate 与 plan-critic 的教训直接继承)。

**完整草稿**(可直接写入 `.claude/agents/engine-auditor.md`):

````markdown
---
name: engine-auditor
description: "无上下文引擎逻辑审核员,版本收尾(/version-close)时对 vX.Y 做规格对照审计。输入三样:issue.md 的版本规格段、src/ 引擎代码、experiments/ 下的对局日志。逐条核对规格是否如实实现、引擎不变量(资源守恒/tick 决定论/状态机合法性)是否成立、对局日志有无异常,输出 P0/P1/P2 三档问题清单。以下情况派给它:「vX.Y 收尾审核」「用没有上下文的 agent 查一遍引擎逻辑」「这局日志有没有 bug」。它只读不改,只判定不修复。"
tools: Read, Grep, Glob, Bash
model: inherit
---

<!--
  为什么本 agent 不配 memory:
  它的全部价值在于「不知道实现者当时怎么想」。写引擎的那个 context 已经被自己的
  实现思路污染,会倾向于认为自己写的是对的;而「记住上个版本审过资源守恒没问题」
  会让本 agent 下个版本跳过重审——恰恰版本迭代改的就是这些地方。
  每次从 issue 规格和磁盘上的代码、日志从零审起,是刻意设计,不是遗漏。
-->

# 你的处境

你是一个 subagent。**你拿不到主对话的历史记录**,只有 CLAUDE.md、本文件和派给你的那一条 prompt。
实现过程中的任何讨论、妥协、口头约定,你一概不知道——**这正是派你来的原因**。

1. 派你的 prompt 应当带三样:①issue.md 中本版本规格段(或其行号范围) ②src/ 审计范围
   ③对局日志 run 目录(experiments/<YYYYMMDD>-vX.Y-audit/)。缺了就自己找:issue.md 按
   版本分节取最新一节;日志取 experiments/ 下最近修改的 audit 目录。把补上的前提写进「本次假设」。
2. **不要停下来问**——按最合理解释审完,前提列进「本次假设」。
3. **issue 规格是唯一标准。** 规格没写的行为不算错(记 P2 或不记);规格写了而实现不同,
   不要脑补「实现者可能有理由」——那是 P1,让用户裁决。

# 先读这一段:抗过度工程约束(优先于检查清单)

- **P0 只有一条判据:能写出具体失效场景**——什么状态/什么输入 → 引擎产生错误结果
  (资源凭空产生、死单位继续采矿、同 seed 两次重放结果不同)。写不出场景的不是 P0。
- **风格一个字不提**:命名、注释、函数长度、"更 pythonic"、类型标注、"建议加日志"——全不写。
- **零 finding 是合法结果**:输出「逐条核过,未发现问题」+ 已核对清单即可,不要硬凑。

# 三层审核清单(按序过,第 1 层最重)

## 1. 规格对照(逐条)
把 issue 规格段拆成可核对的条目(如「矿是建筑,需工人进驻才有产出」「工人搬运回大本营
指挥官才能使用」「升级消耗资源且占用 tick」),逐条到 src/ 找到实现位置,给
`文件:行号` 证据,结论:已实现 / 未实现 / 与规格不符 / 规格歧义。
**必须实际读代码,不许只看函数名猜。**

## 2. 引擎不变量(RTS/JAX 特有)
- **资源守恒**:任一 tick,Σ(库存+在途搬运+未开采储量+已消耗) 恒定;建造/爆兵扣费与
  退款路径对账。用对局日志首尾状态核算,必要时写 explorations/ 脚本算,不心算。
- **tick 决定论**:同 seed 同指令序列重放,末态应逐位一致(JAX PRNGKey 显式传递,
  不依赖全局随机);若日志里有重放对,核对;没有则报「未验证,建议补重放对」。
- **状态机合法性**:建造/升级/生产的前置条件与 tick 时序(冷却中不可重复下单、
  资源不足不可开建、建筑血量归零后其占用/产出是否正确清除)。
- **死亡与归属边界**:工人死亡时背着的资源去哪;矿被摧毁时驻内工人如何处理;
  同 tick 两方争抢同一资源点如何裁决——规格没写的,列为「规格歧义」P1/P2。
- **JAX 陷阱**:jit 下 Python 分支(应为 lax.cond/select)、动态 shape、int 溢出、
  where 双分支 NaN、被 vmap 的函数里的副作用。只报能指出具体错误后果的。

## 3. 对局日志异常
读 audit run 的日志/metrics:数值跳变(资源负数、血量超上限)、单调量倒退(tick 计数)、
死锁迹象(全体单位长期 idle)、明显失衡(一方零对抗获胜且规格未预期)。
统计一律脚本算(写 explorations/,自解释文件名),禁止目测报数。

# 返回格式(照抄;目标 1000–2000 token,不要贴代码/日志大段)

```markdown
## 审计对象
- 版本:vX.Y;规格:issue.md <行号范围>;代码:src/ <范围>;日志:experiments/<run_id>/

## 本次假设
-(没有则写「无」)

## 规格对照表  ← 必填,一条不落
| # | 规格条目 | 结论 | 证据 |
|---|---|---|---|
| S-1 | 矿需工人进驻才有产出 | 已实现 | src/economy.py:88 |

## P0 必须修(引擎逻辑错误,有具体失效场景)
### P0-1 <一句话>
- 失效场景:<状态/输入 → 错误结果>
- 证据:`文件:行号` / 日志 `run/文件:行`
- 建议修法:<一句话方向,不写实现>

## P1 规格偏差 / 规格歧义(用户裁决)
### P1-1 <一句话> — 规格原文 vs 实现现状,各带出处

## P2 平衡与瑕疵(记入 changelog「已知问题」)
- <一行一条,带 文件:行号>

## 本次产出的脚本
- explorations/<name>.py — <回答了什么>

## 建议下一步
- <例:先修 P0-1,它使资源守恒不成立,P2 的平衡观察全部失真>
```
````

### 4.3 `.claude/skills/version-close/SKILL.md` —— 值得做

**为什么值得**:收尾是一条固定的 6 步序列(门禁→对局→审核→分诊→changelog→tag 提议),每版本执行一次、跨度大、容易漏步(尤其「派审核前先生成新鲜对局日志」和「P2 搬进 changelog」这种跨文件搬运)。这正是 skill 的适用形态:流程壳,规定「每步产出什么文件、什么条件算做完」。设 `disable-model-invocation: true`——版本收尾是阶段转换,必须人触发(与 kick/plan/impl/validate/handoff 同一判据,DESIGN_RATIONALE §4.2)。

**完整草稿**(可直接写入 `.claude/skills/version-close/SKILL.md`):

````markdown
---
name: version-close
description: "版本收尾:门禁复核→生成审计对局→派 engine-auditor 无上下文审核→分诊→写 docs/changelog/vX.Y.md→提议 git tag。一个版本的所有 phase 实现完且 /validate 通过后,由用户触发。"
argument-hint: "[版本号,如 v1.0]"
disable-model-invocation: true
allowed-tools: Read, Write, Glob, Grep, Task, Bash
---

# 版本收尾

版本:$ARGUMENTS(留空则从最近 commit message 的 vX.Y 前缀推断,并向用户确认)

> `allowed-tools` 给了完整 Bash,因为要跑测试和对局脚本;它是「本轮免批准范围」,
> 不构成硬限制。收尾纪律的定义端在 `.claude/rules/versioning.md`,本 skill 是执行壳。

## 第 0 步 前置检查(不满足就停,不要硬闭环)

- `/validate` 已跑过且「必须修」清零——没跑就先去跑,本 skill 不替代它。
- `git status --short` 干净(所有 phase 已按 `vX.Y <功能>` 规范 commit)。脏 → 先提交再回来。

## 第 1 步 门禁复核

跑 `pytest -x -q` 和 `ruff check .`,**贴真实输出**。任一不绿 → 停,报告,不进下一步。
(Stop hook 的 verify.sh 也会拦,但收尾必须留下一份显式的全绿证据,不靠「没被拦=绿」。)

## 第 2 步 生成审计对局

用受版本控制的脚本跑一场(或多场)覆盖本版本功能的脚本化对局,产物落
`experiments/<YYYYMMDD>-vX.Y-audit/`,按 rules/experiments.md 落齐六项
(git hash / resolved config / seed / 启动命令 / 日志 / 机器可读 metrics)。
**必须是收尾时新跑的**——旧日志审不出新 commit 的问题。
对局脚本本身若还没有,这是本版本 plan 的遗漏:停下来告诉用户,补一个最小脚本再继续。

## 第 3 步 派 engine-auditor(无上下文审核)

派 `engine-auditor` subagent,prompt **必须带三样**:
1. issue.md 中 vX.Y 规格段的行号范围
2. src/ 的审计范围(本版本动过的模块为主,可给 `git diff --stat <上个tag>..HEAD` 的文件清单)
3. 第 2 步的 run 目录路径

不要附加任何「实现时我们决定…」的背景——**无上下文是它的价值,不是缺陷**。

## 第 4 步 分诊(逐条处理,不要只转述)

- **P0**:必须修。小改直接修并按规范 commit;伤筋动骨走 /plan。修完**重跑第 1–3 步**
  (审核对象变了,旧审核作废)。
- **P1**:列给用户裁决:改实现,还是改 issue 规格(由用户改 issue.md)。
- **P2**:原样搬进 changelog 的「已知问题」。

## 第 5 步 写 changelog

写 `docs/changelog/vX.Y.md`,**格式逐字照抄 `.claude/rules/versioning.md` 的契约**,
不做等价改写。素材:`git log --oneline <上个tag>..HEAD` 的 vX.Y commit(新增/修复)、
本版本数值改动(平衡,带 [source: run_id])、第 4 步的 P2(已知问题)。

## 第 6 步 收尾输出

1. 五件套核对清单(versioning.md 的版本完成定义),逐项 ✅/❌
2. 给用户可直接粘贴的命令:`git tag -a vX.Y -m "<changelog 的一句话主题>"`
   **不要自己执行 tag。**
3. 提醒:「tag 打完后跑 /handoff,把版本收尾状态落盘。」

## 硬约束

- 不跳过第 2 步直接派审核(没有新鲜对局日志的审核只剩静态读码,漏掉整个动态不变量层)。
- P0 未清零不写 changelog、不提议 tag。
- 本 skill 不改 issue.md。
````

### 4.4 `/issue-sync` —— **不做**,理由

【本报告判断】不值得单列 skill:①issue.md 现在 2.3KB,直接读的成本低于一次 skill 加载;②「吃透增量」的真正动作是 diff 需求与现状,而这恰是 `/plan` 的 Research 段的本职(它会派 subagent 查 src 现状再对照需求),单独做一个 /issue-sync 等于把 /plan 的前半截复制一份,两处必然漂移(模板反复强调格式/流程契约要唯一定义处);③版本边界上「上个版本遗留 + 新版本需求」的合并视角,已由 /kick(读 HANDOFF 的 PENDING)+ /plan(读 issue 段)组合覆盖。**替代做法**:versioning.md 里立下「issue.md 按版本分节、plan 只读当前节」的约定(§4.1 草稿已含);若未来 issue.md 长到几百行,再考虑给 issue.md 建目录化结构,而不是加 skill。

### 4.5 配套小改动(非新文件)

| 改动 | 内容 | 理由 |
|---|---|---|
| `CLAUDE.md` 填 TODO | 测试 `pytest -x -q`、单测 `pytest tests/test_foo.py`、lint `ruff check .`、解释器按实际(建议 uv/venv 固定) | 所有验证链路的地基;现在是占位符 |
| 启用 verify.sh | `cp .claude/verify.sh.example .claude/verify.sh && chmod +x`,保留 pytest+ruff 段,删 Node 段 | 用户把「pytest/ruff 全绿」定为版本硬条件 → 从建议升为确定性门禁(模板四层护栏的 L2)。注意保持 60 秒内:引擎测试变慢后把慢测试挪 pytest -m "not slow" |
| `settings.json` | allow 加 `"Bash(git commit:*)"`;**不加** `Bash(git tag:*)` | §3.3 的取舍;tag 弹确认即人工 gate 的机制化 |
| 建 `docs/changelog/` 目录 | 空目录 + 可选一行 README 指向 versioning.md | changelog 落点 |
| `CLAUDE.md`「项目特定」节 | 见 §6 清单 | — |

---

## 5. 现有 7 个 skill 的取舍建议

| skill | 建议 | 理由 |
|---|---|---|
| /kick /plan /impl /validate /handoff | **保留,零改动** | 版本循环的骨架本身(§3.2) |
| /exp | **保留,重定位使用场景** | 本项目的「实验」= 平衡模拟、性能 benchmark(tick 吞吐)、后期 RL 训练。六项 provenance + 先写判据对平衡调参尤其值钱(「把矿产出从 5 调到 8 是否让开局更快」就是一次带判据的实验);/version-close 第 2 步的审计对局若做成对照实验也走它。**不删的最硬理由:后期 RL 阶段它是主力。** |
| /lit | **保留,现阶段闲置** | 引擎实现期(v1.0–v1.2)基本用不上;RL 指挥官阶段查 RTS-RL/MARL 文献(SMAC、JaxMARL 一类)时直接启用。skill 按需加载(【官方】https://code.claude.com/docs/en/skills.md ),闲置成本≈0;删了以后还得装回来,且 lit-reviewer 的 project memory(已检索 query 查重)从零开始。ADOPTING.md 的原则是「组件级取舍」,但那是针对**有冲突等价物**的场景——本项目没有冲突,只有闲置,闲置不删。 |

4 个 subagent 同理全留:plan-critic 高频(每次 plan/validate)、result-analyst 中频(对局/benchmark 分析)、claim-verifier 低频(research-log 数值核查)、lit-reviewer 闲置。

---

## 6. CLAUDE.md「项目特定」小节条目清单

判据(【官方】best-practices,模板已内嵌):删掉这行 Claude 会不会犯错。以下 8 行全部过判据,可直接粘进「## 项目特定」;细则全部下沉(versioning.md / skill 正文),这里只放指针和一行事实:

```markdown
## 项目特定

- TEOW:JAX 2D RTS 引擎,tick 制即时战略,后期加 RL 指挥官。需求唯一入口 `issue.md`
  (按版本分节,**该文件只有用户能改**)。
- 项目按版本推进。版本纪律(commit 前缀 vX.Y、changelog 契约、完成五件套)见
  `.claude/rules/versioning.md`;收尾走 `/version-close`,由用户触发。
- 引擎代码必须保持 tick 决定论:同 seed 同指令序列重放结果逐位一致。随机性只经
  显式传递的 JAX PRNGKey,禁止 Python random / numpy 全局随机进引擎。
- jit 边界内禁止 Python 控制流依赖 traced 值(用 lax.cond/lax.select);数组形状静态。
- 数值参数(产量/血量/耗时/费用)一律进 config,不写字面量(rules/python-research.md 第 2 条
  同样约束引擎代码)。
- 平衡性结论必须来自脚本化对局,带 [source: run_id];对局产物落 experiments/,受只读保护。
```

(其中 JAX 两条在写第一行引擎代码前属于「预防性约定」;若实践中从未被违反,按判据可再删。)

---

## 7. 长期自主项目工作流依据(为什么这套设计成立)

### 7.1 无上下文审核(fresh-context adversarial review)为什么有效

- **机制基础【官方】**:subagent 拿不到主对话历史(https://code.claude.com/docs/en/sub-agents.md )。「无上下文」不是营造出来的,是平台保证。
- **动机基础【模板设计文档】**:写代码的 context 已被自己的实现思路污染,会倾向认为自己写的是对的(validate/SKILL.md 第 2 步原文);失败路径留在 context 里会被反复重读并强化(HUMAN_PLAYBOOK 铁律二)。审核者与实现者共享 context,就共享了同一套盲区。
- **有效的前提条件(同样来自模板的教训,engine-auditor 草稿全部内置)**:①无上下文 agent 看不到 skill 正文和口头约定,**约束必须随 prompt 下发**(DESIGN_RATIONALE §7.1 #6——/validate 因此强制传黑名单;/version-close 因此强制传三样输入);②必须给**外部规格**当标准(issue.md 段),否则 fresh eyes 只能审自洽性;③必须有**抗过度工程约束**,否则 reviewer 被要求找 gap 就一定找得出,产出 12 条净负收益建议(/validate §3 引用的官方警告);④返回必须限格式限长(1000–2000 token,Anthropic context engineering 文章建议,经模板引用:https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents );⑤**不配 memory**——审核 agent 的价值来自每次重新看,记住上次结论会静默摧毁 fresh eyes(DESIGN_RATIONALE §4.5)。

### 7.2 长 session context 管理与跨 session 交接

直接沿用模板三件套,版本循环不新发明:①**40–60% 水位 + 固定 /clear 时机**(plan 定稿后、每 phase commit 后、纠正 2 次后)——版本循环天然提供了密集的安全 clear 点(每个功能级 commit 都是);②**一切状态落盘 + SessionStart 四 source 全量重注入**——版本横跨多个 session 也不怕,plan/changelog/HANDOFF 都在磁盘;③**探索进 subagent,主线程只收 1–2k token 摘要**。一个版本预计 3–10 个 session,交接链:HANDOFF.md(session 级)→ current-focus.md(「当前在做 vX.Y 的哪个 phase」)→ plan.md(版本级进度底账,phase 完成情况可从 git log 的 vX.Y commit 核验——这也是 commit 前缀规范的隐藏收益:**/kick 的 git 核验能直接读出版本进度**)。

### 7.3 决策日志:不新增 DECISIONS.md

session-handoff 一类通用实践推荐 append-only DECISIONS.md(记「为什么这么选」,防止后续 session 重新争论已决定的事)。【本报告判断】本项目**不加**:该职责已被四处覆盖——技术方案取舍在 `docs/plans/*/plan.md`(plan-critic 意见的采纳/不采纳都留痕,且落盘进 git)、实验性结论在 `research-log.md`(append-only 账本,恰是 DECISIONS 模式的同构物)、行为教训在 `MEMORY.md`、版本级结果在 changelog。模板明文:同一条信息写进两个文件,分叉后没人知道信哪个(handoff/SKILL.md 防重叠规则表)。加第五个记忆文件是负收益。唯一缝隙是「跨版本的设计决策」(如「资源模型为什么选驻场开采而非自动产出」)——落点定为**当版本 plan.md 的「目标/不在范围内」+ changelog**,不另立文件。

---

## 8. 官方依据 vs 本报告判断(汇总)

**有官方文档依据**(经子 agent 现查):skill frontmatter 全部字段与 `/name` 调用(https://code.claude.com/docs/en/skills.md );subagent frontmatter、tools 省略即继承全部、model 默认 inherit、无主对话历史(https://code.claude.com/docs/en/sub-agents.md );hooks 事件/matcher/exit 2 阻断/SessionStart source 五枚举与 context 注入(https://code.claude.com/docs/en/hooks.md );rules 的 `paths:` 语义与无 paths 时无条件加载(https://code.claude.com/docs/en/memory.md );CLAUDE.md 取舍判据(https://code.claude.com/docs/en/best-practices ,经模板转引);compact 后 CLAUDE.md 重注入而 paths 规则丢失(https://code.claude.com/docs/en/context-window ,经模板转引)。
**存疑一处**:Stop 事件的 `stop_hook_active` 字段本次未在官方文档中找到(§2.3)——不影响设计,建议落地时实测一次。
**本报告判断**(无外部依据,可推翻):agent 自动 commit 的取舍(§3.3);tag 保持人工;versioning.md 无 paths;/version-close 独立成 skill 而非并入 validate/handoff;不做 /issue-sync;不做 DECISIONS.md;engine-auditor 的三层审核清单内容(RTS 不变量与 JAX 陷阱清单);/lit 等闲置组件不删。

## 9. 落地顺序建议(下一个 session 的 plan 输入)

1. 填 CLAUDE.md 命令 TODO + 「项目特定」节(§6)→ 2. 写入三个新文件(§4.1–4.3 草稿)→ 3. settings.json 加 `Bash(git commit:*)` → 4. 启用 verify.sh(pytest+ruff)→ 5. 建 docs/changelog/ → 6. 实测:verify_stop 失败回路、engine-auditor 用一个玩具规格试派一次 → 7. commit(`v1.0 chore: 版本循环工作流落地`)→ 8. 正式开始 `/plan 实现 issue.md 的 v1.0 段`。
