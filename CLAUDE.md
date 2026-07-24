# TEOW

纯 JAX 实现的类星际 2D 网格 RTS 引擎(tick 制即时战略:采集/建造/升级/战斗),
v1.x 打磨引擎,v2+ 在其上训练强化学习指挥官(自研 PPO)。需求唯一入口:`issue.md`。

## 命令

```bash
# 全部测试
python3 -m pytest -q
# 单个测试文件
python3 -m pytest -q tests/test_state.py
# lint
ruff check src/ tests/
# 跑实验：务必用这个解释器，不要用裸 python
python3 src/run.py
```

## 硬约束

以下几条优先级高于任何「快一点」的诉求，冲突时停下来问我。

1. IMPORTANT: 所有数值结果必须由 `src/` 下受版本控制的脚本产生。YOU MUST NOT 口算、
   估算或凭印象报任何指标——拿不到真实数字就直说拿不到。
2. IMPORTANT: `experiments/` `results/` `runs/` `outputs/` 四个目录下已有产物一律只读。
   YOU MUST NOT 修改或删除其中任何文件（包括「看起来是废弃的」）；新结果一律写新目录。
   本项目的产物目录是 `experiments/`，一次 run 一个目录 `experiments/<run_id>/`，
   run_id 格式 `YYYYMMDD-<slug>`（例：`experiments/20260722-attn-cache-seed1/`）。
   另外三个只是历史/他处约定，同样受保护，但新产物不要往那里写。
3. 每次实验 YOU MUST 同时记录 git hash、resolved config（展开后的完整配置，不是
   config 路径）、随机种子。三者缺一即视为不可复现，该结果不得被引用。
4. 写进 `research-log.md` 的每条结论 YOU MUST 带标注：`[AI-DRAFT]`（你得出的，未经我
   核验）/ `[HUMAN-VERIFIED]`（只有我能打）/ `[source: <run_id>]`。
5. 引用必须来自真实检索并附 DOI 或 arXiv ID。严禁凭记忆生成 BibTeX。
6. 新想法先在 `explorations/` 里验证，通过之后才允许进 `src/`。

## 工作流

R-P-I-V（Research → Plan → Implement → Validate）对应 7 个 skill：
`/kick` 起手接管上下文，`/plan` 出方案，`/impl` 落地，`/validate` 收口，
`/handoff` 交接；`/exp` 跑实验，`/lit` 查文献。
具体步骤写在各 skill 正文里，用到时才加载，这里不展开。

## 交互语言

面向我的提问、澄清、选项——包括 plan 模式和 AskUserQuestion 工具弹出的问题——一律用中文，
即使本文件和你的工作笔记用的是别的语言。约束的是「我读到什么」，不是「你内部怎么想」。
（改成别的语言：把这一节改掉即可。）

## 压缩

当压缩(compact)时，务必保留：修改过的文件清单、当前 plan 的进度、所有测试命令。

## 项目特定

### 需求入口 issue.md
`issue.md` 是需求唯一入口。用户往「草稿箱」区写口语化想法;你看不懂就问,
**完全吃透后**把它改写进对应版本的规格区(通透版,按版本号组织,不保留初稿)。
实现只以 issue.md 的版本规格为依据,规格与代码冲突时以 issue.md 为准并停下来对齐。

### 版本与 git 纪律
- 版本号 v1.0 / v1.1 / … 推进;**commit 按功能/机制变动提交,频率远高于版本**,
  message 一律以当前版本开头,例:`v1.0 采集一体循环:进驻与载荷状态机`。
- commit 和 push 已获用户永久授权,直接执行不用问(force-push、改历史除外,禁止)。
- **版本收尾四步**(缺一不可):①`python3 -m pytest -q` + `ruff check` 全绿
  ②派**无上下文 agent**(engine-auditor)审核 ③写 `docs/changelog/vX.Y.md`
  (新增/修复/平衡[Config 字段 old→new]/已知问题) ④`git tag vX.Y` 并 push。

### JAX 引擎纪律
- 世界状态 = NamedTuple pytree,全部定容数组(`E_max` 实体表 + alive 掩码),
  死=翻掩码位、生=scatter 进空槽,**绝不 resize**;无 data-dependent 形状。
- 静态地图数据闭包进 `build_step`,不进 state;PRNGKey 线程传入,不进 state,
  不存在 numpy 全局随机等第二随机源。
- tick 语义:世界按 tick 连续演化,建造/训练/采集/移动都消耗真实 tick;
  每 tick 都可下指令,no-op 永远合法。单 tick 结算顺序见 `src/teow/step.py` 头注释。
- **数值参数唯一真源是 `src/teow/config.py` 的 Config dataclass**,代码里出现平衡数字
  字面量即 bug;平衡改动必须走 changelog 的「平衡」区(字段 old→new)。

### 用户离线期决策协议
用户睡觉/离线时你全权推进,但:**碰到不确定的先派 agent 调研、报告落
`docs/plans/<YYYYMMDD>-<slug>/research.md`,再决策;不带不确定性决策**。
所有自主决策 append 进 `docs/DECISIONS.md`(带日期与 [AI-DRAFT] 标注)供用户复核。
