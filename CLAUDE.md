# TEOW

TODO: 一句话说明这个项目在做什么

## 命令

```bash
# 全部测试
# TODO(填测试命令)
# 单个测试文件
# TODO(填测试命令) tests/test_foo.py
# lint
# TODO(填 lint 命令)
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

<!-- 留给你填：本项目独有的坑、命名约定、数据路径、外部服务、不许碰的东西。
     判据：删掉这一行 Claude 会不会犯错？不会就别写。 -->
