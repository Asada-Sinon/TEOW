---
paths: ["experiments/**", "results/**", "runs/**", "outputs/**"]
description: "实验产物区只读纪律，以及 run 目录的必备记录项"
---

# 实验产物区纪律

**本模板的产物目录统一叫 `experiments/`，一个 run 一个目录：`experiments/<run_id>/`，
run_id 格式 `YYYYMMDD-<slug>`（例：`experiments/20260722-attn-cache-seed1`）。**
上面 `paths:` 里另外三个目录（`results/` `runs/` `outputs/`）只是为了兼容老项目已有的布局，
本规则的作用域覆盖它们，但**新建目录一律用 `experiments/`**。

## 自查信号：这条规则被压缩掉了吗

**如果你正准备用 `>`、`>>`、`rm`、`mv`、`cp`、`tee`、`--output`、`sed -i` 之类的方式往 `experiments/` 里写东西或删东西，而你说不出下面「run 目录必备项」有哪几条 —— 说明本规则已经在一次 compact 中丢失了。停下，重新 Read `.claude/rules/experiments.md`，再决定要不要动手。**

带 `paths:` 的规则只在读到匹配文件时才加载，省 context，但**压缩后不会自动重新加载**。上面那句是你的重新加载触发器。

## ⚠️ 这条规则是 Bash 路径上唯一的防线

`.claude/hooks/protect_paths.py` 会拦截写向产物区的 `Edit` / `Write`，**但它只看工具调用里的 `file_path` 字段**。它**拦不住**：

- shell 重定向：`python train.py > experiments/20260722-lr1e-4/log.txt`、`echo ... >> experiments/summary.csv`
- `rm` / `rm -rf` / `mv` / `cp` 覆盖
- 脚本自身的输出路径参数：`--out_dir experiments/20260722-lr1e-4`
- `find ... -delete`、`tar -x` 解到产物区
- 任何在 Bash 里发生的写入

也就是说：**在 Bash 里，没有任何自动护栏会救你。** 这段文字就是防线本身。跑了三天的 run 没有 undo。

## hook 的判定语义：拦「改已有」，放行「建新的」

不要把「产物区被保护」理解成「产物区一个字都写不了」——那会把 `/exp` 的核心步骤堵死。实际语义是：

| 动作 | 结果 |
|---|---|
| 新建 `experiments/<新 run_id>/` 并往里写 git hash / resolved config / seed / 日志 | **放行**（这正是 `/exp` 要做的事） |
| `Edit` / `Write` 一个**已存在**的产物文件（改结果、改日志、改 config） | **阻断**（exit 2） |
| 删除或重命名已有 run | **阻断**（且 `rm`/`mv` 走 Bash，hook 根本看不到 —— 靠本规则自觉） |
| 写 `.env` / `*.lock` / `.git/**` | **无条件阻断**，新建也不行 |

一句话：**已有产物只读，新结果一律写新目录。** 被 hook 拦住时不要换 Bash 绕过去，那是同一条纪律的两面。

## 只读

1. **禁止修改或删除任何已存在的 run。** 包括「顺手整理一下目录」「清掉失败的实验」「把日志压缩一下」「重命名成更规范的格式」—— 一律不做。看着像垃圾的目录也可能是某条结论的唯一支撑（见 `.claude/rules/notes.md` 的 `[source: run_id]`）。
2. **新实验必须新建目录**，绝不复用或覆盖已有目录名。目录名 `experiments/<YYYYMMDD>-<slug>`，slug 用小写英文加短横线并带上区分变量，例如 `experiments/20260722-attn-cache-seed1`。
3. **分析时只读不写。** 分析脚本、中间结果、图表一律写到 `explorations/`，不要写回产物区。
4. 需要清理时：**向用户报告并说明你打算删什么、为什么，由用户自己动手。** 不要代劳，也不要提议绕过护栏。

## run 目录必备项

启动一次新实验时，必须在该 run 目录内落下这六项。缺任何一项，这次 run 后面都无法被追溯和归因：

1. **resolved config** —— 所有默认值展开后的最终配置，不是命令行片段，不是模板
2. **git hash** —— 以及工作区是否 dirty（dirty 就把 `git diff` 一并存下）
3. **seed** —— 显式记录实际使用的值，不能只写「用了默认」
4. **启动命令** —— 原样可复制粘贴的完整命令行
5. **日志** —— stdout/stderr 落盘
6. **metrics** —— 机器可读格式（json / csv / event 文件），不能只存在日志文本里

这六项都是**新建**文件，hook 不会拦；跑完之后它们就变成只读的了。

## 归因

对比两个 run 时，若 resolved config 差异不止一项，**必须明确声明「无法归因到单一变量」**，不得替用户挑出「看起来最重要的那一项」下结论。要归因就补一个受控实验。
