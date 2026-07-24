#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PreToolUse hook：受保护路径护栏（科研场景最重要的一道闸）。

为什么需要它：
  实验产出（experiments/ results/ runs/ outputs/）和原始数据（data/raw/）
  一旦被 agent "顺手整理"掉，是不可逆的 —— 跑了三天的结果没有 undo。
  锁文件（*.lock / package-lock.json）被改则会悄悄污染环境复现性。

两类规则（对齐 CLAUDE.md 硬约束第 2 条「已有产物只读，新结果一律写新目录」）：

  普通规则       —— 命中后**再看目标文件是否已存在**：
                    已存在 → exit 2 阻断（保护的是"不覆盖、不删除已有 run"）；
                    不存在 → 放行（新建允许，否则 /exp 第 3 步连 provenance
                    都落不了盘，而落 provenance 恰恰是硬约束第 3 条要求的）。

  `!` 前缀规则   —— **无条件阻断**，存不存在都拦。
                    用于「新建它本身就危险」的东西：.env / *.lock / .git/**。

规则来源：项目根 .claude/protected-paths.txt（每行一个 glob，可带 `!` 前缀，
# 注释）。该文件不存在时使用下面的内置默认值。
"""

import fnmatch
import json
import os
import sys

# 内置默认保护规则。当 .claude/protected-paths.txt 不存在时生效。
# `!` 前缀 = 无条件阻断；无前缀 = 仅当文件已存在时阻断。
# 必须与 .claude/protected-paths.txt 的内容保持一致。
DEFAULT_PATTERNS = [
    "experiments/**",
    "results/**",
    "runs/**",
    "outputs/**",
    "data/raw/**",
    "!.env",
    "!.env.*",
    "!*.lock",
    "!package-lock.json",
    "!uv.lock",
    "!poetry.lock",
    "!.git/**",
]

RULES_FILE = os.path.join(".claude", "protected-paths.txt")


def project_root(data):
    """确定项目根：优先 CLAUDE_PROJECT_DIR，其次 hook 输入里的 cwd，最后当前目录。"""
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return os.path.abspath(root)


def load_patterns(root):
    """返回 (规则列表, 来源说明)。规则保持原始写法，`!` 前缀在 match 里解析。"""
    path = os.path.join(root, RULES_FILE)
    if not os.path.isfile(path):
        return DEFAULT_PATTERNS, "内置默认"

    patterns = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    except Exception:
        return DEFAULT_PATTERNS, "内置默认（protected-paths.txt 读取失败）"

    if not patterns:
        # 文件存在但全是注释/空行 —— 视为用户主动清空，尊重它，不再兜回默认
        return [], RULES_FILE
    return patterns, RULES_FILE


def parse_rule(raw):
    """拆出 (glob, 是否无条件阻断)。`!` 前缀表示无条件，不是 gitignore 的取反。"""
    if raw.startswith("!"):
        return raw[1:].strip(), True
    return raw, False


def match(rel_path, patterns):
    """命中返回 (原始规则行, glob, 是否无条件)；没命中返回 None。

    fnmatch 的 * 会跨越 /，所以 `experiments/**` 能覆盖任意深度；
    纯文件名匹配则让 `*.lock`、`.env` 这类规则不依赖它在哪一层目录。

    无条件规则优先：先扫一遍 `!` 规则，命中就直接判无条件阻断，
    免得同一个文件先被某条普通规则匹上、又因为"文件不存在"被放行。
    """
    base = os.path.basename(rel_path)
    parsed = [(raw,) + parse_rule(raw) for raw in patterns]

    for unconditional_first in (True, False):
        for raw, glob, unconditional in parsed:
            if unconditional is not unconditional_first:
                continue
            if not glob:
                continue
            if fnmatch.fnmatch(rel_path, glob) or fnmatch.fnmatch(base, glob):
                return raw, glob, unconditional
    return None


def block(rel_path, rule, source, unconditional):
    """stderr 输出中文说明并以 exit 2 阻断（stderr 内容会回给 Claude）。"""
    if unconditional:
        why = (
            "命中的是一条**无条件规则**（`!` 前缀）：这类文件不论是新建还是修改\n"
            "都被禁止 —— 凭空造一个 .env 会让密钥出现在仓库里，手写一个 lock 文件\n"
            "同样会污染环境可复现性。"
        )
        outs = """三条合法出路：
  1. 由用户手动创建/编辑（推荐）—— 保留一次人工确认，最安全。
  2. 锁文件请让包管理器自己生成（`uv lock` / `npm install` 等），不要手写。
  3. 若确属误伤：临时把 .claude/protected-paths.txt 里那条规则注释掉，
     改完后立刻恢复，不要长期留着。"""
    else:
        why = (
            "命中的是一条**仅保护已有文件**的规则：该文件**已经存在**。\n"
            "这类路径通常是实验产出或原始数据，覆盖/删除往往不可撤销 ——\n"
            "跑了三天的结果没有 undo。\n"
            "（注意：往这些目录里**新建**文件是允许的，护栏不会拦。）"
        )
        outs = """三条合法出路：
  1. 换一个**新的**输出路径（推荐，也是 CLAUDE.md 硬约束第 2 条的要求：
     「已有产物只读，新结果一律写新目录」）。例如换一个新的 run_id 目录。
  2. 由用户手动编辑 —— 保留一次人工确认。
  3. 若确属误伤：临时把 .claude/protected-paths.txt 里那条规则注释掉，
     改完后立刻恢复，不要长期留着。"""

    msg = f"""[受保护路径] 本次写入被工作流护栏阻断。

  命中文件：{rel_path}
  命中规则：{rule}
  规则来源：{source}

{why}

{outs}

请不要绕过此护栏，先向用户说明情况并征求确认。"""
    sys.stderr.write(msg + "\n")
    sys.exit(2)


def main():
    raw = sys.stdin.read()
    data = json.loads(raw)

    file_path = (data.get("tool_input") or {}).get("file_path") or ""
    if not file_path:
        return

    root = project_root(data)
    abs_path = os.path.abspath(os.path.join(root, file_path))
    try:
        rel_path = os.path.relpath(abs_path, root)
    except ValueError:
        rel_path = file_path
    # 统一成 posix 分隔符，规则写法才不用管平台
    rel_path = rel_path.replace(os.sep, "/")

    patterns, source = load_patterns(root)
    hit = match(rel_path, patterns)
    if not hit:
        return  # 未命中：静默放行

    rule, _glob, unconditional = hit
    if unconditional:
        block(rel_path, rule, source, True)

    # 普通规则：只有目标已经存在时才是"覆盖已有产物"，才阻断。
    # 用绝对路径判断，避免 hook 的工作目录不是项目根时判错。
    if os.path.exists(abs_path):
        block(rel_path, rule, source, False)
    # 新建文件：放行


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # exit 2 必须原样传出去，不能被下面的兜底吞掉
    except Exception:
        # 护栏自身出错时选择放行而不是误杀：这是 PreToolUse，
        # 误杀会让 agent 完全无法工作，而漏判只是回到没有护栏的状态。
        pass
    sys.exit(0)
