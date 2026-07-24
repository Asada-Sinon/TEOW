#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostToolUse hook：文件被 Edit/Write 之后自动格式化 + lint。

设计原则（这套模板要能装到任意项目，所以必须"零依赖、零假设"）：
  1. 只用 Python 3 标准库，不依赖 jq。
  2. 所有外部工具（ruff/black/prettier/shfmt...）都先用 shutil.which 探测，
     不存在就静默跳过 —— 绝不因为目标项目没装某个工具就报错。
  3. lint 报错**不阻断**（永远 exit 0）。错误通过 stdout 的
     hookSpecificOutput.additionalContext 注入到 Claude 的 context，
     让它自己看到并决定要不要修。阻断（exit 2）留给 protect_paths.py。
  4. 全程 try/except 兜底：hook 自身挂掉不能拖垮主流程。
"""

import json
import os
import shutil
import subprocess
import sys

# 单个子进程超时（秒）。格式化工具正常都是毫秒级，超过就说明卡住了。
TIMEOUT = 30
# 注入 context 的错误文本上限，防止一个刷屏的 lint 输出吃掉整个上下文预算。
MAX_CONTEXT_CHARS = 2000

# 走 prettier 的扩展名
PRETTIER_EXTS = {".js", ".ts", ".tsx", ".jsx", ".json", ".css", ".md"}


def run(cmd, cwd=None):
    """跑一个子进程，返回 (returncode, 合并后的输出)。任何异常都当作"跳过"。"""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        # 超时 / 工具突然消失 / 编码问题 —— 一律当没发生过
        return 0, ""


def handle_python(path):
    """Python：优先 ruff（快且格式化+lint 一体），退化到 black + flake8。"""
    problems = []

    if shutil.which("ruff"):
        # ruff format 只管排版，不会改语义
        run(["ruff", "format", path])
        # --fix 先自动修能修的，剩下修不了的才是真问题
        code, out = run(["ruff", "check", "--fix", path])
        if code != 0 and out.strip():
            problems.append(out.strip())
        return problems

    # 没有 ruff 就试老组合
    ran_any = False
    if shutil.which("black"):
        run(["black", "--quiet", path])
        ran_any = True
    if shutil.which("flake8"):
        code, out = run(["flake8", path])
        ran_any = True
        if code != 0 and out.strip():
            problems.append(out.strip())

    if not ran_any:
        # 两套都没有 —— 静默跳过，不是错误
        return []
    return problems


def handle_prettier(path):
    """前端系文件：prettier。本地装了就直接用，否则借 npx。"""
    if shutil.which("prettier"):
        code, out = run(["prettier", "--write", path])
        if code != 0 and out.strip():
            return [out.strip()]
        return []

    if shutil.which("npx"):
        # --no-install 关键：不加的话 npx 会去联网下载 prettier，
        # 在离线 / 内网 / CI 环境里会卡满 30 秒然后超时。
        code, out = run(["npx", "--no-install", "prettier", "--write", path])
        # 包没装时 npx 直接失败，这属于"工具不存在"，静默跳过而不是报错
        if code != 0:
            low = out.lower()
            if "could not determine executable" in low or "not found" in low:
                return []
            if out.strip():
                return [out.strip()]
    return []


def handle_shell(path):
    """Shell 脚本：shfmt（没有就跳过；shellcheck 噪音太大，这里不强上）。"""
    if shutil.which("shfmt"):
        run(["shfmt", "-w", path])
    return []


def emit_context(text):
    """把 lint 错误注入 Claude 的 context（不阻断，只提醒）。"""
    text = text.strip()
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n...(输出过长已截断)"
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                "自动格式化后仍存在 lint 问题（未阻断，请自行判断是否修复）：\n" + text
            ),
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


def main():
    raw = sys.stdin.read()
    data = json.loads(raw)  # 畸形输入交给外层 except 兜底

    file_path = (data.get("tool_input") or {}).get("file_path") or ""
    if not file_path:
        return
    if not os.path.isfile(file_path):
        # 文件可能已被后续操作删掉/移走，不是错误
        return

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        problems = handle_python(file_path)
    elif ext in PRETTIER_EXTS:
        problems = handle_prettier(file_path)
    elif ext == ".sh":
        problems = handle_shell(file_path)
    else:
        return  # 不认识的类型，什么都不做

    if problems:
        emit_context("\n".join(problems))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 兜底：hook 出任何意外都必须安静退出，绝不打断主流程
        pass
    sys.exit(0)
