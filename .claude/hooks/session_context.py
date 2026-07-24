#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionStart hook：把项目持久状态注入会话 context。

这是整套工作流的关键件。SessionStart 的 stdout 会被原样加进 context，
所以这里输出的是**纯文本**（不是 JSON）。

为什么重要：
  compact（上下文压缩）和 /clear 都会让对话历史消失或变成有损摘要。
  但 SessionStart 在这之后会**重新触发**，于是 HANDOFF.md /
  current-focus.md / MEMORY.md 里的关键状态每次都能原样回到 context。
  写进文件的东西比留在对话里的东西活得久 —— 这是整套工作流的地基。

  source 枚举有四个：startup / resume / clear / compact。
  本工作流把 /clear 当成高频固定动作（纠正 2 次即 clear、plan→impl 之间必 clear、
  phase 之间 clear），所以 settings.json 的 matcher 必须四个都覆盖，
  漏掉 clear 等于这套工作流最常走的那条路上没有状态注入。

任何异常都静默 exit 0 且不输出：注入失败顶多少点上下文，
但 hook 报错会让会话启动就带一堆噪音。
"""

import json
import os
import subprocess
import sys

# 各文件注入上限（字符）与超限时保留哪一头。
#
# 截断方向必须跟着**每个文件自己的书写约定**走，方向反了就等于精准切掉最有用的部分：
#
#   HANDOFF.md      —— 保留 head。HANDOFF 的约定是「新会话结束时在最上面加一条，
#                      最新的在前」（见模板 HANDOFF.md 顶部规矩）。保留尾部会把
#                      最新那次交接连同 PENDING 一起切掉，正好切反。
#   MEMORY.md       —— 保留 tail。MEMORY 是累积式的，新条目**追加在后**；
#                      而且文件顶部是一大段格式说明（不是内容），保留 head 会
#                      注入一堆"怎么写 MEMORY"的元信息、一条真教训都留不下。
#   current-focus.md—— 保留 head。它不是时间序列，是一份「当前目标 / 为什么做 /
#                      完成判据 / 不做什么」的快照，重要度自上而下递减；
#                      而且约定是「方向变了就整个重写，不要往下堆」，
#                      所以头部永远是当前那件事。
#
# 值为 (上限字符数, "head" | "tail", 截断说明里附的一句提示或 None)。
# 提示是给模型看的：让它知道自己丢的是哪一头，别把残缺内容当全文。
LIMITS = {
    "HANDOFF.md": (3000, "head", "本文件最新的交接写在最上面，保留的就是最新几次"),
    os.path.join(".context", "current-focus.md"): (1000, "head", None),
    "MEMORY.md": (2000, "tail", "本文件新教训追加在最后，保留的就是最新几条"),
}

GIT_STATUS_LINES = 20   # git status --short 最多显示几行
GIT_LOG_COUNT = 3       # 显示最近几条 commit
GIT_TIMEOUT = 10


def project_root(data):
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return os.path.abspath(root)


def read_capped(path, limit, keep="tail", hint=None):
    """读文件并截断到 limit 字符。文件不存在/空返回 None。

    keep="head" 保留开头（最新的写在最上面、或重要度自上而下递减的文件）；
    keep="tail" 保留结尾（新内容追加在后的文件）。
    截断处一定要留一行说明，否则模型会把残缺内容当成全文。
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) <= limit:
        return text

    tail_hint = ("；" + hint) if hint else ""
    if keep == "head":
        return text[:limit].rstrip() + "\n...(以上为本文件开头 %d 字符，后文已截断%s)..." % (
            limit,
            tail_hint,
        )
    return "...(前文已截断，以下为本文件最后 %d 字符%s)...\n" % (
        limit,
        tail_hint,
    ) + text[-limit:].lstrip()


def git(root, args):
    """跑一条 git 命令，失败返回 None（非 git 仓库属于正常情况）。"""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").rstrip()


def git_section(root):
    """组装版本库状态：分支 + 工作区变更 + 最近提交。"""
    if git(root, ["rev-parse", "--is-inside-work-tree"]) is None:
        return None

    parts = []

    branch = git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        parts.append("当前分支：%s" % branch)

    status = git(root, ["status", "--short"])
    if status is not None:
        lines = status.splitlines()
        if lines:
            shown = lines[:GIT_STATUS_LINES]
            block = "\n".join(shown)
            if len(lines) > GIT_STATUS_LINES:
                block += "\n...(还有 %d 处变更未显示)" % (len(lines) - GIT_STATUS_LINES)
            parts.append("工作区变更（git status --short）：\n" + block)
        else:
            parts.append("工作区变更（git status --short）：\n(干净)")

    log = git(root, ["log", "--oneline", "-n", str(GIT_LOG_COUNT)])
    if log:
        parts.append("最近 %d 条提交：\n%s" % (GIT_LOG_COUNT, log))

    if not parts:
        return None
    return "\n\n".join(parts)


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    root = project_root(data)
    source = data.get("source") or ""

    sections = []  # [(标题, 正文)]

    # compact 场景专属提醒：压缩刚发生，最需要重申的规则放这里。
    # 只在 compact 时读 —— startup/clear/resume 时 rules 和子目录 CLAUDE.md
    # 都还会正常加载，不需要重申；只有 compact 会把它们揉没。
    # 保留 head：这是人手写的短文件，越靠前越重要。
    if source == "compact":
        note = read_capped(
            os.path.join(root, ".claude", "compact-reminder.txt"), 2000, "head"
        )
        if note:
            sections.append(("压缩后提醒 .claude/compact-reminder.txt", note))

    handoff = read_capped(os.path.join(root, "HANDOFF.md"), *LIMITS["HANDOFF.md"])
    if handoff:
        sections.append(("交接文档 HANDOFF.md", handoff))

    focus_rel = os.path.join(".context", "current-focus.md")
    focus = read_capped(os.path.join(root, focus_rel), *LIMITS[focus_rel])
    if focus:
        sections.append(("当前焦点 .context/current-focus.md", focus))

    memory = read_capped(os.path.join(root, "MEMORY.md"), *LIMITS["MEMORY.md"])
    if memory:
        sections.append(("长期记忆 MEMORY.md", memory))

    repo = git_section(root)
    if repo:
        sections.append(("版本库状态", repo))

    # 什么都没有就彻底闭嘴，别往 context 里塞空壳
    if not sections:
        return

    out = ["===== 项目持久状态（自动注入）====="]
    out.append(
        "以下为项目持久状态，由 SessionStart hook 自动注入。"
        "每次 /clear、恢复会话或压缩(compact)之后都会重新注入，"
        "可信度高于对话历史。"
    )
    for title, body in sections:
        out.append("")
        out.append("----- %s -----" % title)
        out.append(body)
    out.append("")
    out.append("===== 持久状态结束 =====")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
