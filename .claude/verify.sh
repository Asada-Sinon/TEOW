#!/usr/bin/env bash
# ============================================================
# TEOW 完成门禁 —— 由 .claude/hooks/verify_stop.py 在每轮结束时调用
#   退出码 0 = 通过;非 0 = 最后 60 行输出回给 Claude 去修
#   临时跳过:SKIP_VERIFY=1 claude
#   ★ 保持 60 秒以内:慢测试(训练冒烟等)标 @pytest.mark.slow 并在此排除 ★
# ------------------------------------------------------------
set -uo pipefail

fail=0

if [ -f pyproject.toml ]; then
  if command -v pytest >/dev/null 2>&1; then
    echo "==> pytest -x -q (排除 slow)"
    pytest -x -q -m "not slow" || fail=1
  fi

  if command -v ruff >/dev/null 2>&1; then
    echo "==> ruff check src/ tests/"
    ruff check src/ tests/ || fail=1
  fi
fi

# 禁止遗留调试断点
if git grep -nE 'breakpoint\(\)|import pdb' -- '*.py' >/dev/null 2>&1; then
  echo "发现遗留的调试断点："
  git grep -nE 'breakpoint\(\)|import pdb' -- '*.py'
  fail=1
fi

exit "$fail"
