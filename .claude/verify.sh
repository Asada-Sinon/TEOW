#!/usr/bin/env bash
# ============================================================
# TEOW 完成门禁 —— 由 .claude/hooks/verify_stop.py 在每轮结束时调用
#   退出码 0 = 通过;非 0 = 最后 60 行输出回给 Claude 去修
#   临时跳过:SKIP_VERIFY=1 claude
#   ★ 保持 60 秒以内:慢测试(训练冒烟等)标 @pytest.mark.slow 并在此排除 ★
# ------------------------------------------------------------
set -uo pipefail

fail=0

# 依赖装在项目 .venv 里(uv sync),裸 pytest/ruff 不在 PATH 上——必须走 .venv/bin,
# 否则 command -v 探测失败会静默跳过,门禁形同虚设。
VENV_BIN="$(dirname "$0")/../.venv/bin"

if [ -f pyproject.toml ] && [ -d "$VENV_BIN" ]; then
  if [ -x "$VENV_BIN/pytest" ]; then
    echo "==> pytest -x -q (排除 slow, CPU)"
    # 门禁跑 CPU:逻辑判定与 GPU 等价,但省掉每个 Config 变体的 GPU jit 编译
    # (实测 GPU 5min vs CPU 14s,GPU 会爆掉本门禁 60 秒预算)
    JAX_PLATFORMS=cpu "$VENV_BIN/pytest" -x -q -m "not slow" || fail=1
  fi

  if [ -x "$VENV_BIN/ruff" ]; then
    echo "==> ruff check src/ tests/"
    "$VENV_BIN/ruff" check src/ tests/ || fail=1
  fi
fi

# 禁止遗留调试断点
if git grep -nE 'breakpoint\(\)|import pdb' -- '*.py' >/dev/null 2>&1; then
  echo "发现遗留的调试断点："
  git grep -nE 'breakpoint\(\)|import pdb' -- '*.py'
  fail=1
fi

exit "$fail"
