#!/usr/bin/env bash
# ============================================================
# TEOW 完成门禁 —— 由 .claude/hooks/verify_stop.py 在每轮结束时调用
#   退出码 0 = 通过;非 0 = 最后 60 行输出回给 Claude 去修
#   临时跳过:SKIP_VERIFY=1 claude
#
#   ★ 预算 ~180 秒(verify_stop.py 的 TIMEOUT=300)★
#   本门禁跑的是**核心子集**,不是全套。依据(2026-08-02 实测,用户拍板):
#     · 每个测试用不同 Config → 各自触发整套 step 的 JAX 重编译,单个引擎测试
#       就要 35-119 秒;同一个测试两轮之间能从 85.79s 飘到 118.96s(CPU 竞争),
#       所以预算必须留余量,不能贴着 TIMEOUT 排。
#     · 全套 119 个即使 `-n 8` 并行也要 1158 秒 = TIMEOUT 的 3.9 倍。而超时的后果
#       **不是拦截、是静默放行**(verify_stop.py:72-77 捕获 TimeoutExpired 后 exit 0),
#       排一个必定超时的门禁 = 每轮假绿灯,比不排更危险。
#   全套的职责在 /version-close 与人工,别指望本脚本:
#     JAX_PLATFORMS=cpu .venv/bin/pytest -q -m "not slow" -n 8   # 119 passed, ~19min
#     JAX_PLATFORMS=cpu .venv/bin/pytest -q -m slow              # 4 passed,   ~4min
# ------------------------------------------------------------
set -uo pipefail

fail=0

# 依赖装在项目 .venv 里(uv sync),裸 pytest/ruff 不在 PATH 上——必须走 .venv/bin。
VENV_BIN="$(dirname "$0")/../.venv/bin"

# ★ venv 缺失必须报错,绝不静默跳过 ★
# 2026-08-02 踩过:.venv 整个不见了,旧版的 `[ -d "$VENV_BIN" ]` 判假 → 整块被跳过
# → fail 保持 0 → 门禁连续多轮报绿,实际一个检查都没跑。
# 缺环境是「门禁失效」,不是「通过」,必须让它响。
if [ ! -d "$VENV_BIN" ]; then
  echo "门禁失效:$VENV_BIN 不存在,一个检查都没跑。"
  echo "重建环境(uv 不在系统 PATH,用绝对路径):"
  echo "  ~/.local/bin/uv sync && ~/.local/bin/uv pip install pytest-xdist"
  exit 1
fi

# --- 静态检查先行:秒级,失败就没必要再烧 ~172 秒跑测试 ---
if [ -f pyproject.toml ] && [ -x "$VENV_BIN/ruff" ]; then
  echo "==> ruff check src/ tests/"
  "$VENV_BIN/ruff" check src/ tests/ || fail=1
fi

# 禁止遗留调试断点
if git grep -nE 'breakpoint\(\)|import pdb' -- '*.py' >/dev/null 2>&1; then
  echo "发现遗留的调试断点："
  git grep -nE 'breakpoint\(\)|import pdb' -- '*.py'
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "(静态检查未过,跳过测试子集以节省 ~172 秒;修完下轮自动跑)"
  exit "$fail"
fi

# --- 核心测试子集 ---
if [ -f pyproject.toml ] && [ -x "$VENV_BIN/pytest" ]; then
  echo "==> pytest 核心子集 (CPU, 实测 ~172s)"
  # 跑 CPU:逻辑判定与 GPU 等价,但省掉每个 Config 变体的 GPU jit 编译
  # (MEMORY.md [LEARN:env]:同套测试 GPU 5min vs CPU 14s)
  #
  # 子集选择依据(2026-08-02 实测单测耗时):
  #   test_state.py                           1 个 0.45s —— 不触发 step 编译,白送
  #   test_combat_win.py                      3 个  122s —— 胜负判定/硬帽不和局/同归于尽
  #   test_determinism::test_scan_runs_with_other_seed   49s —— 决定论(便宜的那个)
  # 刻意排除:
  #   test_economy.py                        10 个 >400s —— 单文件就爆预算
  #   test_determinism::test_bitwise_identical_same_seed  86-119s 且波动最大
  # 增删子集前先实测耗时,别凭感觉加——加错一个文件就把门禁推进"必定超时"区。
  JAX_PLATFORMS=cpu "$VENV_BIN/pytest" -x -q \
    tests/test_state.py \
    tests/test_combat_win.py \
    tests/test_determinism.py::test_scan_runs_with_other_seed || fail=1
fi

exit "$fail"
