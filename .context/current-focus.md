# 当前焦点

## 当前状态

**v1.0–v2.1 全部收官**,12 个 tag 已打,远程 tag 到 v2.1。引擎完工,RL 停在「正式开训」
门口:v2.1 试训除险坐实**纯 PPO 冷启动学不动**,BC 暖启 + 真正训练 = **v2.2**
(用户 2026-07-28 在线拍板选项 B)。

⚠ **issue.md 还没有 v2.2 规格**——需求唯一入口里没有 BC/正式训练那一节,草稿箱空。
v2.2 开工前要先走草稿箱协议把它补进规格区(实现只以规格区为依据)。

## 本次 session(2026-08-02 → 08-03):环境重建 + 门禁修复

- **环境原本是坏的**:`.venv` 整个不存在、`uv` 不在 PATH、系统 python 3.10(项目要 3.12),
  跑不了任何测试或实验。已重建:`~/.local/bin/uv sync` → Python 3.12.13 / jax 0.6.2 /
  numpy 2.5.1;CPU 与 GPU 双后端均可用(**RTX 5090 sm_120 与 jax 0.6.2 兼容**已实测);
  `pytest-xdist` 3.8.0 用 `uv pip install` 单独装(照 DECISIONS 2026-07-27 不进 pyproject,
  否则下次 `uv sync` 会移除)。
- **门禁全绿已独立复现**:`119 passed in 1158.14s`(`-n 8`)+ `slow 4 passed in 220.92s`
  + `ruff All checks passed`。HANDOFF 里那句「119 + slow 4 全绿」成立。
- **门禁本身修好**(commit `ed51df1`):verify.sh 原有两条路都通向静默放行(venv 缺失跳过
  / 全套必定超时),改成 venv 缺失即 fail + 核心子集(实测 117s,预算 300s)。详见
  MEMORY `[LEARN:tooling] 门禁「超时」和「缺环境」都不是拦截`。
- 文档收口:根 `CHANGELOG.md` 补齐 v1.3–v2.1 九条;MEMORY 补 4 条教训(门禁失效 /
  PBRS 库存诱导囤积 / 冷启动需 BC / FFA 座位轮转)。

## 待办

1. ⚠ **v2.1 的 8 个 run 目录不存在**——`20260727-v21-balanced-rr`、`20260728-v21-rr2/rr3/
   rr4/throughput/train-vsrandom/train-fix1/train-fix2`,磁盘上没有、git 历史里从未提交
   (`git log --all --diff-filter=A -- 'experiments/*v21*'` 为空),而 changelog v2.1 与
   research-log 引用它们。按硬约束 3 这些数字**不可复现、不得被引用**。现在环境和 GPU
   都可用,重跑可行(吞吐 bench 最便宜,三次训练最贵),**跑哪几个待用户定**。
2. ⚠ **push 无权限**:`git push origin main` → 403
   (`Permission to Asada-Sinon/TEOW.git denied to michaelfanisme`,gh 已登录同名账号)。
   历史 commit 与 tag 都推送成功过,所以是权限/token scope 后来变了。本地领先 origin/main,
   待用户处理凭据(`gh auth refresh -s repo` 或确认该用哪个账号)。

## 纪律

- experiments/ 现有产物只读;平衡数字只进 `config.py`;commit 点名文件、**禁 `git add -A`**。
- 全套测试(门禁不代跑,只跑核心子集):
  `JAX_PLATFORMS=cpu .venv/bin/pytest -q -m "not slow" -n 8`(~19min)
  `JAX_PLATFORMS=cpu .venv/bin/pytest -q -m slow`(~4min)
- **uv 不在系统 PATH**,一律用 `~/.local/bin/uv`。
- 测试/门禁 CPU,批量 eval/训练才 GPU(MEMORY `[LEARN:env]`)。
