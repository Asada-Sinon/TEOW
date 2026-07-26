# 当前焦点

## 当前目标

**v2.0 RL 不训练 JAX 骨架**(通宵推进 v1.8→v2.0;用户 2026-07-26 已批准 plan 并离线,全权自主)。
**v1.8 + v1.9 已五件套收官**(tag v1.8 / v1.9;异界之门 + 10 风格指挥官 + 评测分层)。
**v2.0 调研 + 设计文档已成**(docs/plans/20260727-v2-rl-approach/),剩 JAX 骨架(空跑不训练)+ 五件套。

主 hook:`~/.claude/plans/v1-8-v2-0-plan-plan-hook-fluffy-matsumoto.md`(总控)+
`docs/plans/20260726-v18-gate-commanders/`(v1.8 详版)。
阶段链:v1.8 ✅ → **v1.9(评测/筛选,进行中)** → v2.0(调研 ✅ + 骨架待做)。

## v1.8 收官摘要(已 commit + tag)

- 异界之门 sudden-death(`gate.py`)+ 10 风格参数化指挥官(`commanders/`)+ 吞吐 bench
  (GPU vmap B64 ~4000 env-tick/s)+ 评测脚手架(`eval_commanders_v18.py`)。
- 117 pytest(`-n 8`)+ ruff 绿;engine-auditor P0 零(P2 修:对怪开火进 cd / 离场清怪血)。
- commits 7c03a61 / d39dfcf / bb5feac / 177ff86 / 8a5d2e1 + 收尾 chore + tag v1.8。

## v1.9 第一件事(PENDING)

用 `explorations/eval_commanders_v18.py` 跑全 10 风格**综合评测**(更多 seed + 座位排列 +
round-robin),按质量判据筛选(胜率分布 / 非退化 / 风格覆盖 / 自适应 / 碾压 random / gate 到达率);
改不好就删;验证后**脚手架提升进 `src/teow/eval.py`**(供 v2.0)。
**注意**:防御建筑对怪已进 cd(勿回退);gate_open_tick 对 rush-vs-develop 平衡敏感,按需扫。

## v2.0(调研 ✅,骨架待做)

`notes/papers/` 19 篇;**推荐 = 自研 JAX PPO + v1.9 脚本课程(易→难)+ BC 暖启 + 势函数塑形
(Ng1999)**;自对弈后置;临摹 PureJaxRL + JaxMARL。「AMP」对离散 RTS≈GAIL。
待做:设计文档(docs/plans/)+ obs/reward/PPO 骨架(explorations/,空跑不训练)+ 课程分档。

## 纪律

- experiments/ 现有产物只读;平衡数字只进 config.py;commit 点名文件、禁 `git add -A`。
- pytest 用 `-n 8`(已装 pytest-xdist,未进 pyproject);GPU 批量 eval / 单环境门禁 CPU。
- 三版体量大,够不到就干净交接下 session(用户已预期跨 session)。
