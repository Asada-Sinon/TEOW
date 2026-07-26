# PPO 实现细节三件套(实现比算法更决定成败)

本笔记合并三篇「PPO/on-policy 到底哪些实现细节在起作用」的工作。

## 1. The 37 Implementation Details of Proximal Policy Optimization
- **citekey**: huang2022ppodetails
- **载体**: ICLR 2022 Blog Track(OpenReview id=Hl6jCqIp2j);**无 DOI/arXiv** `[未验证:博客,非 arXiv]`
- **作者/年**: Huang, Dossa, Raffin, Kanervisto, Wang — 2022
- paper says:逐条复现 PPO 的 37 个实现细节(vectorized env、advantage normalization、
  value clipping、正交初始化、学习率退火、GAE、reward/obs scaling、梯度裁剪等),
  并给出与原实现对齐的可复现代码。

## 2. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
- **arXiv**: 2006.05990 — Andrychowicz 等, 2020 (Google Brain)
- paper says:在 5 个连续控制环境训练 >25 万 agent,系统消融 50+ 个设计选择,
  给出经验建议(如价值网络归一化、初始化、优势归一化、每步更新数等的推荐取值)。

## 3. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO
- **arXiv**: 2005.12729 — Engstrom 等, 2020 (MIT)
- paper says:PPO 相对 TRPO 的多数收益其实来自「code-level optimizations」
  (reward scaling、value clipping、正交初始化、LR annealing 等),而非 clip 目标本身。

## 与本项目的关系 [AI-DRAFT]
本项目**自研 JAX PPO**,这三篇是「不踩坑清单」。关键落地项:advantage normalization、
value function clipping、正交初始化 + tanh、obs/reward 归一化、全局梯度裁剪、LR 退火。
v2.0 骨架的 PPO 循环应把这些做成显式可开关的 config 项,验证「空跑一步」时逐项对照。
[AI-DRAFT] 这批细节比「选 PPO 还是 IMPALA」更决定最终能不能训出东西。
