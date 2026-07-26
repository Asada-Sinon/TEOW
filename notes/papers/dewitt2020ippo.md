# Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? (IPPO)

- **citekey**: dewitt2020ippo
- **arXiv**: 2011.09533
- **作者/年**: de Witt, Gupta, Makoviychuk, Torr, Sun, Whiteson 等 — 2020
- **状态**: 精读(摘要+已知内容)

## 问题
MARL 是否一定需要集中式(joint)价值分解(QMIX 等)?独立学习(每 agent 自己一套
value/policy,把别人当环境的一部分)够不够?

## 核心做法(paper says)
IPPO = 每个 agent 独立跑 PPO,只估计自己的局部价值函数,无集中 critic、无通信、
无价值分解。

## 实验/结论(paper says)
在 SMAC 多个地图上,IPPO 与 SOTA 的联合学习方法相当甚至更好,几乎不用调参。
作者推测其强表现来自对某些环境非平稳性的鲁棒性。

## 局限 [AI-DRAFT]
- 独立学习理论上有非平稳/收敛无保证的问题;IPPO 是「经验上惊人地好」,非有理论背书。

## 与本项目的关系 [AI-DRAFT]
和 yu2021mappo 构成关键对照:**要不要集中 critic**。IPPO 说「很多时候不用,独立 PPO
就够」——对本项目意味着可先用最简单的「共享策略 + 每 player 独立价值(甚至只用终局
胜负)」起步,把集中 critic 当作可选增量,而非必须先搞定的前置工程。降低 v2.0 骨架复杂度。
