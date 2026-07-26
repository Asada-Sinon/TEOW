# The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games (MAPPO)

- **citekey**: yu2021mappo
- **arXiv**: 2103.01955 (NeurIPS 2022 D&B Track)
- **作者/年**: Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu — 2021/2022
- **状态**: 精读(摘要+已知内容)

## 问题
多智能体领域普遍认为 PPO 样本效率差、不如 off-policy(QMIX/MADDPG)。本文反驳这一成见。

## 核心做法(paper says)
MAPPO = PPO + CTDE(集中训练/分散执行):每 agent 共享一个 actor(去中心化,只看
局部 obs),但用一个**集中式 critic**(可看全局状态)估值以缓解非平稳。给出若干实现
要点(value normalization、输入表征、clip、GAE 等)。

## 实验/结论(paper says)
在 MPE、SMAC、Google Research Football、Hanabi 四个基准上,极少调参、无领域特化改动,
即达到与 off-policy SOTA 相当或更优的最终回报与样本效率。

## 局限 [AI-DRAFT]
- 主要在**合作**同队设定验证;本项目是 4 人 FFA 竞争 + 自对弈,集中 critic 的
  「全局状态」定义与信度分配需重新设计(competitive 下没有共享团队回报)。

## 与本项目的关系 [AI-DRAFT]
关键参考。本项目「同类型实体共享策略网 + 集中 critic」的设计正是 MAPPO 范式。
可直接借鉴其实现要点清单。但 FFA 竞争设定要把「集中 critic」理解为「每个 player
一个 critic 看全局」,而非合作共享 critic。见 dewitt2020ippo 的独立学习对比。
