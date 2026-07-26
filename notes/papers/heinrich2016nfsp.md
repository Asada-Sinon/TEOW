# Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP)

- **citekey**: heinrich2016nfsp
- **arXiv**: 1603.01121
- **作者/年**: Heinrich, Silver — 2016 (UCL/DeepMind)
- **状态**: 略读(摘要+已知内容)

## 问题
朴素自对弈在博弈里会震荡/不收敛(追着最新对手打,遗忘旧策略)。要一个能逼近 Nash
均衡的深度自对弈方法。

## 核心做法(paper says)
NFSP = Fictitious Self-Play + 神经网络。每 agent 两张网:
- RL 网络:学对「对手历史平均策略」的近似最佳响应;
- 监督网络:学自己历史行为的**平均策略**(对最佳响应做时间平均)。
行动时按概率在两者间混合。用 reservoir sampling 维护历史。

## 实验/结论(paper says)
Leduc poker 上逼近 Nash;常规 RL 方法发散。Limit Texas Hold'em 上接近专家级。

## 局限 [AI-DRAFT]
- 面向零和不完全信息博弈的 Nash 收敛;本项目是 4 人 FFA(非零和、非两人),Nash 概念
  不直接适用。
- 「平均策略网络」增加实现与显存开销。

## 与本项目的关系 [AI-DRAFT]
提供 Q2 的核心教训:**自对弈要对抗「历史平均」而非「最新自己」以防遗忘**。这一原则
比 NFSP 具体算法更值得移植——即使不实现双网络,也应保留一个「历史对手快照池」并从中
采样,而非只和最新 checkpoint 对打。
