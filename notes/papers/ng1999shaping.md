# Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping

- **citekey**: ng1999shaping
- **载体**: ICML 1999, pp.278-287. ACM DL id 10.5555/645528.657613
  `[未验证:该 10.5555 前缀是 ACM 对早期无正式 DOI 论文的占位标识,非可解析 DOI;
  本文早于 arXiv,无 arXiv ID]`
- **作者/年**: Ng, Harada, Russell — 1999 (UC Berkeley)
- **状态**: 精读(摘要+已知内容)

## 问题
给 MDP 加 shaping 奖励能加速学习,但乱加会改变最优策略(学到「刷分」而非解任务)。
什么形式的 shaping 保证**最优策略不变**?

## 核心做法(paper says)
证明:形如 **F(s,s') = γΦ(s') − Φ(s)**(势函数差,potential-based reward shaping, PBRS)
的附加奖励,是保持最优策略不变的**充分且必要**形式。Φ 可任意选(通常取对真值的启发式估计)。

## 实验/结论(paper says)
在网格世界等上,用 PBRS 大幅加速收敛且不改变最优解。

## 局限 [AI-DRAFT]
- 只保证「最优策略」不变;有限训练预算下,Φ 选得差仍可能误导中间学习。
- 需要一个合理的势函数 Φ(对状态价值的先验)。

## 与本项目的关系 [AI-DRAFT]
**回应「奖励怎么设计不被 hack」的理论底线**:本项目终局稀疏胜负奖励太稀,想加经济/
兵力/伤害等中间奖励时,应尽量写成**势函数差**形式(如 Φ = 己方兵力价值 − 敌方兵力价值),
而非直接累加「每采一矿 +1」这类可被刷的项。这样既缓解稀疏性又不改变「赢」这个最优目标。
是 v2.0 奖励设计文档的必引原则。与 pan2022rewardhacking 互补(一个给正解形式,一个给反面警示)。
