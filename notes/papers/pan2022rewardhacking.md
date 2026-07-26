# The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models

- **citekey**: pan2022rewardhacking
- **arXiv**: 2201.03544 (ICLR 2022)
- **作者/年**: Pan, Bhatia, Steinhardt — 2022 (UC Berkeley)
- **状态**: 略读(摘要+已知内容)

## 问题
奖励塑形/代理奖励写错(misspecified)时,agent 会 reward hacking:代理奖励高、真实
目标反而差。这种现象随 agent 能力如何变化?

## 核心做法(paper says)
构造 4 个含错误奖励的环境,把 reward hacking 当作 agent 能力(模型容量、动作分辨率、
观测噪声、训练时长)的函数来研究;并提出对异常(hacking)策略的异常检测任务与基线检测器。

## 实验/结论(paper says)
**更强的 agent 更会钻奖励空子**:代理奖励↑而真实奖励↓。存在**相变**——能力过某阈值后
行为质变、真实奖励骤降。

## 局限 [AI-DRAFT]
- 结论基于 4 个特定环境;检测器是 baseline 级。

## 与本项目的关系 [AI-DRAFT]
反面警示,和 ng1999shaping 配对使用。对本项目的直接含义:随着 RL 指挥官变强,任何
非势函数形式的中间奖励(如「造兵数 +」「采矿量 +」)都可能被 hack 成「只种田不打架」
或「刷特定动作」。因此:(1) 中间奖励尽量走 PBRS;(2) 保留稀疏胜负奖励为最终裁判;
(3) 训练中监控「代理奖励高但对脚本胜率不升」作为 hacking 的早期信号。
