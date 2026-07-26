# Generative Adversarial Imitation Learning (GAIL)

- **citekey**: ho2016gail
- **arXiv**: 1606.03476 (NeurIPS 2016)
- **作者/年**: Ho, Ermon — 2016 (Stanford)
- **状态**: 精读(摘要+已知内容)

## 问题
从专家轨迹学策略,不访问专家、无环境奖励信号。传统 IRL 先学奖励再 RL,两步且昂贵。

## 核心做法(paper says)
把模仿学习表述为**占用度量匹配**(occupancy measure matching, JS 散度)。
判别器 D 学着区分专家 (s,a) 与策略 (s,a);策略用 −log D 作为奖励做 RL(TRPO/PPO),
即 GAN 式对抗训练。免去显式 IRL 的内层 RL 循环。

## 实验/结论(paper says)
连续控制任务上显著优于既有 model-free 模仿方法,能从少量专家轨迹恢复接近专家的策略。

## 局限 [AI-DRAFT]
- 对抗训练不稳定(GAN 通病),判别器/策略需要平衡;
- 学的是「分布上像专家」,不保证超过专家;若脚本对手本身次优,GAIL 会继承其上限。

## 与本项目的关系 [AI-DRAFT]
这是「用脚本指挥官」这条路的**正牌对抗式代表**(peng2021amp 是其动画特化版)。
可行用法:判别器区分「RL 轨迹」vs「v1.9 高质量脚本轨迹」,把判别分当 shaping 奖励,
与稀疏胜负奖励相加用 PPO 训 —— 缓解稀疏奖励冷启动。
[AI-DRAFT] 但比 BC 暖启动复杂且不稳;建议优先级低于「BC 暖启动 + PPO」,列为进阶备选。
