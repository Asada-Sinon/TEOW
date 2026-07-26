# Proximal Policy Optimization Algorithms

- **citekey**: schulman2017ppo
- **arXiv**: 1707.06347
- **作者/年**: Schulman, Wolski, Dhariwal, Radford, Klimov — 2017 (OpenAI)
- **状态**: 精读(摘要+已知内容)

## 问题
策略梯度方法要么样本效率低(vanilla PG),要么实现复杂(TRPO 需二阶/共轭梯度)。
想要一个「一阶、易实现、又能限制每步策略更新幅度」的算法。

## 核心做法(paper says)
clipped surrogate objective:对重要性采样比 r_t(θ) 做 clip 到 [1-ε,1+ε],
用 min(clipped, unclipped) 近似一个 trust region,只用一阶梯度。多 epoch 复用同一批
on-policy 数据做 minibatch SGD。配 GAE 估计优势。

## 实验/结论(paper says)
MuJoCo 连续控制 + Atari。在样本复杂度、简单度、wall-time 之间取得好平衡,优于同期
在线策略梯度法。

## 局限(paper says + [AI-DRAFT])
- on-policy,样本效率不如 off-policy;但可大规模并行采样弥补。
- clip 系数、GAE λ、minibatch/epoch 数等超参对性能影响大(见 huang2022ppodetails)。

## 与本项目的关系 [AI-DRAFT]
本项目单卡 + JAX vmap 大批量采样(B≈64, ~4000 env-tick/s)+ 稀疏终局奖励,PPO 的
「牺牲样本效率换实现简单 + 靠并行吞吐补回来」正好匹配硬件画像。是 v2.0 的默认首选算法骨架。
