# IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

- **citekey**: espeholt2018impala
- **arXiv**: 1802.01561 (ICML 2018)
- **作者/年**: Espeholt, Soyer, Munos, Simonyan, Mnih, ... Kavukcuoglu — 2018 (DeepMind)
- **状态**: 精读(摘要+已知内容)

## 问题
分布式 RL 中 actor 采样与 learner 更新解耦后,actor 的策略滞后于 learner(policy-lag),
数据变成 off-policy,直接用会有偏。要在保持高吞吐(数千机器)的同时纠偏。

## 核心做法(paper says)
V-trace off-policy 修正:用截断重要性采样比(ρ̄, c̄ 两个截断阈值)对 n-step 回报做
修正,得到有界方差、可控偏差的价值目标。actor 只管产轨迹,learner 批量消费。
吞吐达 ~250k FPS,比单机 A3C 快 >30 倍。

## 实验/结论(paper says)
DMLab-30、Atari 多任务。单一 agent 架构可扩展到数千机器而不牺牲数据效率。

## 局限 [AI-DRAFT]
- V-trace 是为「大规模异步、policy-lag 明显」设计;本项目 JAX 同步 vmap 采样几乎无 lag,
  V-trace 的纠偏收益变小。
- 分布式 actor-learner 架构对单卡项目是过度设计。

## 与本项目的关系 [AI-DRAFT]
备选而非首选。V-trace 的价值在于「若将来做 async / 大 rollout 复用导致 off-policy」。
单卡同步 vmap 场景下,PPO 的 clip 已足够,不必先上 IMPALA。V-trace 思想可留作日后
提升样本复用率时的工具。
