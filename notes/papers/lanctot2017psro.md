# A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO)

- **citekey**: lanctot2017psro
- **arXiv**: 1711.00832 (NeurIPS 2017)
- **作者/年**: Lanctot, Zambaldi, Gruslys, Lazaridou, Tuyls, Perolat, Silver, Graepel — 2017 (DeepMind)
- **状态**: 精读(摘要+已知内容)

## 问题
独立 RL(InRL)会对训练时的对手策略过拟合,泛化差。要一个统一框架来生成鲁棒策略组合。

## 核心做法(paper says)
PSRO(Policy-Space Response Oracles):迭代地
(1) 对当前对手**策略混合**训练一个近似最佳响应(oracle,用深度 RL),
(2) 把新策略加入种群,用经验博弈论分析(EGTA)在种群的收益矩阵上算 meta-strategy
(用什么概率混合各策略当对手)。
统一了 InRL、iterated best response、double oracle、fictitious play。

## 实验/结论(paper says)
在部分可观测协调博弈、Leduc poker 等上,得到比 InRL 更少过拟合、更能泛化的策略。

## 局限 [AI-DRAFT]
- 每轮要训一个「新最佳响应」并维护 N×N 收益矩阵,轮数一多算力线性/二次膨胀;
  完整 PSRO 对单卡偏重。

## 与本项目的关系 [AI-DRAFT]
是 Q2「种群方法避免策略坍缩」的理论骨架。对本项目最实用的**简化版**:
把 v1.9 筛出的脚本指挥官当作固定的初始种群,meta-strategy 用简单的
「按对当前 RL 胜率加权」(≈ PFSP),先不做「每轮训新 oracle 入池」的完整循环。
即:借 PSRO 的对手采样思想,不背 PSRO 的完整算力。
