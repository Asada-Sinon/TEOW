# Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)

- **citekey**: vinyals2019alphastar
- **DOI**: 10.1038/s41586-019-1724-z (Nature, 2019)
- **作者/年**: Vinyals, Babuschkin, Czarnecki, Mathieu, ... Silver — 2019 (DeepMind)
- **状态**: 精读(摘要+已知内容)

## 问题
StarCraft II:巨大动作空间、长时程、部分可观测、需要多样策略且无占优纯策略(石头剪刀布式
博弈循环)。纯自对弈会「策略循环/遗忘」。

## 核心做法(paper says)
1. 用 97.1 万人类replay做**监督学习**初始化策略(BC 暖启动),奠定策略多样性;
2. **League training**:维护一个不断扩张的智能体联盟,含 main agents、
   main exploiters(专门打当前 main 的弱点)、league exploiters(打整个联盟的弱点);
   用 Prioritized Fictitious Self-Play(PFSP,按对手胜率加权采样对手)避免遗忘与循环。
3. 网络:LSTM + self-attention + pointer network,139M 参数。

## 实验/结论(paper says)
三个种族均达 Grandmaster,超过 99.8% 人类天梯玩家。

## 局限 [AI-DRAFT]
- 算力是 TPU 集群 + 数周;单卡 4090 **完全无法照搬** league 规模。
- 依赖大规模人类 replay 暖启动——本项目无人类数据,但有脚本指挥官可类比。

## 与本项目的关系 [AI-DRAFT]
方法论金标准,但要「降规格移植」:(a) 用脚本指挥官替代人类 replay 做 BC 暖启动 +
固定对手池;(b) exploiter/PFSP 的**思想**(按对手胜率优先采样、专门针对性对手)可在
单卡上做一个「轻量联盟」——few main + 脚本池,而非 DeepMind 的完整 league。
是 Q2「最便宜又能避免循环」的答案的直接灵感来源。
