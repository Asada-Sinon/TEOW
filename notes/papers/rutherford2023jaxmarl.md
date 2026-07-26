# JaxMARL: Multi-Agent RL Environments (and Algorithms) in JAX

- **citekey**: rutherford2023jaxmarl
- **arXiv**: 2311.10090 (AAMAS 2024)
- **代码**: github.com/FLAIROx/JaxMARL(含 IPPO/MAPPO 的 JAX 实现 + SMAX 环境)
- **作者/年**: Rutherford, Ellis, ... Foerster 等 — 2023 (Oxford FLAIR 等)
- **状态**: 精读(摘要+已知内容;代码为主要参考物)

## 问题/定位
第一个「易用 + GPU 高效」的开源 MARL 代码库:多环境 + 多基线算法全 JAX 化,
端到端训练比既有(CPU)方案快至多 ~12500x(其宣称)。

## 核心做法(paper says)
- 涵盖合作/竞争/混合、离散/连续、CTDE/zero-shot 的多种环境(MPE、Overcooked、Hanabi、
  Multi-Agent Brax、STORM 等);
- 引入 **SMAX**:向量化、去掉 StarCraft II 引擎依赖的简化版 SMAC,可 GPU 加速,
  「解锁 self-play / meta-learning」等方向;
- **提供 JAX 版 IPPO 与 MAPPO 的参考实现**。

## 局限 [AI-DRAFT]
- 是基准/库,非 RTS;其环境非本项目的引擎。价值在**算法实现与自对弈脚手架**可临摹。

## 与本项目的关系 [AI-DRAFT]
**多智能体那一半的首要参考**——补齐 lu2022purejaxrl 缺的多 agent 部分。
本项目该照抄的是:JaxMARL 里 **IPPO/MAPPO 的 JAX 实现结构**(如何把多 agent 维度
塞进 vmap、共享参数、集中 critic 的 batch 组织),以及 SMAX 的自对弈/对手池组织方式。
是「PPO(dewitt2020ippo / yu2021mappo)理论 + PureJaxRL 单 agent 模板」之间的桥。
