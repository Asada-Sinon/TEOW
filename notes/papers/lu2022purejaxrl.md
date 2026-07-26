# PureJaxRL / Discovered Policy Optimisation

- **citekey**: lu2022purejaxrl
- **arXiv**: 2210.05639 (NeurIPS 2022) — 论文名 "Discovered Policy Optimisation"
- **代码**: github.com/luchris429/purejaxrl (端到端 JAX PPO 参考实现)
- **作者/年**: Lu, Kuba, Letcher, Metz, de Witt, Foerster — 2022 (Oxford FLAIR)
- **状态**: 精读(摘要+已知内容;代码为主要参考物)

## 问题/定位
把**整个 RL 循环(环境 + 算法)全放 GPU**、用 jit+vmap 编译成单个 kernel,消除
CPU-GPU 传输,单卡即可跑「以前要集群」的实验。

## 核心做法(paper says + repo [AI-DRAFT])
- PureJaxRL:纯 JAX 的 PPO(和 env)端到端编译,单卡多环境并行,较 PyTorch 标准实现
  **快 >1000x**(其宣称)。
- 论文 DPO 本身是「用元学习发现新的策略优化目标」,但 PureJaxRL 代码库作为**可照抄的
  单文件 JAX PPO 模板**价值更大。
- **关键限制:PureJaxRL 是单智能体的**,不原生支持多智能体/自对弈。

## 局限 [AI-DRAFT]
- 单 agent;FFA 自对弈需自行扩展(共享策略 + 多 player 维度进 vmap)。
- 「1000x」是相对慢 baseline 的宣称,别当本项目的绝对预期。

## 与本项目的关系 [AI-DRAFT]
**v2.0 JAX PPO 骨架的首要临摹对象**:其 train loop 结构(env.reset/step 全 jit、
`jax.lax.scan` 跑 rollout、vmap 批环境、minibatch 更新)正是本项目要的。
本项目已有 JAX 引擎(step 可 vmap),缺的就是这套 PPO 外壳——直接对着 PureJaxRL 搭,
再把「单 agent」改成「同类型实体共享策略 + 多 player」。
