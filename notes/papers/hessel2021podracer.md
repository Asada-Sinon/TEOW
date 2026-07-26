# Podracer architectures for scalable Reinforcement Learning (Anakin / Sebulba)

- **citekey**: hessel2021podracer
- **arXiv**: 2104.06272
- **作者/年**: Hessel, Kroiss, Clark, Kemaev, Quan, Keck, Viola, van Hasselt — 2021 (DeepMind)
- **状态**: 略读(摘要+已知内容)

## 问题/定位
如何在加速器(TPU Pod)上把 RL 训练做得可扩展、高效、可复现。

## 核心做法(paper says)
两种架构:
- **Anakin**:环境用 JAX 写、能整个跑在加速器上时——把 acting 与 learning **全部编译进
  加速器**,用 vmap/pmap 批量并行,无 host-device 往返。
- **Sebulba**:环境只能在 CPU(如 Atari)时的 decomposed actor-learner 方案。

## 结论(paper says)
论证 TPU/加速器很适合可扩展 RL;Anakin 模式对「JAX 原生环境」尤其高效。

## 局限 [AI-DRAFT]
- 论文语境是 TPU Pod;但 Anakin 的「全流程上加速器」原则对单张 GPU 同样成立
  (PureJaxRL 本质就是 GPU 上的 Anakin 范式)。

## 与本项目的关系 [AI-DRAFT]
给本项目的架构选择提供命名与背书:本项目引擎是**JAX 原生、可 vmap** → 属于 Anakin
适用场景,应把「采样 + PPO 更新」全留在 GPU 上、用 scan/vmap 编译,**不要**搞
CPU actor + GPU learner 的分布式(那是 Sebulba/IMPALA 的场景,单卡无必要)。
即:架构上认准 Anakin 范式,实现上照 PureJaxRL。
