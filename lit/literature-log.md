# 文献索引

**本文件当前是空模板，表里还没有任何真实文献。** 下面只有格式说明和一段被注释掉的示例。

一篇一行。**citekey 和 DOI/arXiv ID 必须来自真实检索**，凭记忆写等于伪造。
细读笔记单独放 `notes/papers/<citekey>.md`，这张表只做索引。

列顺序固定，不要增删或调换：
`citekey | 标题 | 年份 | 状态 | 与本项目的关系 | 笔记路径`

状态取值：`待读` / `略读` / `精读` / `已复现` / `已弃`（写清为什么弃）。
「与本项目的关系」写实质关联——用了它的方法 / 是我们的基线 / 结论与我们冲突 ——
不要写「相关」。

| citekey | 标题 | 年份 | 状态 | 与本项目的关系 | 笔记路径 |
| --- | --- | --- | --- | --- | --- |
| schulman2017ppo | Proximal Policy Optimization Algorithms (arXiv:1707.06347) | 2017 | 精读 | v2.0 首选算法:单卡+vmap大批量+稀疏奖励与PPO画像契合 | notes/papers/schulman2017ppo.md |
| espeholt2018impala | IMPALA: Scalable Distributed Deep-RL w/ V-trace (arXiv:1802.01561) | 2018 | 精读 | 备选;V-trace面向async policy-lag,单卡同步vmap收益小 | notes/papers/espeholt2018impala.md |
| yu2021mappo | The Surprising Effectiveness of PPO in Cooperative MA Games (arXiv:2103.01955) | 2021 | 精读 | 共享actor+集中critic(CTDE)范式来源;但为合作设定,FFA需改 | notes/papers/yu2021mappo.md |
| dewitt2020ippo | Is Independent Learning All You Need in SMAC? (arXiv:2011.09533) | 2020 | 精读 | 对照MAPPO:独立PPO常够用,可降v2.0骨架复杂度 | notes/papers/dewitt2020ippo.md |
| huang2022ppodetails | PPO实现细节三件套(37 details/2006.05990/2005.12729) | 2020-22 | 精读 | 自研JAX PPO的不踩坑清单 | notes/papers/huang2022ppodetails.md |
| vinyals2019alphastar | Grandmaster level in StarCraft II (DOI:10.1038/s41586-019-1724-z) | 2019 | 精读 | league/PFSP金标准;需降规格移植到单卡+脚本池 | notes/papers/vinyals2019alphastar.md |
| lanctot2017psro | A Unified Game-Theoretic Approach to MARL (PSRO) (arXiv:1711.00832) | 2017 | 精读 | 种群防坍缩理论骨架;简化版=脚本池+胜率加权对手采样 | notes/papers/lanctot2017psro.md |
| heinrich2016nfsp | Deep RL from Self-Play in Imperfect-Info Games (NFSP) (arXiv:1603.01121) | 2016 | 略读 | 教训:自对弈打历史平均而非最新自己以防遗忘 | notes/papers/heinrich2016nfsp.md |
| berner2019openaifive | Dota 2 with Large Scale Deep RL (OpenAI Five) (arXiv:1912.06680) | 2019 | 精读 | 最强"PPO+自对弈够用"证据;也是规模警示(单卡不可照搬纯from-scratch) | notes/papers/berner2019openaifive.md |
| peng2021amp | AMP: Adversarial Motion Priors (arXiv:2104.02180, DOI:10.1145/3450626.3459670) | 2021 | 精读 | 用户问的"AMP":本质是GAIL动画特化版,离散RTS不适配;仅借"style+task奖励混合"思路 | notes/papers/peng2021amp.md |
| ho2016gail | Generative Adversarial Imitation Learning (arXiv:1606.03476) | 2016 | 精读 | "用脚本"路线的正牌对抗式模仿;比BC暖启动复杂,列进阶备选 | notes/papers/ho2016gail.md |
| ross2011dagger | A Reduction of Imitation Learning to No-Regret Online Learning (DAgger) (arXiv:1011.0686) | 2011 | 精读 | 脚本=可随时查询的oracle,DAgger在本项目异常可行;蒸馏得BC暖启动 | notes/papers/ross2011dagger.md |
| ng1999shaping | Policy Invariance Under Reward Transformations (PBRS) (ICML1999) | 1999 | 精读 | 奖励设计理论底线:中间奖励走势函数差,不改"赢"这一最优目标 | notes/papers/ng1999shaping.md |
| pan2022rewardhacking | The Effects of Reward Misspecification (arXiv:2201.03544) | 2022 | 略读 | 反面警示:agent越强越会hack;配PBRS+稀疏胜负兜底 | notes/papers/pan2022rewardhacking.md |
| narvekar2020curriculum | Curriculum Learning for RL Domains: Survey (arXiv:2003.04960) | 2020 | 略读 | 支撑"易→难曲线"而非全程最难;v1.9分档脚本=现成课程 | notes/papers/narvekar2020curriculum.md |
| lu2022purejaxrl | PureJaxRL / Discovered Policy Optimisation (arXiv:2210.05639) | 2022 | 精读 | v2.0 JAX PPO骨架首要临摹对象;但单agent,需自扩多player | notes/papers/lu2022purejaxrl.md |
| rutherford2023jaxmarl | JaxMARL: Multi-Agent RL Environments in JAX (arXiv:2311.10090) | 2023 | 精读 | 多agent那半的参考:JAX版IPPO/MAPPO实现+SMAX自对弈脚手架 | notes/papers/rutherford2023jaxmarl.md |
| hessel2021podracer | Podracer architectures (Anakin/Sebulba) (arXiv:2104.06272) | 2021 | 略读 | 架构背书:JAX原生env→认准Anakin(全流程上GPU),别搞分布式actor-learner | notes/papers/hessel2021podracer.md |
| freeman2021brax | Brax — Differentiable Physics Engine in JAX (arXiv:2106.13281) | 2021 | 略读 | 次要:其JAX PPO/SAC实现作交叉参考 | notes/papers/freeman2021brax.md |

<!-- 示例（安装后请删除这整块）
以下为格式示例，不是真实文献。citekey、标题、arXiv ID 全部虚构，
任何 agent 都不得引用它，更不得据此生成 BibTeX。

| example2024method | Example Title of the Paper (arXiv:2401.00000) | 2024 | 略读 | 我们 baseline 的出处，第 4 节的评测协议直接沿用 | notes/papers/example2024method.md |
-->

<!-- 真实文献行追加到上面那张表的表头下面。 -->
