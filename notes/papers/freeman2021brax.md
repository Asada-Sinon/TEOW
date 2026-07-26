# Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation

- **citekey**: freeman2021brax
- **arXiv**: 2106.13281 (NeurIPS 2021 D&B Track)
- **代码**: github.com/google/brax(含 JAX 版 PPO/SAC/ES)
- **作者/年**: Freeman, Frey, Raichuk, Girgin, Mordatch, Bachem — 2021 (Google)
- **状态**: 略读(摘要+已知内容;代码为参考物)

## 问题/定位
在加速器上做大规模刚体物理仿真 + RL,环境与算法**同设备编译**,无缝并行。

## 核心做法(paper says)
JAX 写的可微物理引擎;**提供 PPO、SAC、ES、APG 的 JAX 重实现**,与环境一起编译在同一
设备上,随加速器线性扩展。

## 局限 [AI-DRAFT]
- 领域是连续控制物理仿真,与离散 RTS 无关;价值仅在**其 JAX PPO/SAC 实现**作代码参考。

## 与本项目的关系 [AI-DRAFT]
次要参考:当 lu2022purejaxrl 的 PPO 模板不够时,Brax 的 `training/agents/ppo` 是另一份
成熟、工程化更完整(含 checkpoint、eval、归一化)的 JAX PPO 实现,可交叉对照实现细节。
非首要,列为「实现时备查」。

---
## 附:Q5 其它 JAX 生态(未单独建 note,仅登记,避免凭记忆展开)
- **gymnax** (Lange, 2022, GitHub 软件引用,无 arXiv):JAX 版 gym API,单智能体环境集。
- **Jumanji** (arXiv:2306.09884):JAX 环境套件 `[未验证:作者/年仅见检索标题]`。
- **Stoix / rejax / Mava**:JAX(MA)RL 算法库,均以 GitHub 为主 `[未验证:未做单独检索]`。
以上除 gymnax/Jumanji 的 ID 外,不据记忆写作者年份;需要时再逐一检索确认。
