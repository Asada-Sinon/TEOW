# v2.0 RL 方案设计:一版最优方案(调研 + 不训练的 JAX 骨架)

> **性质与标注**:本文件是 v2.0「RL 算法调研 + 不训练的 JAX 骨架」的**设计文档**(issue.md `## v2.0`
> 要求的「一版最优方案」)。除另有 `[source: <run_id>]` 的实测数字外,**通篇结论均为 `[AI-DRAFT]`**
> ——AI 综合已入库文献与真实引擎类型得出的设计推断,尚未经用户核验,更未经训练验证。
> - **provenance**:撰写于 git `a3ddadf`(2026-07-27);引擎类型以该 commit 的 `src/teow/` 为准。
> - **文献**:引用一律用 citekey,DOI/arXiv 见 `lit/literature-log.md`(19 篇,已入库,commit `a3ddadf`)。
>   本文**不新增任何引用**;凡「未检索到直接支撑」处明写,绝不拿沾边文献顶替(`.claude/rules/notes.md`)。
> - **本版边界**:issue.md `## v2.0` 明文「**本版不开始训练**」。本文只给方案 + 骨架规格;真正训练是 v2.1+。
> - **数值纪律**:唯一实测硬数字是吞吐 `[source: 20260726-v18-bench]`,且带重要 caveat(见 §1.2 / §8)。
>   games/day 等训练吞吐指标**尚未实测**,本文一律不口算(硬约束 #1),留作 v2.1 必测项。

---

## §0 TL;DR — 推荐的一版最优方案

**一句话:自研 JAX PPO + 脚本指挥官课程(易→难)+ BC/DAgger 暖启动 + 势函数塑形(PBRS),自对弈作为后置层。**

| 维度 | 选定方案 | 主要依据(citekey) |
|---|---|---|
| 核心算法 | **自研 JAX PPO**(clipped surrogate + GAE) | schulman2017ppo, berner2019openaifive |
| 工程范式 | **Anakin**(采样+更新全留 GPU,`scan`/`vmap`),照抄 **PureJaxRL** 单 agent 模板 + **JaxMARL** 多 agent 结构 | hessel2021podracer, lu2022purejaxrl, rutherford2023jaxmarl |
| 冷启动 | **BC 暖启动 → DAgger 蒸馏**(脚本=可任意查询的 oracle,协变量漂移可消) | ross2011dagger, vinyals2019alphastar |
| 奖励 | **稀疏 FFA 名次终局 + 退火 PBRS 塑形**(投入价值差),非 PBRS 中间项一律不用 | ng1999shaping, pan2022rewardhacking |
| 策略/价值 | **实体共享 actor(按类型共享参数)+ IPPO 独立 critic 起步**,MAPPO 每-player 中心 critic 作可选增量 | dewitt2020ippo, yu2021mappo |
| 对手 | **v1.9 筛选的脚本指挥官分难度档课程**,阈值触发升档;**快照池 + PFSP-lite 自对弈**为后置层 | narvekar2020curriculum, lanctot2017psro, heinrich2016nfsp, vinyals2019alphastar |
| 明确不用 | IMPALA/V-trace(单卡同步 vmap 无 policy-lag);纯 from-scratch 自对弈(算力不足);纯模仿(受脚本天花板封顶) | espeholt2018impala, berner2019openaifive |

**「用户问的 PPO 还是 AMP 还是不用 RL」的正面回答(详见 §1.3)**:选 **PPO**。AMP(peng2021amp)在离散 RTS
里退化为「对抗式模仿脚本(≈GAIL)」,真正的岔路是「**PPO(+可选对抗式模仿)vs 纯模仿**」,而非「PPO vs AMP」;
纯模仿被脚本天花板封顶,故用模仿做**暖启动/正则**、用 PPO 做**超越**。

---

## §1 算法选型

### 1.1 为什么是 PPO(而不是别的 RL)

本项目的硬件/任务画像:**单卡(4090)+ JAX 原生可 vmap 引擎 + 稀疏终局奖励 + 数千 tick 长时程 + 4 人 FFA**。

- **PPO 与该画像高度契合**(schulman2017ppo):一阶、实现简单、靠 clip 近似 trust region;其唯一硬伤
  「on-policy 样本效率低」正好被本项目的**大批量并行采样**弥补——实测 GPU vmap B≈64 达 **4025.8 env-tick/s**
  `[source: 20260726-v18-bench]`(见 §1.2 caveat)。berner2019openaifive 是「PPO+自对弈就够用」最强证据点
  (Dota2 超人,算法就是朴素 PPO+GAE,无花哨层次/模仿架构)。
- **工程范式认准 Anakin**(hessel2021podracer):引擎是 JAX 原生、可整体上 GPU,应把「采样+PPO 更新」全部
  编译进 GPU、用 `jax.lax.scan`/`vmap`,**不要**搞 CPU-actor+GPU-learner 的分布式(那是 Sebulba/IMPALA
  场景,单卡无必要)。实现直接临摹 **PureJaxRL**(lu2022purejaxrl,单 agent JAX PPO 单文件模板)+ **JaxMARL**
  (rutherford2023jaxmarl,补齐多 agent:如何把「多 player + 共享参数 + 中心 critic」塞进 vmap/scan)。
- **明确排除 IMPALA/V-trace**(espeholt2018impala):V-trace 纠的是异步大规模下的 policy-lag;单卡同步 vmap
  采样几乎无 lag,PPO 的 clip 已足够。V-trace 思想留作「日后提高样本复用率导致 off-policy」时的工具。
- **备查**:freeman2021brax 的 `training/agents/ppo` 是另一份工程更完整(含 checkpoint/eval/归一化)的 JAX PPO
  实现,当 PureJaxRL 模板不够时交叉对照。

### 1.2 吞吐现实与 caveat(决定训练可行性,必须诚实)

实测 `[source: 20260726-v18-bench]`(bench.json):
- 单 CPU **101.39** tick/s;CPU vmap **无益**(B32/B128 env 176.7/145.8 tick/s,per-world 反降)——**勿用 CPU vmap**。
- **GPU vmap B64 = 4025.8 env-tick/s**(per-world 62.9)是单卡甜点;B256/B1024 反降(2719/2840)。

**两条 caveat(必须带进 v2.1,不得当成训练吞吐)**:
1. 该 bench 的 git hash 是 `3834f36`(**dirty,异界之门 gate 阶段尚未接入 step**),故**未含 gate 逐 tick 开销**;
   v1.8 收尾时曾记「异界之门令 step 编译变慢,串行全套 >40min」(docs/DECISIONS.md 2026-07-27)——**含 gate 的
   真实 env-tick/s 必然更低,须重测**。
2. 该数是**纯引擎**吞吐;RL rollout 每 tick 还要跑**策略网前向**(+训练时反向),又一层开销。
- **结论**:`games/day` 训练预算 `[AI-DRAFT]` **本文不估算**(硬约束 #1),列为 v2.1 头号必测项(§8-8)。

### 1.3 正面回答「PPO 还是 AMP 还是不用 RL」

- **AMP(peng2021amp)不是独立候选**。AMP 本体是连续控制物理动画的「风格奖励(判别器打分)+任务奖励」,其
  「风格」定义在低层物理 state 转移的自然度上。本项目是**离散 116 维指令流**,不存在「动作自然度」这一物理量;把
  它塞进 AMP 判别器,得到的是「像不像脚本指挥官的战术分布」——那**本质就是 GAIL**(ho2016gail)。AMP 相对 GAIL
  的动画特化增益在这里≈0。**真正的岔路是「PPO(+可选对抗式模仿)vs 纯模仿」,不是「PPO vs AMP」**。AMP 唯一可迁移
  内核 =「判别器 style 奖励 + task 奖励一起用 PPO 训」这个**组合思路**(RTS 版:胜负 task 奖励 +「打得像高质量
  脚本」的判别器奖励),列为 §4 的可选设计元素。
- **「不用 RL、纯蒸馏/模仿脚本指挥官」评估(issue.md 明文要求正面评估)**:
  - **优点**:脚本是**可任意查询的 oracle**(`controller(state,key)->actions[N]`),DAgger(ross2011dagger)在此
    异常可行——最大落地障碍「人类专家难在线标注」在本项目**不存在**;可无漂移地把脚本蒸馏进网络。
  - **致命短板**:纯模仿(BC/GAIL)学的是「分布上像专家」,**被专家天花板封顶**(ho2016gail 局限)。而 v1.9 的
    脚本指挥官有已知硬伤:上帝视角、`target_mode` 目前 **inert**(只 attack-move 最近敌 HQ,movement.py 硬编码,
    见 DECISIONS 2026-07-27 P1-1)、启发式可被针对。纯模仿只会**继承这些缺陷**,无法超越。
  - **裁决 `[AI-DRAFT]`**:**用模仿做暖启动/正则,用 PPO 做超越**。即 BC/DAgger 先把策略拉到「≈最强脚本」省掉
    稀疏奖励冷启动,再让 PPO 用胜负奖励往上走。**不选纯模仿**。

### 1.4 为什么课程先于自对弈、为什么 BC 暖启动在这里便宜

- **课程先于自对弈**(narvekar2020curriculum):稀疏终局奖励 + 数千 tick,从零自对弈会因**冷启动无信号**在本
  算力下学不动(berner2019openaifive 的规模是本项目的好几个数量级,不可照搬纯 from-scratch)。v1.9 的分档脚本
  = 现成的**手工课程任务序列**:先在最弱档拿稳胜率(奠定基本操作),再按学习进度升档,比「一上来最难/自对弈」稳。
- **BC 暖启动在这里近乎白捡**:脚本是确定性 oracle,可对**任意** state 产出「专家动作」→ 直接监督蒸馏,无需
  AlphaStar 那样的人类 replay(vinyals2019alphastar 用 97 万人类局做 SL 初始化;本项目用脚本平替这一步)。
  暖启动权重 ≈ 最强脚本,再交 PPO。

---

## §2 观测编码(obs):`WorldState` → 定形策略输入

**总原则**:`WorldState` 是定容 pytree(实体表 `[N=256]` + 子表),天然适合做成**逐实体 token + 全局向量**,
且**以行动方玩家 p 为中心(egocentric)**,让一张共享网服务 4 个座位。

### 2.1 结构:逐实体 token `[N, F]` + alive 掩码 + 全局向量 `[G]`

```
obs_p = {
  entity_tokens : f32 [N=256, F]     # 每个实体一条特征 token
  entity_mask   : bool [N]           # = state.alive(死槽停泊在无害值,靠掩码挑)
  global_vec    : f32 [G]            # 全局态势(资源/升级/tick/门态/怪压/存活)
}
```
- **egocentric 化**:把行动方 p 的行块重排到最前(`owner_of_slots` 决定行块),或直接用 `[is_own, is_enemy]`
  标志位 + **相对己方 HQ 的坐标平移**标注视角。地图是四重对称(v1.5 规格),理论上可几何旋转到规范帧让 4 座位
  完全等价——但这是**优化项**,起步先用「平移 + own/enemy 标志」的最简正确做法(几何旋转列 §8 开放问题)。
- **参数共享红利**:一张 egocentric 共享网服务 4 个座位 ⇒ 每个 env 天然产 4 份玩家轨迹,既是样本效率也直接
  贴合自对弈(§6)。

### 2.2 F 的大致内容(逐实体,`[AI-DRAFT]`,精确 spec 见 §8-1)

| 组 | 特征(来源字段/函数) | 维度约 |
|---|---|---|
| 存活/视角 | `alive`;`is_own=(owner==p)`、`is_enemy`;(可选 opponent-id embedding) | 3 |
| 类型 | `etype` → **学习 embedding**(优于 29 维 one-hot,省 F);辅以 `is_harvester/is_combat/is_building/is_air`(cfg 派生表) | 8+4 |
| 位置 | `pos`(row,col)→ 己方 HQ 相对坐标 + 地图归一化坐标 + 到己方 HQ 距离(经 `cell_of` 约定,格心=整数) | 5 |
| 血/级 | `hp / max_hp_of(state,cfg,owner)`(用 stats.py 真源);`level/7` | 2 |
| 意图 | `order` one-hot(IDLE/HARVEST/BUILD/MOVE/ATTACK/GARRISON = 6) | 6 |
| 态位 | `inside`、`aboard>=0`、`atk_cd>0` 三个离场/冷却旗;`cargo/carry_by_type`;`phase` | 5 |
| 生产 | `btimer>0`(在产)+ `btimer` 归一化(HQ/兵营/营/在建自由格建筑) | 2 |
| 战力 | `atk_of(state,cfg,owner)/norm`、`atk_range_by_type/norm`(stats.py/cfg 派生,含升级增益) | 2 |

**F ≈ 36–48**(`[AI-DRAFT]`,取整到便于 SIMD 的值;确切维度 = 开放问题 §8-1)。关键纪律:**血量/攻击/等级等派生量
一律复用 `stats.py` 的 `max_hp_of/atk_of/effective_level/etype_idx`**,不在 obs 里重算(避免第二真源)。

**怪物子表 `monster_* [P,Mmax=64]`**:阵营隔离,只有行 p 的怪与玩家 p 交互。起步做法 = 把「己方怪压力」**摘要进
全局向量**(见 §2.3);升级项 = 把 Mmax 条怪 token 拼进 entity_tokens(带 `is_monster` 标志)。overtime 前怪为空,
起步摘要足够。

### 2.3 全局向量 `G` 的大致内容(`[AI-DRAFT]`)

- 己方 `resources[p]`(ore,water)归一化;(可选)全 P 家 `resources`(P×2=8)
- 己方 `upgrades[p, :]`(N_LINES=11 条线级 /7);己方基地级 `level[hq_slot(p)]/7`
- `tick/episode_len`;**门态**:`tick>=gate_open_tick`(bool)+ overtime 已过拍数归一化
- **己方怪压力**:存活怪数 `/Mmax`、怪总 hp 归一化、最近怪到己方 HQ 距离(overtime 生死攸关)
- **各家 HQ 存活掩码 `[P=4]`**(谁还在局里 —— FFA 名次/选敌的关键)
- 己方经济/军力摘要:工人数、己方已建资源点数、按类型的军队构成计数(归一化)

`G ≈ 40–55`(`[AI-DRAFT]`)。

### 2.4 actor head:三种结构的成本/收益(单 GPU)

| 方案 | 机制 | 成本(N=256,B=64,单 4090) | 收益 | 建议 |
|---|---|---|---|---|
| **A. 逐实体共享 MLP** | 每 token 过同一张 MLP → 116 logits;type 作输入特征 | 最低,O(N·F·H) | 简单、vmap 友好、参数最省 | **起步选它** |
| B. 按类型共享 head | 每类型一套 head(~29),按 etype dispatch | 略高、参数多 | 类型专精 | 收益边际,可选 |
| C. 跨实体注意力 | token 互相 attend(AlphaStar 式实体编码器) | O(N²·H),N²=65k,**单卡可承受**但每层加一块 | 单位间**协同**(合围/掩护) | **第一个升级项** |

**推荐路线 `[AI-DRAFT]`**:**A + 全局上下文注入**(把 entity_tokens 的 mean-pool 作为「态势」拼回每个 token,让
逐实体决策看得到聚合信息)起步;跑通后加**一层跨实体注意力(C)**作第一个升级。AlphaStar(vinyals2019alphastar)
用 self-attention + pointer network,但那是 TPU 集群 139M 参;单卡先小网(§7)。「实体共享策略网」(issue.md v2+
封存方案)= 方案 A/B 的参数共享语义。

---

## §3 动作与掩码

- **输出**:策略对每个实体产 `[N=256, 116]` logits(116 = `n_actions(cfg)`,已实测 default cfg:Nn=20/F=3 → **116**)。
- **掩码**:引擎第一天就输出 `legality_mask(state,cfg,mapdata,owner) -> bool[N,116]`(actions.py)。策略把非法位
  置 `-inf` 后 softmax 采样——这正是 **invalid-action masking**,`random`/`scripted` 控制器已在用同一套
  (controller.py:`random_actions` 的 Gumbel-argmax on masked logits)。**RL 直接复用,零改引擎**。
- **NOOP 恒合法**(`mask[:,A_NOOP]=True`)⇒ 掩码 softmax 永不退化;死槽/矿内(`inside`)/舱内(`aboard>=0`)
  只 NOOP 合法。
- **参数化 action id 铁律**:116 里大段是**参数化块**——`a_build(k)`、`a_harvest(k,cfg)`、`a_train_unit(t,cfg)`、
  `a_research_line(line,cfg)`、`a_garrison_node(k,cfg)`、`a_garrison_flag(j,cfg)`、`a_build_defense(t,cfg)`… id 随
  `n_nodes/max_flags/N_LINES` 变。**策略只输出 116 logits;解码/分析动作时一律走 `a_*()` helper,绝不 hardcode
  数字 id**(actions.py 头注释:退役 id 永久保号非法,旧轨迹可解——硬编码会踩雷)。
- **损失掩码(易漏但要紧,`[AI-DRAFT]`)**:PPO 更新只应对「己方 alive 且 actable」的实体计 log-prob/熵/优势;
  被迫 NOOP 的死槽/离场槽**不进策略损失**(否则大量确定性 NOOP 稀释梯度)。
- **军旗/落位的巧思对 RL 友好**:插旗/建自由格建筑「落在脚下」,动作表**不带坐标参数**(actions.py),故动作空间
  不随地图膨胀——116 维固定,利于 RL。

---

## §4 奖励设计

### 4.1 稀疏终局(唯一「真」奖励,最终裁判)

- 引擎 `_end_tick` 产 `winner ∈ [0,P-1]`(**v1.8 异界之门必分胜负,无和局**;done 后冻结)。
- **FFA 名次奖励 `[AI-DRAFT]`**:玩家 p 的终局奖励按**名次**给,而非仅胜/负——
  `r_terminal(p) = 1 - 2·(rank_p - 1)/(P - 1)`(第 1=+1,末位=-1,P=4 时为 +1/+0.33/-0.33/-1)。
  - **名次来源(需 rollout 记录)**:引擎只存最终 `winner` 与末态 `hq_alive`,**不直接记淘汰顺序**。rollout scan
    body 需**逐 tick 记录各家 HQ 存活翻转的 tick**(`hq_alive[P]` 的首次 False),末了由淘汰 tick 排名次。**便宜**
    (每 tick 一个 `[P]` bool 的 min-tick 归约)。退而求其次:只用 `+1 胜/-1 非胜`(最简,信号更稀)。名次定义
    列开放问题 §8-2。

### 4.2 势函数塑形(PBRS):缓解稀疏、且可证不改最优

- **形式(ng1999shaping,充要保最优不变)**:附加奖励 `F(s,s') = γ·Φ(s') − Φ(s)`。势函数差**电报级抵消**
  (telescoping),整段轨迹的塑形总和只依赖首末 Φ,不改「赢」这一最优解。
- **Φ 候选 `[AI-DRAFT]`(投入价值差)**:以「**投入资源价值**」为统一货币(每单位/建筑在 cfg 都有 ore+water 造价):
  ```
  Value_p(s) = stockpile_p + Σ_{i∈own_p, alive} cost(type_i)·(hp_i / max_hp_i)      # 库存 + 折血的存量投入
  Φ_p(s)     = squash( Value_p(s) − aggregate_{q≠p} Value_q(s) )                     # 与对手的价值差
  ```
  - `cost(type)` 用 `train_cost_*_by_type` / 建筑造价(cfg 真源);`hp_i/max_hp_i` 用 `stats.max_hp_of`。
  - `squash` = tanh 或除以常数归一,防早期价值差量纲炸掉塑形。
  - **FFA aggregate 的信度分配微妙点 `[AI-DRAFT]`**:`q≠p` 有 3 个对手,`aggregate` 取 **max(最强对手)**、
    **mean**、还是**逐对手项**?取 max ⇒ 鼓励压制当前领先者(FFA 合理);取 mean ⇒ 平滑。**倾向 max**,但确切
    形式列开放问题 §8-3。
- **退火**:塑形系数 `β` 随训练线性/指数**退火到 0**——虽然 PBRS 理论上最优不变、无需退火,但**有限训练预算**下
  Φ 选得不完美仍会误导中间学习(ng1999 局限);退火让末期纯粹优化胜负。同时保留 §4.1 稀疏项**恒在**做最终裁判。

### 4.3 反 reward-hacking(必须显式防)

- **警示(pan2022rewardhacking)**:agent 越强越会钻奖励空子,且存在**相变**(能力过阈值后真实回报骤降)。对本
  项目:任何**非 PBRS 的中间奖励**(「+1/采矿」「+1/造兵」)都会被 hack 成「只种田不打架」或「刷特定动作」。
- **三条铁律 `[AI-DRAFT]`**:
  1. 中间奖励**只走 PBRS**(§4.2 电报级抵消,不可刷);**禁止**直接累加可刷的计数项。
  2. **稀疏胜负恒为最终裁判**(§4.1 永不退火)。
  3. **训练中监控 hacking 早期信号**:「塑形/代理奖励升,但对固定脚本对手的胜率不升」⇒ 疑似 hack,即查。
- **重要不确定性 `[AI-DRAFT]`**:ng1999 的最优不变性是**单智能体**结论;推广到 4 人 FFA(马尔可夫博弈)保的是
  否是均衡不变,**本文未检索到直接支撑多智能体 PBRS 的入库文献**(不臆造引用)。缓解:退火 + 稀疏终局兜底,并把
  「FFA 下 PBRS 的理论保证」列为需补文献的开放问题(§8)。
- AMP/GAIL 的**判别器 style 奖励**(§1.3)是可选的第三类塑形:「打得像高质量脚本」,与 task 奖励相加用 PPO 训;
  比 BC 复杂且不稳(GAN 通病,ho2016gail),优先级低于「BC 暖启动 + PBRS」,列进阶备选。

---

## §5 策略/价值结构

- **actor:实体共享(按类型共享参数)**。§2.4 方案 A/B——同类型实体走同一套参数,type 作输入或 type-indexed head。
  与 issue.md v2+ 封存方案「实体共享策略网」一致;MAPPO(yu2021mappo)的「共享 actor、只看局部 obs」范式。
- **critic:IPPO 独立价值起步**(dewitt2020ippo)。每个座位从自己的 egocentric obs 估 `V_p`(参数与其他座位共享,
  输入不同)。dewitt2020ippo 的经验:独立 PPO 在 SMAC 常与联合学习相当甚至更好、几乎不调参 ⇒ **先用最简「共享
  策略 + 每-player 独立价值」起步,把中心 critic 当可选增量,而非前置工程**(降 v2.0 骨架复杂度)。
- **MAPPO 中心 critic 作可选增量**(yu2021mappo)。中心 critic 看**全局(非 egocentric)状态**估值,缓解多 agent
  非平稳。**FFA 关键修正**:yu2021mappo 是**合作**设定(共享团队回报);FFA 是竞争,「中心 critic」须理解为
  **「每个 player 一个中心 critic 看全局、估该 player 自己的回报」**,而非合作共享 critic。执行时(CTDE):训练用
  全局态、执行只用局部——本项目 v1 无迷雾(全图可见),CTDE 的「分散执行」约束更松,中心 critic 更易接。
- **判断 `[AI-DRAFT]`**:起步 IPPO;若观测到价值估计方差大/非平稳明显再升 MAPPO 中心 critic。critic 的 FFA 精确
  定义(名次回报 vs 价值差回报,是否看对手内部态)列开放问题 §8-2。

---

## §6 对手课程 + 自对弈

### 6.1 课程:v1.9 分档脚本指挥官(易→难)

- **对手池**:v1.8 造、v1.9 筛的 **10 风格指挥官**(profile.py):`balanced`(=`scripted` 别名)、`boomer`(种田)、
  `rusher`(爆狗偷家)、`turtle`(龟缩防守)、`timing`(一波流)、`harasser`(骚扰)、`airtech`(空军科技)、
  `tempo`(中期强攻)、`counter`(随机应变/最佳响应)、`chaos`(概率混沌)。接口 `controller(state,key)->actions[N]`
  与 RL 策略**同签名**(环境零改动)。
- **课程曲线(narvekar2020curriculum)**:`random`(最弱地板)→ 弱档 → 中档 → 强档;**按胜率阈值触发升档**(如对
  当前档滑窗胜率 >0.7 才放行下一档),而非固定时间表。座位编排:起步 **1 个 RL 座 vs 3 个同档脚本**;逐步混入更
  多 RL 座(过渡到 §6.2 自对弈)。
- **难度排序 = 占位符(必须由数据填)**:确切分档来自 v1.9 round-robin
  `[source: 20260727-v19-roundrobin]`——该 run 已在 research-log 预注册(criteria 先写,git `a3ddadf`),
  但**结果尚未落盘**(撰文时 `experiments/20260727-v19-roundrobin/` 目录尚不存在,待 result-analyst 读
  `{summary.md,games.jsonl}` 填)。**本文不从该 run 引用任何数字**,只把它标为待填来源。
  - **临时假设 `[AI-DRAFT]`(待 v1.9 数据推翻/确认,勿引用为定论)**:从 v18 P3/P4 初评
    `[source: 20260727-v18-eval-p3]` `[source: 20260727-v18-p4-gate]` 看,`rusher` 偏强(round-robin 常胜、靠消灭
    @~1080–1869 拍取胜)、`boomer`/`turtle` 偏被动(靠门 @~4182 兜底决胜)。**但「对脚本互相的胜率」≠「对 RL 的
    训练难度」**,二者未必同序——课程难度须以「RL 对该档的实际胜率曲线」重新标定,不能照搬 round-robin 名次。
  - **占位表(v1.9 出数后填)**:

    | 难度档 | 指挥官(待填) | vs random 胜率 | round-robin 胜率 | 备注 |
    |---|---|---|---|---|
    | L0 地板 | random | — | — | 冷启动地板 |
    | L1 易 | _待 v1.9 填_ | | | |
    | L2 中 | _待 v1.9 填_ | | | |
    | L3 难 | _待 v1.9 填_ | | | |

### 6.2 自对弈(**后置层**,非起步)

- **原则(必须遵守,否则遗忘/循环)**:
  - heinrich2016nfsp 教训:自对弈要打**历史平均**而非**最新的自己**——即使不实现 NFSP 双网络,也要保留一个
    **历史对手快照池**、从中采样,而非只和最新 checkpoint 对打。
  - lanctot2017psro:种群防过拟合/坍缩的理论骨架;本项目取**简化版**——v1.9 脚本 = 固定初始种群,meta-strategy
    用「按对当前 RL 胜率加权」(≈PFSP),**先不做**「每轮训新 oracle 入池」的完整 PSRO 循环(单卡偏重)。
  - vinyals2019alphastar 的 **PFSP**(按对手胜率优先采样)+ exploiter **思想**可在单卡做一个「轻量联盟」(few
    main + 脚本池 + 己方快照),而非 DeepMind 完整 league(TPU 集群×数周,单卡不可照搬——也是 berner2019 的规模警示)。
- **落地形态 `[AI-DRAFT]`**:训练中**周期性快照** RL 策略入冻结池;每局对手从 `{脚本课程档} ∪ {历史快照}` 采样,
  权重 ∝ PFSP-lite(赢得越多的对手采得越少,专攻打不过的)。这是 **v2.1+** 的事,v2.0 骨架只需把「对手 = 可插拔
  controller」这一点留好(controller.py 的 `make_joint_controller` 已支持任意 P 家 controller 组合)。

---

## §7 JAX 骨架规格(`explorations/rl_skeleton_v20.py`)

> **目标(issue.md v2.0)**:选定算法(obs/奖励/PPO 循环)用 JAX 自研到「**能空跑一步、验证正确**」,**不做真正训练**。
> 骨架落 `explorations/`(`.claude/rules/python-research.md` #6:新想法先在 explorations 验证,过了才进 src)。

### 7.1 必须包含的 6 件套

1. **`RLConfig` dataclass(先于一切)**:所有超参走它,**代码里禁止字面量**(python-research.md #2);**显式设种子**
   (#1)。字段见 §7.2。
2. **obs 编码器 `encode_obs(state, cfg, player) -> (entity_tokens[N,F], entity_mask[N], global_vec[G])`**:§2 的落地;
   派生量复用 `stats.py`;`vmap` over player(4 座位)与 over batch。
3. **奖励函数 `reward_fn(state, next_state, cfg, player) -> f32`**:§4 的落地 = 稀疏名次终局(用逐 tick HQ 存活推名次)
   + `β·(γ·Φ(s')−Φ(s))`;Φ 用投入价值差。返回 per-player。
4. **共享参数 actor-critic 小网 `ActorCritic`**(tiny,§2.4 方案 A):entity MLP trunk(F→H→H,H≈64,1–2 层)→
   mean-pool 态势 → 拼回 → **actor head → `[N,116]` logits**(加 `legality_mask` 的 -inf 掩码)+ **critic head →
   标量 `V_p`**。正交初始化 + tanh(huang2022ppodetails)。参数按实体/座位共享。
5. **vmap 批量 rollout(扩 `eval.matchup_runner`)**:eval.py 头注释已明示路径——把 `matchup_runner.one()` 里的
   `make_scan` 换成**收集轨迹的 scan body**:每 tick 除 `step` 外,emit `(obs, action, logp, value, reward, done)`;
   RL 策略占 1 座、脚本占其余 3 座(§6.1);B 环境 `vmap` over seeds。批量/决定论/终局冻结语义不变(step.py 的
   `jax.tree.map(where done)` 冻结、`make_scan` 的 `scan` 已保证)。
6. **PPO/GAE 损失 + 一次 update step**:从 `(reward, value, done)` 算 **GAE(λ)** 优势;**clipped surrogate + value
   clipping + entropy bonus**;**优势归一化**;对**真实批量 rollout 数据**跑**恰好一次** minibatch 更新(用
   `optax`),**断言 loss 有限、各张量形状对**。**到此为止——不写真正训练循环**(对齐 issue.md「不要开始训练」)。

### 7.2 config 旋钮(huang2022ppodetails 的「实现细节」,做成显式可开关)

自研 JAX PPO 的**不踩坑清单**(这批细节比「选 PPO 还是别的」更决定成败,huang2022ppodetails / 2006.05990 / 2005.12729):

| 旋钮 | 说明 |
|---|---|
| `clip_eps` | surrogate clip ε(如 0.2) |
| `gae_lambda`, `gamma` | GAE λ 与折扣 γ(长时程 γ 需高) |
| `norm_adv` | 优势归一化(minibatch 级) |
| `clip_vloss`, `vf_coef` | value function clipping + 价值损失权重 |
| `ent_coef` | 熵奖励(探索;掩码动作空间下按合法集算熵) |
| `max_grad_norm` | 全局梯度裁剪 |
| `lr`, `anneal_lr` | 学习率 + 线性退火 |
| `update_epochs`, `num_minibatches` | on-policy 数据复用 |
| `norm_obs`, `norm_reward` | obs/reward 归一化/缩放 |
| `orthogonal_init` | 正交初始化 + tanh |
| `shaping_beta`, `anneal_shaping` | §4.2 PBRS 系数与退火 |
| `num_envs(B)`, `num_steps` | rollout 批量与长度(B≈64 是引擎甜点 `[source: 20260726-v18-bench]`) |

**验收判据(骨架「正确空跑一步」)`[AI-DRAFT]`**:①`encode_obs` 形状恒定、无 data-dependent shape;②masked logits 在
非法位 = -inf 且 NOOP 恒有限;③rollout 决定论(同 seed 同轨迹,与引擎一致);④GAE/loss 全有限、无 NaN;⑤一次
update 后参数确实更新、形状不变;⑥`ruff` + 一个 `pytest` smoke 绿。

---

## §8 开放问题(v2.1 训练前须定/须测)

1. **obs F 精确 spec**:§2.2 的确切维度、etype 用 embedding 还是 one-hot、是否几何旋转到规范帧利用四重对称、怪物
   token 摘要 vs 全量。**建议**:先 embedding + 摘要跑通,再逐项消融。
2. **FFA critic / 名次奖励精确定义**:名次奖励靠逐 tick HQ 存活推淘汰序(需 rollout 记录)——确认口径;critic 是
   IPPO 独立还是 MAPPO 每-player 中心;回报是「名次」还是「价值差积分」。
3. **Φ 精确形式**:投入价值的 `cost` 权重、`hp` 折算、`aggregate_{q≠p}` 取 max/mean/逐对手;squash 与量纲;`β`
   退火曲线。**未定即可能 hack**(pan2022)。
4. **多智能体 PBRS 的理论保证**:ng1999 是单 agent;**须补检索** FFA/马尔可夫博弈下 PBRS 是否保均衡的工作(本文
   未检索到入库支撑,不臆造)。在补上前,靠退火 + 稀疏终局兜底。
5. **课程难度阈值 + 分档**:占位表(§6.1)须由 `20260727-v19-roundrobin` 填;升档胜率阈值、座位混合比例、是否
   `random` 起步都要试。**「脚本互相胜率 ≠ 对 RL 难度」**,须以 RL 实际胜率曲线重标定。
6. **BC 暖启动 vs DAgger**:先纯 BC(单次蒸馏)够不够,还是必须 DAgger 迭代消漂移(ross2011dagger);用哪个/哪些
   脚本当 teacher(单个最强 vs 混合多风格)。
7. **自对弈池机制**:快照频率、PFSP-lite 权重公式、池容量、何时从「纯课程」切到「课程+自对弈」;是否需要 exploiter。
8. **吞吐 → games/day(硬约束 #1:必测不可估)**:§1.2 的 4025.8 env-tick/s **不含 gate、不含策略网前向/反向**;
   v2.1 须用**含 gate + 网在环**的真实 rollout **实测** env-tick/s 与 games/day,再定 B、rollout 长度与训练预算。
   **本文不给该数字**。

---

## 附:引用(citekey → 见 `lit/literature-log.md`,DOI/arXiv 齐全)

schulman2017ppo(PPO)· lu2022purejaxrl(JAX PPO 模板)· rutherford2023jaxmarl(JAX MARL/IPPO/MAPPO)·
yu2021mappo(MAPPO/中心 critic)· dewitt2020ippo(IPPO)· huang2022ppodetails(PPO 实现细节三件套)·
vinyals2019alphastar(SL 暖启动 + league/PFSP)· lanctot2017psro(种群/PSRO)· heinrich2016nfsp(历史平均自对弈)·
berner2019openaifive(PPO+自对弈就够用 + 规模警示)· peng2021amp(AMP≈GAIL,离散 RTS 不适配)· ho2016gail(对抗式模仿)·
ross2011dagger(脚本=oracle,DAgger 消漂移)· ng1999shaping(PBRS 保最优)· pan2022rewardhacking(reward hacking 警示)·
narvekar2020curriculum(课程易→难)· espeholt2018impala(V-trace,单卡不必)· hessel2021podracer(Anakin 全上 GPU)·
freeman2021brax(JAX PPO 备查)。
