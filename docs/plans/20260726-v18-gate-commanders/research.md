# v1.8 调研:异界之门 + 多风格指挥官(引擎事实参考)

调研方式:4 个并行 Explore subagent(控制器/动作空间、胜负·config、基建·文档、
RL·批量就绪度)+ 主 context 亲读 step.py/combat.py/controller.py + plan-critic 对抗核对。
所有断言带 `文件:行号`。HEAD `de0cfd2`(v1.7)。

## 1. 相关文件位置

- **tick 顺序唯一权威**:`step.py:1-25` 头注释。9 阶段:production→special_tasks→
  construction→harvest→apply_orders→paid_orders→start_constructions→movement→combat→
  cleanup_deaths→`_end_tick`(`step.py:70-85`)。终局冻结 `step.py:83-85`(done 后恒等)。
- **胜负判定**:全在 `_end_tick`(`step.py:50-62`)。`winner` 编码:-1 进行中/0..P-1 胜者/
  P 和局(`state.py:85`)。两处判和:`step.py:59`(n_alive==0 同帧互灭)、`step.py:60`
  (tick>=episode_len 超时)。`episode_len` 默认 6000(`config.py:100`)。
- **控制器**:`controller.py`(449 行)。接口 `fn(state,key)->actions[N=256]`;实现
  `random_actions`(:43)、`scripted_actions`(:54-421,单一宏观 AI,`del key` 决定论);
  注册点 `make_controller`(:424-433,现仅 random/scripted);合并 `make_joint_controller`
  (:436)+`merge_actions`(:37,按 owner 行块取各家输出)。
- **动作空间**:`actions.py`。116 个整数动作(0-115,参数化于 cfg);`A_NOOP/A_ATTACK`
  等常量+`a_*()` 构造器;`legality_mask`(:265-539);非法→NOOP(:542-550)。**A_ATTACK
  是无参标量 id(:91)**。
- **config**:`config.py`(605 行,单个 flat frozen dataclass,~200 标量字段+@property 派生
  32 型表)。AI 旋钮 `ai_worker_target/ai_attack_threshold/ai_base_level_target/
  ai_upgrade_reserve`(:376-382)。无预设,变体走 `dataclasses.replace(Config(),**over)`。
- **战斗**:`combat.py`。`combat_tick`(:53-194):owner 别敌(`:75-77`
  `owner[:,None]!=owner[None,:]`)、`incoming[N]` 累加(`:92`)、`hp=clip(hp-incoming+heal,
  0,maxhp)`(`:192`)。`cleanup_deaths`(:197-295):`hq_dead[owner]` 淘汰清场(`:210-212`)。
- **移动**:`movement.py`。距离场松弛 `_relax_fields`(:39-58,scan length=k_iters=h+w);
  attack-move 目标**硬编码最近敌 HQ**(`:116-123` argmin(hq_dvals));梯度采样 `BIG_DIST`
  截断(`:209` minimum(dyn,4*(h+w)));per-entity vmap 采样(:215-218)。
- **state**:`state.py`。`WorldState`(NamedTuple,:36-85,定容 pytree,owner 隐式行块);
  `init_state`(:126-161,不吃 key);`hq_slot`(:99);`cell_of`(:88)。
- **stats**:`max_hp_of`(:124)/`atk_of`(:131)/`type_tables`(:143)/`physical_damage`(:137)
  ——均按 [N] 表 etype×level 索引(**怪物需自带 hp/atk,不能复用**)。
- **地图**:`map.py`。`build_map` **硬要求 n_players==4**(:75-76);`dist_fields[Nn+P,H,W]`
  (:40,到每 node/HQ 的 BFS 场,静态闭包);中心=网格中点。
- **入口/评测**:`run.py` `cmd_play`(:109-161,host 循环 `for t in range(episode_len)`:129)、
  `cmd_bench`(:169-181 单环境)、`write_provenance`(:87-102 六件套)、`state_to_numpy`
  (:105)。`make_scan`(step.py:90-106,固定 length scan)。**无**批量 vmap harness、**无**
  锦标赛/胜率、**无** obs/reward 支架。
- **评测参考**:`explorations/exp_v17_duel.py`(无菌单位对决,非整局;provenance/summary
  模式可复用)。

## 2. 数据如何流动

- 控制器每 tick 出 `actions[N]`(只填自家槽,别家 NOOP)→ `apply_orders` 掩非法为 NOOP、
  改写 `state.order` → 各 step 阶段按 standing order 驱动移动/采集/生产/战斗(NOOP≠idle)。
- 决定论 = f(init_state, 指令序列, key 序列);key 每 tick split 两处仲裁(抢格/抢点)。
  **无 numpy 全局随机**(全 `jax.random.*`)。
- 引擎 vmap-ready:定容 pytree+分支无跳转+无 host callback+无 data-dependent shape;
  `jax.vmap(step,in_axes=(0,0,0))` 直接可批(共享 map 免费)。终局冻结让 ragged 长度批
  scan 正确。

## 3. 关键设计判断(经 plan-critic 核对)

- **异界之门 Approach A**(独立怪物子表 `monster_*[P,Mmax]`):引擎层核对通过——怪自带
  hp/atk、用现成静态 HQ 场、不碰撞、清怪折进 cleanup_deaths(`combat.py:210-212` hq_dead)。
  阵营隔离＝只在 owner==p 维度结算。
- **怪物战斗＝独立 `monster_combat_tick` 阶段**(combat 与 cleanup 间):**不能**并入
  combat_tick 的 incoming(`combat.py:192` 已消费);从同一 pre-pass 快照算两侧、一次 apply。
- **胜负最小改动**:保留 `episode_len` 为硬帽/scan 边界(零改 run.py:129 + make_scan + 7 处
  引用),新增 `gate_open_tick`(<episode_len)触发门开;删 `step.py:60` 超时和局;overtime
  在既有边界内跑到 `n_alive==1→argmax`。测试 blast radius ≥5:改 `test_combat_win.py:70`、
  `test_elimination.py:50/58`(draw 已废),核 `test_scripted_v13/14/16`、`test_scripted_upgrades`。
- **FFA attack_tgt**(P2b,可延后):A_ATTACK 无参、目标硬编码最近敌 HQ(movement.py:116-123
  会覆盖 target_cell)→ 选敌须加 `attack_tgt int8[N]` state 字段+改热路径。核心 roster 先用
  默认最近敌 HQ。
- **吞吐**(头号风险):单环境 64×64/4p 未测,v1.3 @24×24 仅 57-66 tick/s,64×64 距离场
  估 ~43×→ 恐极慢;v1.9/v2.0 全靠 vmap+GPU(4090/24GB)。`Mmax` 是批量显存乘子
  (`[B,P,Mmax,N]` B256/M64≈2.1GB;`[B,36,64,64]` 场 buffer≈1.5GB/个主导)。

## 4. 既有模式与约束

- 平衡数字只进 config.py;怪物/门数值走新 Config 字段,初值 [AI-DRAFT] 模拟校准。
- 定容数组、死=翻掩码、生=scatter 进空槽,绝不 resize;无 data-dependent 形状。
- 新想法先 explorations/ 验证再进 src/;实验 `JAX_PLATFORMS=cpu .venv/bin/python`,批量才 GPU。
- 指挥官须 branchless-JAX(进 scan/vmap);随机性用线程 key(现 scripted `del key`,新指挥官可用)。
- scripted×scripted 对 seed 不敏感、镜像先手偏置(research-log 已记)→ 评测须多风格非对称+
  多座位排列。
