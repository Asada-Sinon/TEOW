# v1.8 — 实施计划(异界之门 + 多风格指挥官)

plan-critic 状态:主计划已过一轮,无 BLOCKER 残留(B-1 rollout 边界、M-2 monster_combat
阶段、M-1 attack_tgt + 3 MINOR 已吸收)。**阶段权威＝主计划**
`~/.claude/plans/v1-8-v2-0-plan-plan-hook-fluffy-matsumoto.md` 的 v1.8 节;本文只补数值草案、
关键设计决策清单与规格解释,不重复阶段正文。引擎事实见同目录 `research.md`。

## 关键设计决策(均记 DECISIONS [AI-DRAFT];用户在线拍板的三项不带 [AI-DRAFT])

- **D0 异界之门＝Approach A 独立怪物子表**(不动 owner-by-row 主表)。
- **D1 怪物战斗独立 `monster_combat_tick` 阶段**(combat 与 cleanup 间),同一 pre-pass 快照
  算两侧、一次 apply(玩家侧走各自 incoming/clip、怪侧 `monster_hp=clip(hp-inc,0,∞)`)。
- **D2 胜负最小改动**:`episode_len` 留作硬帽/scan 边界(零改 run.py/make_scan),新增
  `gate_open_tick`(<episode_len)触发门开,删超时和局。
- **D3 怪物不参与单位互推**(沿目标玩家 HQ 场慢速 descent,`BIG_DIST` 截断)。
- **D4 指挥官策略参数＝代码常量 StrategyProfile**(非 Config);共享宏观抽 `commanders/macro.py`。
- **D5 FFA attack_tgt** 独立 P2b、可延后;核心 roster 先用默认最近敌 HQ。

## 数值草案([AI-DRAFT],P4 用 `exp_v18_gate_tune.py` 校准,落 Config)

| 字段 | 初值 | 说明 |
|---|---|---|
| `gate_open_tick` | 4000 | 门开时刻(< episode_len);旧「超时」语义迁移到此 |
| `episode_len`(硬帽) | 6000→**7000** | scan/host 边界;给 ~3000 overtime 余量,不够再调 |
| `monster_spawn_interval` | 50 | 每 50 拍一波 |
| `monster_wave_count` | 2 | 每玩家每波怪数 |
| `Mmax` | 64 | 每玩家怪容量(批量显存旋钮) |
| `monster_hp_base` | 50 | HP 基(生成时 = base + slope×overtime,**无上限**) |
| `monster_hp_slope` | 0.5 | HP 每 overtime 拍增量(线性;P4 调此控收敛速度) |
| `monster_atk_base` | 5 | 攻击基 |
| `monster_atk_slope` | 0.005 | 攻击增速(小) |
| `monster_atk_cap` | 20 | **攻击上限** |
| `monster_speed` | 0.15 | 慢速(格/tick;P4 对齐单位速度表校准) |
| `monster_melee_range` | 1.5 | 近战判定半径(欧氏,对齐近战单位) |

**P4 可测目标**:门开后 overtime 中位 ≤ ~1500 拍且 < (episode_len−gate_open_tick);防御型
不必然赢、进攻型不必然输;四家压力对称(同参数生成可断言)。

## 规格解释(issue.md v1.8 → 实现口径)

- 「场地中央」= 网格中点(map.py 六边形中心);四家怪从中心向各自 HQ 场下降(阵营隔离,
  互不碰撞穿插)。
- 「强度生成时定死」= 每只怪 spawn 时把 hp/atk 写死进 `monster_hp/monster_atk`,之后不随
  overtime 变(只有新 spawn 的更强)。
- 「某阵营死则其怪离场」= cleanup_deaths 里 `hq_dead[p]` → `monster_alive[p,:]=False`。
- 「近战」= 只在 `monster_melee_range` 内结算;怪只打 owner==p 的存活实体(单位+HQ+建筑),
  p 的可攻击实体打相邻怪(复用 can-target 的「可打单位」维?否——怪不是实体表项,单独判:
  p 侧凡 atk>0 且在 range 内即可击怪,不分对空/对地——怪是地面近战靶,简单处理,记 DECISIONS)。

## 指挥官 roster(P3,先多造后筛;详见主计划)

rusher / boomer / turtle / timing / harasser / airtech / tempo / counter /(chaos)。每个:
branchless-JAX、上帝视角条件判据、**自适应回退**(主战术失败转型)、注册 make_controller、
冒烟 vs random 必碾压。
