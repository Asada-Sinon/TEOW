# JAX 游戏/RTS/Gridworld 先例调研 — TEOW 引擎设计参考

日期:2026-07-25。调研目标:为「纯 JAX(jit/vmap/lax.scan,全静态形状)2D 网格 RTS 引擎」提炼先例中的静态形状惯用法与坑。参考骨架:本机 `/home/michael/workspace/pi05/temp/alicization/underworld/`(NamedTuple pytree + `build_step(cfg, terrain)` 闭包 + `lax.scan`)。

---

## 0. 先例一览

| 项目 | 形态 | 与 TEOW 的相关性 | 出处 |
|---|---|---|---|
| JaxMARL **SMAX** | 连续 2D 星际微操(非网格) | 攻击目标选择、scatter-add 伤害结算、死亡掩码、动作掩码 | https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax |
| **Craftax / Craftax-Classic** | 单人网格采集/合成/战斗 | 网格状态表示、定容 mob 表+掩码、占用图、朝向交互、spawn 入槽 | https://github.com/MichaelTMatthews/Craftax |
| **MA-Craftax** | 多人 Craftax | 多玩家同格问题的存在性证明+吞吐数据 | https://arxiv.org/abs/2511.04904 、https://github.com/BaselOmari/MA-Craftax |
| Jumanji **RobotWarehouse / Cleaner** | 多机器人网格 | 碰撞的「另一种解法」(撞了就终局,不仲裁) | https://github.com/instadeepai/jumanji |
| **gigastep** | 3D 连续多智能体 | 吞吐上限参考(10^9 SPS) | https://github.com/mlech26l/gigastep |
| **Parabellum** | SMAX 魔改的战争模拟(地形/爆炸半径) | 证明 SMAX 骨架可扩展成"更像 RTS"的东西 | https://github.com/syrkis/parabellum |
| **gymnax** | 环境基类 | autoreset 标准写法 | https://github.com/RobertTLange/gymnax |

要点先说:**没有找到现成的"完整 RTS(经济+建造+战斗)纯 JAX 实现"**——SMAX 只有战斗微操,Craftax 只有单人采集,TEOW 的"工人经济循环 + 建造训练 + 战斗"组合是要自己拼的,但每个子件都有可直接照抄的先例模式。

---

## 1. 逐格移动 + 碰撞仲裁

### 先例做法

- **SMAX**(`smax_env.py`):连续坐标,不是网格。移动 = `pos + move_vec * velocity * dt`,越界 clamp;重叠用 `_push_units_away` 软碰撞(两两距离矩阵 → `relu(radius/dist - 1)` 重叠量 → 位置互推),死单位不动(`jnp.where(alive, new_pos, pos)`)。**对网格 RTS 不直接适用**,但"提议→修正"的两段式值得借鉴。
- **Craftax**(`craftax_classic/game_logic.py`):网格 + 布尔占用图 `state.mob_map`。mob 移动用 **`lax.scan` 串行逐 mob 更新**:每个 mob 依次检查 `valid = in_bounds & ~in_wall & ~in_mob & ~in_lava`,合法则清旧格、写新格。串行处理天然消灭同格竞态——先处理的赢。注意 Craftax 里 mob 移动**全是 scan、没用 vmap**,靠 vmap 大量 env 找回并行度。
- **Jumanji RobotWarehouse/Cleaner**:也是 `lax.scan` 逐 agent 顺序更新,但碰撞不仲裁——检测到碰撞**直接终局**(`done = collision | horizon`)。这是 benchmark 环境的偷懒解,RTS 不可用。

### `.at[]` scatter 的竞态语义(核心事实)

- `arr.at[idx].add(v)`:重复索引**确定性累加**——伤害结算、资源入库放心用。
- `arr.at[idx].set(v)`:重复索引**结果未定义**(XLA 不保证哪个写入生效)——占用图绝不能用裸 `set` 让多个单位抢同一格。
- `arr.at[idx].min(v)` / `.max(v)`:重复索引下是确定性的 reduce——**这是同格仲裁的正解**。

### TEOW 推荐:并行提议 + scatter-min 仲裁(单 pass,全向量化)

```
# 每 tick,对全部 2*E_max 个单位(死的提议自己原地):
prop_cell = cell_index(pos + dir)                    # [N] 提议目标格(非法/不动则=当前格)
prio      = random_permutation(key, N)               # 每 tick 重洗的优先级,防低下标恒赢
occ_now   = full(n_cells, INF).at[cur_cell].min(prio) # tick 开始时的占用(静态阻挡另算)
winner    = full(n_cells, INF).at[prop_cell].min(prio)
can_move  = (winner[prop_cell] == prio)              # 赢得目标格
          & (occ_now_by_others[prop_cell] == INF)    # 目标格 tick 初无人(保守规则)
          & passable[prop_cell] & alive
new_pos   = jnp.where(can_move[:, None], prop_pos, pos)
```

- 保守规则「目标格 tick 初必须为空」自动杜绝 A↔B 互换穿越和"跟车链移动"的复杂性;代价是队列前进每格慢一拍,RTS 完全可接受。
- **公平性坑(self-play 特有)**:若平局恒按单位下标裁决,玩家 0 的单位永远赢抢格,PPO self-play 会学出对称性伪差异。每 tick 用 `jax.random.permutation` 生成优先级(或至少在两玩家间交替偏置)。
- 备选:照抄 Craftax 的 `lax.scan` 串行方案。正确性最好写(可支持链式跟进),N=128 的 scan 串行开销在 vmap 数千 env 下通常无所谓;若 profiling 后发现 scan 是瓶颈再换 scatter-min。**建议 v1 直接上 scatter-min**,它和 PPO 大 batch rollout 的亲和性更好(无 128 步串行依赖链)。

---

## 2. 寻路

### 先例结论:没人做真寻路

- SMAX:连续空间直线走+互推,无寻路。
- Craftax:mob 对玩家**贪心一步**(近了朝玩家方向挪,被挡就停/随机),玩家由策略网络自己"学会寻路"。
- Jumanji 路由类环境:把寻路本身留给 RL 策略。
- Parabellum 加了地形/障碍,但同样无经典寻路。

### TEOW 推荐:预计算 BFS 距离场 + 贪心下降(强烈推荐,成本极低)

v1 的"朝目标贪心一步(可被挡)"**能用但有硬伤**:凹形障碍(建筑群、矿区拐角)会让工人卡死,采集循环是 TEOW 的经济主轴,卡死直接毁掉学习信号。而 TEOW 恰好满足距离场的全部前提:

- 地图静态、目标点集固定且少(各矿点 + 各 HQ,设 G 个);
- 在 **jit 外用 numpy 做 G 次 BFS**,得 `dist_fields: int16[G, H, W]`,连同地图一起**闭包进 build_step**(与 underworld 把 terrain 闭包进 step 完全同一惯用法,不进 scan carry);
- 每 tick 每单位:取自己目标的场 `f = dist_fields[goal_idx[e]]`,对 5 个候选格(4 邻+原地)算 `f[cand] + BIG*blocked[cand]`,**masked argmin** 挑一格作为"提议方向",再进第 1 节的仲裁。被挡时自动选次优邻格,天然绕行。

这是经典 RTS 的 flow-field 寻路,单步 O(1),对静态障碍是精确最短路;动态阻挡(别的单位)靠 argmin 的次优候选 + 抢格失败原地等下 tick 消化。建筑若可在运行时新建而改变通行性:v1 可规定建筑只占非走廊格/不重算场(近似),v2 若需要再做增量方案。**结论:v1 就上距离场,不要用裸贪心。**

---

## 3. 攻击目标选择 + 同时伤害结算(照抄 SMAX)

SMAX 的完整链条(`smax_env.py`,函数名可对照):

1. **动作空间**:离散动作 = 移动 0..4 + 攻击 5..5+n_enemies(每个敌方槽位一个动作)。`get_avail_actions` 给出掩码:`shootable = in_range & attacker_alive & target_alive`。
2. **自动选目标**(TEOW 的"相邻自动战斗"更该用这条,SMAX 连续动作模式里就有):
   ```
   dist   = |pos[i] - enemy_pos|                 # [E, E_enemy]
   score  = dist + 1e8 * (~enemy_alive | ~adjacent | ~self_alive)
   target = argmin(score, axis=-1)               # masked argmin
   has_t  = any(score < 1e8, axis=-1)            # 关键:全无效时 argmin 返回 0,必须用 has_t 门控!
   ```
3. **伤害结算**:所有攻击并行算出 `(attacked_idx, -damage)`,一次 `scatter_add` 进血量(SMAX 用显式 `jax.lax.scatter_add` + `ScatterDimensionNumbers`;等价且更好写的是 `health.at[target].add(-dmg * has_t * alive)`,重复目标确定性累加),再 `maximum(health, 0)`。多打一零竞态。
4. **死亡处理**:不搬运不压缩,纯掩码——`alive = health > 0`;死单位观测置空(`where(alive, feat, 0)`)、动作掩到只剩 no-op、位置冻结。**同 tick 互殴允许同归于尽**(先结算全部伤害再更新 alive),规则简单且无顺序偏置,建议 TEOW 采用同款。

Craftax 侧的玩家打 mob 是 `health.at[index].add(-damage * is_attacking)`,同一个模式。

---

## 4. 采集折返状态机(纯 jnp 组织)

先例中最接近的是 Craftax 的意图逻辑(mob 的"追/逃/游荡")和 underworld 的相位标量,但都没有显式多相位状态机。推荐组织:

```
phase: int8[E]   # 0=TO_MINE, 1=MINING, 2=TO_HQ (仅工人有意义,其他单位恒0)
cargo: int16[E]; mine_slot: int8[E]  # 绑定的矿点id → 决定用哪张距离场

# 每 tick(顺序:先判定转移,再按新相位行动):
at_mine  = dist_fields[mine_of[e]][cell] == 0        # 或 <=1 表示邻格进驻
full_    = cargo >= CAP
at_hq    = dist_fields[hq_of[owner]][cell] == 0
phase = where(phase==0 & at_mine,          1, phase)
phase = where(phase==1 & full_,            2, phase)
phase = where(phase==2 & at_hq,            0, phase)   # 卸货后折返
cargo    = where(phase==1, cargo + rate, cargo)
cargo    = where(phase==2 & at_hq, 0, cargo)           # 卸货
stock    = stock.at[owner].add(cargo * (phase==2 & at_hq))   # 入库 scatter-add
goal_idx = where(phase==2, hq_goal[owner], mine_goal[mine_of])  # 喂给第2节的距离场
moving   = (phase != 1)                                # MINING 时不参与移动/占格可选
```

要点:相位少(≤4)就用 `jnp.where` 链,**不要上 `lax.switch`**(switch 按 batch 元素不能向量化分派,per-entity 异构分支本来就得全算再选,where 链最直白);转移条件互斥时 where 链的先后即优先级,写成"后写的覆盖先写的"要固定顺序并加注释。进驻开采若要单位从地图上"消失",用 `in_building: bool[E]` 掩码把它移出占用图与战斗判定,而不是改坐标——坐标保留,出来时原位吐出(Craftax 熄灭 mob 也是只翻 mask 不动数组)。

建筑建造/训练计时同构:`build_timer: int16[B]`,每 tick 减 1,`done = timer==0` 时产出单位——**入槽用 Craftax 的 spawn 惯用法**:`slot = argmax(~alive_mask)` 找第一个空槽,`can_spawn = ~all(alive_mask)` 门控,`tree_map + where` 写入该槽。E_max 满了就是训练排队(这就是定容表的天然人口上限,和真 RTS 的 supply cap 语义一致,是特性不是 bug)。

---

## 5. JAX 通用惯用法与坑(汇总)

1. **masked argmin/argmax**:`argmin(x + BIG * ~mask)`;**坑**:全 False 掩码时返回下标 0,必须配 `valid = any(mask)` 门控后续所有写入。
2. **scatter 语义**:add/min/max 对重复索引确定;set 未定义。伤害、卸货、占格仲裁全走前者。
3. **死槽位要"停泊"在无害值**:位置合法、目标下标 0、血量 0;所有算式对死槽照算不误,最后 `where(alive, …)` 挑。别让死槽产生 NaN/越界索引(clamp 住)。
4. **autoreset 标准写法**(gymnax `environment.py`,与 underworld 无关但 v2 PPO 必用):
   ```
   obs_st, state_st = step_env(...); obs_re, state_re = reset_env(...)
   state = jax.tree.map(lambda re, st: lax.select(done, re, st), state_re, state_st)
   obs   = lax.select(done, obs_re, obs_st)
   ```
   两支都无条件算;TEOW 地图静态,reset 极便宜,开销可忽略。若 init state 是常量,直接 `where(done, init_leaf, leaf)`,连 reset_env 都省了。
5. **静态地图/距离场闭包进 build_step,不进 scan carry**(underworld `step.py` 注释原话:避免每步拷贝 [n_cells] 字段)。同理,能从 state 便宜重算的派生量不要缓存进 carry(underworld `size_of` 的注释)。
6. **per-entity 向量化**:一等公民是"[N] 数组上的整列运算",两两交互用 [N,N] 广播(N=128 时 16K 项,完全无压力,SMAX 就是全对距离矩阵);只有像"逐 mob 串行占格"这种有写序依赖的才 scan。
7. **`jnp.where` 双分支都会执行**:分支里别放会产生 NaN/越界的裸算(先 clamp 再 where)。
8. **观测/动作掩码要和引擎一起长**:SMAX 的 `get_avail_actions` 从第一天就有;v2 PPO 的 invalid-action-masking 依赖它,v1 就该把「该单位此 tick 的合法动作掩码」做成引擎输出。
9. dtype 纪律:计数/坐标用 int16/int32,掩码 bool,血量资源 int32 或 f32;vmap 数千 env 时省显存的是这些小 dtype。

---

## 6. 性能量级(有公开数据的部分)

- **Craftax**:比 Python 原版 Crafter 快约 **250x**;PPO **10 亿步交互单 GPU 1 小时内**跑完(论文摘要,https://arxiv.org/abs/2402.16801 )。
- **MA-Craftax**:IPPO 4 agent,单张 L40S,2.5 亿 env steps 用时 57 分钟 ≈ **7.3–8 万 env-steps/s(含训练)**(https://arxiv.org/abs/2511.04904 )。
- **JaxMARL/SMAX**:整套训练管线相对 CPU 基线**至多 12500x**;vmap 同步 rollout 数千 env(https://arxiv.org/abs/2311.10090 )。
- **gigastep**:消费级 GPU 上 **10^9 steps/s**(动力学极简、无网格逻辑,视为吞吐天花板参考,https://github.com/mlech26l/gigastep )。
- 单 env(不 vmap)裸 step:各家论文均不单独报告;经验量级是 jit 后 10^4–10^5 SPS/env,vmap 数千 env 后合计 10^7–10^8 SPS(**此行无直接公开数据,是从上述 batch 数据反推的量级判断**)。TEOW 复杂度介于 Craftax 与 SMAX 之间,vmap 后 10^6–10^7 合计 SPS 是合理预期。

---

## 7. TEOW 计划 vs 先例:差异、可抄清单、坑清单

**差异点**(即需要自己发明的部分):
- 完整 RTS 闭环(经济→建造→训练→战斗)无先例,是 SMAX(战斗)+ Craftax(采集/占用图)+ 自研状态机(第 4 节)的拼装。
- 两玩家对称 self-play 在同一 state 里(SMAX 是 ally/enemy 但 enemy 常挂脚本;MA-Craftax 是合作)。用 `owner: int8[N]` 一张表装两家 + per-player 聚合(`stock.at[owner].add(...)`),不要开两套数组。
- 定容 SoA+掩码+闭包静态地图+lax.scan:与 Craftax/underworld 完全同构,**方案本身无需修正**。

**直接照抄**:SMAX 的 masked-argmin 选目标 + scatter-add 同时结算 + 先结算后判死(§3);Craftax 的 `mob_map` 占用图、`argmax(~mask)` 入槽 spawn、`valid & where` 移动合法性链(§1/§4);gymnax autoreset(§5.4);underworld 的 build_step 闭包与 carry 瘦身纪律(§5.5)。

**要避开的坑**(先例踩过或结构上必踩):
1. 占用图用 `.at[].set` 抢格 → 未定义行为;用 scatter-min 仲裁或 scan 串行(§1)。
2. 平局仲裁的下标偏置 → self-play 学出假不对称;每 tick 随机优先级(§1)。
3. 裸贪心寻路在凹障碍卡死工人 → 经济信号损毁;上预计算距离场(§2)。
4. 全无效掩码下 argmin 返回 0 → 幽灵攻击/幽灵移动;一切 masked-arg 后必须门控(§5.1)。
5. 距离场/地图放进 scan carry → 每步白拷贝;闭包(§5.5)。
6. 到处用 `lax.cond/switch` 想"省计算" → batch 下两支都算还多一层开销;小分支一律 where(§5.7)。
7. Jumanji 式"碰撞即终局"别学——那是 benchmark 语义,不是引擎语义。

**v2 前瞻**:动作空间照 SMAX 做成 per-unit 离散(移动+攻击槽位+工作指令),avail_actions 从 v1 就输出;autoreset 进 env 基类;PPO 管线可参考 purejaxrl(https://github.com/luchris429/purejaxrl )与 JaxMARL 的 IPPO。

---

### 主要出处
- SMAX 源码:https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/smax/smax_env.py
- Craftax 源码:https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax_classic/game_logic.py ;论文 https://arxiv.org/abs/2402.16801
- MA-Craftax:https://arxiv.org/abs/2511.04904
- Jumanji RobotWarehouse:https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/routing/robot_warehouse/env.py ;Cleaner 文档 https://instadeepai.github.io/jumanji/environments/cleaner/
- gymnax autoreset:https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py
- gigastep:https://github.com/mlech26l/gigastep (NeurIPS 2023)
- Parabellum:https://github.com/syrkis/parabellum
- JaxMARL 论文:https://arxiv.org/abs/2311.10090
