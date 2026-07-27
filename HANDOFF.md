# HANDOFF

**本文件当前是空模板，还没有任何真实历史。** 下面只有格式说明和一段被注释掉的示例。

这不是文档，是上一个 agent 写给下一个 agent 的信。要短、要具体、只写下次用得上的。
不写背景介绍，不写「本项目旨在……」——那些在 CLAUDE.md 里。

规矩：
- 新会话结束时加一节，**最新的在最上面**。
- 只保留最近 3 节，更旧的直接删掉（历史在 git 里，不用囤在这）。
- `PENDING` 是下一个 agent 开工的第一件事，必须写成可执行的动作，不是「继续优化」。
- 教训不要写这里，写 `MEMORY.md`：HANDOFF 会过期，教训不会。
- 提到实验产物时路径一律 `experiments/<run_id>/`，run_id 格式 `YYYYMMDD-<slug>`。

格式：

```markdown
## Session YYYY-MM-DD
- 完成: ...
- PENDING: ...        ← 下次第一件事
- 坑: ...
```

<!-- 示例（安装后请删除这整块）
以下为格式示例，不是本项目的真实历史。这里出现的日期、文件名、run_id、数字全部虚构，
任何 agent 都不得把它们当作本项目的事实、进度或依据。

## Session 2026-03-14
- 完成: 把 dataloader 的 shuffle 挪到 sampler 层，`tests/test_loader.py` 全绿
  （`pytest -x -q tests/test_loader.py`，17 passed）。
- PENDING: `src/train.py:118` 只存了 config 路径，没落盘 resolved config，违反
  CLAUDE.md 硬约束第 3 条。下次第一件事：把展开后的 dict dump 成
  `experiments/20260314-shuffle-sampler-seed0/config.resolved.yaml`，再补跑一次验证。
- 坑: 直接 `python src/train.py` 用的是系统 python，缺 torch；必须用 CLAUDE.md
  命令区里那个解释器的绝对路径。
-->

---

<!-- 真实的 session 记录从这一行下面开始写，最新的一节永远插在紧挨本行的下面。 -->

## Session 2026-07-28(通宵:v2.1 训练前准备,除险完成)
- 完成: **v2.1 全脚手架 + Phase A/B/C/D + 试训除险**。①训练循环 `explorations/rl_train_v21.py`
  (持续环境分段 rollout done→autoreset 解全长局显存 / epoch×mb 全scan / lr·ent·β 退火 / savez
  ckpt / CurriculumScheduler)+ `rl_eval_v21`(RL-vs-脚本谐波)+ `bench_train_v21` + `diagnose/
  plot_train_v21` + slow test(4 passed:整环路/ckpt bit相等/课程/持续环境reset)。②Phase A 对手池
  洗净:**发现座位偏置**(rr 单shift令座位与对手混淆,result-analyst 核验→弱尾主要真弱)改**全P座位
  轮转**;**turtle/airtech 定向调**修「对random 0军被动」(升本囤钱→降 upgrade_reserve/base_level/
  attack_threshold 腾钱造兵,末军0.1/0.2→1.1);boomer 激进调过头回退。rr4 干净分层 HARD rusher/
  MEDIUM balanced·harasser·tempo·counter/EASY 其余。③Phase C 吞吐 **54-62k games/day**(含gate+网+
  更新,B64T128甜点,训练非瓶颈)。④**Phase D 试训除险(核心)**:管线机器正确,但抓到②问题——
  **reward-hacking**(Φ含全额库存诱导囤钱,已修:库存打折 stockpile_weight)+ **纯PPO冷启动学不动**
  (随机网探索不到结构化动作,3训练+全谱调参 army=0 退化NOOP/STOP)→**正式开训必须BC暖启**。
  五件套:门禁 / engine-auditor(src引擎未改→N/A,profile靠rr4+门禁) / changelog v2.1 / tag v2.1 / 本交接。
- PENDING(正式开训): **先实现 BC 暖启**(explorations,脚本oracle蒸馏让策略从会造兵起步→PPO微调超越;
  plan §1.4/§7 已设计)。BC后重跑 Phase D 看真学习信号(vs random 胜率升+army>0)。**可直接复用**:
  rl_train_v21(持续环境/退火/ckpt/课程全就绪)+ 干净对手池(rr4分层)+ 吞吐甜点(B64T128)+ reward-hacking
  已修的Φ。命令模板见 research-log 20260728-v21-train-fix2 节。
- 坑: ①**PBRS 势函数不能含未投入的库存**(诱导囤积)——stockpile_weight<1;②纯PPO长局稀疏奖励从
  随机网学不动,强塑形救不了冷启动探索,**必须BC**;③FFA评测座位轮转必须**全P**(单shift座位与对手
  混淆;座位偏置全局6000下温和seat0偏强{51,21,34,38});④训练eval全局6000贵,eval_seeds/every要控;
  ⑤rl_train/eval/bench在explorations(未进src),conftest加了explorations path供slow test;⑥骨架改动
  向后兼容(potential默认stockpile_weight=1.0保v2.0 smoke,训练传0.3)。

## Session 2026-07-27(通宵:v1.8 收官,v1.9/v2.0 推进中)
- 完成: **v1.8 五件套收官**——异界之门 sudden-death 必分胜负(`src/teow/gate.py`:阵营隔离怪
  gate_tick 生成+慢速 descent / monster_combat_tick 独立子结算;HP 无上限线性、攻击封顶、慢速
  近战、强度生成时定死、死玩家清怪;_end_tick 删和局+硬帽残血兜底;d39dfcf)+ **10 风格参数化
  指挥官**(`commanders/{profile,base}.py`:StrategyProfile 静态闭包 trace 期分支;levers=经济旋钮/
  tech_focus/comp_bias(counter=最佳响应)/aggression/adaptive(反空转)/stochastic;balanced≡
  scripted 逐位一致;bb5feac)。P0 吞吐 bench:GPU vmap B64 ~4000 env-tick/s(#1 风险解;7c03a61)。
  评测脚手架 `explorations/eval_commanders_v18.py`(胜率矩阵+质量,v1.9 复用)。engine-auditor P0
  零;P2 当场修两处(对怪开火进 cd / 离场清怪血,8a5d2e1);终门禁 117 pytest(`-n 8` 并行~8min)+
  ruff 绿;覆盖局 experiments/20260727-v18-audit-cover(airtech/turtle/boomer/counter,tick5860
  winner0 经异界之门;决定论+隔离+守恒全过)。
- 完成(续): **v1.9 收官**——评测脚手架核心 `matchup_runner` 提升进 `src/teow/eval.py` + test_eval;
  综合评测(experiments/20260727-v19-roundrobin,80 局):10 指挥官全碾压 random、无统治风格、非退化
  (无秒杀/硬帽/和局);难度分层 HARD rusher / MEDIUM 6 / EASY turtle·timing·airtech;**全留不删**
  (弱尾风格独立 + eval 非均衡噪声)。changelog v1.9 + tag。**v2.0 设计文档**已落
  docs/plans/20260727-v2-rl-approach/research.md(9 节)。
- 完成(续 2): **v2.0 收官**——RL 调研(19 篇 notes/papers/)+ 设计文档(9 节,
  docs/plans/20260727-v2-rl-approach/)+ 不训练 JAX PPO 骨架 `explorations/rl_skeleton_v20.py`
  (纯 JAX 手搓 MLP+Adam;F=36/G=43/32821 参;smoke B8×T128 空跑一步 loss 有限、终局名次自检✓,
  **未做真正训练**)。changelog v2.0 + tag。**🎉 v1.8→v2.0 全部五件套收官(tag v1.8/v1.9/v2.0)。**
- PENDING: **v2.1(训练前最终准备,本轮未做)**——全量测试 + 小范围试训。开工先:①**v1.9 弱尾
  followup**(均衡 round-robin ≥8 seed 复核 turtle/timing/airtech + 定向调「0-军被动 gate 胜」);
  ②骨架补 v2.1 训练循环(多步 epoch×minibatch + LR/β anneal 调度 + checkpoint + vs-脚本胜率监控
  + 课程/对手池 wiring + BC 暖启);③**吞吐实测**(gate + net-in-loop → games/day,守硬约束#1、
  不口算);④小范围试训别训出「什么都不会/执行奇怪指令」的指挥官(issue.md v2.1)。设计细节 +
  开放问题见 docs/plans/20260727-v2-rl-approach/research.md §8。
- v1.9 followup(v2.1 前):均衡 round-robin(每对覆盖、≥8 seed)复核弱尾真实强度;定向调 turtle/
  airtech「0-军被动 gate 胜」;确认 22/80 局固定落 length4182 不诱导 RL「摆烂等门」。
- 坑: ①异界之门令 step 编译变慢,**串行全套 pytest >40min**;已装 pytest-xdist(uv pip,未进
  pyproject),**用 `pytest -n 8`**~8min。②GPU 单环境编译极慢(分钟级)但 vmap 批量 run 快;eval/
  bench 用 GPU 批量,单环境/门禁用 CPU。③指挥官策略参数是代码常量(StrategyProfile)非 config;
  复现锚点=git hash+名字+seed。④eval 有 per-matchup try/except 兜底(坏指挥官不杀整轮)。

## Session 2026-07-26(v1.7 数值平衡收官)
- 完成: v1.7 五件套收官(tag 待打)。修训练营升级不补血差 bug(economy.py 自升级
  完成分支加 camp 补血,与哨塔/兵营同构;挂 v1.4-v1.6 四版静默 bug;6498fe2)。
  建通用对决脚手架 explorations/exp_v17_duel.py(用户 2026-07-26 定口径:①单位vs
  单位=原地接战交错摆位纯 stat 交换;②防御建筑=攻防局,同价该守住/~2倍造价该被破,
  攻方起步距离随防御方射程缩放;③water=矿同重)+ audit_v17_invariants.py(加⑪离场
  inside/aboard 血量护栏、⑫龙喷火对建筑折扣量级上界,修 v1.6 已知问题 #35 ab 变量
  遮蔽;d62d61d/36b4a2f)。数值复核结论:近战/哨塔/喷火/激光/攻城/龙对空/奶妈**全
  平衡不动**,仅三处偏离——**法师塔 magetower_atk 14→20 + magetower_period 5→4、
  龙火海 dragon_breath_radius 2.5→4.5**(用户 2026-07-26 在线定案,8408a7b);迫击炮
  数值无解(扫13候选全守不住1×,机制限制:盲区2.5+单发慢炮弹不预判,用户定案接受
  为炮击/攻城支援非点防,记 changelog 已知)。覆盖局 experiments/20260726-v17-audit-
  cover 用新 config 重录(4361tick,7种高阶实体全出场):audit_v17 决定论 2181帧逐位
  一致0失配+25不变量全零;engine-auditor P0/P1零;终门禁 111 pytest+ruff 绿。
- PENDING: ①打 tag v1.7 并 push(本条落盘后立即执行);②用户已在 issue.md 加
  v1.8-v2.1 路线图草稿(v1.8 多风格脚本指挥官/上帝视角条件判据/启发式+随机+概率+
  博弈论;v1.9 筛选高质量对手数据、改不好就删;v2.0 调研RL算法 PPO/AMP 或直接用脚本
  指挥官训练+奖惩项/难度曲线,**只调研不训练**;v2.1 训练前全量测试+小范围试训)——
  工作区 issue.md 故意留脏没提交,**下个 session 第一件事走草稿协议吃透 v1.8 搬进
  规格区**;③用户复核 v1.7 DECISIONS 的 [AI-DRAFT] 三条。
- 坑: JAX 门禁(pytest)别和其它重 CPU 任务并行——本次和覆盖局录制/engine-auditor
  审计重放并行,被拖到 40min(独占约 9min)。GPU 对「每 Config 不同+单环境逐 tick」
  这类负载**慢 12 倍**(实测单对决 GPU 6:57 vs CPU 35s),提速靠多 CPU 进程分片,GPU
  要等 v2 vmap 批量 rollout 才有意义。
