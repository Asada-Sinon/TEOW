# TEOW v1.5 实施计划:六边形四人图 + 栅栏 + 蓝方贴图

(Plan agent 产出,行号按 v1.4 HEAD=061791f 核对;plan-critic 审查:
2 BLOCKER + 2 MAJOR + 3 MINOR 已全部吸收进下文,原文对应处以【critic】标注)

## 设计决策

**D1 n_players 进 Config,默认 4;全引擎泛化,不保留 2 人旧地图。**
n_total = n_players*e_max;owner_of_slots = repeat(arange(P), e_max);map.py 重写为
六边形四人布局(assert n_players==4),引擎核心全按 cfg.n_players 参数化
(P 静态,jit 编译期展开)。现有测试大多经 hq_slot/mapdata 助手,迁 4 人默认。

**D2 winner 编码与淘汰:不加 state 字段。**
winner:-1 进行中,0..P-1 胜者,P=和局。「已淘汰」真源 = ~alive[hq_slot(p)]。
**淘汰清场落 cleanup_deaths**:alive2 后算 hq_dead[P] → elim_mask=hq_dead[owner]
→ alive3;矿泵释放/工地取消/驻守失效等连锁全基于 alive3(现逻辑白拿);
另 flag_active[p]←False。_end_tick 重写:存活 1 家→winner;0 家→和局 P;
超时→和局 P;淘汰不终局。

**D3 ATTACK 目标=最近存活敌方 HQ,每 tick 动态重选。**
movement goal:gather dist_fields[Nn..Nn+P) 在单位格的值,己方与已死 HQ 罚 BIG,
argmin。距离场选近天然绕障;HQ 亡后军队自动流向次近。actions.py 的 enemy_hq
target_cell 同口径;goal_center clip 上界 P-1。
【critic MINOR-2】argmin 平手取最小玩家号,对称局面低序号玩家被优先集火——
接受并记 DECISIONS(等变破平的收益配不上复杂度;field 值整数,真平手极罕见)。

**D4 六边形地图:Klein 四群对称(上下镜像+左右镜像+180°),奇数中心。**
出生点「左上/左下/右上/右下」按镜像配对(180° 单群配不齐四家公平)。玩家 0
(左上)定义布局,1=σh 像,2=σv 像,3=180° 像,(位置,类型) 三操作下自映射。
【critic B-1 修正】64×64 偶数网格无整数镜像轴 → **hex 心取 (31,31),镜像定义
σh: r→62-r、σv: c→62-c,行/列 63 恒在 hex mask 外**;E/W 顶点的矿与水都放在
不动行 r=31 上(E 侧一矿一水,W 侧=σv 像,类型严格自映射;原「(31,c)/(32,c)
镜像对放矿水」构造会把矿的像映到水的位置,自映射不成立,弃)。
平顶 hex mask:|r-cy|≤b 且 |r-cy|+k|c-cx|≤m,界外 passable=False,引擎零改动。
20 点:E/W 不动行 4 点(2矿2水)+ NE 轨道 1矿1水×4 像=8 → 公共 12 +
每家家门 1矿1水×4=8。dist_fields [Nn+P,H,W];旗通道从 Nn+P 起
(movement.py:119/170);n_goals = n_nodes + P(1+max_flags)。
工人出生位:朝地图中心搜索候选格再取像(三像互不重叠断言)。

**D5 栅栏:三独立建筑,完整复用 camp 自由格建筑管线。**
TYPE 19/20/21,BTASK -8/-9/-10;落位=建造者相邻第一空闲格(不落脚下——防
v1.4 活埋类缺陷);structs/grow_specs 各加三行;speed=0 自动硬障碍;不进
can_hit(无攻击);塔/迫击炮打不了栅栏,攻城车专克(自动成立);无数量上限、
无升级(a_upgrade 白名单不加即不可升)。「无半场限制」引擎本无此检查,零代码;
free_in_half 改名 free_in_block 消歧义。栅栏可自闭围死(不可达=场 BIG 原地停,
不做防自闭检查,记 DECISIONS)。数值草案 [AI-DRAFT]:木 HQ2/10矿/20t/HP80/甲0;
石 HQ3/25矿5水/35t/HP200/甲20;铁 HQ5/40矿20水/50t/HP400/甲40。

**D6 per-player 循环两模式**:顺序仲裁类 for p in range(P) 编译期展开;
计数类 stack([p0,p1])[half] → scatter-add `zeros(P).at[own_i].add(cond)[own_i]`。

**D7 e_max 保 64(N=256,N² 矩阵 262KB 无压力);episode_len 3000→6000
[AI-DRAFT]。热点=动态场松弛(36 通道×4096 格×~128 迭代,~43×):P2 末 bench
实测,掉速则 k_iters 提成 Config 字段压到六边形直径。**

**D8 前端四色 + 蓝方 PNG。**
P_COLOR 4 色(蓝/红/绿/琥珀);owner = floor(s/e_max);和局判据 winner===n_players
(meta 下发 n_players);fig/ 17 张 → web/assets/ 改名 cp(大本营→hq_p0.png、
矿井→mine_p0、水井→pump_p0、工人→worker_p0、普通刀斧手→infantry_p0、
技能训练营→camp_p0、狗子→dog_p0、哨塔→tower_p0、大力士工人→strongman_p0、
运输马车→wagon_p0、弓箭手→archer_p0、骑兵→cavalry_p0、重装刀斧手→heavy_p0、
法师→mage_p0、奶妈神官→healer_p0、迫击炮→mortar_p0、军旗→flag_p0);
drawFlag 加 PNG 槽;兵营/攻城车/栅栏无贴图走矢量(TYPE_NAMES+3)。

**D9 脚本 AI 四人化改动极小**:scripted 已按 player 参数化,ATTACK 目标引擎侧
解决;make_joint_controller(names) 变长,merge = stack(acts)[owner, arange(N)];
run.py --p2/--p3;栅栏不教 AI 建(单测覆盖,记 DECISIONS)。

## 分阶段

- **P1 P-泛化**(默认仍 2 人,纯重构保绿):config/state/actions/economy/
  movement/combat/step/controller/server/render/run 全部 P 参数化;
  77 测试不动保绿(critic 核数);test_elimination 雏形(2 人版)。
  【critic B-2】start_constructions 先手仲裁 `bernoulli`(economy.py:429)只出
  0/1,四人下玩家 2/3 平票永败——改 `randint(key,(),0,P)`,加单测:四家工人
  同 tick 到同一无主点,多 seed 统计中签率无系统偏置。
  【critic M-1】P 化清单补:economy.py:474 `clip(node_owner,0,1)`、run.py
  metrics 的 res_p0/p1 与 outcome dict per-player 化。
- **P2 六边形四人地图**(结构主拍):map.py 全重写;默认翻 4 人/64×64/20 点/
  6000;test_map 重写(hex 界外/Klein 自映射/20 点/出生对称/场形状);
  【critic M-1】全测试 sweep 三条并列:①字面量坐标改 (h//2,w//2) 锚定;
  ②winner/和局编码 2→n_players(test_combat_win/test_scripted_v14/run.py:153);
  ③半区切片 `[:e_max]` 与 `[2,…]` 形状字面量。
  【critic M-2】bench 判据:先记 v1.4 基线 tick/s 入 run 目录;4 人 hex 若
  < 基线/8 则 k_iters 提 Config 字段压到 hex 直径,并加「最远可达格场值<BIG」
  断言;再不济记 DECISIONS 降档。
- **P3 栅栏**:config 3 类型+3 任务码+字段;actions 3 build id+掩码;economy
  structs/grow_specs +3;stats +3;test_fence(挡路/可拆/塔打不了/攻城车专克/
  三档解锁/低级仍可建/建造者不活埋/【critic MINOR-3】槽满建栅栏 no-op 不扣费)。
- **P4 前端四色+蓝方贴图**:sprites/render.js/render.py/server;assets cp;
  录四人局 serve 验收。
- **P5 脚本四人局收口**:run.py 四控制器;test_scripted_v15;determinism 4 人;
  episode_len 校准实验;五件套收尾。

## 风险
1. 动态场吞吐 43×(movement.py:177)——bench 后压 k_iters。
2. jit 编译时间:per-node 8→20 × per-player 2→4 展开,trace ~5×;编译 >1min
   才改 vmap。
3. 测试字面量坐标落 hex 界外(症状「不动」难查)——P2 系统 sweep。
4. 淘汰在途状态(工地/炮弹/矿内/旗)——test_elimination 逐项断言。
5. 和局编码 2→4:旧回放靠 resolved_config 区分;旧轨迹不迁移(记 DECISIONS)。

## Config 变更汇总
n_players=4(规格)、grid 64×64(规格「约 64」)、n_nodes=20(规格)、
episode_len=6000 [AI-DRAFT]、栅栏三组字段 [AI-DRAFT]、解锁 2/3/5(规格)、
(条件)field_relax_iters。
