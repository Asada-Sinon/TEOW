# v1.2 360°移动+兵营+哨塔+浏览器前端 — plan

规格:issue.md「## v1.2」节;设计取舍见同目录 research.md。
每 phase = 一个功能级 commit(`v1.2 <功能>: …`),验证贴真实输出。

## 目标

单位连续 360° 移动(建筑/资源点格子锚定不变)、兵营+狗子、哨塔、
浏览器前端矢量渲染(PNG 替换槽+提示词包)、全功能终审收官 v1 引擎阶段。

## 不在范围内

RL(v2)、迷雾(v3)、地形墙、前端 live 对战模式(先做回放观战)、
狗子独立升级线(先吃步兵捆绑线,记 DECISIONS)、兵营 2 级+的内容(待用户定)。

## Phase 0 工具债

- scripted builder 分支超时自愈:BUILD 指令持续 N tick(初值 120)未开工
  → 转 IDLE 重新决策;审计对账脚本补「卸货即死/开工即死」口径;
  run.py 给 random 侧对局也 --record。
- 验证:pytest 全绿;重跑 v1.1 审计脚本于旧 run 目录,原假阳性归零。

## Phase 1 连续移动核心(本版最大、最先)

- **坐标约定先定死(critic S-2,第一个 commit)**:格 (r,c) 覆盖
  [r-0.5, r+0.5),**格心=整数坐标**;唯一 helper `cell_of(pos)=round`
  (单定义处),所有格索引一律经它。
- **触碰面全清单(critic S-1,逐个入 commit 链)**:state.pos→float32;
  movement 全部;economy(occupancy_grid:53 / 建营落格:168 / 到达判定);
  combat(距离);actions(MOVE 目标格 128-132、target_cell 写入 238 的
  dtype——改 float32 防 pytree 结构漂移);controller(dist 索引 72、
  让路逻辑 143-153);production 落地写格心;render/metrics/tests 断言口径。
  **先改测试口径再改实现**,防实现迁就旧断言。
- Config:speed_by_type(工/兵 0.5,狗 0.9)、unit_radius=0.35、
  reach_radius=1.2(入驻/卸货/开工/近战统一)。
- movement.py 重写:场梯度双线性采样(HARVEST/BUILD/ATTACK)或直指(MOVE)
  → candidate = pos+dir*speed → 边界/建筑格沿墙滑动 → 单位圆互推一轮 →
  建筑格硬 clamp 殿后。**双线性采样前对场值 clip 掉 BIG_DIST 哨兵**
  (critic S-3:否则建筑邻域梯度被 BIG 支配,单位倒退不贴墙)。
  **镜像正对互推是精确共线退化**(critic FYI-1:180° 对称开局必触发)
  → 互推加确定性切向 epsilon(按槽号奇偶定向,保决定论)。
  抢格仲裁/候选格/方向序取反退役;动态场保留,静止单位改大 cell_cost
  软障碍,建筑格硬障碍;audit_movement_deadlock.py 后半段重写为通用
  stall 判据(last_moved 段保留)。
- heading 不进 state,由前端从位移推导(记 DECISIONS,防终审当漏项)。
- 验证:tests 全绿(口径已改);同 seed **同后端**逐位决定论(跨后端口径
  收窄记 DECISIONS;**provenance 增记 backend**,审计重放强制与录制同后端
  ——critic S-4);300+ tick scripted 无停摆 + **镜像对撞定向用例** +
  绕障单位 pos 单调接近目标的脚本断言;肉眼 replay 弧线/绕障/挤开。

## Phase 2 解锁表 + 兵营 + 狗子

- **production_tick 多生产者重构(critic B-1,v1.1 代码 NOTE 的兑现,
  本 phase 第一件事)**:同玩家同 tick 多建筑完工时逐完成者仲裁落地格与
  槽位(economy.py:265-283 的单 argmax 改为循环内处理全部完成者);
  训练/研发类 legality 的「空闲」判定补 `btype==0` 门控(actions.py:151
  只查 btimer==0——完工未落地的建筑处于 btype>0,btimer==0 窗口,会被
  新训练单覆写导致已付费单位蒸发)。定向测试:HQ 与兵营同 tick 完工
  各自落地、无覆写无双扣。
- Config:unlock_level_by_type 表替换 camp_unlock_level;TYPE_BARRACKS/
  TYPE_DOG;兵营建造(复用建营机制,A_BUILD_BARRACKS)、训狗
  (A_TRAIN_DOG,兵营 btype 正数复用);狗子吃步兵捆绑线;
  HQ 收窄(菜单不变,HQ 本就只出工/兵)。
- 数值初值:兵营 80/40/120 建造,hp 200;狗子 20/5/30 训练,hp 24 atk 3
  speed 0.9(全 [AI-DRAFT] 记 DECISIONS)。
- scripted:基地 2 级后建兵营,兵营持续出狗;攻击阈值统计狗+步兵。
- 验证:test_barracks.py(解锁/训狗/狗速>步速/狗吃步兵线补血);
  对局中狗群出现且先于步兵成型(便宜快出)。

## Phase 3 哨塔

- TYPE_TOWER;A_BUILD_TOWER(工人,复用自由格建筑机制);塔射程 4.0、
  攻击查表按塔级(tower_atk_by_level/hp_by_level,升级复用 A_UPGRADE);
  combat:塔与单位同路径选目标(射程查表,塔只攻单位不攻建筑?——攻一切,
  简单);塔不可移动。
- scripted:家门口与矿点旁各一塔(位置=复用建营的「相邻空闲格」,
  由守备工人携带指令)。
- 验证:test_tower.py(射程内自动开火/射程外不打/升级增伤);
  对局观察 rush 被塔削弱但未绝育(狗绕塔仍可偷家)。

## Phase 4 浏览器前端(观战/回放)

- src/teow/server.py:FastAPI + /ws,回放模式读 run 目录 npz 逐帧推 JSON;
  web/index.html + web/sprites.js(Path2D 矢量图案:HQ 堡垒/矿镐/泵水滴/
  工人安全帽/步兵盾矛/狗四足/塔楼/营帐,阵营色参数)+ web/render.js
  (Canvas2D,帧间插值,血条,等级角标)。
- 资产槽:web/assets/<name>.png 存在则优先 drawImage;
  docs/sprite-prompts.md 提示词包(统一俯视角/尺寸/风格/命名)。
- pyproject 加 fastapi/uvicorn/websockets(对齐 alicization 版本)。
- 验证:`.venv/bin/python src/run.py serve <run_dir>` 浏览器打开肉眼验收:
  图案可辨识、阵营色正确、连续移动平滑。

## Phase 5 全功能对决 + version-close v1.2

- scripted 全功能对局(升本/营/研发/兵营/狗/塔全出现,metrics 断言)
  → 无上下文终审(v1 收官,审计范围含 v1.0-v1.2 全规格回归)
  → changelog v1.2 → tag → handoff。

## 端到端验证

浏览器里看一场 seed 0 全功能对局回放:狗群走弧线包抄、塔开火、
攀科技与爆狗路线可辨;pytest 全绿;终审 P0 清零。

## 风险与对策

- 连续化改写面大 → Phase 1 单独成 commit 链(pos 迁移/movement/判定半径
  各一 commit),每步全测绿;
- 互推与建筑 clamp 顺序错会把单位挤进墙 → 固定「互推在前、硬 clamp 殿后」
  并加不变量测试(单位永不在建筑格内);
- 前端是新面(JS),风格问题不进 verify.sh 门禁,人眼验收为准。
