# v1.2 360°移动+兵营+哨塔+前端 — research

规格:issue.md「## v1.2」节(用户已确认:连续坐标真 360°、建筑/资源点格子锚定、
渲染走「矢量+PNG 替换槽+提示词包」、matplotlib 不投美术)。

## 现状约束(与 v1.2 相关)

1. **pos 是 int32 格坐标**,贯穿:occupancy_grid(占用图)、movement(候选格+
   抢格仲裁)、combat(Chebyshev≤1)、harvest/build 的 dist==1 到达判定、
   production/建营落地格。连续化=这条主动脉换血,必须一次换干净,
   不留「有的系统看格、有的看浮点」的双真源。
2. **距离场是格上的 int32**(静态 BFS + 动态松弛)。连续单位用它:位置→所在格→
   对场取**双线性插值梯度**得方向向量。SMAX/gigastep 无场直线走;我们保留场
   是为了绕障,梯度采样是 flow-field 的标准用法。
3. **抢格仲裁将消失**:连续移动没有「格」可抢,改 SMAX 的圆形互推
   (两两距离矩阵 → 重叠量 → 位置互推,[N,N] 在 N=128 完全无压力)。
   随之消失的还有:候选格评分、方向序玩家取反(方向连续天然对称)、
   「目标格 tick 初须空」。**建筑仍占格**:单位圆与建筑格做圆-方推离。
4. **决定论**:浮点运算仍决定论(同后端);但 v1.0 的「GPU 录制→CPU 重放逐位
   一致」在 float32 上**不再保证**(累加序不同)。决定论口径收窄为「同后端
   逐位一致」,记 DECISIONS + 审计口径同步。
5. **btype/btimer/paid_orders_pass/解锁/研发**等 v1.1 机制与坐标无关,零改动;
   兵营=第二个「自由格建筑」直接复用建营机制;哨塔=第三个,外加攻击逻辑。
6. **前端**:alicization 有现成 FastAPI+WS+binary 协议+JS 渲染骨架可抄
   (server/app.py, server/protocol.py, web/render.js);TEOW 状态量小
   (N=128 实体),JSON over WS 就够,不必上二进制/WebGL,Canvas2D + Path2D
   矢量图案即可,PNG 槽用 Image 对象覆盖绘制。
7. 工具债三件(HANDOFF PENDING)与本版强相关:审计对账口径要在连续化后重验,
   先修;builder 自愈防脚本活锁在更长对局里更重要,先修。

## 关键设计(建议,plan 落实)

- **数据表示**:`pos` 改 float32 [N,2](建筑写成格中心坐标);新增
  `speed_by_type` 表(格/tick:工人 0.5、步兵 0.5、狗子 0.9 初值——与
  v1.1 的 move_cooldown=2 即 0.5 格/tick 对齐,行为近似保持)。单位半径
  统一 0.35 格(圆不出格,建筑推离简单);`heading` 仅渲染用,不进力学。
- **tick 内移动管线**:①方向 = 场梯度(双线性)或直指目标 ②candidate = pos +
  dir*speed ③建筑/边界 clamp+滑动(先 x 后 y 分量测试,撞墙沿墙滑)
  ④单位互推(一轮即可,SMAX 同款;推完不再判定,轻微重叠自然衰减)。
- **到达半径**:入驻/卸货/开工 = 与目标格中心距离 ≤ 1.2;近战射程 1.2;
  哨塔射程 4.0(全部进 Config)。
- **占用图退役与保留**:动态场的「静止单位=硬障碍」改为「静止单位所在格
  cell_cost 大惩罚」(软化,连续单位可贴身绕过);建筑格仍是硬障碍
  (入 passable 动态版)。
- **兵营/狗子/哨塔**:TYPE_BARRACKS=7, TYPE_DOG=8, TYPE_TOWER=9;解锁表
  `unlock_level_by_type` 替换单字段;兵营训练=复用 btype 正数;哨塔攻击=
  combat 内按射程选目标(与步兵同一条 masked-argmin 路径,射程查表);
  狗子线挂步兵捆绑线还是独立线?——**独立「狗子线」会让 upgrades 表变
  [2,3]**;v1.2 先让狗子吃步兵线(捆绑线语义=「兵种线」),记 DECISIONS,
  用户想分线在草稿箱提。
- **前端**:src/teow/server.py(FastAPI, /ws 推 JSON 帧,回放模式读 npz;
  live 模式后续版本再说,先做回放观战)+ web/index.html + web/sprites.js
  (Path2D 矢量图案,阵营色参数)+ web/render.js(Canvas2D,插值平滑)。
  资产槽:web/assets/<name>.png 存在则 drawImage 覆盖。提示词包
  docs/sprite-prompts.md。

## 风险

- 连续化触碰所有核心测试(到达/相邻断言从格数改半径),测试改写量大,
  必须先改测试口径再改实现,防「实现迁就旧断言」。
- 互推可能把单位推进建筑格 → 推离顺序:先互推再建筑 clamp(建筑硬约束
  最后执行)。
- 哨塔+射程后,「1 宽走廊」类地形仍不存在,该回归用例继续挂起(无地形墙)。
