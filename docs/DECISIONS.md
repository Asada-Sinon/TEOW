# 决策日志(append-only)

用户离线期间 Claude 的自主决策记录。每条带日期与标注;用户复核后可把 [AI-DRAFT]
升级为 [HUMAN-VERIFIED] 或推翻(推翻时 append 新条目引用旧条目,不删旧条目)。

---

- 2026-07-25 [AI-DRAFT] **矿/泵被摧毁时,矿内工人弹出到相邻格、不受伤害**。
  理由:比「陪葬」宽容,避免经济被一波打崩导致对局质量差;实现也更简单(不需要
  结算建筑内单位伤害)。待 v1.0 试跑后复核手感。
- 2026-07-25 [AI-DRAFT] **v1.0 寻路 = 朝目标贪心一步(被挡则原地),不做 A\*/BFS 距离场**。
  理由:24×24 开阔地图障碍极少,贪心足够;若调研报告(jax-rts-engine)显示先例有
  便宜的距离场方案且贪心实测卡死,再升级。
- 2026-07-25 [AI-DRAFT] **超时和局阈值 episode_len=3000 tick**(Config 可调)。
  理由:按默认参数估算一局经济+攻防节奏所需时长的上界,试跑后校准。
- 2026-07-25 [AI-DRAFT] **v1.0 采集节奏初值:矿内开采 20 tick/趟、载荷 10、移动 2 tick/格**。
  理由:新机制无先例参数,取「一趟往返约 60-80 tick、一个满编矿(4 工人)约
  0.5 资源/tick」的量级起步,scripted 对局试跑后校准并记 changelog 平衡区。
- 2026-07-25 [AI-DRAFT] **工作流调研报告的三点建议被否决(用户指示优先)**,报告见
  docs/plans/20260725-workflow-design/research.md:①报告建议 issue.md 归用户独有、
  agent 只读——与用户明确指示「吃透后重写成通透版」冲突,按用户指示执行,但吸收其
  精神:改写必须语义等价、歧义必须先问(已写进 versioning.md);②报告建议 git tag
  留人工 gate——用户已把版本化 git 管理整体授权给 agent 且批准了含 tag 的计划,
  改为 agent 在 /version-close 第 6 步执行;③报告建议不建 DECISIONS.md——但其
  论证未覆盖「用户离线期自主决策留痕复核」场景(本文件的唯一职责),故保留,
  职责限定为离线决策,不做通用设计日志(设计取舍仍落 plan.md/changelog)。
- 2026-07-25 [AI-DRAFT] **出矿不做占位仲裁,允许暂时叠格**。若要求入口格为空才能
  出矿,「满员矿点 + 排队工人站满全部入口格」会形成永久死锁(里面出不来→外面
  进不去)。叠格是瞬态:占用图按「格上有人」计,叠格者后续移动自然散开。矿被拆
  时驻内工人的弹出沿用同一约定。实现:src/teow/economy.py harvest_tick。
- 2026-07-25 [AI-DRAFT] **施工被打断(工人死亡)不退款**;训练扣费在下单瞬间、
  建造扣费在到场开工瞬间(避免「在途死亡退款」的对账复杂度)。
- 2026-07-25 [AI-DRAFT] **寻路升级为「每 tick min-plus 松弛的动态距离场」**,
  静态 BFS 场只用于到达/入驻/卸货判定。原因:纯静态场会被站桩单位永久堵死
  (实测:满载工人被 HQ 旁发呆工人堵死在唯一下坡格,经济归零)。障碍=静态障碍
  +静止单位;移动中单位不算障碍。v2 vmap 训练时需重新评估此项开销。
- 2026-07-25 [AI-DRAFT] **三类经济死锁的修复约定**:①BUILD 指令目标点被别人
  占用即自动转 IDLE(否则产生永久站在矿入口的僵尸建造工);②生产落地斜角格
  优先(正邻 4 格是卸货格,落正邻会堵运矿通道);③脚本 AI 令贴 HQ 正邻格站立
  的空闲单位主动让开。①②是引擎规则,③是脚本行为。
- 2026-07-25 [AI-DRAFT] **verify.sh 门禁强制 JAX_PLATFORMS=cpu**:同套测试
  GPU 5min(每个 Config 变体重新 jit)vs CPU 14s,GPU 会爆 60 秒门禁预算;
  逻辑判定二者等价。正式训练/对局仍走 GPU。
- 2026-07-25 [AI-DRAFT] **Python 钉 3.12、jax[cuda12] 钉 ==0.6.2**(与 alicization
  验证过的组合一致;3.14 无 jaxlib 轮子,浮动版本导致过 30 分钟的无效下载)。
- 2026-07-25 [AI-DRAFT] **audit P0-1(对向工人流死锁)修复方案**:移动中单位从
  「不算障碍」改为动态场**软障碍**(穿其格 +congestion_cost=8),对向流自动分道。
  验证:冻结拖和的 seed1 复跑 733 tick 分胜负;random vs scripted 双向均由
  scripted 获胜(此前 random 靠对方经济冻结取胜);14 测试仍全绿。
- 2026-07-25 [AI-DRAFT] **audit P1-1 裁决**:attack-move 维持「仅步兵、目标固定
  敌方 HQ」的 v1.0 实现,issue.md 规格文字系我改写时的含糊,已澄清成与实现一致
  (通透版是我写的,该歧义不构成对用户原意的偏离;用户原稿未提 attack-move 细节)。
- 2026-07-25 [AI-DRAFT] **v1.1「升级中的建筑照常工作与否」定案**(issue.md:89 要求
  记录,audit v1.1 P1-1 补账):HQ/训练营升级期间**停产停研**(单槽任务队列天然
  串行,升级与训练/研发互斥);**矿/泵升级期间照常产出**(驻矿采集不走任务队列,
  产出按当前等级查表)。依据:前者避免「免费并行」,后者避免升矿变成惩罚。
- 2026-07-25 [AI-DRAFT] **audit v1.1 P0-1 修复**:同玩家同 tick 跨营对同一条线
  并发研发会双倍扣费只得一级 → paid_orders_pass 在扣费前按线去重,只批槽号
  最小的申请(v1.1 控制器不触发,但 v2 RL 会踩,掩码正是给 RL 复用的接口)。
- 2026-07-25 [AI-DRAFT] **seed12「random 胜 scripted」判定为控制器活锁+平衡现象,
  非引擎回归**(audit v1.1 P2 证据链):scripted 唯一空闲工人被 builder 分支钉死
  在「没水建不起的泵」上,其余工人永不空闲不可重派 → 水收入归零 1200+ tick。
  本版只修助燃剂(rich_for_node 改查 成本+储备);builder 分支的自愈(如超时
  改派)留 v1.2 控制器改进,记 changelog 已知问题。
- 2026-07-25 [AI-DRAFT] **v1.2 连续移动的四项定案**:①决定论口径收窄为
  「同后端逐位一致」(pos 改 float32 后跨后端累加序不同;provenance 增记
  backend.txt,审计重放须与录制同后端);②heading 不进 state,由前端从位移
  推导;③近战射程 melee_range=1.5(≈旧 Chebyshev≤1 含对角)与到达半径
  reach_radius=1.2(≈旧 4 邻)是两个常量,不强行统一——统一到 1.2 会砍掉
  对角攻击改变 v1.1 战斗力平衡;④**移动单位不计入寻路场代价**:自己的罚分
  落在自己脚下的格,双线性梯度在原地采样被尖峰支配,实测退化成布朗游走
  (一趟采集 35→140 tick);对向流僵持由圆形互推+槽号奇偶切向分量解决,
  v1.1 的 congestion_cost 字段退役。
- 2026-07-25 [AI-DRAFT] **v1.2 终审分诊**:①P0-1(步兵线研发不给存量狗补血,
  血攻分裂)已修——补血循环重构为「每线多受益类型」,防多条目重复加级;
  ②P1-1 兵营自升级:规格括注「后续级别内容待定」→ 裁决为**顺延**,内容定义
  后再实现(issue.md 已同步);③P2 塔目标偏好(is_building 漏 TYPE_TOWER)
  顺手修;scripted 贫困陷阱(真没钱时挂机)与「驻矿工人停靠格被建筑占格」
  两条记 changelog 已知问题,不阻塞收官。
  test_adjacent_mutual_damage 失败,单跑与后续两次全套(CPU/GPU)均绿,未复现。
  若再现按 P0 排查(int32 scatter-add 理论上确定,不应 flaky)。
- 2026-07-25 [AI-DRAFT] **引擎设计采纳 jax-rts-engine 调研的四项修正**(报告见
  docs/plans/20260725-jax-rts-engine/research.md):①实体表从 [2,E] 双player轴改为
  **单表 [N=2*E_max] + owner 列**(HQ 槽固定 0 与 E_max);②抢格仲裁用
  scatter-min + 每 tick 随机优先级 + 目标格 tick 初须空(防 self-play 下标偏置);
  ③寻路弃裸贪心,改 **jit 外 numpy BFS 预计算距离场**(各资源点+各HQ)闭包进
  build_step,单步 masked-argmin 下降(防凹障碍卡死工人毁掉经济信号);
  ④战斗照抄 SMAX:masked argmin 选邻接目标(全无效必须门控)+ .at[].add 同时
  结算 + 先结算后翻 alive(允许同归于尽)。
- 2026-07-25 [AI-DRAFT] **v1.3 Phase 4「拥有建成兵营」判定 = `btype >= 0`,不采
  plan 字面的 `btype == 0`**。issue.md 规格写「解锁条件 = 拥有兵营(1 级即可)」
  「撤旗挂在兵营上,免费、即时」,均无「空闲」要求;btype==0 是训狗 legality 的
  空闲判定,借用过来会把正在训狗的兵营排除在「拥有」之外(插旗突然非法),并让
  Phase 5 总攻 tick「兵营边训狗边撤旗」被掩码卡死。btype>=0 恰好只排除在建
  (BTASK_BUILD_BARRACKS<0),命中 plan critic m-2 的本意。规格优先于 plan 字面。
- 2026-07-25 [AI-DRAFT] **v1.3 Phase 6 哨塔终值:`tower_atk_by_level` L1 6→3,
  其余等级与造价/血量不动**(用户明确授权 agent 直接决策)。依据
  experiments/20260725-tower-balance-{base,atk4,atk3,cost80-50,hp90}:
  ①config-only 杠杆都翻转不了「1 塔守 ≤5 狗」的结局,只能调交换比——atk3 把
  3-4 狗骚扰的战果从换 1 工人提到换全部 3 工人、清场时间 21→36 tick,是唯一
  真实提高骚扰收益的杠杆,直接命中规格「小规模进攻与骚扰全被轻易化解」;
  ②造价 80/50 会让 scripted 全程不造塔(压制使用而非平衡强度),hp90 各档
  战果零变化,均不采;③「攻击间隔」新机制不进 v1.3(plan 明文默认不做,且
  issue.md v1.4 草稿的多塔上限+迫击炮本就要重做塔体系,届时一并设计)。
- 2026-07-25 [AI-DRAFT] **v1.3 终审 P1-1 裁决:修,不收已知问题**(用户授权
  agent 直接决策收尾)。名额仲裁竞态——同 tick「从 A 改派 k 被 rank 拒 +
  新人抢走 A 的预释放名额」可把 A 顶到 cap+K 且持续存在(审计员纠正 plan
  「瞬时超额」的定性)。裁决理由:规格明文「满员后对该点的采集指令被合法性
  掩码挡掉」,可持续超额直接违反 v1.3 头号机制语义,且 v2 RL 恰好会学会
  利用此洞。修法照 plan 预案:HARVEST 改派旧名额「新指派成功才释放」
  (actions.py 仲裁 hold 口径),代价满员点同 tick 对换互卡一拍;回归用例
  test_harvest_reassign_rejected_keeps_cap。终审 P2 三条记 changelog。
- 2026-07-26 [AI-DRAFT] **v1.4 机制细节六项**(用户睡眠期,离线协议:规格未
  尽处按最合理解释落地,平衡数值均标 [AI-DRAFT] 待 v1.7 复核):
  ①迫击炮死亡时在途弹随之消失(逐实体弹字段的自然语义,免弹道表;炮都没了
  弹再落地反直觉);②迫击炮溅射只伤敌方地面单位——不溅建筑、无友伤(目标
  过滤=伤害过滤,同哨塔「只攻单位」约定;规格只说「对远处的地面单位造成范围
  伤害」);③v1.4 迫击炮不可升级(规格未提;哨塔明文可升级而迫击炮没说,
  从窄解释);④奶妈只奶单位不奶建筑(规格「友军」收窄:建筑修理是另一套
  机制,且在建血量成长与治疗叠加会破「建成=满血」不变量);⑤八线研发共用
  一套成本/耗时表(per-line 差异化无规格依据,留 v1.7 数值版);⑥旧双线研发
  动作 id(11/12+2Nn)退役保号、狗子从步兵捆绑线拆出独立线(推翻 v1.2
  「狗吃步兵线」决策——v1.4 规格「按需升级对应兵种」明文要求按兵种线)。
- 2026-07-26 [AI-DRAFT] **v1.4 自审计 P0:建筑落位活埋矿内工人,修**。
  矿内单位不占格,训练营恰落其入口格 → 出矿弹回被永久困死(硬障碍格内场
  梯度归零),覆盖局实测卡 1800 tick 且吊死该点采集名额(experiments/
  20260726-v14-audit-cover 为出 bug 证据局,cover2 为修后复核局,17 项
  不变量全零)。修法:自由格建筑落位占用图并入矿内单位入口格;单位落地
  (production)不改——单位间短暂叠格由互推散开,是既有约定。
- 2026-07-26 [AI-DRAFT] **v1.4 scripted AI 行为调整三项**(AI 旋钮不属引擎
  规则,为对局质量与审计覆盖服务):①插旗改「第一只狗即插」且总攻期也可插
  ——旧「第 3 只狗插旗」在总攻波次下凑不齐并发狗,插旗时点贴终局,对 jit
  融合浮点微差过敏(scan 版与逐拍版时间线分叉导致测试翻车);②升级决策加
  「升级任务在途不重复下单」门——修「完成拍看旧等级再下一单」竞态,基地
  会被顶超目标一级白烧 600/400;③兵营选型「训数量最少的已解锁兵种」带
  平手偏置(高阶优先、healer 先于 mage、healer/ram 封顶 2)——纯 argmin
  平手恒取狗,ram 永不入队。覆盖型审计局配置:富开局 3500/2200 +
  base_target 5 + threshold 14(experiments/20260726-v14-audit-cover2)。
- 2026-07-26 [AI-DRAFT] **v1.5 机制细节七项**(离线协议):①六边形几何取奇数
  镜像中心 (31,31),σh:r→62-r、σv:c→62-c,行/列 63 恒在界外;E/W 顶点公共
  矿水放不动行 r=31(plan-critic B-1:偶数格无整数镜像轴,原「(31,c)/(32,c)
  对放矿水」会把矿的像映到水的位置,自映射不成立,弃);②开工先手仲裁
  bernoulli→randint(0,P)(critic B-2:四人下玩家 2/3 平票永败);③ATTACK
  最近敌 HQ 用静态 BFS 场值选靶、argmin 平手取最小玩家号(等变破平收益配不上
  复杂度,场值整数真平手极罕见);④episode_len 3000→6000(大图四人局,
  实测默认局 2244 tick 分胜负,余量充足);⑤栅栏可自闭围死不做防呆
  (不可达=场 BIG 原地停,拆墙/绕路是玩家的事);⑥scripted AI 不建栅栏
  (机制由单测覆盖);⑦旧双人回放与 v1.5 不兼容(地图/动作空间换代,靠
  resolved_config 的 n_players 区分;v1.4 动作保号先例不适用)。
- 2026-07-26 [AI-DRAFT] **v1.5 已知现象:镜像位玩家(p1/p3)在 scripted
  对局中战绩系统性偏弱**。引擎地图对称已被 test_map Klein 自映射锁死;
  偏差来自 AI 层非镜像的平手偏好(槽号序、_SPAWN_DIRS 斜角序、让路方向)
  ——镜像世界里同一套偏好不再对称。属 AI 层已知问题记档,v2 RL 自对弈
  不受影响(策略自己学);不改 v1.5(改 AI 平手口径会全量重标定涌现测试)。
- 2026-07-26 [AI-DRAFT] **v1.5 审计口径:淘汰当拍守恒豁免**。同 tick
  「付费获批 → HQ 阵亡 → 全家清场」会把刚落地实体一并抹掉,支出无从与
  实体对账(钱确实花了、买家已出局,引擎语义正确);audit_v15_invariants
  对淘汰当拍的该玩家行豁免守恒检查。
- 2026-07-26 [AI-DRAFT] **v1.6 规格解释与机制细节**(离线协议,plan 八条+实现
  两条):①投石车/飞艇/龙骑兵不开升级线(固定属性)——N_LINES 参与 v1.5 栅栏
  动作 id 公式,扩线即破「退役保号」契约且 upgrades 形状连带;v1.7 要开线以
  N_LINES_LEGACY 固化旧公式再追加新 id;②龙喷火只伤地面**单位**不伤建筑
  (对齐喷火器措辞与迫击炮溅射先例;攻坚职能交给飞艇空投,空军克军队、地面
  克建筑);③投石车只伤单位/无盲区/短弹道(取迫击炮对称);④法师塔/激光炮
  hit_b=0(防御建筑只打单位先例);⑤地雷不挡路(「绕道」=绕触发圈非物理墙,
  挡路=5 枚不可拆的廉价永久栅栏)/占落位格/在建不触发/全图可见/空军不触发;
  ⑥登艇限地面战斗单位(采集单位不可——规格通篇「空降作战」)/reach 内最近
  己方艇/无自动走靠/DROP_ALL 单动作/登艇清指令锚点;⑦同 tick DROP 先于
  BOARD(腾位同拍可用,与撤旗先于插旗同哲学);⑧威胁禁区=敌方 (atk>0&rng>0)
  射程圆 ∪ 喷火器/龙自心圈,不含地雷触发圈(与 combat 攻击者门同源);
  ⑨空军直线转向(六边形凸集连线不出界)/受静态边界/空地零碰撞/不占格;
  ⑩「己方 HQ 附近秒上」=「其余位置即时」的子集(规格区原文无 HQ 豁免语义,
  plan-critic 复核确认非遗漏)。
- 2026-07-26 [AI-DRAFT] **v1.6 覆盖局定标:8000/8000 + target 7 + 阈值 25**
  (explorations/calibrate_v16_coverage.py)。8000/5000 时兵营卡 5 级——
  5→6 升级要 190 水+预备金,水饥荒(基地升级链吃掉 600+400+250 水);
  水预算翻倍后 laser 1950 → airship 3650 → dragon 3875 → catapult 4225 →
  4456 分胜负,七实体全出现 [source: 20260726-v16-audit-cover]。
- 2026-07-26 [AI-DRAFT] **v1.6 scripted AI**:防御建筑链(法师塔/地雷×5/
  喷火器/激光炮,互斥顺延);投石车/飞艇/龙入兵营「最少已解锁兵种」选型
  (cap 99/1/1);登艇/空降 scripted 不使用(状态手术单测覆盖,AI 层战术
  编排留 v2 RL)。
- 2026-07-26 **v1.6 用户复核修订三条(用户在线拍板,覆盖此前 [AI-DRAFT])**:
  ①投石车/飞艇/龙**开升级线,上限 3 级**(推翻「不开线」决策;保号约束用
  N_LINES_LEGACY=8 冻结 v1.5 动作 id 公式解决,新研发 id 追加动作表尾部
  113-115,N_LINES 8→11、upgrades [P,11]);②**龙喷火伤建筑但打折**,
  折扣初值 50% [AI-DRAFT] 待 v1.7 复核(推翻「不伤建筑」解释);③地雷
  「炸了就没、重新部署」语义经用户确认与现实现一致(一次性自毁,cap 按
  存活数计,炸后名额释放),无改动。

## v1.7 数值平衡(2026-07-26)
- 2026-07-26 [AI-DRAFT] **v1.7 方法学**:scripted 自对弈对 seed 不敏感、镜像局先手
  主导 → 整局胜率无平衡信号;改用 explorations/exp_v17_duel.py 的手工无菌对决(用户
  定口径:①单位vs单位=原地接战交错摆位纯 stat 交换;②防御建筑=攻防局,同价该守住、
  ~2倍造价该被破;③water=矿同重)。首版脚手架经济学有系统偏差(20/22 误标超模),
  按用户口径重做后结论可信。全版复核:近战/哨塔/喷火/激光/攻城/龙对空/奶妈**均平衡,
  不动**;仅法师塔超弱、迫击炮超弱、龙火海太小三处偏离。[source: 20260726-v17-duel-v2]
  [source: 20260726-v17-duel-v3]
- 2026-07-26 **用户定案(在线拍板)**:①**法师塔 magetower_atk 14→20、magetower_period
  5→4**(dps 2.8→5.0)——现值攻防局 1× 就守不住,atk18 仍不够、atk20 起守住,用户要
  「atk20 且 dps 再高一点」故 period 降到 4。[source: 20260726-v17-tune-magetower-dps]
  ②**龙火海 dragon_breath_radius 2.5→4.5**——现值火海太小清不完等价地面波,用户要
  「给大点」;龙对纯地面无敌(步兵打不到空军)故放大只加快清场不失衡,地面军须带
  防空反制。[source: 20260726-v17-tune]
  ③**迫击炮数值不动**——扫 13 种 config-only 候选(period/min_range/hp/atk/aoe 全试)
  全部守不住 1×,根因是机制(盲区 2.5 + 单发慢炮弹不预判 → 对移动步兵必然打空);
  用户定案接受迫击炮为**远程炮击/攻城支援建筑(非独立点防)**,不强行用数值救,记
  changelog 已知。[source: 20260726-v17-tune-mortar-aoe]
  ④**龙喷火对建筑 50% 折扣保留**(dragon_breath_bld_percent=50)——用户定位「龙不擅
  拆建筑」,接受廉价哨塔按射程反杀龙的克制关系。
- 2026-07-26 [AI-DRAFT] **训练营升级不补血差 bug 修复**(economy.py:自升级完成分支
  加 camp 补血,与哨塔/兵营同构)——挂 v1.4/v1.5/v1.6 四版的静默 bug,是 camp_hp_by_level
  数值复核前置;修后默认局决定论仍逐位一致(默认局无营升级)。

## v1.8 多风格指挥官 + 异界之门(2026-07-26)
- 2026-07-26 **用户在线拍板(AskUserQuestion)**:①**「必分胜负」＝异界之门 sudden-death**:
  `gate_open_tick` 后场地中央开门,同时向每个存活玩家出**阵营隔离**怪物(打 p 的怪只打 p、
  也只有 p 能打它);怪 **HP 无上限随超时线性增**、**攻击力有上限**、**移速慢**、**近战**、
  **强度生成时定死**(后出更强、已在场不变)、**各阵营压力一致**;**某阵营死则其怪离场**。
  效果:最弱基地先塌、最后存活者胜。②**主战场维持 4 人 FFA 64×64**(1v1 需新地图代码,
  不做)。③**v2.0＝调研+跑通但不训练的 JAX 骨架**(用 jax 自研到「空跑一步验证正确」,不真训)。
- 2026-07-26 [AI-DRAFT] **异界之门用 Approach A 独立怪物子表**(`monster_* [P,Mmax]` 进
  WorldState,不动 owner-by-row 主表)——阵营隔离天然成立(只在 owner==p 维度结算);经
  `plan-critic` 核对 combat/step/movement/cleanup 插入点均成立。
- 2026-07-26 [AI-DRAFT] **怪物战斗独立成 `monster_combat_tick` 阶段**(combat_tick 与
  cleanup_deaths 之间)——不能并入 combat_tick 的 incoming(该累加器 combat.py:192 已被
  hp=clip 消费);怪物战斗自成同帧子结算,写进 step.py 头注释。
- 2026-07-26 [AI-DRAFT] **胜负改造走最小改动**:保留 `episode_len` 为硬帽/scan 边界(零改
  run.py:129 与 make_scan、7 处引用不 rename),新增 `gate_open_tick`(<episode_len)触发门开;
  删 _end_tick 超时和局;overtime 在既有边界内跑到唯一 winner;episode_len 作防御硬帽+残血
  打分兜底(应因怪物升级不可达)。
- 2026-07-26 [AI-DRAFT] **怪物不参与单位互推碰撞**(只受 impassable 约束,沿目标玩家 HQ
  dist_field 慢速 descent 绕障)——决定论+简化。
- 2026-07-26 [AI-DRAFT] **指挥官策略参数走版本控制代码常量(StrategyProfile)而非 Config**
  ——AI 策略非平衡数字,复现锚点＝git hash+指挥官名+seed;共享宏观管线抽到
  `src/teow/commanders/macro.py` 去重。
- 2026-07-26 [AI-DRAFT] **FFA 目标选择(`attack_tgt` 状态字段)独立成 P2b、可延后**——
  A_ATTACK 硬编码最近敌 HQ(movement.py:116-123);核心 roster 先用默认最近敌 HQ 跑通,不阻塞。
- 2026-07-26 [AI-DRAFT] **v1.9 锦标赛脚手架 explorations 验证后提升进 `src/`**(供 v2.0 复用)。

### v1.8 收尾:engine-auditor 裁决(2026-07-27,用户离线按最合理解释)
- 2026-07-27 [AI-DRAFT] **v1.8 engine-auditor P0 清零**:决定论(含怪物子表逐位一致、seed 确进
  key)、阵营隔离(双向×4 家)、必分胜负(winner 恒 ∈[0,3])、资源守恒、无僵尸/泄漏怪——全成立。
- 2026-07-27 [AI-DRAFT] **P1-1 裁决:FFA「选敌」v1.8 接受 nearest-only**。规格「可按最弱/最近/
  威胁最大等」的「可」按可选解读;按敌选择需 attack_tgt 状态字段=P2b(计划/changelog 已延后)。
  `target_mode` 字段保留但注释标明 v1.8 **inert**(base.py 未读),供 P2b 落地,避免「声明不生效」误导。
- 2026-07-27 [AI-DRAFT] **P2 当场修两处**(影响 v1.9 评测有效性/不变量):①`monster_combat_tick`
  对怪开火回写 `atk_cd`(gate.py:否则 period>1 建筑对怪每 tick 连发,虚高龟缩/攻城/空军抗怪力、
  扭曲 v1.9 平衡评测);②离场怪清 `monster_hp`(combat.py:保 monster_hp>0⟺alive)。其余 P2
  (攻击 slope 近惰性、喷火/地雷对怪零伤=spec 未定、boomer 上兵慢=v1.9 筛选范畴、run.py:154
  死和局打印分支无害)记 changelog 已知问题。
- 2026-07-27 [AI-DRAFT] **装 pytest-xdist(`uv pip install`)加速门禁**:异界之门令 step 编译变慢,
  串行全套 >40min(25min 只跑到 61%);`-n 8` 并行 ~8min(117 passed)。dev 工具非数值依赖,未写进
  pyproject(uv sync 会移除),v1.9+ 沿用。

### v1.9 评测 / 筛选(2026-07-27,用户离线)
- 2026-07-27 [AI-DRAFT] **v1.9 筛选结论:10 指挥官全留、按强度分层,不删任一**。综合评测
  (experiments/20260727-v19-roundrobin,result-analyst 分析):全 vs random 1.00、无统治风格
  (max rusher 0.75)、非退化(80 局无秒杀/无撞硬帽/无和局)、风格清晰二分。弱尾(turtle/timing/
  airtech round-robin 零胜)受**非均衡 eval 噪声**影响(出场 4–24 不等、4 seed 薄),且弱者风格仍
  独立→对「多样对手池」有价值,**不删**(用户「改不好才删」;此处非「改不好」而是「样本不足未定」)。
  难度分层供 v2.0 课程:HARD rusher / MEDIUM balanced·harasser·boomer·chaos·counter·tempo /
  EASY turtle·timing·airtech。
- 2026-07-27 [AI-DRAFT] **v1.9 followup(下个 session / v2.1 前)**:①**均衡** round-robin(每 2-
  指挥官对都覆盖、≥8 seed)复核弱尾真实强度;②定向调 turtle/airtech「0-军被动 gate 胜」(疑攻击
  阈值过高/兵种没上场,行为太被动,非引擎 bug);③22/80 局固定落 length4182(宏观局全靠 gate 结算)
  ——训练前确认别让 RL 学到「摆烂等门」。
- 2026-07-27 [AI-DRAFT] **评测脚手架提升进 `src/teow/eval.py`**(`matchup_runner` 批量 rollout 原语,
  v2.0 PPO rollout 复用;`explorations/eval_commanders_v18.py` 留作 CLI 包装 + 聚合)。

## v2.1 训练前准备(2026-07-27,用户睡眠期自主推进)
- 2026-07-27 **用户睡前在线拍板 v2.1 范围/深度**:①深度=**看到学习信号**(试训到 vs 弱对手胜率
  上升/行为不退化,不追求训强);②算力充足可通宵;③**对手池先洗干净**(v1.9 弱尾 followup 作训练
  前置);④**全长局训练**(episode 6000/gate 4000 不动),**明确否决短局**——短局诱导 RL 学退化策略
  (只龟缩耗死脚本一招,训不出完整战术,「短局训出来可能只会这一种方法」);⑤BC 暖启先不加。
  plan:`~/.claude/plans/v2-1-enchanted-quilt.md`。
- 2026-07-27 [AI-DRAFT] **骨架最小改动而非新建超集**:`rl_skeleton_v20` 的 RLConfig 加 7 个 v2.1
  训练字段、`ppo_loss` 加可选 `ent_coef` 参(默认用 rlcfg.ent_coef 常量→不破骨架 smoke;训练传退火
  traced 标量)。理由:避免 RLConfig/ppo_loss 第二真源;改动向后兼容。持续环境 rollout / 训练循环
  作**新文件** `explorations/rl_train_v21.py`(不覆盖骨架 collect_rollout,保 v2.0 smoke)。
- 2026-07-27 [AI-DRAFT] **训练结构 = Python 外循环 + 内层全 lax.scan;持续环境分段 rollout**:carry
  跨 update,`done→where(nstate.done, state0, nstate)` autoreset(打破 step done 冻结)+ elim 归零 +
  跨段 value bootstrap。理由:全长局 6000 一次 rollout 整局爆显存(整批 PPO 反向 logits[M,N,A] 单张
  45GB;分段 T=128 每段 ~0.55GB);Python 外循环便于 eval/ckpt/课程升档(换对手=重编译,升档罕见)。
  γ=0.999 有效视界 1000≪6000,长程 credit 起步靠 PBRS 承载,试训观察必要时调 γ(留 sweep)。
- 2026-07-27 [AI-DRAFT] **Phase A 均衡设计 = covering design + cyclic 座位轮转**:贪心 covering
  design 9 组合覆盖全 45 对指挥官,每组合按序号 cyclic 轮转座位消位置偏置;vs-random 加厚 16 seed。
  取代 v1.9 的 `combos[::step]` 稀疏采样(出场 4–24 不均)。`conftest.py` 加 explorations 到
  sys.path(slow test 测 explorations 训练管线;explorations 本身不在 ruff/门禁范围=沙箱)。
- 2026-07-27 [AI-DRAFT] **Phase A 座位偏置发现 → 修评测方法(全 P 座位轮转)**:均衡 rr 第一次跑
  (`20260727-v21-balanced-rr`,每 covering 组合仅 1 个 cyclic shift)暴露座位偏置——**seat0
  系统偏强**(各座位总胜场 38/8/14/12,全局 6000 下 v1.5 记的镜像位偏置仍在),且 index%4 的
  1-shift 使 **turtle/airtech/chaos 从没坐 seat0**,零胜含人为低估(非纯真弱)。修:每 covering
  组合跑**全 P 座位轮转**(每家在每座位各一次),消偏置重跑(`rr2`);弱尾真实强度以 rr2 定,再判
  是否定向调参 / 分层。方法学教训:FFA 评测**座位轮转必须全排列/全 cyclic**,单 shift 不够。
- 2026-07-28 [AI-DRAFT] **Phase D 试训发现 PBRS reward-hacking → 修(库存打折)**:vs random 首训
  (`20260728-v21-train-vsrandom`)学习失败——army=0 全程、贪心胜率 0.75→0.25、mean_reward≈0、
  pg≈0、loss 被熵项主导。诊断:Φ=`_invested_value` 含**全额库存**——造兵是「库存→单位」cost 守恒
  (Value 无上行)+单位会死(有下行),囤钱 Value 更稳 → RL 学「囤钱不造兵」(**plan §4.3 reward-hacking
  预警命中**)。修:骨架 `_invested_value`/`potential` 加 `stockpile_weight` 参(默认 1.0 保 v2.0
  smoke),RLConfig `stockpile_weight=0.3`(库存打折→造兵产 +0.7cost 上行、囤钱不涨 Φ);训练加
  `--pot-scale`/`--stockpile-weight` CLI,重训用 β 0.1→1.0、pot_scale 300→100 加强+敏感化信号。
  重训 `v21-train-fix1` 验证 army>0。**教训:PBRS 势函数含「未投入资源(库存)」会诱导囤积,势函数
  应只奖励「已转化为战力的投入」。**
- 2026-07-28 [AI-DRAFT] **v2.1 engine-auditor N/A + 收尾裁决**:src 引擎逻辑(step/combat/economy/
  gate/state/actions/movement/config)**未改**,v2.1 全部是 explorations 训练脚手架 + `commanders/
  profile.py` AI 策略数值常量定向调。引擎不变量不受影响(profile 改只让 AI 出不同动作),由 rr4(304 局
  非退化)+ 门禁(119 测试)覆盖 → engine-auditor 判 **N/A**(同 v2.0 先例)。
- 2026-07-28 [AI-DRAFT] **v2.1 收在「管线就绪+除险发现」而非「训出会打的指挥官」**:用户睡前定深度=
  「看到学习信号」,但试训除险发现**纯 PPO 冷启动学不动、需 BC 暖启**(用户 v2.1 明确暂缓 BC)。故
  **未自主实现 BC**(尊重用户「v2.1 不加 BC」+ BC 是正式开训方案应由用户拍板;离线协议:重大不确定
  不自主决策)。issue.md v2.1 核心目标(训练前除险、别训出什么都不会的)**达成**——小范围试训阶段就
  发现纯 PPO 会训废 + 定位根因(reward-hacking 已修 / 冷启动需 BC),避免正式开训才踩坑。BC 暖启 +
  真看到学习信号 = 正式开训(v2.2+),留用户决定。
