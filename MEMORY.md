# MEMORY

**本文件当前是空模板，还没有任何真实教训。** 下面只有格式说明和一段被注释掉的示例。

累积教训。和 `HANDOFF.md` 的分工：
**HANDOFF 是会过期的当前状态，MEMORY 是不会过期的教训。**
「dataloader 改到一半」属于 HANDOFF；「这台机器上 num_workers>4 会静默丢样本」属于 MEMORY。

规矩：
- 累积式，**新条目追加在最后面**。
- 一条一组，用 `### [LEARN:tag]` 起头。tag 是自己起的短分类（env / data / api / perf / tooling …）。
- 写入时机：踩坑并真正定位到原因之后。发现旧条目是错的，就地改掉或删掉，不要叠加。

格式：

```markdown
### [LEARN:tag] 一句话标题
- 现象: 当时看到了什么
- 原因: 真实原因（查证过的，不是猜的）
- 对策: 下次怎么做
- 来源: Session YYYY-MM-DD
```

**禁止为了填表而编造条目。空着比编造便宜——一条假教训会被后面每一个 agent 当真。**
没踩到坑就是没踩到坑，这个文件长期只有两三条是完全正常的。

<!-- 示例（安装后请删除这整块）
以下为格式示例，不是本项目的真实教训。现象、原因、文件名、日期全部虚构，
任何 agent 都不得把它们当作本项目已知的坑或既定结论。

### [LEARN:env] 裸 `python` 指向系统解释器，缺依赖
- 现象: `python src/train.py` 报 `ModuleNotFoundError: torch`，但 `pytest` 能跑。
- 原因: shell 没激活 venv；pytest 是从 venv 里以绝对路径调用的，所以看着正常。
- 对策: 一律用 CLAUDE.md 命令区里那个解释器的绝对路径，不依赖当前 shell 状态。
- 来源: Session 2026-03-14
-->

---

<!-- 真实的 [LEARN:tag] 条目从这一行下面开始，新的追加在最后。 -->

### [LEARN:env] 测试与门禁必须 JAX_PLATFORMS=cpu,训练/对局才用 GPU
- 现象: 同一套 14 项测试 GPU 跑 5min18s,CPU 只要 14s;单环境对局 GPU 27 tick/s
  vs CPU 800+ tick/s。
- 原因: 每个测试用不同 Config → 每个都触发整套 step 的 GPU jit 重编译;单环境
  逐 tick 调用时 GPU kernel launch 开销占主导。GPU 优势要 vmap 批量才体现。
- 对策: verify.sh 与 pytest 一律 `JAX_PLATFORMS=cpu`;v2 vmap rollout 前先 bench。
- 来源: Session 2026-07-25

### [LEARN:tooling] npz 是惰性解压,循环里反复索引 data[k] 会平方级卡死
- 现象: 回放服务 load 1359 帧卡 148 秒、100% CPU,像死循环。
- 原因: np.load(npz) 返回惰性对象,每次 `data["k"]` 都完整解压该数组;
  帧循环内逐元素访问 → O(帧×实体×全量解压)。
- 对策: 用前一次性物化 `{k: raw[k] for k in raw.files}`(148s → 0.1s)。
- 来源: Session 2026-07-25

### [LEARN:tooling] future annotations 下 FastAPI 依赖类型必须模块级导入
- 现象: WebSocket 路由握手一律 403,HTTP 路由正常,最小复现却通过。
- 原因: `from __future__ import annotations` 把注解变字符串,FastAPI 用
  get_type_hints 从模块 globals 解析;`WebSocket` 在函数内局部导入 → 解析
  失败 → 被当成不可满足的依赖直接拒 403。
- 对策: fastapi 类型模块级导入;或该文件不要开 future annotations。
- 来源: Session 2026-07-25

### [LEARN:engine] 网格 RTS 里「静态最短路 + 严格改善 + 不许穿人」三件套必死锁
- 现象: 三次不同形态的经济全冻结:站桩工人堵死唯一下坡格、僵尸建造工占矿入口、
  对向工人流对头互堵(审计 P0-1,最隐蔽,要 300+ tick 才显形)。
- 原因: 寻路场不认识单位就没法绕;指令没有失效清理就会永久站桩;移动单位互不
  让路时严格改善门禁止一切侧移。
- 对策: 动态场(静止单位=硬障碍,移动单位=软障碍 congestion_cost)+ 指令失效
  自动转 IDLE + 落地/让路规避卸货环。改移动逻辑后必须跑 300+ tick 的完整对局
  验证,单元测试测不出这类涌现死锁——无上下文审计 agent 抓出了主 context
  自查漏掉的那个。
- 来源: Session 2026-07-25

### [LEARN:tooling] 用户在线期间 commit 禁用 git add -A
- 现象: v1.3 收尾 commit 用 `git add -A`,把用户刚放进工作区的 fig/ 17 张
  贴图和他正在编辑的 issue.md 草稿中间态一并提交推送了。
- 原因: 用户在线时工作区不是 agent 独占的——用户随时往里放素材、改草稿;
  add -A 把「我验证过的改动」和「用户未定稿的东西」混进同一个 commit,
  且 push 后不可改历史。本次 fig/ 事后被草稿证实是要用的素材,纯属侥幸。
- 对策: commit 一律点名文件清单;凡 status 里有非本轮任务产生的路径,
  先问或先绕开,不许顺手带走。
- 来源: Session 2026-07-25

### [LEARN:tooling] 门禁「超时」和「缺环境」都不是拦截,是静默放行
- 现象: `.venv` 整个不见的那段时间,`.claude/verify.sh` 每轮退出码 0、零输出,
  一路报绿;而环境修好后,全套 119 个测试 `-n 8` 实测 1158s。
- 原因: 两条独立路径都通向 exit 0。① 旧 verify.sh 的 `[ -d "$VENV_BIN" ]` 判假时
  直接跳过整个检查块,`fail` 保持 0 → 缺环境被当成通过;② `verify_stop.py:72-77`
  捕获 `TimeoutExpired` 后是 `exit 0` 放行(TIMEOUT=300s,只有全套耗时的 1/3.9)
  → 排一个必定超时的检查,等于每轮假绿灯,比不排更危险。
- 对策: 门禁必须「缺环境即 fail」而非跳过;进门禁的检查先实测耗时并留余量
  (现子集实测 117s / 预算 300s)。增删子集前实测,别凭感觉——单个引擎测试
  35-119s 且同测试两轮能飘 40%,加错一个文件就推回必定超时区。
- 来源: Session 2026-08-02

### [LEARN:rl] PBRS 势函数含「未投入的库存」会诱导囤积
- 现象: vs random 首训学习失败——army 全程 0、贪心胜率 0.75→0.25、mean_reward≈0、
  pg≈0,loss 被熵项主导。
- 原因: Φ=投入价值里含**全额库存**。造兵是「库存→单位」的 cost 守恒(Φ 没有上行)
  而单位会死(有下行),于是囤钱的 Value 更稳 → 策略学会不造兵。
- 对策: 势函数只奖励「已转化为战力的投入」,库存打折
  (`explorations/rl_skeleton_v20.py:321` `stockpile_weight`,训练用 0.3;默认 1.0
  保 v2.0 smoke 向后兼容)。设计任何塑形项时先问一句:「什么都不做能不能拿到这个分」。
- 来源: Session 2026-07-28(机制已在代码中查证;现象数字依 research-log 记录——
  实验产物 `20260728-v21-train-vsrandom` 已缺失,未能独立复现)

### [LEARN:rl] 纯 PPO 从随机网冷启动学不到结构化动作,加大塑形救不了
- 现象: 三次训练 + 全谱调参(β / pot_scale / stockpile / ent_coef),army 恒 0、
  结构化动作频率→0.000、行为退化成 NOOP/STOP/MOVE。
- 原因: 长局(6000 tick)稀疏延迟奖励下,随机策略几乎采样不到「造兵/采集」这类
  多步组合动作 → PPO 始终没有正样本可强化;塑形只是把零信号放大,不解决探索。
- 对策: 正式开训必须先 BC 暖启(脚本 oracle 蒸馏),让策略从「会造兵」起步,再用
  PPO 微调超越。别指望调参救冷启动。
- 来源: Session 2026-07-28(产物 `20260728-v21-train-fix1/fix2` 已缺失,依 research-log 记录)

### [LEARN:eval] 四人 FFA 评测的座位轮转必须做全排列
- 现象: 均衡 round-robin 第一版跑出 turtle/airtech/chaos 零胜,复核发现这三家
  **从没坐过 seat0**,而各座位总胜场是 38/8/14/12(seat0 系统性偏强)。
- 原因: 用 `index % 4` 做单次 cyclic shift 时,座位与对手组合被绑死,座位偏置
  和真实强度不可分离,弱尾结论里混进了人为低估。
- 对策: 每个对手组合跑**全 P 座位轮转**(每家在每个座位各一次)。位置偏置在全局
  6000 tick 下依然存在(v1.5 就记过镜像位偏弱),不能假设它会被平均掉。
- 来源: Session 2026-07-28(产物已缺失,依 research-log / DECISIONS 记录)
