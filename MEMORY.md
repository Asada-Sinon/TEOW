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
