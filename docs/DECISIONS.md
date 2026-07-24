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
- 2026-07-25 [AI-DRAFT] **引擎设计采纳 jax-rts-engine 调研的四项修正**(报告见
  docs/plans/20260725-jax-rts-engine/research.md):①实体表从 [2,E] 双player轴改为
  **单表 [N=2*E_max] + owner 列**(HQ 槽固定 0 与 E_max);②抢格仲裁用
  scatter-min + 每 tick 随机优先级 + 目标格 tick 初须空(防 self-play 下标偏置);
  ③寻路弃裸贪心,改 **jit 外 numpy BFS 预计算距离场**(各资源点+各HQ)闭包进
  build_step,单步 masked-argmin 下降(防凹障碍卡死工人毁掉经济信号);
  ④战斗照抄 SMAX:masked argmin 选邻接目标(全无效必须门控)+ .at[].add 同时
  结算 + 先结算后翻 alive(允许同归于尽)。
