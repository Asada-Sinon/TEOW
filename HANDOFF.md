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

## Session 2026-07-25(深夜,v1.3 收尾)
- 完成: v1.3 五件套收官打 tag——哨塔定案 tower_atk L1 6→3(c99e03f,用户授权
  agent 决策,依据 experiments/20260725-tower-balance-*);/validate 零必须修;
  两轮终审:P0 零,P1-1 名额仲裁竞态(改派被拒+空位被抢 → 持续 cap+K)修于
  3f255b0(HARVEST 改派旧名额「新指派成功才释放」)并复审关闭,P2 三条进
  changelog;changelog+收尾 8d8e709;终门禁 50 测试+ruff 绿;审计对局
  experiments/20260725-v1.3-audit{,2}/ 决定论逐位一致、12 项不变量全零。
- PENDING: ①用户还在扩写 issue.md 草稿箱(v1.4 兵种树/多塔/迫击炮/飞艇/
  龙骑兵,v1.5 六边形四人图+栅栏,v1.6 防御建筑群;工作区 issue.md 故意留脏
  没提交)——第一件事:读草稿箱走草稿协议吃透 v1.4,注意用户明写「你看这个
  数值怎么样」= 数值要讨论不要自定,且要求维护「以建筑为标题的几级爆什么兵」
  中文细则总结;②用户复核 DECISIONS 新三条 [AI-DRAFT](哨塔定案/P1-1 修/
  收尾裁决);③fig/ 17 张 1024² 贴图已入库,用户草稿明确「只有蓝方贴图,先
  应用到蓝方,其他用矢量图」,v1.4 接 web/assets 替换槽时处理。
- 坑: 收尾期用户可能同时在线编辑 issue.md——commit 一律点名文件,不要
  git add -A(本次把 fig/ 和用户中途的草稿改动一并带进过 commit,事后才由
  草稿证实合意,属侥幸)。
