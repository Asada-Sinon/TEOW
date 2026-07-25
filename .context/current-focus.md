# 当前焦点

## 当前目标

**v1.4 已收官(tag v1.4,五件套齐,终审 P0/P1 零)**。当前:v1.5 实现——
四人泛化(P1,先保 2 人默认绿)→ 六边形地图(P2,翻 4 人/64×64/20 点)→
栅栏(P3)→ 前端四色+蓝方贴图(P4)→ 四人局收口(P5)。
计划:docs/plans/20260726-v15-hex-4p/plan.md(critic 2 BLOCKER 已吸收)。

## 为什么做

用户睡前下达通宵任务:明早要看到 v1.4/v1.5/v1.6 全部收官并 push GitHub。
v1.4 已完成;v1.5 进行中;v1.6(防御建筑群/空中域)排后。

## 完成判据

每版五件套:pytest+ruff 真实全绿、engine-auditor P0 清零、changelog、
git tag+push、handoff 落盘。

## 不做什么

v1.7(数值调整)不碰;新数值一律 [AI-DRAFT] 初值记 DECISIONS 等 v1.7 复核;
不确定的设计先派 agent 调研再决策(离线协议)。
