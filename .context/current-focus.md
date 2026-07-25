# 当前焦点

## 当前目标

**v1.5 已收官(tag v1.5)**。当前:v1.6 实现(最后一版)——
空中域+投石车(P1)→ 四防御建筑(P2)→ 飞艇容器(P3)→ 龙骑兵(P4)→
收尾定标(P5)。
计划:docs/plans/20260726-v16-air-defense/plan.md(critic 4 MAJOR 已吸收)。

## 为什么做

用户睡前下达通宵任务:明早要看到 v1.4/v1.5/v1.6 全部收官并 push GitHub。
v1.4/v1.5 已收官;v1.6 进行中,完成即全部交付。

## 完成判据

每版五件套:pytest+ruff 真实全绿、engine-auditor P0 清零、changelog、
git tag+push、handoff 落盘。

## 不做什么

v1.7(数值调整)不碰;新数值一律 [AI-DRAFT] 初值记 DECISIONS 等 v1.7 复核;
不确定的设计先派 agent 调研再决策(离线协议)。
