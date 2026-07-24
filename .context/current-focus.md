# 当前焦点

## 当前目标

v1.0 引擎核心:纯 JAX 实现两玩家 2D 网格 RTS(采集一体循环/建造训练/战斗/胜负),
scripted vs scripted 能跑完整局并通过无上下文 agent 审核。

## 为什么做

v1 全系列先把引擎做扎实(用户决策,RL 推迟到 v2+);v1.0 是一切后续版本的地基。

## 完成判据

`python3 src/run.py play --p0 scripted --p1 scripted --seed 0` 跑完整局打印 winner;
pytest/ruff 全绿;engine-auditor 审核通过;docs/changelog/v1.0.md 落盘;git tag v1.0。

## 不做什么

RL/PPO(v2+)、升级系统(v1.1)、兵营/哨塔/狗子(v1.2)、浏览器渲染(v1.2)、战争迷雾(v3)、
A* 寻路(v1.0 用贪心一步)。
