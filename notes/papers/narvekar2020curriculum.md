# Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey

- **citekey**: narvekar2020curriculum
- **arXiv**: 2003.04960 (JMLR 21, 2020)
- **作者/年**: Narvekar, Peng, Leonetti, Sinapov, Taylor, Stone — 2020
- **状态**: 略读(摘要+已知内容)

## 问题
把「课程学习」(由易到难地排列任务/样本)系统化:如何为 RL 生成、排序、迁移课程。

## 核心做法(paper says)
提出一个统一框架,把课程学习拆成三要素:**任务生成、任务排序(sequencing)、
知识迁移**;据此对现有 CL 方法按假设/能力/目标分类综述。

## 实验/结论(paper says)
综述性质:归纳出「手工课程 / 基于学习进度自动排序 / teacher-student」等范式及其权衡,
指出自动课程与迁移是开放难点。

## 局限 [AI-DRAFT]
- 2020 年综述,不含近年 PLR/无监督环境设计(UED)等自动课程新进展。

## 与本项目的关系 [AI-DRAFT]
回应「训练对手是开局易还是上难度曲线」:本项目有 v1.9 按难度**分档**的脚本指挥官,
天然是一套**手工课程的任务序列**。综述支持的做法:先对最弱档拿到稳定胜率(奠定基本
操作,避免稀疏奖励下从零学不动),再按 agent 学习进度(如对当前档胜率过阈值)自动
升档,而非固定时间表。这比「一上来就 self-play / 最难对手」更稳。是「易→难曲线」而非
「全程最难」的文献支撑。
