# AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

- **citekey**: peng2021amp
- **arXiv**: 2104.02180 ; **DOI**: 10.1145/3450626.3459670 (ACM TOG / SIGGRAPH 2021)
- **作者/年**: Peng, Ma, Abbeel, Levine, Kanazawa — 2021 (UC Berkeley)
- **状态**: 精读(摘要+已知内容)

## 问题(注意:这是物理动画领域,不是 RTS)
让物理仿真的角色(人形/机器人)既完成任务(task reward)又动作**自然/有风格**,
且不必手工设计模仿的动作序列或高层规划器。

## 核心做法(paper says)
奖励 = task reward + **style reward**。style reward 来自一个**判别器**(GAN 式):
判别器学着区分「参考动作数据集里的 state 转移」与「策略产生的 state 转移」,
判别器给策略的打分(least-squares,数据=+1、策略=−1)作为 style 奖励项,
用 PPO 优化。等价于对**非结构化动作数据集**做对抗式模仿(GAIL 的连续控制/动画特化版),
技能组合自动涌现,无需 motion planner。

## 实验/结论(paper says)
在物理角色上产出高质量、可组合的运动,质量媲美 tracking-based 方法,却能吃大规模
无标注动作片段。

## 局限 / 对本项目的适配性 [AI-DRAFT]（重点,回应用户「PPO 还是 AMP」）
- AMP 的**本体是连续控制的物理角色动画**:style 定义在低层物理 state 转移的「像不像」。
- 本项目是**离散动作 RTS 指挥官**,不存在「动作自然度/风格」这一物理量;把 116 维离散
  指令流塞进 AMP 的判别器,得到的是「像不像脚本指挥官的战术分布」——那**本质就是 GAIL/
  对抗式模仿**(见 ho2016gail),AMP 相对 GAIL 的动画特化增益在这里几乎为零。
- 因此判断:用户说的「AMP」大概率是想问「**能否用对抗式方法模仿脚本指挥官**」。答案是
  「可以,但那条路的正牌代表是 GAIL,不是 AMP」。AMP 的可迁移内核 = 「判别器风格奖励
  与 task 奖励相加,一起用 PPO 训」——这个**组合思路**可借鉴(RTS 版:胜负 task 奖励 +
  「打得像高质量脚本」的判别器奖励),但不要照搬 AMP 的动画设定。

## 与本项目的关系 [AI-DRAFT]
定位为「对抗式模仿(GAIL 系)」的一个变体,不是独立候选。真正要评估的是 GAIL/BC/DAgger
这条「用脚本」的线,AMP 只贡献「style reward + task reward 混合」这一个可选设计元素。
