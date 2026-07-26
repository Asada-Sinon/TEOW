# A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)

- **citekey**: ross2011dagger
- **arXiv**: 1011.0686 (AISTATS 2011)
- **作者/年**: Ross, Gordon, Bagnell — 2011 (CMU)
- **状态**: 精读(摘要+已知内容)

## 问题
朴素行为克隆(BC)有**协变量漂移**:学生犯的小错把它带到专家没访问过的状态,误差
按时间平方累积(compounding error)。

## 核心做法(paper says)
DAgger:迭代地让**当前学生策略**去跑、收集它实际访问到的状态,再由**专家**在这些状态
上标注正确动作,聚合进数据集重训。把模仿学习归约为 no-regret 在线学习,给出线性(而非
平方)误差界。

## 实验/结论(paper says)
在 Super Tux Kart 转向、超级马里奥上显著优于 BC。

## 局限 [AI-DRAFT]
- 需要专家**可在任意查询状态给出动作**(interactive expert)。

## 与本项目的关系 [AI-DRAFT]
**本项目的专家是脚本指挥官,天生可查询任意 state**——`controller(state,player,key)`
就是随叫随到的 oracle。这消除了 DAgger 最大的落地障碍(人类专家难以在线标注)。
因此 DAgger 在这里比在多数项目里更可行:可用它把脚本策略蒸馏进神经网络,得到无漂移的
BC 暖启动权重,再交给 PPO 继续 RL。是「用脚本」路线里最务实的一环。
