# Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five)

- **citekey**: berner2019openaifive
- **arXiv**: 1912.06680
- **作者/年**: Berner 等(OpenAI, 25 人)— 2019
- **状态**: 精读(摘要+已知内容)

## 问题
Dota 2:超长时程(数万步)、高维连续观测/大动作空间、需团队协作与长期信度分配。
能否用「简单算法 + 极大规模」拿下?

## 核心做法(paper says)
- 算法就是 **PPO**(带 GAE),**没有**用花哨的层次/模仿架构;
- 大规模分布式 self-play,batch ≈ 每 2 秒 200 万帧,训练约 10 个月,
  用「surgery」在网络/环境改动后继续训练;
- 靠自对弈从零涌现出复杂团队战术。

## 实验/结论(paper says)
2019-04 击败 Dota2 世界冠军 OG,首个在电竞击败世界冠军的 AI。证明**自对弈 + 规模化 PPO**
可在极难任务上达超人水平。

## 局限 [AI-DRAFT]
- 770±50 PFlops/s·days 级算力——单卡 4090 差好几个数量级,时间尺度不可比。
- 长时程靠「巨 batch + 高 γ + reward shaping」硬扛,样本效率极低。

## 与本项目的关系 [AI-DRAFT]
最强的「PPO + 自对弈就够用」证据点,支持把 PPO 定为 v2.0 首选。但也是**规模警示**:
本项目episode 数千 tick、稀疏终局奖励,若照搬「纯自对弈从零涌现」会因算力不足而学不动;
因此更应走「脚本对手暖启动 + 中间奖励塑形」的低算力路线,而非纯 from-scratch self-play。
