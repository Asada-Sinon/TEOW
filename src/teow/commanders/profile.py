"""StrategyProfile:多风格指挥官的策略参数(版本控制代码常量,非 config 平衡数字)。

profile 是静态闭包(bake 进 make_controller 的 partial),故 base.commander_actions 里对
profile 字段的 Python 分支在 trace 期解析——可以用丰富的 Python if/elif 选策略,只有依赖
state 的量用 jnp。字段语义即 base.py 的实现契约(见每字段注释)。

数值均 [AI-DRAFT] 起步,P4/v1.9 校准/筛选后定。
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class StrategyProfile:
    name: str
    # ---- 经济/生产(泛化现 cfg.ai_* 四旋钮;base 用 profile 值覆盖 cfg 默认)----
    worker_target: int = 8        # 采集单位低于此数则 HQ 优先补(经济投入)
    attack_threshold: int = 6     # 战斗单位攒到此数发起总攻(越低越 rush)
    base_level_target: int = 3    # 升本目标等级(越高越偏科技/后期)
    upgrade_reserve: int = 40     # 升级/研发预备金(越高越谨慎/留军费)
    # ---- 科技路线:影响「建造征调顺序」与升本积极性 ----
    #   balanced 现默认链;military 早出兵营/多兵;air 冲兵营高级(飞艇/龙);
    #   defense 优先防御建筑(塔/迫击炮/法师塔/喷火/激光);economy 多矿泵少军。
    tech_focus: str = "balanced"
    # ---- 兵种构成偏好:影响兵营选型的 tie_bias ----
    #   balanced 现默认;dogs 偏狗/轻骑(骚扰);ranged 偏弓/法(风筝);
    #   heavy 偏重甲/近战(硬)；air 偏飞艇/龙;siege 偏攻城/投石(拆家);
    #   counter 动态最佳响应——按观测到的敌方最强兵种选克制兵种。
    comp_bias: str = "balanced"
    # ---- 进攻风格 ----
    #   rush 早压;timing 攒到配比一波;defensive 守+反击;harass 少量早扰;allin 倾家一波。
    aggression: str = "timing"
    # ---- FFA 选敌:**P2b 延后,v1.8 未接线**——base.py 尚不读本字段,全部 attack-move 打
    #   最近敌 HQ(movement.py:116-123 硬编码);harasser/counter 声明 weakest 仅为 P2b 预留
    #   意图,待 attack_tgt 状态字段落地才生效(审计 P1-1,用户离线按「可」=可选解读接受)----
    target_mode: str = "nearest"  # nearest / weakest / leader(v1.8 inert)
    # ---- 自适应回退(反「无意义行为」):主战术失败(军队被清空且已过窗口)→ 转宏观运营 ----
    adaptive: bool = True
    # ---- 概率性:>0 时用线程 key 随机化部分 build 选择(制造对局多样性)----
    stochastic: float = 0.0


# 复现锚点 = git hash + 指挥官名 + seed。名字即 make_controller 的 dispatch key。
PROFILES: dict[str, StrategyProfile] = {
    # 默认平衡(= 现 cfg.ai_* 值;"scripted" 别名保持向后兼容,现有测试/CLI 不破)
    "balanced": StrategyProfile("balanced"),
    "scripted": StrategyProfile("scripted"),  # 与 balanced 同参(别名)

    # 种田流:重经济、晚而强的一波;高预备金稳健(v2.1 试 attack_threshold 16→11/reserve 80→45
    # 过头——种田流早出弱兵送死,vsRandom 胜率 1.00→0.88;回退原值,末军 0.9=攒兵到门是风格
    # 非极端被动,用户只点名 turtle/airtech [source: 20260728-v21-balanced-rr3])
    "boomer": StrategyProfile(
        "boomer", worker_target=12, attack_threshold=16, base_level_target=6,
        upgrade_reserve=80, tech_focus="economy", comp_bias="heavy",
        aggression="timing"),

    # 爆狗偷家 rush:极少工人、极低阈值早压;失败自适应转型(反无意义空转)
    "rusher": StrategyProfile(
        "rusher", worker_target=5, attack_threshold=3, base_level_target=2,
        upgrade_reserve=15, tech_focus="military", comp_bias="dogs",
        aggression="rush", adaptive=True),

    # 龟缩防守:堆防御建筑、守+反击(异界之门利好防守型);v2.1 定向调:降 attack_threshold
    # 22→10 + base_level_target 7→6 + upgrade_reserve 100→45,腾钱造兵让「守+反击」真有反击,
    # 不再全投升本囤钱→0 军等门(保留 defense/ranged 龟缩风格)[source: 20260728-v21-balanced-rr2]
    "turtle": StrategyProfile(
        "turtle", worker_target=9, attack_threshold=10, base_level_target=6,
        upgrade_reserve=45, tech_focus="defense", comp_bias="ranged",
        aggression="defensive"),

    # 一波流 timing:攒到中期配比再全军突
    "timing": StrategyProfile(
        "timing", worker_target=8, attack_threshold=13, base_level_target=4,
        upgrade_reserve=50, tech_focus="military", comp_bias="heavy",
        aggression="timing"),

    # 骚扰流:少量快速单位早扰敌经济;低阈值多波
    "harasser": StrategyProfile(
        "harasser", worker_target=7, attack_threshold=4, base_level_target=3,
        upgrade_reserve=30, tech_focus="military", comp_bias="dogs",
        aggression="harass", target_mode="weakest", adaptive=True),

    # 空军科技:冲兵营高级,飞艇/龙制空;v2.1 定向调:降 base_level_target 7→6(飞艇兵营6门槛,
    # 早出中级空军)+ attack_threshold 10→8 + upgrade_reserve 70→45,让空军真上场作战不再等门
    # [source: 20260728-v21-balanced-rr2]
    "airtech": StrategyProfile(
        "airtech", worker_target=9, attack_threshold=8, base_level_target=6,
        upgrade_reserve=45, tech_focus="air", comp_bias="air",
        aggression="timing"),

    # 中期强攻 tempo:早于 timing 的持续压制,army-value 节奏
    "tempo": StrategyProfile(
        "tempo", worker_target=7, attack_threshold=7, base_level_target=4,
        upgrade_reserve=35, tech_focus="military", comp_bias="balanced",
        aggression="rush", adaptive=True),

    # 随机应变/博弈论:动态最佳响应——按敌方最强兵种选克制兵种构成
    "counter": StrategyProfile(
        "counter", worker_target=8, attack_threshold=11, base_level_target=5,
        upgrade_reserve=50, tech_focus="military", comp_bias="counter",
        target_mode="weakest", adaptive=True),

    # 概率混沌:随机化 build,制造多样对局(弱结构,靠随机性覆盖策略空间)
    "chaos": StrategyProfile(
        "chaos", worker_target=8, attack_threshold=8, base_level_target=4,
        upgrade_reserve=40, tech_focus="balanced", comp_bias="balanced",
        aggression="timing", stochastic=1.0, adaptive=True),
}
