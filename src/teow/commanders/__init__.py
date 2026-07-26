"""多风格脚本指挥官(v1.8):共享宏观管线 + StrategyProfile 参数化策略层。

设计:一个 branchless-JAX 的参数化指挥官 `commander_actions(st,cfg,mapdata,owner,player,
key,profile)`(base.py)+ 若干 StrategyProfile(profile.py)。profile 是**静态闭包**
(与 player 一样 bake 进 functools.partial),故对 profile 字段的 Python 分支在 trace 期
解析(不产生 traced 控制流),只有依赖 state 的逻辑用 jnp.where —— 这让「多风格」既丰富又
保持可进 scan/vmap。指挥官策略参数是版本控制代码常量(非平衡数字,不进 config;复现锚点
= git hash + 指挥官名 + seed)。
"""

from .base import commander_actions
from .profile import PROFILES, StrategyProfile

__all__ = ["PROFILES", "StrategyProfile", "commander_actions"]
