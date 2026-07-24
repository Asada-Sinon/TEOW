"""TEOW:纯 JAX 的类星际 2D 网格 RTS 引擎(v1.x 引擎,v2+ RL 指挥官)。"""

from .config import Config
from .map import MapData, build_map
from .state import WorldState, init_state
from .step import build_step, make_scan, new_world

__all__ = ["Config", "MapData", "WorldState", "build_map", "build_step",
           "init_state", "make_scan", "new_world"]
