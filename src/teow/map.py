"""静态地图:通行格、资源点布局、HQ 位置、BFS 距离场。

全部在 jit 外用 numpy 构建一次,连同距离场一起闭包进 build_step(不进 state、
不进 scan carry——照 alicization terrain 与调研报告 §5.5 的纪律)。

寻路方案(docs/plans/20260725-jax-rts-engine/research.md §2):目标点集固定且少
(n_nodes 个资源点 + 2 个 HQ),对每个目标做一次 BFS 得 dist_fields[G,H,W];
单位每 tick 对 4 邻 + 原地做 masked argmin 下降,即经典 flow-field 寻路,
对静态障碍是精确最短路,被挡时自动绕行。裸贪心会在凹障碍卡死工人,不用。

对称性:地图在 180° 旋转(r,c)->(H-1-r,W-1-c) 下 (位置,类型) 集合自映射——
近家点与双角公共点的旋转像类型均保持不变,严格镜像公平(v1.3 起公共点为
左下 1矿1水 + 右上其旋转像;v1.0-v1.2「中央公共点旋转换类型」的取舍已退役)。
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from .config import RES_ORE, RES_WATER, Config


class MapData(NamedTuple):
    """静态地图数据(numpy,构建后只读)。goal 编号:0..n_nodes-1 = 资源点,
    n_nodes+p = 玩家 p 的 HQ。"""

    passable: np.ndarray     # bool [H,W]  资源点格与 HQ 格不可通行
    node_pos: np.ndarray     # int32 [Nn,2]
    node_type: np.ndarray    # int32 [Nn]  RES_ORE / RES_WATER
    hq_pos: np.ndarray       # int32 [2,2]
    spawn_pos: np.ndarray    # int32 [2,start_workers,2]  初始工人站位
    dist_fields: np.ndarray  # int32 [Nn+2,H,W]  到各静态 goal 的 BFS 步数;
    #                          不可达=BIG_DIST。旗的动态通道不在此(movement 拼接)
    goal_seeds: np.ndarray   # bool  [Nn+2,H,W]  静态 goal 种子格(movement 动态场用)

    @property
    def n_nodes(self) -> int:
        return int(self.node_pos.shape[0])


BIG_DIST = 10**6  # 不可达哨兵;masked argmin 的 BIG 惩罚也用它


def _rot(rc: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    """180° 旋转。玩家 1 的所有布局 = 玩家 0 布局的旋转像。"""
    return (h - 1 - rc[0], w - 1 - rc[1])


def _bfs_field(passable: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    """从 src 出发经可通行格的 4 邻 BFS 距离场。src 本身不可通行也置 0
    (单位的目标是走到 dist==1 的相邻格,而不是踩上目标)。"""
    h, w = passable.shape
    dist = np.full((h, w), BIG_DIST, dtype=np.int32)
    dist[src] = 0
    q: deque[tuple[int, int]] = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] == BIG_DIST:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def build_map(cfg: Config) -> MapData:
    """确定性布局(只由 cfg 决定,不吃随机):

    - HQ0 (3,3),HQ1 = 旋转像 (H-4,W-4)。
    - 每家附近 1 矿 1 水(距 HQ 约 5-6 格,「靠近但不贴脸」,留出防守纵深);
    - 公共点左下 1 矿 1 水,右上为其旋转像(类型不变),构成抢点/扩张博弈(v1.3)。
    布局按 24×24 标定;其他尺寸按比例缩放并 clip 到界内。
    """
    h, w = cfg.grid_h, cfg.grid_w
    if h < 12 or w < 12:
        raise ValueError(f"grid 至少 12×12,收到 {h}×{w}")

    def scale(rc: tuple[int, int]) -> tuple[int, int]:
        r = min(max(round(rc[0] * h / 24), 0), h - 1)
        c = min(max(round(rc[1] * w / 24), 0), w - 1)
        return (r, c)

    hq0 = scale((3, 3))
    hq1 = _rot(hq0, h, w)

    # 玩家 0 的近家点 + 左下公共对;玩家 1 侧/右上取旋转像(类型一律不变,
    # (位置,类型) 集合在 180° 旋转下自映射,严格公平——见文件头「对称性」)
    ore0 = scale((2, 8))
    water0 = scale((8, 2))
    publ_ore = scale((17, 4))    # 左下公共矿(v1.3 双角布局,初值可调)
    publ_water = scale((20, 7))  # 左下公共水
    node_pos_list = [ore0, water0, _rot(ore0, h, w), _rot(water0, h, w),
                     publ_ore, publ_water,
                     _rot(publ_ore, h, w), _rot(publ_water, h, w)]
    node_type_list = [RES_ORE, RES_WATER, RES_ORE, RES_WATER,
                      RES_ORE, RES_WATER, RES_ORE, RES_WATER]
    if cfg.n_nodes != len(node_pos_list):
        raise ValueError(f"v1.3 布局固定 8 个资源点,cfg.n_nodes={cfg.n_nodes} 不支持")

    node_pos = np.asarray(node_pos_list, dtype=np.int32)
    node_type = np.asarray(node_type_list, dtype=np.int32)
    hq_pos = np.asarray([hq0, hq1], dtype=np.int32)

    # 唯一性检查:任何两个占位(HQ/资源点)不得重叠
    occupied = {tuple(p) for p in node_pos_list} | {hq0, hq1}
    if len(occupied) != len(node_pos_list) + 2:
        raise ValueError("地图布局重叠:HQ 或资源点撞格")

    passable = np.ones((h, w), dtype=bool)
    for r, c in node_pos_list:
        passable[r, c] = False
    passable[hq0] = False
    passable[hq1] = False

    # 初始工人:HQ 东侧一列可通行格(不够则绕 8 邻找),两家旋转对称
    def worker_spawns(hq: tuple[int, int]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        candidates = [(hq[0] + dr, hq[1] + dc)
                      for dc in (1, 2) for dr in (-1, 0, 1, 2)]
        for rc in candidates:
            if 0 <= rc[0] < h and 0 <= rc[1] < w and passable[rc] and rc not in out:
                out.append(rc)
            if len(out) == cfg.start_workers:
                return out
        raise ValueError("HQ 周围可通行格不足以放下初始工人")

    sp0 = worker_spawns(hq0)
    sp1 = [_rot(rc, h, w) for rc in sp0]  # 严格旋转像,保证两家出生位对称
    for rc in sp1:
        if not passable[rc]:
            raise ValueError("玩家 1 工人出生位不可通行(布局 bug)")
    spawn_pos = np.asarray([sp0, sp1], dtype=np.int32)

    # 距离场:每个资源点一张 + 每个 HQ 一张
    goals = node_pos_list + [hq0, hq1]
    fields = [_bfs_field(passable, tuple(p)) for p in goals]
    dist_fields = np.stack(fields).astype(np.int32)
    goal_seeds = np.zeros((len(goals), h, w), dtype=bool)
    for g, rc in enumerate(goals):
        goal_seeds[g][tuple(rc)] = True

    return MapData(passable=passable, node_pos=node_pos, node_type=node_type,
                   hq_pos=hq_pos, spawn_pos=spawn_pos, dist_fields=dist_fields,
                   goal_seeds=goal_seeds)
