"""静态地图(v1.5:六边形四人布局):通行格、资源点、HQ、BFS 距离场。

全部在 jit 外用 numpy 构建一次,连同距离场一起闭包进 build_step(不进 state、
不进 scan carry)。

几何(v1.5,plan D4 + critic B-1):
- 方格网格上雕**平顶六边形**可行区:中心取奇数镜像轴 (cy,cx),
  |dr| ≤ b 且 |dc| ≤ a·(1 − |dr|/(2b));界外 passable=False。
- 对称群 = Klein 四群:σh: r→2cy−r,σv: c→2cx−c,ρ=σhσv(180°)。
  玩家 0(左上斜边中点附近)定义全部布局,玩家 1=σh 像(左下)、
  玩家 2=σv 像(右上)、玩家 3=ρ 像(右下);(位置,类型) 集合在三个操作下
  严格自映射(E/W 顶点的公共矿水放在不动行 r=cy 上,σv 把 E 对映到 W)。
- 资源点 20 个:每家家门 1矿1水(8)+ E/W 顶点各 1矿1水(4)+
  NE 轨道 1矿1水 的四元轨道(8)。编号:0..7 = 玩家 p 的 (矿,水)
  按 p 序(与 v1.3「0/1=玩家0 近家」惯例兼容),8..19 公共。

寻路:每目标一张 BFS 场(资源点 + P 个 HQ),单位下降场;旗的动态通道
在 movement 拼接。布局字面量属地图几何,不属平衡数值(map.py 惯例)。
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from .config import RES_ORE, RES_WATER, Config


class MapData(NamedTuple):
    """静态地图数据(numpy,构建后只读)。goal 编号:0..n_nodes-1 = 资源点,
    n_nodes+p = 玩家 p 的 HQ。"""

    passable: np.ndarray     # bool [H,W]  界外/资源点格/HQ 格不可通行
    node_pos: np.ndarray     # int32 [Nn,2]
    node_type: np.ndarray    # int32 [Nn]  RES_ORE / RES_WATER
    hq_pos: np.ndarray       # int32 [P,2]
    spawn_pos: np.ndarray    # int32 [P,start_workers,2]  初始工人站位
    dist_fields: np.ndarray  # int32 [Nn+P,H,W]  到各静态 goal 的 BFS 步数;
    #                          不可达=BIG_DIST。旗的动态通道不在此(movement 拼接)
    goal_seeds: np.ndarray   # bool  [Nn+P,H,W]  静态 goal 种子格

    @property
    def n_nodes(self) -> int:
        return int(self.node_pos.shape[0])


BIG_DIST = 10**6  # 不可达哨兵;masked argmin 的 BIG 惩罚也用它


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
    """确定性六边形四人布局(只由 cfg 决定,不吃随机)。要求 n_players=4、
    n_nodes=20、方形网格 ≥32(布局参数按网格尺寸比例推导)。"""
    h, w = cfg.grid_h, cfg.grid_w
    if h != w or h < 32:
        raise ValueError(f"v1.5 六边形布局要求方形网格且 ≥32×32,收到 {h}×{w}")
    if cfg.n_players != 4:
        raise ValueError(f"v1.5 布局固定 4 玩家,cfg.n_players={cfg.n_players}")
    if cfg.n_nodes != 20:
        raise ValueError(f"v1.5 布局固定 20 资源点,cfg.n_nodes={cfg.n_nodes}")

    # 镜像轴取奇数中心(critic B-1):h=64 → cy=31,镜像 r→62−r,行/列 63
    # 恒在 hex 外
    cy = (h - 2) // 2 if h % 2 == 0 else (h - 1) // 2
    cx = cy
    a = cx - 3                      # E/W 顶点半宽
    b = round(a * 0.866)            # 上下平边半高(正六边形比例)
    b = min(b, cy - 1)

    def sh(rc):  # σh 上下镜像
        return (2 * cy - rc[0], rc[1])

    def sv(rc):  # σv 左右镜像
        return (rc[0], 2 * cx - rc[1])

    def rho(rc):  # 180°
        return (2 * cy - rc[0], 2 * cx - rc[1])

    def in_hex(rc) -> bool:
        dr, dc = abs(rc[0] - cy), abs(rc[1] - cx)
        return dr <= b and dc <= a * (1 - dr / (2 * b))

    # 六边形 passable 掩码
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    drm, dcm = np.abs(rr - cy), np.abs(cc - cx)
    passable = (drm <= b) & (dcm <= a * (1 - drm / (2 * b)))

    def frac(fy: float, fx: float) -> tuple[int, int]:
        """按 (b,a) 比例取整的中心相对坐标。"""
        return (cy + round(fy * b), cx + round(fx * a))

    # 玩家 0(左上斜边中点附近,往里收让出建设纵深)
    hq0 = frac(-0.46, -0.64)
    home_ore0 = (hq0[0] - 5, hq0[1] + 4)
    home_water0 = (hq0[0] + 6, hq0[1] - 3)
    # 公共点:E 顶点近旁(放不动行 r=cy 上,σv 生成 W 侧)+ NE 轨道
    e_ore = (cy, cx + round(0.86 * a))
    e_water = (cy, cx + round(0.68 * a))
    ne_ore = frac(-0.875, 0.39)
    ne_water = frac(-0.75, 0.54)

    for rc in (hq0, home_ore0, home_water0, e_ore, e_water, ne_ore, ne_water):
        if not in_hex(rc):
            raise ValueError(f"布局点 {rc} 落在六边形外(网格 {h},a={a},b={b})")

    # 玩家像:1=σh(左下) 2=σv(右上) 3=ρ(右下)
    ops = (lambda rc: rc, sh, sv, rho)
    hq_list = [op(hq0) for op in ops]
    node_pos_list: list[tuple[int, int]] = []
    node_type_list: list[int] = []
    for op in ops:                       # 0..7 家门点(p 序,矿前水后)
        node_pos_list += [op(home_ore0), op(home_water0)]
        node_type_list += [RES_ORE, RES_WATER]
    # 8..11:E/W 顶点(E 定义,σv 生成 W;不动行上 σh 自映射)
    node_pos_list += [e_ore, e_water, sv(e_ore), sv(e_water)]
    node_type_list += [RES_ORE, RES_WATER, RES_ORE, RES_WATER]
    # 12..19:NE 轨道四元像
    for op in ops:
        node_pos_list += [op(ne_ore), op(ne_water)]
        node_type_list += [RES_ORE, RES_WATER]

    assert len(node_pos_list) == cfg.n_nodes

    # (位置,类型) 集合在三个对称操作下自映射(布局公平性的机器可查口径)
    pt = {(tuple(p), t) for p, t in zip(node_pos_list, node_type_list, strict=True)}
    for op in (sh, sv, rho):
        assert {(op(p), t) for p, t in pt} == pt, "资源点 (位置,类型) 未自映射"
        assert {op(p) for p in hq_list} == set(hq_list), "HQ 集未自映射"

    node_pos = np.asarray(node_pos_list, dtype=np.int32)
    node_type = np.asarray(node_type_list, dtype=np.int32)
    hq_pos = np.asarray(hq_list, dtype=np.int32)

    occupied = {tuple(p) for p in node_pos_list} | {tuple(p) for p in hq_list}
    if len(occupied) != len(node_pos_list) + len(hq_list):
        raise ValueError("地图布局重叠:HQ 或资源点撞格")

    for r, c in node_pos_list:
        passable[r, c] = False
    for r, c in hq_list:
        passable[r, c] = False

    # 初始工人:HQ 朝地图中心方向的候选格里取前 start_workers 个,
    # 各玩家用各自的像(严格对称)
    def worker_spawns(hq: tuple[int, int], op) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        # 玩家 0 的候选序(朝中心=右下方向);像玩家直接映射同一候选序
        cands0 = [(hq0[0] + dr, hq0[1] + dc)
                  for dc in (1, 2, 3) for dr in (1, 0, 2, -1, 3)]
        for rc0 in cands0:
            rc = op(rc0)
            if (0 <= rc[0] < h and 0 <= rc[1] < w and passable[rc]
                    and rc not in out):
                out.append(rc)
            if len(out) == cfg.start_workers:
                return out
        raise ValueError("HQ 周围可通行格不足以放下初始工人")

    spawns = [worker_spawns(hq_list[p], ops[p]) for p in range(4)]
    spawn_pos = np.asarray(spawns, dtype=np.int32)

    # 距离场:每个资源点一张 + 每个 HQ 一张
    goals = node_pos_list + hq_list
    fields = [_bfs_field(passable, tuple(p)) for p in goals]
    dist_fields = np.stack(fields).astype(np.int32)
    goal_seeds = np.zeros((len(goals), h, w), dtype=bool)
    for g, rc in enumerate(goals):
        goal_seeds[g][tuple(rc)] = True

    return MapData(passable=passable, node_pos=node_pos, node_type=node_type,
                   hq_pos=hq_pos, spawn_pos=spawn_pos, dist_fields=dist_fields,
                   goal_seeds=goal_seeds)
