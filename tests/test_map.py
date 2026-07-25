"""地图:形状、对称性、距离场正确性。"""

import numpy as np

from teow.config import RES_ORE, RES_WATER, Config
from teow.map import BIG_DIST, build_map


def rot(rc, h, w):
    return (h - 1 - rc[0], w - 1 - rc[1])


def test_shapes_and_types():
    cfg = Config()
    m = build_map(cfg)
    assert m.passable.shape == (cfg.grid_h, cfg.grid_w)
    assert m.node_pos.shape == (cfg.n_nodes, 2)
    # MapData 只含静态场(Nn+2);n_goals 含旗动态通道,由 movement 每 tick 拼接
    assert m.dist_fields.shape == (cfg.n_nodes + 2, cfg.grid_h, cfg.grid_w)
    # 每家附近 1矿1水 + 左下/右上公共各 1矿1水(v1.3 双角布局)
    assert list(m.node_type) == [RES_ORE, RES_WATER, RES_ORE, RES_WATER,
                                 RES_ORE, RES_WATER, RES_ORE, RES_WATER]


def test_rotational_symmetry():
    cfg = Config()
    m = build_map(cfg)
    h, w = cfg.grid_h, cfg.grid_w
    assert tuple(m.hq_pos[1]) == rot(tuple(m.hq_pos[0]), h, w)
    # 玩家 1 的近家点是玩家 0 近家点的旋转像(类型一致)
    assert tuple(m.node_pos[2]) == rot(tuple(m.node_pos[0]), h, w)
    assert tuple(m.node_pos[3]) == rot(tuple(m.node_pos[1]), h, w)
    # 公共点:右上(6,7)是左下(4,5)的旋转像,类型不变(严格镜像)
    assert tuple(m.node_pos[6]) == rot(tuple(m.node_pos[4]), h, w)
    assert tuple(m.node_pos[7]) == rot(tuple(m.node_pos[5]), h, w)
    assert m.node_type[6] == m.node_type[4]
    assert m.node_type[7] == m.node_type[5]
    # 出生位旋转对称
    for a, b in zip(m.spawn_pos[0], m.spawn_pos[1], strict=True):
        assert tuple(b) == rot(tuple(a), h, w)


def test_obstacles_impassable():
    cfg = Config()
    m = build_map(cfg)
    for r, c in m.node_pos:
        assert not m.passable[r, c]
    for r, c in m.hq_pos:
        assert not m.passable[r, c]


def test_dist_fields():
    cfg = Config()
    m = build_map(cfg)
    for g in range(cfg.n_nodes + 2):  # 静态场:资源点 + HQ(旗动态场不在 MapData)
        f = m.dist_fields[g]
        src = m.node_pos[g] if g < cfg.n_nodes else m.hq_pos[g - cfg.n_nodes]
        assert f[src[0], src[1]] == 0
        # 与目标 4 邻的可通行格距离恰为 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = src[0] + dr, src[1] + dc
            if 0 <= r < cfg.grid_h and 0 <= c < cfg.grid_w and m.passable[r, c]:
                assert f[r, c] == 1
        # 所有可通行格都可达(开阔地图上不应有 BIG_DIST)
        assert np.all(f[m.passable] < BIG_DIST)
