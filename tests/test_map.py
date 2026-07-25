"""v1.5 六边形四人地图:形状/边界/Klein 四群对称/出生位/BFS 场。"""

import numpy as np

from teow.config import RES_ORE, RES_WATER, Config
from teow.map import BIG_DIST, build_map


def _ops(h, w):
    cy = (h - 2) // 2 if h % 2 == 0 else (h - 1) // 2
    cx = cy
    return (lambda rc: (2 * cy - rc[0], rc[1]),      # σh
            lambda rc: (rc[0], 2 * cx - rc[1]),      # σv
            lambda rc: (2 * cy - rc[0], 2 * cx - rc[1]))  # ρ=180°


def test_shapes_and_counts():
    cfg = Config()
    m = build_map(cfg)
    assert m.node_pos.shape == (20, 2)
    assert m.hq_pos.shape == (4, 2)
    assert m.spawn_pos.shape == (4, cfg.start_workers, 2)
    assert m.dist_fields.shape == (20 + 4, cfg.grid_h, cfg.grid_w)
    # 类型配比:每家 1矿1水 ×4 + 公共 6矿6水 = 10 矿 10 水
    assert int(np.sum(m.node_type == RES_ORE)) == 10
    assert int(np.sum(m.node_type == RES_WATER)) == 10
    # 家门点编号约定:0..7 = 玩家 p 的 (矿,水) 按 p 序
    for p in range(4):
        assert m.node_type[2 * p] == RES_ORE
        assert m.node_type[2 * p + 1] == RES_WATER
        d = np.linalg.norm(m.node_pos[2 * p] - m.hq_pos[p])
        assert 3.5 <= d <= 10, f"玩家 {p} 家门矿距 HQ {d:.1f} 不在近旁"


def test_hex_boundary():
    cfg = Config()
    m = build_map(cfg)
    h, w = cfg.grid_h, cfg.grid_w
    # 四角在六边形外
    for rc in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        assert not m.passable[rc], f"角格 {rc} 应在六边形外"
    # 中心可行
    assert m.passable[31, 31]
    # 行/列 63(镜像轴外余量)整行整列不可行
    assert not m.passable[h - 1, :].any()
    assert not m.passable[:, w - 1].any()


def test_klein_symmetry_self_mapping():
    """(位置,类型) 集合与 HQ 集在 σh/σv/180° 三操作下严格自映射(公平性)。"""
    cfg = Config()
    m = build_map(cfg)
    pt = {(tuple(p), int(t)) for p, t in zip(m.node_pos, m.node_type, strict=True)}
    hqs = {tuple(int(x) for x in p) for p in m.hq_pos}
    for op in _ops(cfg.grid_h, cfg.grid_w):
        assert {(op(p), t) for p, t in pt} == pt
        assert {op(p) for p in hqs} == hqs
    # passable 掩码本身对称
    for op in _ops(cfg.grid_h, cfg.grid_w):
        mapped = np.zeros_like(m.passable)
        idx = np.argwhere(m.passable)
        for r, c in idx:
            rr, cc = op((int(r), int(c)))
            mapped[rr, cc] = True
        assert (mapped == m.passable).all()


def test_spawns_are_symmetric_images_and_passable():
    cfg = Config()
    m = build_map(cfg)
    sh, sv, rho = _ops(cfg.grid_h, cfg.grid_w)
    s0 = [tuple(int(x) for x in rc) for rc in m.spawn_pos[0]]
    assert [tuple(int(x) for x in rc) for rc in m.spawn_pos[1]] == [sh(rc) for rc in s0]
    assert [tuple(int(x) for x in rc) for rc in m.spawn_pos[2]] == [sv(rc) for rc in s0]
    assert [tuple(int(x) for x in rc) for rc in m.spawn_pos[3]] == [rho(rc) for rc in s0]
    for p in range(4):
        for rc in m.spawn_pos[p]:
            assert m.passable[tuple(rc)], f"出生位 {rc} 不可通行"


def test_bfs_fields_reach_everywhere():
    cfg = Config()
    m = build_map(cfg)
    for g in range(m.dist_fields.shape[0]):
        f = m.dist_fields[g]
        assert (f[m.passable] < BIG_DIST).all(), f"场 {g} 有不可达可行格"
        # 种子格值 0
        seed = np.argwhere(m.goal_seeds[g])[0]
        assert f[tuple(seed)] == 0
