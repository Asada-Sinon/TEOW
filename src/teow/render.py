"""matplotlib 渲染与回放(v1 保底调试用;正经浏览器前端在 v1.2)。

从 run 目录读 trajectory.npz(run.py play --record 产出),逐帧画:
格子地图、资源点(◆矿 ●水,着色=归属)、HQ(大方块)、矿/泵(方块)、
工人(小圆,驻内的画在点旁,带载荷显示)、步兵(三角)、资源 HUD。
"""

from __future__ import annotations

import pathlib

import numpy as np

from .config import (
    TYPE_BARRACKS,
    TYPE_CAMP,
    TYPE_DOG,
    TYPE_HQ,
    TYPE_INFANTRY,
    TYPE_MINE,
    TYPE_PUMP,
    TYPE_TOWER,
    TYPE_WORKER,
)

P_COLOR = {0: "#1f77b4", 1: "#d62728", -1: "#999999"}  # 蓝=玩家0,红=玩家1,灰=无主


def _draw_frame(ax, frame: dict, e_max: int, passable: np.ndarray,
                node_pos: np.ndarray, node_type: np.ndarray):
    h, w = passable.shape
    ax.clear()
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    alive = frame["alive"]
    etype = frame["etype"]
    pos = frame["pos"]
    inside = frame["inside"]
    owner = np.concatenate([np.zeros(e_max, int), np.ones(e_max, int)])

    # 资源点底色(归属)
    for k, (r, c) in enumerate(node_pos):
        col = P_COLOR[int(frame["node_owner"][k])]
        marker = "D" if node_type[k] == 0 else "o"
        ax.plot(c, r, marker, ms=14, mfc="none", mec=col, mew=2)

    # 军旗(v1.3;active 才画:阵营色三角小旗+杆。旧 run 无 flag 字段则跳过)
    flag_active = frame.get("flag_active")
    if flag_active is not None:
        for p in (0, 1):
            for j in range(flag_active.shape[1]):
                if flag_active[p][j]:
                    fr, fc = frame["flag_pos"][p][j]
                    col = P_COLOR[p]
                    ax.plot([fc, fc], [fr + 0.35, fr - 0.45], color=col, lw=1.5)
                    ax.plot(fc + 0.14, fr - 0.3, ">", ms=6, color=col)

    level = frame.get("level")
    for i in np.flatnonzero(alive):
        r, c = pos[i]
        col = P_COLOR[int(owner[i])]
        t = etype[i]
        # 建筑等级角标(v1.1;等级 1 不标,减少视觉噪音)
        if (level is not None and int(level[i]) > 1
                and t in (TYPE_HQ, TYPE_MINE, TYPE_PUMP, TYPE_CAMP,
                          TYPE_BARRACKS, TYPE_TOWER)):
            ax.text(c + 0.45, r - 0.45, str(int(level[i])), fontsize=7,
                    color=col, ha="left", va="bottom", weight="bold")
        if t == TYPE_HQ:
            ax.plot(c, r, "s", ms=16, color=col)
        elif t in (TYPE_MINE, TYPE_PUMP):
            ax.plot(c, r, "s", ms=10, color=col, alpha=0.8)
        elif t == TYPE_CAMP:
            # 在建(btype<0)半透明,建成实心
            building = frame["btype"][i] < 0
            ax.plot(c, r, "p", ms=12, color=col, alpha=0.4 if building else 0.9)
        elif t == TYPE_BARRACKS:
            building = frame["btype"][i] < 0
            ax.plot(c, r, "H", ms=12, color=col, alpha=0.4 if building else 0.9)
        elif t == TYPE_TOWER:
            building = frame["btype"][i] < 0
            ax.plot(c, r, "*", ms=13, color=col, alpha=0.4 if building else 0.9)
        elif t == TYPE_DOG:
            ax.plot(c, r, "v", ms=5, color=col)
        elif t == TYPE_WORKER:
            ms = 4 if inside[i] else 6
            alpha = 0.4 if inside[i] else 1.0
            ax.plot(c, r, "o", ms=ms, color=col, alpha=alpha)
            if frame["cargo"][i] > 0:
                ax.plot(c, r - 0.3, ".", ms=3, color="#e6b800")
        elif t == TYPE_INFANTRY:
            ax.plot(c, r, "^", ms=7, color=col)

    res = frame["resources"]
    # 标题用 ASCII:matplotlib 默认字体没有 CJK 字形,中文会渲染成豆腐块
    up = frame.get("upgrades")
    up_txt = ""
    if up is not None:
        up_txt = (f"   P0 lv{up[0][0]}/{up[0][1]}  P1 lv{up[1][0]}/{up[1][1]}")
    ax.set_title(
        f"tick {int(frame['tick'])}   "
        f"P0 ore {res[0][0]} water {res[0][1]}  |  "
        f"P1 ore {res[1][0]} water {res[1][1]}{up_txt}",
        fontsize=9)


def replay_run(run_dir: pathlib.Path, fps: int = 20, save: str | None = None) -> None:
    import json

    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt

    from .config import Config
    from .map import build_map

    data = np.load(run_dir / "trajectory.npz")
    cfg_dict = json.loads((run_dir / "resolved_config.json").read_text())
    cfg = Config(**cfg_dict)
    m = build_map(cfg)

    n_frames = data["tick"].shape[0]
    frames = [{k: data[k][i] for k in data.files if k != "record_every"}
              for i in range(n_frames)]

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(i):
        _draw_frame(ax, frames[i], cfg.e_max, m.passable, m.node_pos, m.node_type)
        return []

    a = anim.FuncAnimation(fig, update, frames=n_frames,
                           interval=1000 / fps, blit=False, repeat=False)
    if save:
        if save.endswith(".gif"):
            a.save(save, writer=anim.PillowWriter(fps=fps))
        else:
            a.save(save, fps=fps)
        print(f"已保存 {save}")
    else:
        plt.show()
