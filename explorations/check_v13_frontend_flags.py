"""v1.3 Phase 7 验证:load_replay 数据契约含军旗 + matplotlib 抽查一帧渲染出旗。

回答的问题:
1. server.load_replay(run_dir) 的帧里是否出现过非空 `flags`(格式 [p, r, c])?
2. render._draw_frame 对含旗帧能否画出旗(存 PNG 供人工抽查)?

用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/check_v13_frontend_flags.py <run_dir>
(默认 experiments/20260725-v13-frontend-demo;只读 run 目录,PNG 写在本脚本
同级 explorations/output/ 下,不碰产物目录。)
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from teow.config import Config
from teow.map import build_map
from teow.render import _draw_frame
from teow.server import load_replay

ROOT = pathlib.Path(__file__).resolve().parent.parent
run_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "experiments" / "20260725-v13-frontend-demo")

# ---- 1. 数据契约:任一帧含非空 flags,且元素形如 [p, r, c] ----
replay = load_replay(run_dir)
flagged = [fr for fr in replay["frames"] if fr["flags"]]
assert flagged, "所有帧 flags 均为空——数据契约未含旗"
sample = flagged[0]["flags"][0]
assert len(sample) == 3 and sample[0] in (0, 1), f"flags 元素格式不对: {sample}"
first, last = flagged[0], flagged[-1]
print(f"[OK] load_replay 契约含旗: {len(flagged)}/{len(replay['frames'])} 帧有旗, "
      f"首现 tick={first['tick']} 末现 tick={last['tick']}, 样例={sample}")

# ---- 2. matplotlib 抽查:对一含旗帧渲染存 PNG ----
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
m = build_map(cfg)
raw = np.load(run_dir / "trajectory.npz")
data = {k: raw[k] for k in raw.files}
i = first["i"]
frame = {k: data[k][i] for k in data if k != "record_every"}
assert frame["flag_active"].any(), "抽查帧 flag_active 全 False,与契约矛盾"

fig, ax = plt.subplots(figsize=(7, 7))
_draw_frame(ax, frame, cfg.e_max, m.passable, m.node_pos, m.node_type)
out = pathlib.Path(__file__).resolve().parent / "output"  # gitignore: explorations/**/output/
out.mkdir(exist_ok=True)
png = out / "v13_flag_frame.png"
fig.savefig(png, dpi=110)
print(f"[OK] 含旗帧(i={i}, tick={frame['tick']})已渲染: {png}")
