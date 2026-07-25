"""v1.3 终审补充:seed7 audit 局 p1 在 t=1265 后 ~319 tick 零卸货,是战损还是死锁?
逐帧(record_every=2)统计 p1 存活工人数/矿泵数/工人 order 分布/矿泵点归属。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v13_p1_tail_stall.py <run_dir>
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from teow.config import TYPE_MINE, TYPE_PUMP, TYPE_WORKER, Config

run_dir = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
raw = np.load(run_dir / "trajectory.npz")
alive, etype, order = raw["alive"], raw["etype"], raw["order"]
tick = raw["tick"]; node_owner = raw["node_owner"]; node_ent = raw["node_ent"]
p1 = slice(cfg.e_max, cfg.n_total)
prev = None
for f in range(alive.shape[0]):
    t = int(tick[f])
    if t < 1200:
        continue
    w = alive[f, p1] & (etype[f, p1] == TYPE_WORKER)
    nw = int(w.sum())
    od = np.bincount(order[f, p1][w].astype(int), minlength=6).tolist() if nw else []
    mines = int(((node_owner[f] == 1) & (node_ent[f] >= 0)).sum())
    cur = (nw, od, mines)
    if cur != prev or f == alive.shape[0] - 1:
        print(f"t={t} p1工人={nw} order分布(IDLE,HARV,BUILD,MOVE,ATT,GAR)={od} p1已建矿泵={mines}")
    prev = cur
