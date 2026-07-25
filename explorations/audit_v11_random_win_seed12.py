"""v1.1 专项:seed 12 random(p0) 胜 scripted(p1) 的因果链判定。
问题:①经济死锁回归 ②新机制引擎 bug ③平衡现象?
重放并记录:p1 水泵/矿的生灭与归属、p1 水库存、p1 兵力/工人数、HQ hp、
升级/研发/建营支出事件、HARVEST 工人停滞、每 100 tick 摘要。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v11_random_win_seed12.py
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import numpy as np
from teow.config import TYPE_HQ, TYPE_INFANTRY, TYPE_MINE, TYPE_PUMP, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.state import ORDER_HARVEST, PH_MINING, owner_of_slots
from teow.step import new_world

run_dir = pathlib.Path(__file__).resolve().parent.parent / "experiments/20260725-v1.1-audit-random"
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
assert cfg.seed == 12
state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller("random", "scripted", cfg, m))
owner = np.asarray(owner_of_slots(cfg))
node_type = np.asarray(m.node_type)
print(f"node_type={node_type.tolist()} node_pos={np.asarray(m.node_pos).tolist()} hq={np.asarray(m.hq_pos).tolist()}")

prev = {k: np.asarray(v) for k, v in state._asdict().items()}
stall = np.zeros(cfg.n_total, np.int64); stall_max = 0
water_ge_pump = 0   # p1 水 >= 30(能重建泵)的 tick 数(泵没了之后)
p1_pump_alive_last = -1
events = []
for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}
    # 资源点归属变化
    for k in range(cfg.n_nodes):
        if cur["node_owner"][k] != prev["node_owner"][k]:
            events.append((t, f"点{k}({'水' if node_type[k] else '矿'}) 归属 {int(prev['node_owner'][k])} -> {int(cur['node_owner'][k])}"))
    # p1 是否有活泵(node_owner==1 且水点且 ent 建成)
    p1_pump = any(cur["node_owner"][k] == 1 and node_type[k] == 1 and cur["node_ent"][k] >= 0
                  for k in range(cfg.n_nodes))
    if p1_pump: p1_pump_alive_last = t
    else:
        if cur["resources"][1, 1] >= 30: water_ge_pump += 1
    # 单位死亡事件(建筑)
    died = prev["alive"] & ~cur["alive"]
    for i in np.nonzero(died)[0]:
        et = int(prev["etype"][i])
        if et in (1, 2, 3, 6):
            events.append((t, f"p{owner[i]} 建筑死亡 slot={i} etype={et} pos={prev['pos'][i].tolist()}"))
    hv = (cur["alive"] & ~cur["inside"] & (cur["order"] == ORDER_HARVEST)
          & (cur["phase"] != PH_MINING) & (cur["etype"] == TYPE_WORKER))
    same = (cur["pos"] == prev["pos"]).all(axis=1)
    stall = np.where(hv & same, stall + 1, 0); stall_max = max(stall_max, int(stall.max()))
    if t % 200 == 0:
        sel1 = cur["alive"] & (owner == 1)
        ni = int((sel1 & (cur["etype"] == TYPE_INFANTRY)).sum())
        nw = int((sel1 & (cur["etype"] == TYPE_WORKER)).sum())
        sel0 = cur["alive"] & (owner == 0)
        ni0 = int((sel0 & (cur["etype"] == TYPE_INFANTRY)).sum())
        nw0 = int((sel0 & (cur["etype"] == TYPE_WORKER)).sum())
        print(f"t={t} p1: res={cur['resources'][1].tolist()} inf={ni} w={nw} hq_hp={int(cur['hp'][cfg.e_max])} baseL={int(cur['level'][cfg.e_max])} | "
              f"p0: res={cur['resources'][0].tolist()} inf={ni0} w={nw0} hq_hp={int(cur['hp'][0])}")
    prev = cur
    if bool(state.done): break

print(f"\n结束 tick={int(state.tick)} winner={int(state.winner)}")
print(f"p1 最后有活泵的 tick = {p1_pump_alive_last}")
print(f"泵没了之后 p1 水>=30(可重建泵)的 tick 数 = {water_ge_pump}")
print(f"HARVEST 在途工人最长连续不动 = {stall_max} tick")
print("事件:")
for t, e in events: print(f"  t={t} {e}")
