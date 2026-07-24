"""v1.0 审计(复审轮扩展):①同 seed 重放 vs trajectory.npz 逐位一致(tick 决定论)
②逐 tick 资源守恒对账:Δ库存 = 卸货入账 - 训练扣费 - 建造扣费
③经济停摆诊断:最后一次卸货 tick、每玩家最大卸货间隔(停摆段)、
  每个在场 HARVEST 工人(非采矿相)连续不动的最长 tick 数(死锁症状)、末态工人指令分布。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_conservation_determinism.py <run_dir>
run_dir 需含 resolved_config.json 与 trajectory.npz(command.txt 提供控制器与 seed)。
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import jax
import numpy as np

from teow.actions import node_costs, unit_costs
from teow.config import TYPE_HQ, TYPE_INFANTRY, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.state import ORDER_HARVEST, ORDER_IDLE, owner_of_slots
from teow.step import new_world

run_dir = pathlib.Path(sys.argv[1])
cfg_d = json.loads((run_dir / "resolved_config.json").read_text())
cfg = Config(**cfg_d)
cmd = (run_dir / "command.txt").read_text().split()
p0 = cmd[cmd.index("--p0") + 1] if "--p0" in cmd else "scripted"
p1 = cmd[cmd.index("--p1") + 1] if "--p1" in cmd else "scripted"
print(f"run={run_dir.name} p0={p0} p1={p1} seed={cfg.seed}")

state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(p0, p1, cfg, m))
owner = np.asarray(owner_of_slots(cfg))
ucost = np.asarray(unit_costs(cfg))          # [2,2] worker/infantry
ncost = np.asarray(node_costs(cfg, m))       # [Nn,2]
node_type = np.asarray(m.node_type)

traj = None
tp = run_dir / "trajectory.npz"
if tp.exists():
    traj = np.load(tp)
    rec_every = int(traj["record_every"])
    print(f"trajectory.npz 帧数={traj['tick'].shape[0]} record_every={rec_every}")

prev = {k: np.asarray(v) for k, v in state._asdict().items()}
violations = 0
mismatch_frames = 0
frames_checked = 0
last_deposit = [-1, -1]
last_deposit_by_res = [[-1, -1], [-1, -1]]   # [player][ore/water]
max_gap = [(0, -1, -1), (0, -1, -1)]         # [player] (gap, from_t, to_t) 首次卸货起算
first_deposit = [-1, -1]
stall_len = np.zeros(cfg.n_total, np.int64)  # 在场 HARVEST 非采矿相连续不动 tick 数
stall_max = np.zeros(cfg.n_total, np.int64)
from teow.state import PH_MINING  # noqa: E402

for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}

    # ---- 决定论:与录制帧逐位比较(录制在 t%record_every==0 时机)----
    if traj is not None and t % rec_every == 0 and frames_checked < traj["tick"].shape[0]:
        f = frames_checked
        for k in cur:
            if not np.array_equal(np.asarray(traj[k][f]), cur[k]):
                if mismatch_frames == 0:
                    print(f"[决定论] 首个不一致: 帧{f}(t={t}) 字段 {k}")
                mismatch_frames += 1
                break
        frames_checked += 1

    # ---- 守恒对账 ----
    # 卸货 = 存活于前后两帧的工人 cargo 减少量,记到 owner/cargo_type(前帧)
    alive_both = prev["alive"] & cur["alive"]
    drop = np.where(alive_both, prev["cargo"].astype(np.int64) - cur["cargo"], 0)
    drop = np.maximum(drop, 0)
    dep = np.zeros((2, 2), np.int64)
    np.add.at(dep, (owner, prev["cargo_type"].astype(int)), drop)
    # 训练开工 = HQ btimer 增加
    spend = np.zeros((2, 2), np.int64)
    for i in np.nonzero((cur["btimer"] > prev["btimer"]) & (cur["etype"] == TYPE_HQ))[0]:
        u = 0 if cur["btype"][i] == TYPE_WORKER else 1
        spend[owner[i]] += ucost[u]
    # 建造开工 = node_build_timer 增加
    for k in np.nonzero(cur["node_build_timer"] > prev["node_build_timer"])[0]:
        spend[cur["node_owner"][k]] += ncost[k]
    delta = cur["resources"].astype(np.int64) - prev["resources"]
    if not np.array_equal(delta, dep - spend):
        violations += 1
        if violations <= 5:
            print(f"[守恒] t={t} Δ={delta.tolist()} 卸货={dep.tolist()} 扣费={spend.tolist()}")
    for p in (0, 1):
        if dep[p].sum() > 0:
            if first_deposit[p] < 0:
                first_deposit[p] = t
            elif t - last_deposit[p] > max_gap[p][0]:
                max_gap[p] = (t - last_deposit[p], last_deposit[p], t)
            last_deposit[p] = t
        for r in (0, 1):
            if dep[p][r] > 0:
                last_deposit_by_res[p][r] = t

    # ---- 死锁症状:在场 HARVEST 工人(非采矿相)位置连续不变的时长 ----
    harv_ob = (cur["alive"] & ~cur["inside"] & (cur["order"] == ORDER_HARVEST)
               & (cur["phase"] != PH_MINING) & (cur["etype"] == TYPE_WORKER))
    same = (cur["pos"] == prev["pos"]).all(axis=1)
    stall_len = np.where(harv_ob & same, stall_len + 1, 0)
    stall_max = np.maximum(stall_max, stall_len)

    prev = cur
    if bool(state.done):
        break

print(f"\n结束 tick={int(state.tick)} winner={int(state.winner)} "
      f"res={np.asarray(state.resources).tolist()}")
print(f"决定论: 帧对比 {frames_checked} 帧, 不一致 {mismatch_frames}")
print(f"守恒: 违例 tick 数 = {violations}")
print(f"最后卸货 tick: p0={last_deposit[0]} p1={last_deposit[1]} (末 tick={int(state.tick)})")
print(f"分资源最后卸货 [p][ore,water]: {last_deposit_by_res}")
print(f"最大卸货间隔(停摆段): p0={max_gap[0]} p1={max_gap[1]}")
print(f"HARVEST 在途工人最长连续不动: {int(stall_max.max())} tick "
      f"(slot={int(stall_max.argmax())})")

# ---- 末态经济诊断 ----
st = {k: np.asarray(v) for k, v in state._asdict().items()}
print(f"node_owner={st['node_owner'].tolist()} node_ent={st['node_ent'].tolist()} "
      f"build_timer={st['node_build_timer'].tolist()}")
for p in (0, 1):
    sel = (owner == p) & st["alive"]
    nw = int(((st["etype"] == TYPE_WORKER) & sel).sum())
    ni = int(((st["etype"] == TYPE_INFANTRY) & sel).sum())
    harv = sel & (st["order"] == ORDER_HARVEST)
    idle = sel & (st["order"] == ORDER_IDLE) & (st["etype"] == TYPE_WORKER)
    hq = p * cfg.e_max
    print(f"p{p}: workers={nw} inf={ni} harv={int(harv.sum())} idle_w={int(idle.sum())} "
          f"hq_hp={int(st['hp'][hq])} hq_btype={int(st['btype'][hq])} "
          f"hq_btimer={int(st['btimer'][hq])}")
    for i in np.nonzero(sel & (st["etype"] == TYPE_WORKER))[0]:
        print(f"  w{i} pos={st['pos'][i].tolist()} order={int(st['order'][i])} "
              f"ph={int(st['phase'][i])} tn={int(st['target_node'][i])} "
              f"cargo={int(st['cargo'][i])} inside={bool(st['inside'][i])} "
              f"timer={int(st['mine_timer'][i])}")
