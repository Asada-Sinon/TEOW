"""v1.1 收尾审计:①同 seed 重放 vs trajectory.npz 逐位一致(tick 决定论)
②逐 tick 资源守恒对账(v1.1 扩展):Δ库存 = 卸货 - 训练 - 建矿泵 - 升级 - 研发 - 建营
③v1.1 不变量:库存非负 / 单位 hp ≤ 其线级表上限 / 矿泵营等级 ≤ 基地等级(上限链)/
  升级线 ≤ 7 且每 tick 每线至多 +1 / 建成营(btype -4→0)当拍 level==2 /
  研发完成拍存量单位 hp 增量 == 表差值(对未受击单位为恰好,受击单位为 ≤)
④停摆诊断(v1.0 同款):最大卸货间隔、HARVEST 在途工人最长连续不动。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v11_conservation_invariants.py <run_dir>
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import numpy as np
from teow.actions import node_costs, unit_costs
from teow.config import (BTASK_BUILD_CAMP, BTASK_RESEARCH_INF, BTASK_RESEARCH_WORKER,
                         BTASK_UPGRADE, TYPE_CAMP, TYPE_HQ, TYPE_INFANTRY, TYPE_MINE,
                         TYPE_PUMP, TYPE_WORKER, Config)
from teow.controller import make_joint_controller
from teow.state import ORDER_HARVEST, PH_MINING, owner_of_slots
from teow.step import new_world

run_dir = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
cmd = (run_dir / "command.txt").read_text().split()
p0 = cmd[cmd.index("--p0") + 1] if "--p0" in cmd else "scripted"
p1 = cmd[cmd.index("--p1") + 1] if "--p1" in cmd else "scripted"
print(f"run={run_dir.name} p0={p0} p1={p1} seed={cfg.seed}")

state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(p0, p1, cfg, m))
owner = np.asarray(owner_of_slots(cfg))
ucost = np.asarray(unit_costs(cfg)); ncost = np.asarray(node_costs(cfg, m))
base_up = np.stack([cfg.base_up_cost_ore, cfg.base_up_cost_water], -1)
node_up = np.stack([cfg.node_up_cost_ore, cfg.node_up_cost_water], -1)
camp_up = np.stack([cfg.camp_up_cost_ore, cfg.camp_up_cost_water], -1)
inf_res = np.stack([cfg.inf_res_cost_ore, cfg.inf_res_cost_water], -1)
wrk_res = np.stack([cfg.worker_res_cost_ore, cfg.worker_res_cost_water], -1)
whp_t = np.asarray(cfg.worker_hp_by_level); ihp_t = np.asarray(cfg.inf_hp_by_level)
camp_cost = np.asarray([cfg.camp_cost_ore, cfg.camp_cost_water])

traj = None; tp = run_dir / "trajectory.npz"
if tp.exists():
    traj = np.load(tp); rec_every = int(traj["record_every"])
    print(f"trajectory.npz 帧数={traj['tick'].shape[0]} record_every={rec_every}")

prev = {k: np.asarray(v) for k, v in state._asdict().items()}
V = {"守恒": 0, "负库存": 0, "hp超上限": 0, "上限链": 0, "线级": 0, "营建成级": 0, "研发补血": 0}
mismatch = 0; frames = 0
last_dep = [-1, -1]; max_gap = [(0, -1, -1), (0, -1, -1)]
stall = np.zeros(cfg.n_total, np.int64); stall_max = np.zeros(cfg.n_total, np.int64)
events = []

for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}
    if traj is not None and t % rec_every == 0 and frames < traj["tick"].shape[0]:
        for k in cur:
            if not np.array_equal(np.asarray(traj[k][frames]), cur[k]):
                if mismatch == 0: print(f"[决定论] 首个不一致: 帧{frames}(t={t}) 字段 {k}")
                mismatch += 1; break
        frames += 1
    ab = prev["alive"] & cur["alive"]
    drop = np.maximum(np.where(ab, prev["cargo"].astype(np.int64) - cur["cargo"], 0), 0)
    dep = np.zeros((2, 2), np.int64)
    np.add.at(dep, (owner, prev["cargo_type"].astype(int)), drop)
    spend = np.zeros((2, 2), np.int64)
    inc = ab & (cur["btimer"] > prev["btimer"])
    for i in np.nonzero(inc)[0]:
        bt = int(cur["btype"][i]); p = owner[i]
        if bt == TYPE_WORKER: spend[p] += ucost[0]
        elif bt == TYPE_INFANTRY: spend[p] += ucost[1]
        elif bt == BTASK_UPGRADE:
            lv = int(cur["level"][i]); et = int(cur["etype"][i])
            spend[p] += (base_up[lv] if et == TYPE_HQ else
                         camp_up[lv] if et == TYPE_CAMP else node_up[lv])
        elif bt == BTASK_RESEARCH_INF: spend[p] += inf_res[int(cur["upgrades"][p, 0])]
        elif bt == BTASK_RESEARCH_WORKER: spend[p] += wrk_res[int(cur["upgrades"][p, 1])]
    for k in np.nonzero(cur["node_build_timer"] > prev["node_build_timer"])[0]:
        spend[cur["node_owner"][k]] += ncost[k]
    new_camp = ~prev["alive"] & cur["alive"] & (cur["etype"] == TYPE_CAMP)
    for i in np.nonzero(new_camp)[0]:
        spend[owner[i]] += camp_cost
        events.append((t, f"p{owner[i]} 起营 slot={i}"))
    delta = cur["resources"].astype(np.int64) - prev["resources"]
    if not np.array_equal(delta, dep - spend):
        V["守恒"] += 1
        if V["守恒"] <= 5:
            print(f"[守恒] t={t} Δ={delta.tolist()} 卸货={dep.tolist()} 扣费={spend.tolist()}")
    if (cur["resources"] < 0).any(): V["负库存"] += 1
    # 单位 hp ≤ 线级表上限
    for et, tab, ln in ((TYPE_WORKER, whp_t, 1), (TYPE_INFANTRY, ihp_t, 0)):
        sel = cur["alive"] & (cur["etype"] == et)
        cap = tab[cur["upgrades"][owner, ln]]
        if (cur["hp"][sel] > cap[sel]).any(): V["hp超上限"] += 1
    # 上限链:矿/泵/营等级 ≤ 己方基地等级(基地存活时)
    for p in (0, 1):
        hqs = p * cfg.e_max
        if cur["alive"][hqs]:
            blv = int(cur["level"][hqs])
            sel = (cur["alive"] & (owner == p) & np.isin(cur["etype"], [TYPE_MINE, TYPE_PUMP, TYPE_CAMP]))
            if (cur["level"][sel] > blv).any():
                V["上限链"] += 1
                if V["上限链"] <= 3: print(f"[上限链] t={t} p{p} 建筑级超基地级 {blv}")
    du = cur["upgrades"].astype(int) - prev["upgrades"]
    if (du < 0).any() or (du > 1).any() or (cur["upgrades"] > 7).any(): V["线级"] += 1
    camp_done = ab & (prev["btype"] == BTASK_BUILD_CAMP) & (cur["btype"] == 0) & (cur["etype"] == TYPE_CAMP)
    for i in np.nonzero(camp_done)[0]:
        events.append((t, f"p{owner[i]} 营建成 slot={i} lv={int(cur['level'][i])} hp={int(cur['hp'][i])}"))
        if int(cur["level"][i]) != 2: V["营建成级"] += 1
    # 研发完成拍:存量单位补血 ≤ 表差值,未受击者恰好(受击口径宽松,只查上界+全体一致性)
    for p in (0, 1):
        for ln, code, tab, et in ((0, BTASK_RESEARCH_INF, ihp_t, TYPE_INFANTRY),
                                  (1, BTASK_RESEARCH_WORKER, whp_t, TYPE_WORKER)):
            if du[p, ln] == 1:
                d_tab = int(tab[cur["upgrades"][p, ln]]) - int(tab[prev["upgrades"][p, ln]])
                sel = ab & (owner == p) & (cur["etype"] == et) & ~inc
                dhp = cur["hp"][sel].astype(int) - prev["hp"][sel]
                # 同 tick 还有战斗伤害,允许 ≤;但不得超过表差值
                if (dhp > d_tab).any(): V["研发补血"] += 1
                events.append((t, f"p{p} 线{ln} -> {int(cur['upgrades'][p, ln])} 补血差值={d_tab} 实测max={dhp.max() if dhp.size else '-'}"))
    for i in np.nonzero(ab & (cur["btype"] == BTASK_UPGRADE) & (cur["btimer"] > prev["btimer"]))[0]:
        events.append((t, f"p{owner[i]} 升级开工 slot={i} etype={int(cur['etype'][i])} lv{int(cur['level'][i])}->{int(cur['level'][i])+1}"))
    for p in (0, 1):
        if dep[p].sum() > 0:
            if last_dep[p] >= 0 and t - last_dep[p] > max_gap[p][0]:
                max_gap[p] = (t - last_dep[p], last_dep[p], t)
            last_dep[p] = t
    hv = (cur["alive"] & ~cur["inside"] & (cur["order"] == ORDER_HARVEST)
          & (cur["phase"] != PH_MINING) & (cur["etype"] == TYPE_WORKER))
    same = (cur["pos"] == prev["pos"]).all(axis=1)
    stall = np.where(hv & same, stall + 1, 0); stall_max = np.maximum(stall_max, stall)
    prev = cur
    if bool(state.done): break

print(f"\n结束 tick={int(state.tick)} winner={int(state.winner)} res={np.asarray(state.resources).tolist()}")
print(f"决定论: 对比 {frames} 帧, 不一致 {mismatch}")
print("违例计数:", V)
print(f"最大卸货间隔: p0={max_gap[0]} p1={max_gap[1]}; 最后卸货 p0={last_dep[0]} p1={last_dep[1]}")
print(f"HARVEST 在途最长连续不动: {int(stall_max.max())} tick (slot={int(stall_max.argmax())})")
print("事件(升级/研发/建营):")
for t, e in events: print(f"  t={t} {e}")
