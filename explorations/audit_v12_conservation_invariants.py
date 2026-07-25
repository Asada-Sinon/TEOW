"""v1.2 收官审计:在 v1.1 对账脚本上扩展(v1.1 版对 v1.2 run 会假阳性且用格距口径):
①同 seed 重放 vs trajectory.npz 逐位一致(tick 决定论,须与录制同后端 cpu)
②逐 tick 资源守恒:Δ库存 = 卸货 - 训练(工/兵/狗) - 建矿泵 - 升级(含塔) - 研发
   - 建营/兵营/哨塔;「卸货即死/开工即死」口径改欧氏 reach_radius(v1.2)
③v1.1 不变量:负库存/hp≤线级上限(含狗)/上限链(矿泵营塔≤基地级)/线级步进/营建成级
④v1.2 专项:单位任意帧不站硬障碍格(静态不可通行/建筑格)/狗步长>步兵步长可观测
   /建筑掉血必有敌方近战单位在 melee_range 内(塔只攻单位的交叉核对)
⑤停摆:最大卸货间隔、HARVEST 在途最长连续不动
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v12_conservation_invariants.py <run_dir>
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import numpy as np
from teow.actions import node_costs, unit_costs
from teow.config import (BTASK_BUILD_BARRACKS, BTASK_BUILD_CAMP, BTASK_BUILD_TOWER,
                         BTASK_RESEARCH_INF, BTASK_RESEARCH_WORKER, BTASK_UPGRADE,
                         TYPE_BARRACKS, TYPE_CAMP, TYPE_DOG, TYPE_HQ, TYPE_INFANTRY,
                         TYPE_MINE, TYPE_PUMP, TYPE_TOWER, TYPE_WORKER, Config)
from teow.controller import make_joint_controller
from teow.state import ORDER_BUILD, ORDER_HARVEST, PH_MINING, PH_TO_HQ, owner_of_slots
from teow.step import new_world

run_dir = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
cmd = (run_dir / "command.txt").read_text().split()
p0 = cmd[cmd.index("--p0") + 1] if "--p0" in cmd else "scripted"
p1 = cmd[cmd.index("--p1") + 1] if "--p1" in cmd else "scripted"
print(f"run={run_dir.name} p0={p0} p1={p1} seed={cfg.seed} backend={jax.default_backend()}")

state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(p0, p1, cfg, m))
owner = np.asarray(owner_of_slots(cfg))
ucost = np.asarray(unit_costs(cfg)); ncost = np.asarray(node_costs(cfg, m))
base_up = np.stack([cfg.base_up_cost_ore, cfg.base_up_cost_water], -1)
node_up = np.stack([cfg.node_up_cost_ore, cfg.node_up_cost_water], -1)
camp_up = np.stack([cfg.camp_up_cost_ore, cfg.camp_up_cost_water], -1)
tower_up = np.stack([cfg.tower_up_cost_ore, cfg.tower_up_cost_water], -1)
inf_res = np.stack([cfg.inf_res_cost_ore, cfg.inf_res_cost_water], -1)
wrk_res = np.stack([cfg.worker_res_cost_ore, cfg.worker_res_cost_water], -1)
whp_t = np.asarray(cfg.worker_hp_by_level); ihp_t = np.asarray(cfg.inf_hp_by_level)
dhp_t = np.asarray(cfg.dog_hp_by_level)
camp_cost = np.asarray([cfg.camp_cost_ore, cfg.camp_cost_water])
bar_cost = np.asarray([cfg.barracks_cost_ore, cfg.barracks_cost_water])
twr_cost = np.asarray([cfg.tower_cost_ore, cfg.tower_cost_water])
dog_cost = np.asarray([cfg.dog_cost_ore, cfg.dog_cost_water])
NN = cfg.n_nodes
npos = np.asarray(m.node_pos, np.float64)
hqpos = np.asarray(m.hq_pos, np.float64)
passable = np.asarray(m.passable)
speed_t = np.asarray(cfg.speed_by_type)
BUILDING_TYPES = [TYPE_HQ, TYPE_MINE, TYPE_PUMP, TYPE_CAMP, TYPE_BARRACKS, TYPE_TOWER]
UNIT_TYPES = [TYPE_WORKER, TYPE_INFANTRY, TYPE_DOG]

traj = None; tp = run_dir / "trajectory.npz"
if tp.exists():
    raw = np.load(tp); traj = {k: raw[k] for k in raw.files}
    rec_every = int(traj["record_every"])
    print(f"trajectory.npz 帧数={traj['tick'].shape[0]} record_every={rec_every}")

prev = {k: np.asarray(v) for k, v in state._asdict().items()}
V = {"守恒": 0, "负库存": 0, "hp超上限": 0, "上限链": 0, "线级": 0, "营建成级": 0,
     "站硬障碍格": 0, "建筑掉血无近战敌": 0}
mismatch = 0; frames = 0
last_dep = [-1, -1]; max_gap = [(0, -1, -1), (0, -1, -1)]
stall = np.zeros(cfg.n_total, np.int64); stall_max = np.zeros(cfg.n_total, np.int64)
max_step = {TYPE_WORKER: 0.0, TYPE_INFANTRY: 0.0, TYPE_DOG: 0.0}
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
    # ---- ② 守恒对账 ----
    drop = np.maximum(np.where(ab, prev["cargo"].astype(np.int64) - cur["cargo"], 0), 0)
    dep = np.zeros((2, 2), np.int64)
    np.add.at(dep, (owner, prev["cargo_type"].astype(int)), drop)
    # 卸货即死(欧氏口径 v1.2):tick 初带货、TO_HQ 相、HARVEST、距己方 HQ ≤ reach
    died = prev["alive"] & ~cur["alive"]
    d_hq = np.linalg.norm(prev["pos"].astype(np.float64) - hqpos[owner], axis=-1)
    dep_dead = (died & (prev["cargo"] > 0) & (d_hq <= cfg.reach_radius)
                & (prev["phase"] == PH_TO_HQ) & (prev["order"] == ORDER_HARVEST))
    np.add.at(dep, (owner[dep_dead], prev["cargo_type"][dep_dead].astype(int)),
              prev["cargo"][dep_dead].astype(np.int64))
    spend = np.zeros((2, 2), np.int64)
    inc = ab & (cur["btimer"] > prev["btimer"])
    for i in np.nonzero(inc)[0]:
        bt = int(cur["btype"][i]); p = owner[i]
        if bt == TYPE_WORKER: spend[p] += ucost[0]
        elif bt == TYPE_INFANTRY: spend[p] += ucost[1]
        elif bt == TYPE_DOG: spend[p] += dog_cost
        elif bt == BTASK_UPGRADE:
            lv = int(cur["level"][i]); et = int(cur["etype"][i])
            spend[p] += (base_up[lv] if et == TYPE_HQ else
                         camp_up[lv] if et == TYPE_CAMP else
                         tower_up[lv] if et == TYPE_TOWER else node_up[lv])
        elif bt == BTASK_RESEARCH_INF: spend[p] += inf_res[int(cur["upgrades"][p, 0])]
        elif bt == BTASK_RESEARCH_WORKER: spend[p] += wrk_res[int(cur["upgrades"][p, 1])]
    for k in np.nonzero(cur["node_build_timer"] > prev["node_build_timer"])[0]:
        spend[cur["node_owner"][k]] += ncost[k]
    # 开工即死(欧氏口径):付得起+可认领+tick 初在 reach 内 ⇒ 必已开工,当拍取消=沉没
    d_nd = np.linalg.norm(
        prev["pos"].astype(np.float64)
        - npos[np.clip(prev["target_node"].astype(int), 0, NN - 1)], axis=-1)
    for i in np.nonzero(died & (prev["order"] == ORDER_BUILD)
                        & (prev["target_node"] >= 0) & (d_nd <= cfg.reach_radius))[0]:
        k = int(prev["target_node"][i])
        if (prev["node_build_timer"][k] == 0 and cur["node_build_timer"][k] == 0
                and prev["node_owner"][k] == -1 and cur["node_owner"][k] == -1
                and bool(np.all(prev["resources"][owner[i]] >= ncost[k]))):
            spend[owner[i]] += ncost[k]
    for stype, scost, name in ((TYPE_CAMP, camp_cost, "营"),
                               (TYPE_BARRACKS, bar_cost, "兵营"),
                               (TYPE_TOWER, twr_cost, "哨塔")):
        new_s = ~prev["alive"] & cur["alive"] & (cur["etype"] == stype)
        for i in np.nonzero(new_s)[0]:
            spend[owner[i]] += scost
            events.append((t, f"p{owner[i]} 起{name} slot={i}"))
    delta = cur["resources"].astype(np.int64) - prev["resources"]
    if not np.array_equal(delta, dep - spend):
        V["守恒"] += 1
        if V["守恒"] <= 5:
            print(f"[守恒] t={t} Δ={delta.tolist()} 卸货={dep.tolist()} 扣费={spend.tolist()}")
    if (cur["resources"] < 0).any(): V["负库存"] += 1
    # ---- ③ v1.1 不变量 ----
    for et, tab, ln in ((TYPE_WORKER, whp_t, 1), (TYPE_INFANTRY, ihp_t, 0),
                        (TYPE_DOG, dhp_t, 0)):
        sel = cur["alive"] & (cur["etype"] == et)
        cap = tab[cur["upgrades"][owner, ln]]
        if (cur["hp"][sel] > cap[sel]).any(): V["hp超上限"] += 1
    for p in (0, 1):
        hqs = p * cfg.e_max
        if cur["alive"][hqs]:
            blv = int(cur["level"][hqs])
            sel = (cur["alive"] & (owner == p)
                   & np.isin(cur["etype"], [TYPE_MINE, TYPE_PUMP, TYPE_CAMP, TYPE_TOWER]))
            if (cur["level"][sel] > blv).any():
                V["上限链"] += 1
                if V["上限链"] <= 3: print(f"[上限链] t={t} p{p} 建筑级超基地级 {blv}")
    du = cur["upgrades"].astype(int) - prev["upgrades"]
    if (du < 0).any() or (du > 1).any() or (cur["upgrades"] > 7).any(): V["线级"] += 1
    camp_done = ab & (prev["btype"] == BTASK_BUILD_CAMP) & (cur["btype"] == 0) & (cur["etype"] == TYPE_CAMP)
    for i in np.nonzero(camp_done)[0]:
        events.append((t, f"p{owner[i]} 营建成 slot={i} lv={int(cur['level'][i])} hp={int(cur['hp'][i])}"))
        if int(cur["level"][i]) != 2: V["营建成级"] += 1
    for code, tname in ((BTASK_BUILD_BARRACKS, "兵营"), (BTASK_BUILD_TOWER, "哨塔")):
        for i in np.nonzero(ab & (prev["btype"] == code) & (cur["btype"] == 0))[0]:
            events.append((t, f"p{owner[i]} {tname}建成 slot={i} lv={int(cur['level'][i])} hp={int(cur['hp'][i])}"))
    for i in np.nonzero(ab & (cur["btype"] == BTASK_UPGRADE) & (cur["btimer"] > prev["btimer"]))[0]:
        events.append((t, f"p{owner[i]} 升级开工 slot={i} etype={int(cur['etype'][i])} lv{int(cur['level'][i])}->{int(cur['level'][i])+1}"))
    for p in (0, 1):
        for ln, code in ((0, BTASK_RESEARCH_INF), (1, BTASK_RESEARCH_WORKER)):
            if du[p, ln] == 1:
                events.append((t, f"p{p} 线{ln} -> {int(cur['upgrades'][p, ln])}"))
    # ---- ④ v1.2 专项 ----
    on_board = cur["alive"] & ~cur["inside"]
    is_unit_c = on_board & np.isin(cur["etype"], UNIT_TYPES)
    cells = np.round(cur["pos"]).astype(int)
    cells = np.clip(cells, 0, [cfg.grid_h - 1, cfg.grid_w - 1])
    bmask = np.zeros((cfg.grid_h, cfg.grid_w), bool)
    is_bld_c = on_board & np.isin(cur["etype"], BUILDING_TYPES)
    bmask[cells[is_bld_c][:, 0], cells[is_bld_c][:, 1]] = True
    bad_cell = is_unit_c & (~passable[cells[:, 0], cells[:, 1]] | bmask[cells[:, 0], cells[:, 1]])
    if bad_cell.any():
        V["站硬障碍格"] += 1
        if V["站硬障碍格"] <= 5:
            print(f"[硬障碍] t={t} slots={np.nonzero(bad_cell)[0].tolist()} pos={cur['pos'][bad_cell].tolist()}")
    both = ab & ~prev["inside"] & ~cur["inside"] & np.isin(cur["etype"], UNIT_TYPES)
    dstep = np.linalg.norm(cur["pos"].astype(np.float64) - prev["pos"], axis=-1)
    for et in max_step:
        sel = both & (cur["etype"] == et)
        if sel.any(): max_step[et] = max(max_step[et], float(dstep[sel].max()))
    # 建筑掉血 ⇒ 必有敌方近战单位(tick 初在场)在 melee_range 内(塔只攻单位交叉核对)
    bld_hit = ab & np.isin(cur["etype"], BUILDING_TYPES) & (cur["hp"] < prev["hp"] +
        np.where((cur["btype"] < 0) & (cur["btimer"] > 0), 0, 0))  # 在建成长为正,掉血仍需敌
    dd = np.linalg.norm(cur["pos"][:, None, :].astype(np.float64)
                        - cur["pos"][None, :, :], axis=-1)
    melee_enemy = (prev["alive"][None, :] & ~cur["inside"][None, :]
                   & np.isin(cur["etype"], UNIT_TYPES)[None, :]
                   & (owner[:, None] != owner[None, :]) & (dd <= cfg.melee_range + 1e-6))
    no_src = bld_hit & ~melee_enemy.any(axis=1)
    if no_src.any():
        V["建筑掉血无近战敌"] += 1
        if V["建筑掉血无近战敌"] <= 5:
            print(f"[建筑掉血] t={t} slots={np.nonzero(no_src)[0].tolist()} etype={cur['etype'][no_src].tolist()}")
    # ---- ⑤ 停摆 ----
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
print(f"最大单步位移(狗应>步兵): { {k: round(v, 4) for k, v in max_step.items()} } "
      f"(speed表 工/兵={speed_t[TYPE_WORKER]}/{speed_t[TYPE_INFANTRY]} 狗={speed_t[TYPE_DOG]})")
print(f"最大卸货间隔: p0={max_gap[0]} p1={max_gap[1]}; 最后卸货 p0={last_dep[0]} p1={last_dep[1]}")
print(f"HARVEST 在途最长连续不动: {int(stall_max.max())} tick (slot={int(stall_max.argmax())})")
print("事件(升级/研发/建营/兵营/哨塔):")
for t, e in events: print(f"  t={t} {e}")
