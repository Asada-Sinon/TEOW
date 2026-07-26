"""v1.7 数值平衡审计:在 audit_v16_invariants.py 上加两条护栏 + 修 v1.6 已知问题 #35。
v1.7 新增:
 ⑪离场护栏——inside/aboard 离场实体(矿内工人、舱内乘员)血量任何一拍不得变化
   (直接护住 combat on_board 门控:防「投石车带弹上艇弹仍爆到舱内」类回归,
   比重建弹道 splash 几何更稳健)。
 ⑫建筑掉血伤害上界——建筑单拍掉血 ≤ 在射程内、可打建筑的敌方攻击者伤害之和
   (龙喷火对建筑按 dragon_breath_bld_percent 折扣计),把 v1.6 的检查⑤从「有无
   来源」加强到「量级不超模」,直接护住喷火折扣([AI-DRAFT])回归。
 修复:v1.6 已知问题 #35——原 375 行 `ab=cur["aboard"]` 遮蔽 156 行 both-alive
   `ab`,致 398 行 new_shot 误用 aboard;本版把 aboard 变量改名 abrd。
--- 以下沿用 v1.6 ---
⑨aboard 一致性——乘员的载具必须 alive 且为飞艇、空军自身 aboard 恒 -1、
   reboard_lock∈[0,lockout];⑩空军不入 building_cells(占格语义);
   雷一次性(存量只减不增,除新建);建筑落地对账加四防御建筑。
⑧淘汰不变量——HQ 亡的玩家不得有任何 alive 实体、其旗必须全灭;
   胜负编码 -1/0..P-1/P;栅栏经 speed 表自动进建筑集(挡路/可拆对账口径不变)。
①同 seed 重放 vs trajectory.npz 逐位一致(tick 决定论,须与录制同后端 cpu)
②逐 tick 资源守恒:Δ库存 = 卸货 - 训练(11 种单位,train 表) - 建矿泵
   - 升级(基地/矿泵/营/兵营/塔) - 研发(八线共用表) - 建营/兵营/哨塔/迫击炮
③不变量:负库存 / hp≤stats 表上限(全类型,含治疗封顶核对) / 上限链
   (矿泵营兵营塔≤基地级) / 线级步进(8 线) / 营建成级=2
④移动:单位任意帧不站硬障碍格(单位=speed 表>0,自动覆盖九新类型)
⑤建筑掉血必有「射程内可打建筑的敌方单位」(v1.4 口径:攻城车/弓/法师远程,
   塔与迫击炮不打建筑——交叉核对目标过滤)
⑥v1.3 名额/驻守/军旗全套沿用(名额口径类型无关,大力士马车同计)
⑦v1.4 专项:采集单位(工/大力士/马车)不得处于 GARRISON;
   shell_timer∈[0,flight] 且仅迫击炮非零;atk_cd∈[0,period-1];
   迫击炮不可升级(level 恒 1);盲区:迫击炮开火锁点距炮位 ≥ min_range
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v17_invariants.py <run_dir>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import numpy as np

from teow.actions import node_costs
from teow.config import (
    BTASK_BUILD_BARRACKS,
    BTASK_BUILD_CAMP,
    BTASK_BUILD_MORTAR,
    BTASK_BUILD_TOWER,
    BTASK_RESEARCH_BASE,
    BTASK_UPGRADE,
    N_LINES,
    N_TYPES,
    TYPE_BARRACKS,
    TYPE_CAMP,
    TYPE_HQ,
    TYPE_MINE,
    TYPE_MORTAR,
    TYPE_PUMP,
    TYPE_STRONGMAN,
    TYPE_TOWER,
    TYPE_WAGON,
    TYPE_WORKER,
    Config,
)
from teow.controller import make_joint_controller
from teow.state import (
    ORDER_BUILD,
    ORDER_GARRISON,
    ORDER_HARVEST,
    PH_MINING,
    PH_TO_HQ,
    owner_of_slots,
)
from teow.stats import hp_table
from teow.step import new_world

run_dir = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
cmd = (run_dir / "command.txt").read_text().split()
p0 = cmd[cmd.index("--p0") + 1] if "--p0" in cmd else "scripted"
p1 = cmd[cmd.index("--p1") + 1] if "--p1" in cmd else "scripted"
P = cfg.n_players
names = ([p0, p1] + ["scripted"] * P)[:P]
if "--p2" in cmd:
    names[2] = cmd[cmd.index("--p2") + 1]
if "--p3" in cmd:
    names[3] = cmd[cmd.index("--p3") + 1]
print(f"run={run_dir.name} players={names} seed={cfg.seed} backend={jax.default_backend()}")

state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(*names, cfg=cfg, mapdata=m))
owner = np.asarray(owner_of_slots(cfg))
ncost = np.asarray(node_costs(cfg, m))
base_up = np.stack([cfg.base_up_cost_ore, cfg.base_up_cost_water], -1)
node_up = np.stack([cfg.node_up_cost_ore, cfg.node_up_cost_water], -1)
camp_up = np.stack([cfg.camp_up_cost_ore, cfg.camp_up_cost_water], -1)
bar_up = np.stack([cfg.barracks_up_cost_ore, cfg.barracks_up_cost_water], -1)
tower_up = np.stack([cfg.tower_up_cost_ore, cfg.tower_up_cost_water], -1)
line_res = np.stack([cfg.line_res_cost_ore, cfg.line_res_cost_water], -1)
tco = np.asarray(cfg.train_cost_ore_by_type)
tcw = np.asarray(cfg.train_cost_water_by_type)
camp_cost = np.asarray([cfg.camp_cost_ore, cfg.camp_cost_water])
bar_cost = np.asarray([cfg.barracks_cost_ore, cfg.barracks_cost_water])
twr_cost = np.asarray([cfg.tower_cost_ore, cfg.tower_cost_water])
mor_cost = np.asarray([cfg.mortar_cost_ore, cfg.mortar_cost_water])
fw_cost = np.asarray([cfg.fence_wood_cost_ore, cfg.fence_wood_cost_water])
fs_cost = np.asarray([cfg.fence_stone_cost_ore, cfg.fence_stone_cost_water])
fi_cost = np.asarray([cfg.fence_iron_cost_ore, cfg.fence_iron_cost_water])
mt_cost = np.asarray([cfg.magetower_cost_ore, cfg.magetower_cost_water])
lm_cost = np.asarray([cfg.landmine_cost_ore, cfg.landmine_cost_water])
fl_cost = np.asarray([cfg.flamer_cost_ore, cfg.flamer_cost_water])
ls_cost = np.asarray([cfg.laser_cost_ore, cfg.laser_cost_water])
is_air_t = np.asarray(cfg.is_air_by_type)
NN = cfg.n_nodes
npos = np.asarray(m.node_pos, np.float64)
hqpos = np.asarray(m.hq_pos, np.float64)
passable = np.asarray(m.passable)
speed_t = np.asarray(cfg.speed_by_type)
slots_tbl = np.asarray(cfg.harvest_slots_by_level, np.int64)
htab = np.asarray(hp_table(cfg))                      # [32,8]
line_of = np.asarray(cfg.line_of_type)
rng_t = np.asarray(cfg.atk_range_by_type)
hit_b_t = np.asarray(cfg.can_hit_buildings_by_type)
period_t = np.asarray(cfg.atk_period_by_type)
UNIT_TYPES = [t for t in range(N_TYPES) if speed_t[t] > 0]
BUILDING_TYPES = [t for t in range(N_TYPES)
                  if speed_t[t] == 0 and t != 0 and htab[t].max() > 0]
HARVESTER_TYPES = [TYPE_WORKER, TYPE_STRONGMAN, TYPE_WAGON]

traj = None
tp = run_dir / "trajectory.npz"
if tp.exists():
    raw = np.load(tp)
    traj = {k: raw[k] for k in raw.files}
    rec_every = int(traj["record_every"])
    print(f"trajectory.npz 帧数={traj['tick'].shape[0]} record_every={rec_every}")

prev = {k: np.asarray(v) for k, v in state._asdict().items()}
V = {"守恒": 0, "负库存": 0, "hp超上限": 0, "上限链": 0, "线级": 0, "营建成级": 0,
     "站硬障碍格": 0, "建筑掉血无有效攻击者": 0,
     "采集名额超额": 0, "旗超上限": 0, "GARRISON引用无效": 0, "撤旗悬空引用": 0,
     "采集单位驻守": 0, "shell异常": 0, "cd异常": 0, "迫击炮升级": 0,
     "盲区开火": 0, "淘汰残余": 0, "淘汰残旗": 0,
     "aboard悬空": 0, "空军在艇上": 0, "回艇锁越界": 0, "超载": 0,
     "离场血量变动": 0, "建筑掉血超上界": 0}
mismatch = 0
frames = 0
last_dep = [-1] * 9
max_gap = [(0, -1, -1)] * 9  # 用前 P 个
stall = np.zeros(cfg.n_total, np.int64)
stall_max = np.zeros(cfg.n_total, np.int64)
gar_ticks = 0
flag_ticks = 0
first_gar = -1
first_flag = -1
mortar_shots = 0
events = []

for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}
    if traj is not None and t % rec_every == 0 and frames < traj["tick"].shape[0]:
        for k in cur:
            if not np.array_equal(np.asarray(traj[k][frames]), cur[k]):
                if mismatch == 0:
                    print(f"[决定论] 首个不一致: 帧{frames}(t={t}) 字段 {k}")
                mismatch += 1
                break
        frames += 1
    ab = prev["alive"] & cur["alive"]
    # ---- ② 守恒对账 ----
    drop = np.maximum(np.where(ab, prev["cargo"].astype(np.int64) - cur["cargo"], 0), 0)
    dep = np.zeros((P, 2), np.int64)
    np.add.at(dep, (owner, prev["cargo_type"].astype(int)), drop)
    died = prev["alive"] & ~cur["alive"]
    d_hq = np.linalg.norm(prev["pos"].astype(np.float64) - hqpos[owner], axis=-1)
    dep_dead = (died & (prev["cargo"] > 0) & (d_hq <= cfg.reach_radius)
                & (prev["phase"] == PH_TO_HQ) & (prev["order"] == ORDER_HARVEST))
    np.add.at(dep, (owner[dep_dead], prev["cargo_type"][dep_dead].astype(int)),
              prev["cargo"][dep_dead].astype(np.int64))
    spend = np.zeros((P, 2), np.int64)
    inc = ab & (cur["btimer"] > prev["btimer"])
    for i in np.nonzero(inc)[0]:
        bt = int(cur["btype"][i])
        p = owner[i]
        if bt > 0:                     # 在训单位:成本查 train 表(11 种通吃)
            spend[p] += (tco[bt], tcw[bt])
        elif bt == BTASK_UPGRADE:
            lv = int(cur["level"][i])
            et = int(cur["etype"][i])
            spend[p] += (base_up[lv] if et == TYPE_HQ else
                         camp_up[lv] if et == TYPE_CAMP else
                         bar_up[lv] if et == TYPE_BARRACKS else
                         tower_up[lv] if et == TYPE_TOWER else node_up[lv])
        elif BTASK_RESEARCH_BASE - (N_LINES - 1) <= bt <= BTASK_RESEARCH_BASE:
            line = BTASK_RESEARCH_BASE - bt
            spend[p] += line_res[int(cur["upgrades"][p, line])]
    for k in np.nonzero(cur["node_build_timer"] > prev["node_build_timer"])[0]:
        spend[cur["node_owner"][k]] += ncost[k]
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
                               (TYPE_TOWER, twr_cost, "哨塔"),
                               (TYPE_MORTAR, mor_cost, "迫击炮"),
                               (19, fw_cost, "木栅栏"),
                               (20, fs_cost, "石栅栏"),
                               (21, fi_cost, "铁栅栏"),
                               (22, mt_cost, "法师塔"),
                               (23, lm_cost, "地雷"),
                               (24, fl_cost, "喷火器"),
                               (25, ls_cost, "激光炮")):
        new_s = ~prev["alive"] & cur["alive"] & (cur["etype"] == stype)
        for i in np.nonzero(new_s)[0]:
            spend[owner[i]] += scost
            events.append((t, f"p{owner[i]} 起{name} slot={i}"))
    delta = cur["resources"].astype(np.int64) - prev["resources"]
    # 淘汰当拍豁免:同 tick「付费获批 → HQ 被打死 → 全家清场」会把刚落地的
    # 实体一并抹掉,支出无从对账(钱确实花了,买家已出局;引擎语义正确)
    elim_now = np.asarray([prev["alive"][q * cfg.e_max]
                           and not cur["alive"][q * cfg.e_max]
                           for q in range(P)])
    delta = np.where(elim_now[:, None], 0, delta)
    dep = np.where(elim_now[:, None], 0, dep)
    spend = np.where(elim_now[:, None], 0, spend)
    if not np.array_equal(delta, dep - spend):
        V["守恒"] += 1
        if V["守恒"] <= 5:
            print(f"[守恒] t={t} Δ={delta.tolist()} 卸货={dep.tolist()} 扣费={spend.tolist()}")
    if (cur["resources"] < 0).any():
        V["负库存"] += 1
    # ---- ③ hp ≤ stats 表上限(全类型;治疗封顶交叉核对)----
    et_c = np.clip(cur["etype"].astype(int), 0, N_TYPES - 1)
    line_c = line_of[et_c]
    lv_eff = np.where(line_c >= 0,
                      cur["upgrades"][owner, np.clip(line_c, 0, N_LINES - 1)],
                      np.clip(cur["level"].astype(int), 0, 7))
    # 在建建筑 hp < 满血是常态,上限校验只对「不在建」实体
    cap_hp = htab[et_c, np.clip(lv_eff, 0, 7)]
    chk = cur["alive"] & (cur["btype"] >= 0)
    if (cur["hp"][chk] > cap_hp[chk]).any():
        V["hp超上限"] += 1
        if V["hp超上限"] <= 5:
            bad = chk & (cur["hp"] > cap_hp)
            print(f"[hp超上限] t={t} slots={np.nonzero(bad)[0].tolist()} "
                  f"hp={cur['hp'][bad].tolist()} cap={cap_hp[bad].tolist()}")
    for p in range(P):
        hqs = p * cfg.e_max
        if cur["alive"][hqs]:
            blv = int(cur["level"][hqs])
            sel = (cur["alive"] & (owner == p)
                   & np.isin(cur["etype"],
                             [TYPE_MINE, TYPE_PUMP, TYPE_CAMP, TYPE_BARRACKS,
                              TYPE_TOWER]))
            if (cur["level"][sel] > blv).any():
                V["上限链"] += 1
                if V["上限链"] <= 3:
                    print(f"[上限链] t={t} p{p} 建筑级超基地级 {blv}")
    du = cur["upgrades"].astype(int) - prev["upgrades"]
    if (du < 0).any() or (du > 1).any() or (cur["upgrades"] > 7).any():
        V["线级"] += 1
    camp_done = (ab & (prev["btype"] == BTASK_BUILD_CAMP) & (cur["btype"] == 0)
                 & (cur["etype"] == TYPE_CAMP))
    for i in np.nonzero(camp_done)[0]:
        events.append((t, f"p{owner[i]} 营建成 slot={i} lv={int(cur['level'][i])}"))
        if int(cur["level"][i]) != 2:
            V["营建成级"] += 1
    for code, tname in ((BTASK_BUILD_BARRACKS, "兵营"), (BTASK_BUILD_TOWER, "哨塔"),
                        (BTASK_BUILD_MORTAR, "迫击炮")):
        for i in np.nonzero(ab & (prev["btype"] == code) & (cur["btype"] == 0))[0]:
            events.append((t, f"p{owner[i]} {tname}建成 slot={i}"))
    for p in range(P):
        for ln in range(N_LINES):
            if du[p, ln] == 1:
                events.append((t, f"p{p} 线{ln} -> {int(cur['upgrades'][p, ln])}"))
    # ---- ④ 硬障碍格 ----
    on_board = cur["alive"] & ~cur["inside"]
    is_unit_c = on_board & np.isin(cur["etype"], UNIT_TYPES)
    cells = np.round(cur["pos"]).astype(int)
    cells = np.clip(cells, 0, [cfg.grid_h - 1, cfg.grid_w - 1])
    bmask = np.zeros((cfg.grid_h, cfg.grid_w), bool)
    # 地雷不挡路(v1.6 引擎口径:building_cells 显式排雷)
    is_bld_c = (on_board & np.isin(cur["etype"], BUILDING_TYPES)
                & (cur["etype"] != 23))
    bmask[cells[is_bld_c][:, 0], cells[is_bld_c][:, 1]] = True
    air_c = is_air_t[np.clip(cur["etype"].astype(int), 0, 31)].astype(bool)
    bad_ground = (is_unit_c & ~air_c
                  & (~passable[cells[:, 0], cells[:, 1]]
                     | bmask[cells[:, 0], cells[:, 1]]))
    bad_air = is_unit_c & air_c & ~passable[cells[:, 0], cells[:, 1]]
    bad_cell = bad_ground | bad_air
    if bad_cell.any():
        V["站硬障碍格"] += 1
        if V["站硬障碍格"] <= 5:
            print(f"[硬障碍] t={t} slots={np.nonzero(bad_cell)[0].tolist()}")
    # ---- ⑤ 建筑掉血须有「射程内可打建筑的敌方单位」----
    bld_hit = ab & np.isin(cur["etype"], BUILDING_TYPES) & (cur["hp"] < prev["hp"])
    dd = np.linalg.norm(cur["pos"][:, None, :].astype(np.float64)
                        - cur["pos"][None, :, :], axis=-1)
    et_prev = np.clip(prev["etype"].astype(int), 0, N_TYPES - 1)
    # v1.6 修订:龙喷火伤建筑(打折)——自心圈也算有效攻击来源
    sa_t = np.asarray(cfg.self_aoe_radius_by_type)
    breath_src = ((prev["etype"] == 28)
                  & (sa_t[np.clip(prev["etype"].astype(int), 0, 31)] > 0))
    src_rng = np.where(breath_src, sa_t[np.clip(prev["etype"].astype(int),
                                                0, 31)],
                       rng_t[et_prev])
    can_hit_me = (prev["alive"][None, :] & ~prev["inside"][None, :]
                  & ((hit_b_t[et_prev] > 0) | breath_src)[None, :]
                  & (owner[:, None] != owner[None, :])
                  & (dd <= src_rng[None, :] + 0.75))  # 双方各动一步的余量
    no_src = bld_hit & ~can_hit_me.any(axis=1)
    if no_src.any():
        V["建筑掉血无有效攻击者"] += 1
        if V["建筑掉血无有效攻击者"] <= 5:
            print(f"[建筑掉血] t={t} slots={np.nonzero(no_src)[0].tolist()} "
                  f"etype={cur['etype'][no_src].tolist()}")
    # ---- 停摆 ----
    for p in range(P):
        if dep[p].sum() > 0:
            if last_dep[p] >= 0 and t - last_dep[p] > max_gap[p][0]:
                max_gap[p] = (t - last_dep[p], last_dep[p], t)
            last_dep[p] = t
    hv = (cur["alive"] & ~cur["inside"] & (cur["order"] == ORDER_HARVEST)
          & (cur["phase"] != PH_MINING) & np.isin(cur["etype"], HARVESTER_TYPES))
    same = (cur["pos"] == prev["pos"]).all(axis=1)
    stall = np.where(hv & same, stall + 1, 0)
    stall_max = np.maximum(stall_max, stall)
    # ---- ⑥ v1.3 名额/驻守/军旗 ----
    tn = np.clip(cur["target_node"].astype(int), 0, NN - 1)
    assigned = cur["alive"] & (cur["order"] == ORDER_HARVEST) & (cur["target_node"] >= 0)
    acnt = np.zeros(NN, np.int64)
    np.add.at(acnt, tn[assigned], 1)
    ent_k = cur["node_ent"].astype(int)
    lvl_k = cur["level"][np.clip(ent_k, 0, cfg.n_total - 1)].astype(int)
    cap = np.where(ent_k >= 0, slots_tbl[np.clip(lvl_k, 0, 7)], 0)
    if (acnt > cap).any():
        V["采集名额超额"] += 1
        if V["采集名额超额"] <= 5:
            print(f"[名额] t={t} 指派={acnt.tolist()} 名额={cap.tolist()}")
    if (cur["flag_active"].sum(axis=1) > cfg.max_flags).any():
        V["旗超上限"] += 1
    gid = cur["garrison_id"].astype(int)
    gar = cur["alive"] & (cur["order"] == ORDER_GARRISON)
    bad_gar = gar & ((gid < 0) | (gid > NN + cfg.max_flags))
    node_t = gar & (gid >= 1) & (gid <= NN)
    gk = np.clip(gid - 1, 0, NN - 1)
    bad_gar |= node_t & ((cur["node_ent"][gk] < 0) | (cur["node_owner"][gk] != owner))
    if bad_gar.any():
        V["GARRISON引用无效"] += 1
    flag_ref = cur["alive"] & (gid >= NN + 1) & (gid <= NN + cfg.max_flags)
    fj = np.clip(gid - NN - 1, 0, cfg.max_flags - 1)
    dangling = flag_ref & ~cur["flag_active"][owner, fj]
    if dangling.any():
        V["撤旗悬空引用"] += 1
    if gar.any():
        gar_ticks += 1
        if first_gar < 0:
            first_gar = t
    if cur["flag_active"].any():
        flag_ticks += 1
        if first_flag < 0:
            first_flag = t
    # ---- ⑦ v1.4 专项 ----
    if (gar & np.isin(cur["etype"], HARVESTER_TYPES)).any():
        V["采集单位驻守"] += 1
    flight_t = np.asarray(cfg.shell_flight_by_type)
    is_sheller = cur["alive"] & (flight_t[np.clip(cur["etype"].astype(int),
                                                  0, 31)] > 0)
    sh = cur["shell_timer"].astype(int)
    sh_cap = flight_t[np.clip(cur["etype"].astype(int), 0, 31)]
    if ((sh < 0) | (sh > sh_cap)).any() or (sh[~is_sheller] != 0).any():
        V["shell异常"] += 1
    cd = cur["atk_cd"].astype(int)
    if ((cd < 0) | (cd > period_t[et_c] - 1)).any():
        V["cd异常"] += 1
    if ((cur["alive"] & (cur["etype"] == TYPE_MORTAR))
            & (cur["level"] != 1)).any():
        V["迫击炮升级"] += 1
    # ---- ⑨ v1.6 aboard/空军不变量(v1.7 修 #35:aboard 变量改名 abrd,
    # 不再遮蔽 156 行的 both-alive ab)----
    abrd = cur["aboard"].astype(int)
    has_ab = cur["alive"] & (abrd >= 0)
    car = np.clip(abrd, 0, cfg.n_total - 1)
    if (has_ab & (~cur["alive"][car] | (cur["etype"][car] != 27))).any():
        V["aboard悬空"] = V.get("aboard悬空", 0) + 1
    if (cur["alive"] & is_air_t[np.clip(cur["etype"].astype(int), 0, 31)]
            & (abrd >= 0)).any():
        V["空军在艇上"] = V.get("空军在艇上", 0) + 1
    rl = cur["reboard_lock"].astype(int)
    if ((rl < 0) | (rl > cfg.reboard_lockout)).any():
        V["回艇锁越界"] = V.get("回艇锁越界", 0) + 1
    # 飞艇载员 ≤ 容量
    paxn = np.zeros(cfg.n_total, np.int64)
    np.add.at(paxn, car[has_ab], 1)
    if (paxn > cfg.airship_capacity).any():
        V["超载"] = V.get("超载", 0) + 1
    # ---- ⑧ v1.5 淘汰不变量 ----
    for p in range(P):
        if not cur["alive"][p * cfg.e_max]:
            if (cur["alive"] & (owner == p)).any():
                V["淘汰残余"] = V.get("淘汰残余", 0) + 1
            if cur["flag_active"][p].any():
                V["淘汰残旗"] = V.get("淘汰残旗", 0) + 1
    new_shot = (ab & is_sheller & (prev["shell_timer"] == 0)
                & (cur["shell_timer"] > 0))
    for i in np.nonzero(new_shot)[0]:
        mortar_shots += 1
        d_lock = float(np.linalg.norm(cur["shell_target"][i] - cur["pos"][i]))
        min_r = (cfg.mortar_min_range
                 if int(cur["etype"][i]) == TYPE_MORTAR else 0.0)
        if d_lock < min_r - 0.75:   # 目标当拍还能动一步的余量
            V["盲区开火"] += 1
            print(f"[盲区] t={t} slot={i} 锁点距={d_lock:.2f}")
        if mortar_shots <= 3:
            events.append((t, f"p{owner[i]} 迫击炮开火 lock={cur['shell_target'][i].tolist()}"))
    # ---- ⑪ v1.7 离场护栏:inside/aboard 离场实体血量任何一拍不得变化 ----
    prev_off = prev["alive"] & (prev["inside"] | (prev["aboard"] >= 0))
    cur_off = cur["alive"] & (cur["inside"] | (cur["aboard"] >= 0))
    off_both = prev_off & cur_off
    if (off_both & (cur["hp"] != prev["hp"])).any():
        V["离场血量变动"] += 1
        if V["离场血量变动"] <= 5:
            bad = off_both & (cur["hp"] != prev["hp"])
            print(f"[离场血量] t={t} slots={np.nonzero(bad)[0].tolist()} "
                  f"hp {prev['hp'][bad].tolist()}->{cur['hp'][bad].tolist()}")
    # ---- ⑫ v1.7 龙喷火对建筑折扣量级护栏:仅当某建筑掉血且在场敌方攻击者
    # 全是龙喷火(breath_src)时,校验掉血 ≤ 在场龙数 × 折扣物理伤害(否则说明
    # 喷火对建筑用了全额而非 dragon_breath_bld_percent 折扣)。can_hit_me/breath_src
    # 复用检查⑤;有其它攻击者混入时跳过(交由⑤保「有无来源」)----
    bld_base_disc = max(
        1, cfg.dragon_breath_atk * cfg.dragon_breath_bld_percent // 100)
    armor_by_t = np.asarray(cfg.armor_by_type)
    for i in np.nonzero(bld_hit)[0]:
        srcs = np.nonzero(can_hit_me[i])[0]
        if len(srcs) == 0 or not bool(np.all(breath_src[srcs])):
            continue
        per = max(1, (bld_base_disc * (100 - int(armor_by_t[et_c[i]])) + 99) // 100)
        drop = int(prev["hp"][i]) - int(cur["hp"][i])
        if drop > len(srcs) * per:
            V["建筑掉血超上界"] += 1
            if V["建筑掉血超上界"] <= 5:
                print(f"[喷火折扣] t={t} slot={i} 掉血={drop} "
                      f"上界={len(srcs) * per}(龙×{len(srcs)}×{per})")
    prev = cur
    if bool(state.done):
        break

print(f"\n结束 tick={int(state.tick)} winner={int(state.winner)} "
      f"res={np.asarray(state.resources).tolist()}")
print(f"决定论: 对比 {frames} 帧, 不一致 {mismatch}")
print("违例计数:", V)
print(f"迫击炮总开火次数: {mortar_shots}")
print(f"最大卸货间隔: p0={max_gap[0]} p1={max_gap[1]}; 最后卸货 p0={last_dep[0]} p1={last_dep[1]}")
print(f"HARVEST 在途最长连续不动: {int(stall_max.max())} tick")
print(f"驻守出现: {gar_ticks} tick(首次 t={first_gar}); "
      f"军旗激活: {flag_ticks} tick(首次 t={first_flag})")
print("事件摘要(前 60 条):")
for tt, e in events[:60]:
    print(f"  t={tt} {e}")
