"""v1.7 通用成本归一对决脚手架:找超模单位/建筑。

背景:scripted×scripted 全局对局对 seed 不敏感(movement/controller 都 `del key`)、
镜像局先手主导,「跑整局比胜率」无平衡信号(见 research-log.md 20260725-tower-balance)。
平衡信号只能来自**手工无菌对决局**:等资源投入下两组单位直接对打,看谁胜、余多少。

本脚本泛化 exp_tower_balance_v13.py 的场景 A:
- 用 n_players=2 的隔离竞技场(A 群=p0 行块、B 群=p1 行块,归属由行块决定);
- 保留双方默认 HQ(远在角上),撤掉默认工人做无菌局。**双方 HQ 必须存活**,否则
  combat.cleanup_deaths 的「HQ 亡即淘汰清场」会当拍清空该玩家全部手工摆的兵;
- 成本归一:scal(t)=cost_ore + water_weight*cost_water,count=max(1,round(budget/scal));
  也可在 matchup 里对某侧指定固定 count(防御建筑 vs 进攻波用);
- 摆位:两簇沿 hq0→hq1 轴对称分居中心两侧,机动单位下 ORDER_ATTACK 向中间对进汇合,
  防御建筑/远程可站桩(ORDER_IDLE)靠 combat 自动选靶;所有 spawn 落 passable 格、
  避开 node/hq 格、不重叠;
- 逐 tick 早停:一方 count 归零 / st.done / 超时;胜负 = 对方全灭且己方有余,超时按
  余血占比(hp_ratio)给强弱信号;
- 等级:每侧可带 level,spawn 时设 upgrades[owner,line]=level(战斗单位强度真源)+
  hp=hp_table[type,level];建筑设 state.level。

用法(务必 .venv/bin/python,CPU 跑):
  JAX_PLATFORMS=cpu .venv/bin/python explorations/exp_v17_duel.py
  烟测(不污染 experiments/):… --smoke --out-root /tmp/xxx
  只跑部分:… --filter melee   (matchup name 子串过滤)
  指定预算/超时:… --budget 240 --max-ticks 600
"""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 实验/门禁一律 CPU(MEMORY LEARN:env)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from teow.actions import A_NOOP
from teow.config import (
    TYPE_AIRSHIP,
    TYPE_ARCHER,
    TYPE_BARRACKS,
    TYPE_CATAPULT,
    TYPE_DOG,
    TYPE_DRAGON,
    TYPE_FLAMER,
    TYPE_HEALER,
    TYPE_HEAVY,
    TYPE_HQ,
    TYPE_INFANTRY,
    TYPE_LASER,
    TYPE_LCAV,
    TYPE_MAGE,
    TYPE_MAGETOWER,
    TYPE_MORTAR,
    TYPE_RAM,
    TYPE_TOWER,
    Config,
)
from teow.state import ORDER_ATTACK, ORDER_IDLE, hq_slot
from teow.stats import hp_table
from teow.step import new_world

# ---- 对决竞技场:默认 4 人图(v1.5 布局固定 4 玩家),只用 p0(A)/p1(B),
# 移除 p2/p3(HQ+工人)。ORDER_ATTACK 只找存活敌方 HQ,故 p2/p3 死后 A→p1、
# B→p0 唯一路由,天然对进汇合(数值与 n_players 无关)----
WATER_WEIGHT = 1.0          # 成本归一里 1 水 = 1 矿当量(写进 provenance,可调)
DEFAULT_BUDGET = 240        # 每侧 ore 当量预算(cost-normalize 用)
DEFAULT_MAX_TICKS = 600
BASE_OVERRIDE = {}          # 竞技场基底(默认 Config;matchup override 叠加于此)

NAME2TYPE = {
    "infantry": TYPE_INFANTRY, "dog": TYPE_DOG, "archer": TYPE_ARCHER,
    "lcav": TYPE_LCAV, "heavy": TYPE_HEAVY, "mage": TYPE_MAGE,
    "healer": TYPE_HEALER, "ram": TYPE_RAM, "catapult": TYPE_CATAPULT,
    "dragon": TYPE_DRAGON, "airship": TYPE_AIRSHIP,
    "tower": TYPE_TOWER, "mortar": TYPE_MORTAR, "magetower": TYPE_MAGETOWER,
    "flamer": TYPE_FLAMER, "laser": TYPE_LASER, "barracks": TYPE_BARRACKS,
    "hq": TYPE_HQ,
}


def unit_cost(cfg, t: int) -> tuple[int, int]:
    """(ore, water) 单个该类型的建造/训练成本。单位查 train 表,建筑查各自字段。"""
    ore = int(cfg.train_cost_ore_by_type[t])
    wat = int(cfg.train_cost_water_by_type[t])
    if ore or wat:
        return ore, wat
    bld = {
        TYPE_TOWER: (cfg.tower_cost_ore, cfg.tower_cost_water),
        TYPE_MORTAR: (cfg.mortar_cost_ore, cfg.mortar_cost_water),
        TYPE_MAGETOWER: (cfg.magetower_cost_ore, cfg.magetower_cost_water),
        TYPE_FLAMER: (cfg.flamer_cost_ore, cfg.flamer_cost_water),
        TYPE_LASER: (cfg.laser_cost_ore, cfg.laser_cost_water),
        TYPE_BARRACKS: (cfg.barracks_cost_ore, cfg.barracks_cost_water),
        TYPE_HQ: (cfg.hq_hp, 0),   # HQ 无造价,用 hp 作名义当量(一般固定 count=1)
    }
    o, w = bld.get(t, (0, 0))
    return int(o), int(w)


def resolve_count(cfg, spec: dict, budget: float) -> int:
    """side spec = {"type":name,"level":L,"count":N|None}。count=None → 成本归一。"""
    t = NAME2TYPE[spec["type"]]
    if spec.get("count") is not None:
        return int(spec["count"])
    ore, wat = unit_cost(cfg, t)
    scal = ore + WATER_WEIGHT * wat
    if scal <= 0:
        return 1
    return max(1, round(budget / scal))


# ---- 对决清单:每项 {name, A, B, budget?, override?};A/B = side spec ----
def side(type_: str, level: int = 1, count=None) -> dict:
    return {"type": type_, "level": level, "count": count}


MATCHUPS = [
    # 近战互克(等投入)
    dict(name="melee_inf_vs_dog", A=side("infantry"), B=side("dog")),
    dict(name="melee_heavy_vs_lcav", A=side("heavy"), B=side("lcav")),
    dict(name="melee_heavy_vs_inf", A=side("heavy"), B=side("infantry")),
    dict(name="melee_lcav_vs_dog", A=side("lcav"), B=side("dog")),
    dict(name="melee_inf_vs_lcav", A=side("infantry"), B=side("lcav")),
    # 远程 vs 近战(魔法应克重甲 heavy_armor=60,是设计意图)
    dict(name="ranged_archer_vs_inf", A=side("archer"), B=side("infantry")),
    dict(name="ranged_archer_vs_dog", A=side("archer"), B=side("dog")),
    dict(name="ranged_mage_vs_heavy", A=side("mage"), B=side("heavy")),
    dict(name="ranged_mage_vs_inf", A=side("mage"), B=side("infantry")),
    # 攻城 vs 建筑(攻城应高效拆建筑)
    dict(name="siege_ram_vs_tower", A=side("ram"), B=side("tower", count=1)),
    dict(name="siege_ram_vs_barracks", A=side("ram"), B=side("barracks", count=1)),
    dict(name="siege_ram_vs_hq", A=side("ram"), B=side("hq", count=1)),
    # 防御建筑 vs 进攻波(1 建筑 vs 等 budget 的兵)
    dict(name="def_tower_vs_dogwave", A=side("tower", count=1), B=side("dog")),
    dict(name="def_mortar_vs_infwave", A=side("mortar", count=1), B=side("infantry")),
    dict(name="def_magetower_vs_infwave",
         A=side("magetower", count=1), B=side("infantry")),
    dict(name="def_flamer_vs_dogwave", A=side("flamer", count=1), B=side("dog")),
    dict(name="def_laser_vs_heavy", A=side("laser", count=1), B=side("heavy", count=1)),
    # 龙(6 级线,level=1 起;line_cap=3)
    dict(name="dragon_vs_airship", A=side("dragon"), B=side("airship")),
    dict(name="dragon_vs_infwave", A=side("dragon"), B=side("infantry")),
    dict(name="dragon_vs_tower", A=side("dragon"), B=side("tower", count=1)),
    dict(name="dragon_vs_barracks",
         A=side("dragon"), B=side("barracks", count=1)),  # 验证喷火对建筑 5 折
    # 辅助超模排查:等投入下奶妈是否让步兵线性翻盘
    dict(name="support_infhealer_vs_inf",
         A=[side("infantry"), side("healer")], B=side("infantry")),
]


def new_run_dir(out_root: pathlib.Path, slug: str) -> pathlib.Path:
    date = datetime.date.today().strftime("%Y%m%d")
    d = out_root / f"{date}-{slug}"
    i = 1
    while d.exists():
        i += 1
        d = out_root / f"{date}-{slug}-{i}"
    d.mkdir(parents=True)
    return d


def write_provenance(run_dir: pathlib.Path, cfg, seeds, extra: dict) -> None:
    """三件套(git hash / resolved config / seed)+ backend + 命令行 + 实验设计常量。"""
    if cfg is not None:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    (run_dir / "seeds.txt").write_text("\n".join(str(s) for s in seeds) + "\n")
    (run_dir / "backend.txt").write_text(jax.default_backend() + "\n")
    (run_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    (run_dir / "design.json").write_text(
        json.dumps(extra, indent=2, ensure_ascii=False))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=ROOT).stdout
    (run_dir / "git.txt").write_text(f"{head}\ndirty: {bool(dirty)}\n{dirty}")
    if dirty:
        diff = subprocess.run(["git", "diff"], capture_output=True, text=True,
                              cwd=ROOT).stdout
        (run_dir / "git.diff").write_text(diff)


def _pick_cells(passable: np.ndarray, forbidden: set, center, n: int,
                claimed: set) -> list:
    """从 center 向外螺旋找 n 个 passable、非 forbidden、未被占的格,返回浮点格心列表。"""
    h, w = passable.shape
    cr, cc = int(round(center[0])), int(round(center[1]))
    out = []
    r = 0
    while len(out) < n and r < max(h, w):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if max(abs(dr), abs(dc)) != r:      # 只取当前环
                    continue
                rr, ccc = cr + dr, cc + dc
                if not (0 <= rr < h and 0 <= ccc < w):
                    continue
                if not passable[rr, ccc]:
                    continue
                if (rr, ccc) in forbidden or (rr, ccc) in claimed:
                    continue
                claimed.add((rr, ccc))
                out.append((float(rr), float(ccc)))
                if len(out) >= n:
                    return out
        r += 1
    return out


def _spawn(st, slot, etype, pos, hp, level, order):
    return st._replace(
        alive=st.alive.at[slot].set(True),
        etype=st.etype.at[slot].set(jnp.asarray(etype, jnp.int8)),
        pos=st.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
        hp=st.hp.at[slot].set(jnp.asarray(hp, jnp.int32)),
        level=st.level.at[slot].set(jnp.asarray(level, jnp.int8)),
        order=st.order.at[slot].set(jnp.asarray(order, jnp.int8)),
    )


def battlefield(mapdata):
    """竞技场几何:passable/forbidden 格集 + A/B 两簇中心(沿 hq0→hq1 轴对进)。"""
    passable = np.asarray(mapdata.passable)
    forbidden = {tuple(int(x) for x in p) for p in mapdata.node_pos}
    forbidden |= {tuple(int(x) for x in p) for p in mapdata.hq_pos}
    hq0 = np.asarray(mapdata.hq_pos[0], np.float32)
    hq1 = np.asarray(mapdata.hq_pos[1], np.float32)
    # A 群居 hq0 侧、B 群居 hq1 侧,各距中心一段,对进汇合
    a_center = hq0 + 0.40 * (hq1 - hq0)
    b_center = hq0 + 0.60 * (hq1 - hq0)
    return passable, forbidden, a_center, b_center


def build_state(cfg, mapdata, state0, mtc, budget: float):
    """按 matchup 摆好一个 state,返回 (state, a_slots, b_slots, meta)。"""
    passable, forbidden, a_center, b_center = battlefield(mapdata)
    htab = hp_table(cfg)
    line_of = cfg.line_of_type
    st = state0
    # 无菌化:撤 p0/p1 默认工人(两家 HQ 留着,做对进锚点);彻底移除 p2/p3
    # (HQ+工人,让 ORDER_ATTACK 只剩 p0↔p1 唯一路由)
    for p in (0, 1):
        for i in range(cfg.start_workers):
            st = st._replace(alive=st.alive.at[hq_slot(p, cfg) + 1 + i].set(False))
    for p in range(2, cfg.n_players):
        base = hq_slot(p, cfg)
        for i in range(cfg.start_workers + 1):   # +1 含 HQ 本身(0 号槽)
            st = st._replace(alive=st.alive.at[base + i].set(False))

    def sides(spec):
        return spec if isinstance(spec, list) else [spec]

    claimed = set()
    meta = {"invest": {}, "counts": {}}
    slots_by_side = {}
    for who, center, order in (("A", a_center, ORDER_ATTACK),
                               ("B", b_center, ORDER_ATTACK)):
        p = 0 if who == "A" else 1
        base = hq_slot(p, cfg)
        specs = sides(mtc[who])
        # 先设该侧所有出现线的 upgrades level(战斗单位强度/血量真源)
        for spec in specs:
            t = NAME2TYPE[spec["type"]]
            ln = int(line_of[t])
            if ln >= 0:
                st = st._replace(
                    upgrades=st.upgrades.at[p, ln].set(
                        jnp.asarray(spec["level"], jnp.int8)))
        # 再逐单位 spawn
        my_slots = []
        inv_o = inv_w = 0
        cnts = {}
        slot_cursor = base + 6      # 跳过 HQ(0)与工人区
        for spec in specs:
            t = NAME2TYPE[spec["type"]]
            lv = int(spec["level"])
            n = resolve_count(cfg, spec, budget)
            # 建筑站桩(IDLE),机动单位对进(ATTACK);射程>0 的远程即使 IDLE 也自动开火
            is_bld = float(cfg.speed_by_type[t]) == 0.0
            od = ORDER_IDLE if is_bld else order
            cells = _pick_cells(passable, forbidden, center, n, claimed)
            hpv = int(htab[t, lv])
            for cell in cells:
                st = _spawn(st, slot_cursor, t, cell, hpv, lv, od)
                my_slots.append(slot_cursor)
                slot_cursor += 1
            o1, w1 = unit_cost(cfg, t)
            inv_o += o1 * len(cells)
            inv_w += w1 * len(cells)
            cnts[spec["type"]] = len(cells)
        slots_by_side[who] = my_slots
        meta["invest"][who] = [inv_o, inv_w]
        meta["counts"][who] = cnts
    return st, slots_by_side["A"], slots_by_side["B"], meta


def run_duel(cfg, step_fn, mapdata, state0, mtc, budget, max_ticks, seed=0):
    st, a_slots, b_slots, meta = build_state(cfg, mapdata, state0, mtc, budget)
    a_arr = jnp.asarray(a_slots, jnp.int32)
    b_arr = jnp.asarray(b_slots, jnp.int32)
    a_hp0 = int(jnp.sum(st.hp[a_arr]))
    b_hp0 = int(jnp.sum(st.hp[b_arr]))
    noop = jnp.full(cfg.n_total, A_NOOP, jnp.int32)
    key = jax.random.PRNGKey(seed)
    outcome, tick = "timeout", max_ticks
    for t in range(max_ticks):
        key, sub = jax.random.split(key)
        st = step_fn(st, noop, sub)
        a_left = int(jnp.sum(st.alive[a_arr]))
        b_left = int(jnp.sum(st.alive[b_arr]))
        if a_left == 0 or b_left == 0 or bool(st.done):
            tick = t + 1
            outcome = ("B_win" if a_left == 0 and b_left > 0 else
                       "A_win" if b_left == 0 and a_left > 0 else
                       "both_dead")
            break
    a_left = int(jnp.sum(st.alive[a_arr]))
    b_left = int(jnp.sum(st.alive[b_arr]))
    a_hp = int(jnp.sum(st.hp[a_arr]))
    b_hp = int(jnp.sum(st.hp[b_arr]))
    if outcome == "timeout":
        # 未分胜负:按余血占比给强弱信号
        ra = a_hp / max(a_hp0, 1)
        rb = b_hp / max(b_hp0, 1)
        outcome = ("A_ahead" if ra - rb > 0.15 else
                   "B_ahead" if rb - ra > 0.15 else "even")
    return {
        "name": mtc["name"], "outcome": outcome, "resolved_tick": tick,
        "budget": budget,
        "A_count": meta["counts"]["A"], "B_count": meta["counts"]["B"],
        "A_invest": meta["invest"]["A"], "B_invest": meta["invest"]["B"],
        "A_left": a_left, "B_left": b_left,
        "A_hp0": a_hp0, "B_hp0": b_hp0, "A_hp": a_hp, "B_hp": b_hp,
        "A_hp_ratio": round(a_hp / max(a_hp0, 1), 3),
        "B_hp_ratio": round(b_hp / max(b_hp0, 1), 3),
        "override": mtc.get("override", {}),
    }


def flag_super(row: dict) -> str:
    """疑似超模标记:一边全灭对方且余兵占比高,或余血占比压倒。"""
    a_alive_ratio = row["A_left"] / max(sum(row["A_count"].values()), 1)
    b_alive_ratio = row["B_left"] / max(sum(row["B_count"].values()), 1)
    if row["outcome"] == "A_win" and a_alive_ratio >= 0.5:
        return "A≫B?"
    if row["outcome"] == "B_win" and b_alive_ratio >= 0.5:
        return "B≫A?"
    if row["A_hp_ratio"] - row["B_hp_ratio"] > 0.7:
        return "A>B?"
    if row["B_hp_ratio"] - row["A_hp_ratio"] > 0.7:
        return "B>A?"
    return ""


def summarize(rows: list) -> str:
    head = (
        "# v1.7 对决矩阵汇总\n\n"
        "成本归一:count=budget/(ore+w·water);胜负=对方全灭且己方有余,超时按余血占比。\n"
        "「疑似超模」= 全灭对方且余兵≥50%,或余血占比差>0.7。\n\n"
        "| 对决 | A×n | B×n | 结果 | tick | A余/B余 | A血比/B血比 | 超模 |\n"
        "|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        acnt = ",".join(f"{k}{v}" for k, v in r["A_count"].items())
        bcnt = ",".join(f"{k}{v}" for k, v in r["B_count"].items())
        lines.append(
            f"| {r['name']} | {acnt} | {bcnt} | {r['outcome']} | "
            f"{r['resolved_tick']} | {r['A_left']}/{r['B_left']} | "
            f"{r['A_hp_ratio']}/{r['B_hp_ratio']} | {flag_super(r)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--slug", default="v17-duel-matrix")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑前 2 个 matchup(不污染 experiments/ 请配 --out-root)")
    ap.add_argument("--filter", default="", help="matchup name 子串过滤")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)

    mtcs = MATCHUPS
    if args.filter:
        mtcs = [m for m in mtcs if args.filter in m["name"]]
    if args.smoke:
        mtcs = mtcs[:2]

    t0 = time.time()
    run_dir = new_run_dir(out_root, args.slug)
    rows = []
    # 每 matchup 用自己的 cfg(可带 override),各自 new_world(map/编译按 cfg)
    with open(run_dir / "duel.jsonl", "w", encoding="utf-8") as fh:
        for mtc in mtcs:
            cfg = dataclasses.replace(
                Config(), **BASE_OVERRIDE, **mtc.get("override", {}))
            state0, _, step_fn, mapdata = new_world(cfg)
            bud = mtc.get("budget", args.budget)   # 高成本单位可在 matchup 里抬预算
            row = run_duel(cfg, step_fn, mapdata, state0, mtc,
                           bud, args.max_ticks, args.seed)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {row['name']}: {row['outcome']} @{row['resolved_tick']} "
                  f"A{row['A_left']}/B{row['B_left']} "
                  f"血比 {row['A_hp_ratio']}/{row['B_hp_ratio']} "
                  f"{flag_super(row)}", flush=True)

    # provenance(用第一个 matchup 的 cfg 做代表落 resolved,design.json 记全部设计常量)
    rep_cfg = dataclasses.replace(Config(), **BASE_OVERRIDE)
    write_provenance(run_dir, rep_cfg, (args.seed,), extra=dict(
        water_weight=WATER_WEIGHT, budget=args.budget, max_ticks=args.max_ticks,
        base_override=BASE_OVERRIDE,
        matchups=[{k: v for k, v in m.items()} for m in mtcs]))
    (run_dir / "summary.md").write_text(summarize(rows))
    print("\n" + summarize(rows))
    print(f"总耗时 {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
