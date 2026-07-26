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
TYPE2NAME = {v: k for k, v in NAME2TYPE.items()}


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


def resolve_side(cfg, spec, budget: float) -> list:
    """side spec = 单个 {"type","level","count"} 或其列表(混合兵种)。返回展开的
    [(type, level), ...](每单位一项)。混合兵种把 budget 均分给各子兵种,使整侧
    总投入 ≈ budget(修:此前每子兵种各吃满 budget → 混合侧超投,如 6步兵+3奶妈)。"""
    specs = spec if isinstance(spec, list) else [spec]
    sub = budget / len(specs)
    units = []
    for s in specs:
        t = NAME2TYPE[s["type"]]
        if s.get("count") is not None:
            n = int(s["count"])
        else:
            ore, wat = unit_cost(cfg, t)
            scal = ore + WATER_WEIGHT * wat
            n = 1 if scal <= 0 else max(1, round(sub / scal))
        units += [(t, int(s["level"]))] * n
    return units


# ---- 对决清单:每项 {name, A, B, budget?, override?};A/B = side spec ----
def side(type_: str, level: int = 1, count=None) -> dict:
    return {"type": type_, "level": level, "count": count}


# 口径(用户 2026-07-26 定):①单位 vs 单位 = 原地接战(交错摆位,IDLE,纯血量/
# 攻击交换,去掉行军/风筝伪影);②防御建筑 = 攻防局(建筑守自家 HQ 前,机动方阵前
# ORDER_ATTACK 收拢聚火,引擎在射程处自停),建筑同等造价该赢、~2 倍造价该被攻破;
# ③water_weight=1(水矿同重)。防御建筑造价(scal=矿+水):塔80/迫140/法师塔120/
# 喷火130/激光210——def 局按 1×(该赢)与 2×(该破)两档设 budget。
MATCHUPS = [
    # ① 近战互克(原地接战,等投入 budget 240)
    dict(name="melee_inf_vs_dog", A=side("infantry"), B=side("dog")),
    dict(name="melee_heavy_vs_lcav", A=side("heavy"), B=side("lcav")),
    dict(name="melee_heavy_vs_inf", A=side("heavy"), B=side("infantry")),
    dict(name="melee_lcav_vs_dog", A=side("lcav"), B=side("dog")),
    dict(name="melee_inf_vs_lcav", A=side("infantry"), B=side("lcav")),
    # ① 远程 vs 近战(原地接战:点对点交换,验证魔法克重甲 heavy_armor=60)
    dict(name="ranged_archer_vs_inf", A=side("archer"), B=side("infantry")),
    dict(name="ranged_archer_vs_dog", A=side("archer"), B=side("dog")),
    dict(name="ranged_mage_vs_heavy", A=side("mage"), B=side("heavy")),
    dict(name="ranged_mage_vs_inf", A=side("mage"), B=side("infantry")),
    # ② 攻城拆建筑(攻防局,等造价:攻城应高效拆建筑)
    dict(name="siege_ram_vs_tower", A=side("ram"), B=side("tower", count=1), budget=80),
    dict(name="siege_ram_vs_barracks",
         A=side("ram"), B=side("barracks", count=1), budget=120),
    dict(name="siege_ram_vs_hq", A=side("ram", count=2), B=side("hq", count=1)),
    # ② 防御建筑 vs 进攻波(攻防局)——1×造价该守住 / 2×造价该被攻破
    dict(name="def_tower_vs_dog_1x", A=side("tower", count=1), B=side("dog"), budget=80),
    dict(name="def_tower_vs_dog_2x", A=side("tower", count=1), B=side("dog"), budget=160),
    dict(name="def_mortar_vs_inf_1x",
         A=side("mortar", count=1), B=side("infantry"), budget=140),
    dict(name="def_mortar_vs_inf_2x",
         A=side("mortar", count=1), B=side("infantry"), budget=280),
    dict(name="def_magetower_vs_inf_1x",
         A=side("magetower", count=1), B=side("infantry"), budget=120),
    dict(name="def_magetower_vs_inf_2x",
         A=side("magetower", count=1), B=side("infantry"), budget=240),
    dict(name="def_flamer_vs_dog_1x",
         A=side("flamer", count=1), B=side("dog"), budget=130),
    dict(name="def_flamer_vs_dog_2x",
         A=side("flamer", count=1), B=side("dog"), budget=260),
    dict(name="def_laser_vs_heavy_1x",
         A=side("laser", count=1), B=side("heavy"), budget=210),
    dict(name="def_laser_vs_heavy_2x",
         A=side("laser", count=1), B=side("heavy"), budget=420),
    # 龙(6 级线,level=1;line_cap=3)
    dict(name="dragon_vs_airship", A=side("dragon"), B=side("airship"), budget=640),
    dict(name="dragon_vs_infwave", A=side("dragon"), B=side("infantry"), budget=320),
    dict(name="dragon_vs_tower", A=side("dragon"), B=side("tower", count=1), budget=80),
    dict(name="dragon_vs_barracks",  # 攻防局:验证喷火对建筑 5 折的拆建筑速度
         A=side("dragon"), B=side("barracks", count=1), budget=120),
    # ① 辅助超模排查:等投入(240 均分)下奶妈是否让步兵翻盘
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


def _arena(mapdata):
    passable = np.asarray(mapdata.passable)
    forbidden = {tuple(int(x) for x in p) for p in mapdata.node_pos}
    forbidden |= {tuple(int(x) for x in p) for p in mapdata.hq_pos}
    return passable, forbidden


def _invest(cfg, units):
    """整侧总投入 [ore,water] 与按类型名计数 dict(从展开后的 units 列表算)。"""
    o = w = 0
    cnt = {}
    for t, _ in units:
        o1, w1 = unit_cost(cfg, t)
        o += o1
        w += w1
        name = TYPE2NAME.get(t, str(t))
        cnt[name] = cnt.get(name, 0) + 1
    return [o, w], cnt


def build_state(cfg, mapdata, state0, mtc, budget: float):
    """按 matchup 摆好一个 state,返回 (state, a_slots, b_slots, meta)。
    模式(用户 2026-07-26 口径):一方全建筑 → 攻防局(建筑守自家 HQ 前 IDLE,
    机动方阵前 ORDER_ATTACK 收拢聚火,引擎在射程处自停);否则 → 原地接战
    (两军交错紧凑摆位,全 IDLE,纯血量/攻击交换,无行军无风筝)。"""
    passable, forbidden = _arena(mapdata)
    htab = hp_table(cfg)
    speed = cfg.speed_by_type
    st = state0
    # 无菌化:撤 p0/p1 默认工人(两家 HQ 留着做锚点);彻底移除 p2/p3
    for p in (0, 1):
        for i in range(cfg.start_workers):
            st = st._replace(alive=st.alive.at[hq_slot(p, cfg) + 1 + i].set(False))
    for p in range(2, cfg.n_players):
        base = hq_slot(p, cfg)
        for i in range(cfg.start_workers + 1):   # +1 含 HQ 本身(0 号槽)
            st = st._replace(alive=st.alive.at[base + i].set(False))

    a_units = resolve_side(cfg, mtc["A"], budget)   # [(type,level),...]
    b_units = resolve_side(cfg, mtc["B"], budget)
    a_bld = all(float(speed[t]) == 0 for t, _ in a_units)
    b_bld = all(float(speed[t]) == 0 for t, _ in b_units)
    # upgrades = 战斗单位强度/血量真源
    for p, units in ((0, a_units), (1, b_units)):
        for t, lv in units:
            ln = int(cfg.line_of_type[t])
            if ln >= 0:
                st = st._replace(
                    upgrades=st.upgrades.at[p, ln].set(jnp.asarray(lv, jnp.int8)))

    hq0 = np.asarray(mapdata.hq_pos[0], np.float32)
    hq1 = np.asarray(mapdata.hq_pos[1], np.float32)
    claimed = set()
    ca, cb = hq_slot(0, cfg) + 6, hq_slot(1, cfg) + 6   # 每家跳过 HQ(0)+工人区

    def place(units, cells, order, cursor):
        nonlocal st
        slots = []
        for (t, lv), cell in zip(units, cells):
            st = _spawn(st, cursor, t, cell, int(htab[t, lv]), lv, order)
            slots.append(cursor)
            cursor += 1
        return slots

    if a_bld != b_bld:
        mode = "assault"
        # 建筑侧=防御(守自家 HQ 前 IDLE),机动侧=进攻(阵前 ORDER_ATTACK)
        if a_bld:
            dp, du, dhq, dc = 0, a_units, hq0, ca
            xp, xu, xhq, xc = 1, b_units, hq1, cb
        else:
            dp, du, dhq, dc = 1, b_units, hq1, cb
            xp, xu, xhq, xc = 0, a_units, hq0, ca
        axis = xhq - dhq
        axis = axis / (float(np.linalg.norm(axis)) + 1e-6)
        d_center = dhq + 4.0 * axis            # 建筑摆自家 HQ 前 4 格
        x_center = d_center + 6.0 * axis        # 攻方在建筑前 6 格,ATTACK 收拢聚火
        d_cells = _pick_cells(passable, forbidden, d_center, len(du), claimed)
        x_cells = _pick_cells(passable, forbidden, x_center, len(xu), claimed)
        d_slots = place(du, d_cells, ORDER_IDLE, dc)
        x_slots = place(xu, x_cells, ORDER_ATTACK, xc)
        a_slots, b_slots = ((d_slots, x_slots) if a_bld else (x_slots, d_slots))
    else:
        mode = "brawl"
        # 原地接战:两军交错摆在中心紧凑区,全 IDLE
        center = 0.5 * (hq0 + hq1)
        cells = _pick_cells(passable, forbidden, center,
                            len(a_units) + len(b_units), claimed)
        seq = []          # 交错序列 → 空间上你中有我
        ai = bi = 0
        while ai < len(a_units) or bi < len(b_units):
            if ai < len(a_units):
                seq.append(("A", a_units[ai]))
                ai += 1
            if bi < len(b_units):
                seq.append(("B", b_units[bi]))
                bi += 1
        a_slots, b_slots = [], []
        for (who, (t, lv)), cell in zip(seq, cells):
            if who == "A":
                st = _spawn(st, ca, t, cell, int(htab[t, lv]), lv, ORDER_IDLE)
                a_slots.append(ca)
                ca += 1
            else:
                st = _spawn(st, cb, t, cell, int(htab[t, lv]), lv, ORDER_IDLE)
                b_slots.append(cb)
                cb += 1

    inv_a, cnt_a = _invest(cfg, a_units)
    inv_b, cnt_b = _invest(cfg, b_units)
    meta = {"mode": mode, "invest": {"A": inv_a, "B": inv_b},
            "counts": {"A": cnt_a, "B": cnt_b}}
    return st, a_slots, b_slots, meta


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
        "budget": budget, "mode": meta["mode"],
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
        "「疑似超模」= 全灭对方且余兵≥50%,或余血占比差>0.7。\n"
        "模式:brawl=原地接战(单位vs单位) / assault=攻防局(建筑防御)。\n\n"
        "| 对决 | 模式 | A×n | B×n | 投入A/B | 结果 | tick | A余/B余 | A血比/B血比 | 超模 |\n"
        "|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        acnt = ",".join(f"{k}{v}" for k, v in r["A_count"].items())
        bcnt = ",".join(f"{k}{v}" for k, v in r["B_count"].items())
        inv = f"{sum(r['A_invest'])}/{sum(r['B_invest'])}"
        lines.append(
            f"| {r['name']} | {r['mode']} | {acnt} | {bcnt} | {inv} | "
            f"{r['outcome']} | {r['resolved_tick']} | {r['A_left']}/{r['B_left']} | "
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
