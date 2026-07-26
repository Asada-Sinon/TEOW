"""v1.7 定值扫参:对确认超弱的迫击炮/法师塔扫候选 override,找满足「防御建筑
1×造价守住(A_win)、2×造价被攻破(B_win)」的最小改动。

用户 2026-07-26 定案:①迫击炮是建筑,按攻防局判,应能自守等价近战;②法师塔加 atk;
③龙保持 50% 折扣、不擅拆建筑,不动。复用 exp_v17_duel 的 run_duel/摆位机制。

复核标准(与主矩阵一致):
- 迫击炮:1× budget=140(4 步兵)应 A_win,2× budget=280(7 步兵)应 B_win。
- 法师塔:1× budget=120(3 步兵)应 A_win,2× budget=240(6 步兵)应 B_win。

用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/exp_v17_tune.py
      烟测:… --smoke --out-root /tmp/xxx
"""

import argparse
import dataclasses
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "explorations"))

import json  # noqa: E402

from exp_v17_duel import new_run_dir, run_duel, side, write_provenance  # noqa: E402

from teow.config import Config  # noqa: E402
from teow.step import new_world  # noqa: E402

# 候选 override(每项一个 dict);base={} 作对照(应两档皆 B_win=守不住)
MORTAR_CANDS = {
    "base": {},
    "period25": dict(mortar_atk_period=25),
    "period15": dict(mortar_atk_period=15),
    "minr1.0": dict(mortar_min_range=1.0),
    "minr1.0_period25": dict(mortar_min_range=1.0, mortar_atk_period=25),
    "hp250_period25": dict(mortar_hp=250, mortar_atk_period=25),
    "atk50_period20": dict(mortar_atk=50, mortar_atk_period=20),
}
MAGETOWER_CANDS = {
    "base": {},
    "atk18": dict(magetower_atk=18),
    "atk20": dict(magetower_atk=20),
    "atk24": dict(magetower_atk=24),
    "atk28": dict(magetower_atk=28),
}
# 龙对地面=大范围火海(用户 2026-07-26:火海范围可给大)。dragon_breath_radius
# 现 2.5;龙对纯地面无敌(步兵打不到空军),调大半径只让清场更快更广、不失衡
# (地面军须带防空反制)。目标:1 龙对等价 8 步兵应 A_win(清光火海覆盖)。
DRAGON_CANDS = {
    "base": {},
    "r3.5": dict(dragon_breath_radius=3.5),
    "r4.0": dict(dragon_breath_radius=4.0),
    "r4.5": dict(dragon_breath_radius=4.5),
    "r5.0": dict(dragon_breath_radius=5.0),
}


def run_def(override, bld_type, budget, max_ticks):
    """跑一个「防御建筑 count=1 vs 步兵波」攻防局,返回结果行。"""
    cfg = dataclasses.replace(Config(), **override)     # 默认 4 人图
    state0, _, step_fn, mapdata = new_world(cfg)
    mtc = dict(name=f"{bld_type}_b{budget}", A=side(bld_type, count=1),
               B=side("infantry"))
    return run_duel(cfg, step_fn, mapdata, state0, mtc, budget, max_ticks)


def run_dragon(override, budget, max_ticks):
    """1 龙 vs 等价步兵波(brawl 原地接战),测火海清场。"""
    cfg = dataclasses.replace(Config(), **override)
    state0, _, step_fn, mapdata = new_world(cfg)
    mtc = dict(name=f"dragon_b{budget}", A=side("dragon"), B=side("infantry"))
    return run_duel(cfg, step_fn, mapdata, state0, mtc, budget, max_ticks)


def sweep_dragon(cands, budget, max_ticks):
    rows = []
    for cname, ov in cands.items():
        r = run_dragon(ov, budget, max_ticks)
        ok = r["outcome"] == "A_win"    # 龙清光步兵波
        rows.append(dict(building="dragon", cand=cname, override=ov,
                         x1_outcome=r["outcome"], x1_hp=r["A_hp_ratio"],
                         x1_Aleft=r["A_left"], x1_Bleft=r["B_left"],
                         x2_outcome="-", x2_hp="-", x2_Aleft="-", x2_Bleft="-",
                         resolved_tick=r["resolved_tick"], meets=ok))
        print(f"  [dragon] {cname:18s} {r['outcome']}@{r['resolved_tick']} "
              f"龙血{r['A_hp_ratio']} 步兵余{r['B_left']} "
              f"{'✓清场' if ok else ''}", flush=True)
    return rows


def sweep(name, cands, bld_type, b1x, b2x, max_ticks):
    rows = []
    for cname, ov in cands.items():
        r1 = run_def(ov, bld_type, b1x, max_ticks)
        r2 = run_def(ov, bld_type, b2x, max_ticks)
        ok = (r1["outcome"] == "A_win" and r2["outcome"] == "B_win")
        row = dict(building=name, cand=cname, override=ov,
                   x1_outcome=r1["outcome"], x1_hp=r1["A_hp_ratio"],
                   x1_Aleft=r1["A_left"], x1_Bleft=r1["B_left"],
                   x2_outcome=r2["outcome"], x2_hp=r2["A_hp_ratio"],
                   x2_Aleft=r2["A_left"], x2_Bleft=r2["B_left"],
                   meets=ok)
        rows.append(row)
        print(f"  [{name}] {cname:18s} 1×={r1['outcome']}(塔血{r1['A_hp_ratio']}) "
              f"2×={r2['outcome']}(波余{r2['B_left']}) {'✓达标' if ok else ''}",
              flush=True)
    return rows


def summarize(rows):
    lines = ["# v1.7 迫击炮/法师塔补强扫参\n",
             "标准:1×造价该 A_win(建筑守住)、2×造价该 B_win(被攻破)。\n",
             "| 建筑 | 候选 | override | 1× | 1×塔血 | 2× | 2×波余 | 达标 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ov = ", ".join(f"{k}={v}" for k, v in r["override"].items()) or "现值"
        lines.append(
            f"| {r['building']} | {r['cand']} | {ov} | {r['x1_outcome']} | "
            f"{r['x1_hp']} | {r['x2_outcome']} | {r['x2_Bleft']} | "
            f"{'✓' if r['meets'] else ''} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--slug", default="v17-tune")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=800)
    args = ap.parse_args()

    t0 = time.time()
    mc = ({"base": {}, "period25": MORTAR_CANDS["period25"]} if args.smoke
          else MORTAR_CANDS)
    gc = ({"base": {}, "atk20": MAGETOWER_CANDS["atk20"]} if args.smoke
          else MAGETOWER_CANDS)
    rows = []
    print("== 迫击炮扫参(1×=140/4步兵,2×=280/7步兵)", flush=True)
    rows += sweep("mortar", mc, "mortar", 140, 280, args.max_ticks)
    print("== 法师塔扫参(1×=120/3步兵,2×=240/6步兵)", flush=True)
    rows += sweep("magetower", gc, "magetower", 120, 240, args.max_ticks)
    dc = ({"base": {}, "r4.0": DRAGON_CANDS["r4.0"]} if args.smoke
          else DRAGON_CANDS)
    print("== 龙火海半径扫参(1 龙 vs 8 步兵 @320,目标 A_win 清场)", flush=True)
    rows += sweep_dragon(dc, 320, args.max_ticks)

    run_dir = new_run_dir(pathlib.Path(args.out_root), args.slug)
    write_provenance(run_dir, Config(), (0,), extra=dict(
        mortar_cands=MORTAR_CANDS, magetower_cands=MAGETOWER_CANDS,
        criterion="1x A_win & 2x B_win"))
    (run_dir / "sweep.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (run_dir / "summary.md").write_text(summarize(rows))
    print("\n" + summarize(rows))
    print(f"总耗时 {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
