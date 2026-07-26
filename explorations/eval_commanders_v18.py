"""v1.8/v1.9 指挥官评测:GPU vmap 批量跑整局,算胜率矩阵 + 质量指标。

口径(research-log 20260726-v18-bench 定):单卡 GPU vmap 批量是海量 rollout 正路
(B≈64 ~4000 env-tick/s)。一个 matchup(4 家指挥官名的元组)编译一次、vmap over seeds
批量跑;跨 matchup 循环聚合。scripted×scripted 对 seed 不敏感/镜像先手偏置(见
research-log 20260725-tower-balance)→ 评测靠**不同风格非对称** + **多座位排列**取真分布。

质量指标(v1.9 筛选判据):胜率(vs random / round-robin)、对局时长分布、异界之门到达率
(overtime)、退化标志(never 建军 / 输给 random)。

用法(需 GPU;**不**设 JAX_PLATFORMS):
  .venv/bin/python explorations/eval_commanders_v18.py --slug v18-eval
  --roster boomer,rusher,turtle,timing,harasser,airtech,tempo,counter,chaos
  烟测:… --smoke --out-root /tmp/xxx
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import itertools
import json
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from teow.config import Config
from teow.controller import make_joint_controller
from teow.map import build_map
from teow.state import init_state
from teow.step import build_step, make_scan


def new_run_dir(out_root: pathlib.Path, slug: str) -> pathlib.Path:
    day = datetime.date.today().strftime("%Y%m%d")
    base = out_root / f"{day}-{slug}"
    d, i = base, 2
    while d.exists():
        d = out_root / f"{day}-{slug}-{i}"
        i += 1
    d.mkdir(parents=True)
    return d


def write_provenance(d, cfg, seeds, extra):
    (d / "resolved_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    (d / "seeds.txt").write_text("\n".join(str(s) for s in seeds))
    (d / "backend.txt").write_text(jax.default_backend())
    (d / "command.txt").write_text(shlex.join(sys.argv))
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        porc = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
        (d / "git.txt").write_text(f"{h}\ndirty: {bool(porc.strip())}\n{porc}")
        if porc.strip():
            (d / "git.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT).decode())
    except Exception as e:  # noqa: BLE001
        (d / "git.txt").write_text(f"git 不可用: {e}")
    (d / "design.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False))


# ---- 一个 matchup:vmap over seeds,取每局末态指标 ----
def make_matchup_runner(cfg, names):
    """返回 run(seeds[int32 S]) -> dict of [S] 指标数组(winner/length/gate_reached/
    每家 n_army/hq_alive)。names 长度 = cfg.n_players。"""
    mapdata = build_map(cfg)
    step_fn = build_step(cfg, mapdata)
    joint = make_joint_controller(*names, cfg=cfg, mapdata=mapdata)
    scan = make_scan(step_fn, joint)
    state0 = init_state(cfg, mapdata)
    P, e = cfg.n_players, cfg.e_max
    is_combat = jnp.asarray(cfg.is_combat_by_type, bool)

    def one(seed):
        st, _, _ = scan(state0, jax.random.PRNGKey(seed), cfg.episode_len)
        et = jnp.clip(st.etype.astype(jnp.int32), 0, 31)
        army = (st.alive & is_combat[et]).reshape(P, e).sum(axis=1)     # [P]
        hq_alive = st.alive.reshape(P, e)[:, 0]                          # [P] HQ 在 0 号槽
        return dict(winner=st.winner.astype(jnp.int32), length=st.tick,
                    gate_reached=(st.tick >= cfg.gate_open_tick).astype(jnp.int32),
                    army=army, hq_alive=hq_alive.astype(jnp.int32))

    run = jax.jit(jax.vmap(one))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="v18-eval")
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--roster", default="boomer,rusher,turtle,timing,harasser,"
                                        "airtech,tempo,counter,chaos")
    ap.add_argument("--seeds", type=int, default=24)      # 每 matchup 局数(vmap B)
    ap.add_argument("--episode-len", type=int, default=6000)
    ap.add_argument("--gate-open", type=int, default=4000)
    ap.add_argument("--rr-matchups", type=int, default=12)  # round-robin 采样的 matchup 数
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.episode_len, args.gate_open, args.rr_matchups = 4, 900, 600, 3

    roster = args.roster.split(",")
    cfg = dataclasses.replace(Config(), episode_len=args.episode_len,
                              gate_open_tick=args.gate_open)
    seeds = jnp.arange(args.seeds, dtype=jnp.int32)
    rows = []

    def run_and_log(names, tag):
        try:
            run = make_matchup_runner(cfg, names)
            out = jax.tree.map(lambda x: x.tolist(), run(seeds))
        except Exception as e:  # noqa: BLE001 一个坏指挥官不该杀掉整轮
            rows.append(dict(tag=tag, names=list(names),
                             error=f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        for s in range(args.seeds):
            w = out["winner"][s]
            rows.append(dict(tag=tag, names=list(names), seed=int(seeds[s]),
                             winner=w, length=out["length"][s],
                             gate=out["gate_reached"][s],
                             winner_name=(names[w] if 0 <= w < len(names) else "none"),
                             army=[out["army"][s][p] for p in range(cfg.n_players)]))
        print(f"[ok] {tag}: {names}", flush=True)

    # ---- Phase A:每个指挥官 seat0 vs random×3 → vs-random 胜率 + 建军 sanity ----
    print("=== Phase A: 各指挥官 vs random×3 ===", flush=True)
    for c in roster:
        run_and_log((c, "random", "random", "random"), f"vsRandom:{c}")

    # ---- Phase B:round-robin 采样(4 个不同风格 + 座位轮转)→ 相对强度 ----
    print("=== Phase B: round-robin 采样 ===", flush=True)
    combos = list(itertools.combinations(roster, cfg.n_players))
    step = max(1, len(combos) // args.rr_matchups)
    for combo in combos[::step][:args.rr_matchups]:
        run_and_log(tuple(combo), "rr:" + "|".join(combo))

    # ---- 聚合 ----
    d = new_run_dir(pathlib.Path(args.out_root), args.slug)
    write_provenance(d, cfg, [int(s) for s in seeds], dict(
        roster=roster, seeds=args.seeds, episode_len=args.episode_len,
        gate_open=args.gate_open, note="v1.8/v1.9 指挥官评测"))
    (d / "games.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    # vs-random 胜率 + 平均时长 + gate 到达率 + 退化标志
    lines = ["# 指挥官评测汇总\n", f"roster: {roster}  seeds/matchup: {args.seeds}\n",
             "\n## vs random×3(seat0 胜率 / 平均时长 / gate 到达率 / 平均建军)\n",
             "| 指挥官 | 胜率 | 平均时长 | gate率 | 末军均值 | 退化? |",
             "|---|---|---|---|---|---|"]
    for c in roster:
        rr = [r for r in rows if r["tag"] == f"vsRandom:{c}" and "winner" in r]
        if not rr:
            lines.append(f"| {c} | — | — | — | — | ❌崩溃 |")
            continue
        n = len(rr)
        wr = sum(1 for r in rr if r["winner_name"] == c) / n
        avg_len = sum(r["length"] for r in rr) / n
        gate_rate = sum(r["gate"] for r in rr) / n
        avg_army = sum(r["army"][0] for r in rr) / n
        degen = "⚠️" if (wr < 0.5 or avg_army < 1) else ""
        lines.append(f"| {c} | {wr:.2f} | {avg_len:.0f} | {gate_rate:.2f} | "
                     f"{avg_army:.1f} | {degen} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\n=== 写入 {d} ===\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
