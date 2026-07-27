"""v2.1 Phase A:均衡 round-robin 复核弱尾(turtle/timing/airtech)+ vs-random 加厚。

动机(v1.9 已知问题):v1.9 的 round-robin 是 `combos[::step][:rr_matchups]` 稀疏采样,
各指挥官出场 4–24 局不等 + 仅 4 seed → 尾部排名含噪、turtle/timing/airtech「零胜」不可靠。
本脚本改成**均衡采样**:

  1. round-robin 用 **covering design**——贪心选一组 4-指挥官组合,使**每一对指挥官(C(10,2)=45
     对)都至少在某个组合里共现**;每个组合按其序号做 **cyclic 座位轮转**(i%P)使各家座位分布
     均衡(消 v1.5 记的镜像位 p1/p3 偏弱);每组合 ≥8 seed。
  2. vs-random 加厚到 ≥16 seed(v1.9 是 4)。

只产出 `games.jsonl`(原始每局)+ provenance;聚合(座位均衡后的 rr 胜率、退化判据、干净分层)
见 `agg_v21_balanced.py`,由 result-analyst 跑。数字一律脚本产出,不手算(CLAUDE.md 硬约束 #1)。

用法(需 GPU,**不**设 JAX_PLATFORMS,批量 vmap 才快):
  .venv/bin/python explorations/eval_balanced_v21.py --slug v21-balanced-rr
  smoke(小局快验):… --smoke --out-root /tmp/xxx
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
from teow.eval import matchup_runner  # 提升后的正版 rollout 原语(src/teow/eval.py)

ROSTER = ["balanced", "boomer", "rusher", "turtle", "timing",
          "harasser", "airtech", "tempo", "counter", "chaos"]


def covering_design(n: int, k: int) -> list[tuple[int, ...]]:
    """贪心 covering design:返回一组 k-子集(索引),使 {0..n-1} 的每一对都至少被某子集覆盖。
    每次选「覆盖最多尚未覆盖对」的子集,平手取字典序最小(确定性,可复现)。"""
    all_combos = list(itertools.combinations(range(n), k))
    needed = set(itertools.combinations(range(n), 2))
    chosen: list[tuple[int, ...]] = []
    remaining = list(all_combos)
    while needed:
        def gain(c):
            return len(needed & set(itertools.combinations(c, 2)))
        best = max(remaining, key=lambda c: (gain(c), tuple(-x for x in c)))
        chosen.append(best)
        needed -= set(itertools.combinations(best, 2))
        remaining.remove(best)
    return chosen


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="v21-balanced-rr")
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--vsr-seeds", type=int, default=16)   # vs-random 每指挥官局数
    ap.add_argument("--rr-seeds", type=int, default=8)     # round-robin 每组合局数
    ap.add_argument("--episode-len", type=int, default=6000)
    ap.add_argument("--gate-open", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.vsr_seeds, args.rr_seeds, args.episode_len, args.gate_open = 3, 2, 900, 600

    cfg = dataclasses.replace(Config(), episode_len=args.episode_len,
                              gate_open_tick=args.gate_open)
    P = cfg.n_players
    rows = []

    def run_and_log(names, tag, n_seeds):
        seeds = jnp.arange(n_seeds, dtype=jnp.int32)
        try:
            out = jax.tree.map(lambda x: x.tolist(), matchup_runner(cfg, names)(seeds))
        except Exception as e:  # noqa: BLE001 一个坏 matchup 不该杀掉整轮
            rows.append(dict(tag=tag, names=list(names),
                             error=f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        for s in range(n_seeds):
            w = out["winner"][s]
            rows.append(dict(tag=tag, names=list(names), seed=int(seeds[s]),
                             winner=w, length=out["length"][s], gate=out["gate_reached"][s],
                             winner_name=(names[w] if 0 <= w < len(names) else "none"),
                             army=[out["army"][s][p] for p in range(P)]))
        print(f"[ok] {tag}", flush=True)

    # ---- vs-random 加厚(指挥官固定 seat0 vs random×(P-1))----
    print(f"=== vs random×{P-1}(seeds={args.vsr_seeds})===", flush=True)
    for c in ROSTER:
        run_and_log((c, *(["random"] * (P - 1))), f"vsRandom:{c}", args.vsr_seeds)

    # ---- 均衡 round-robin:covering design + cyclic 座位轮转 ----
    design = covering_design(len(ROSTER), P)
    print(f"=== 均衡 round-robin:{len(design)} 组合 × cyclic 座位 × seeds={args.rr_seeds} ===",
          flush=True)
    for i, combo in enumerate(design):
        names = tuple(ROSTER[j] for j in combo)
        shift = i % P                                   # cyclic 座位轮转(座位均衡)
        names = names[shift:] + names[:shift]
        run_and_log(names, "rr:" + "|".join(names), args.rr_seeds)

    # ---- 落盘(games.jsonl + provenance;聚合交 agg_v21_balanced.py)----
    d = new_run_dir(pathlib.Path(args.out_root), args.slug)
    write_provenance(d, cfg, list(range(max(args.vsr_seeds, args.rr_seeds))), dict(
        roster=ROSTER, vsr_seeds=args.vsr_seeds, rr_seeds=args.rr_seeds,
        n_rr_combos=len(design), covering_design=[list(c) for c in design],
        episode_len=args.episode_len, gate_open=args.gate_open,
        note="v2.1 Phase A 均衡 round-robin 复核弱尾 + vs-random 加厚"))
    (d / "games.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n_games = sum(1 for r in rows if "winner" in r)
    n_fail = sum(1 for r in rows if "error" in r)
    print(f"\n=== 写入 {d} ===")
    print(f"总 {n_games} 局(vsRandom {len(ROSTER)}×{args.vsr_seeds} + "
          f"rr {len(design)}×{args.rr_seeds});失败 matchup {n_fail}")
    print("聚合:.venv/bin/python explorations/agg_v21_balanced.py "
          f"{d / 'games.jsonl'}")


if __name__ == "__main__":
    main()
