"""聚合 v2.1 Phase A 均衡评测(读一个 games.jsonl,只读产物,数字全脚本算)。

回答:(1) vs-random 胜率/时长/gate/建军;(2) **座位均衡后**的 round-robin 相对强度排名
(= v2.0 课程干净分层);(3) 座位分布/偏置检查(验证 cyclic 轮转是否奏效);(4) 弱尾专项
(turtle/timing/airtech:rr 胜率 + 是否仍「0 军被动等门胜」);(5) 非退化(秒杀/hard-cap/gate/draw)。

判据阈值沿用 v1.9(agg_v19_roundrobin.py):vs-random wr<0.90=fail;rr wr>0.85=统治(平衡 flag)、
wr<0.10=太弱(fix/delete flag)、w==0=从不胜。

用法:.venv/bin/python explorations/agg_v21_balanced.py <path/to/games.jsonl>
"""

from __future__ import annotations

import bisect
import json
import statistics
import sys
from collections import defaultdict

ROSTER = ["balanced", "boomer", "rusher", "turtle", "timing",
          "harasser", "airtech", "tempo", "counter", "chaos"]
WEAK_TAIL = ["turtle", "timing", "airtech"]   # v1.9 疑弱尾,本轮复核对象


def load(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_ln"] = i
            rows.append(r)
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: agg_v21_balanced.py <games.jsonl>")
    rows = load(sys.argv[1])
    errs = [r for r in rows if "error" in r]
    games = [r for r in rows if "winner" in r]
    vsr = [r for r in games if r["tag"].startswith("vsRandom:")]
    rr = [r for r in games if r["tag"].startswith("rr:")]
    P = len(rows and games[0]["army"]) if games else 4
    print(f"rows={len(rows)} games={len(games)} vsRandom={len(vsr)} rr={len(rr)} "
          f"failed_matchups={len(errs)} P={P}")
    if errs:
        for e in errs:
            print(f"  [FAIL] {e['tag']}: {e['error']}")
    print(f"winner values: vsRandom={sorted(set(r['winner'] for r in vsr))} "
          f"rr={sorted(set(r['winner'] for r in rr))}")

    # ================= vs random =================
    print("\n===== vs random (指挥官@seat0) =====")
    print(f"{'cmd':10} {'n':>3} {'wr':>5} {'avglen':>7} {'gate':>5} {'avgArmy':>7} "
          f"{'lmin':>5} {'lmax':>5}")
    vsr_stats = {}
    for c in ROSTER:
        g = [r for r in vsr if r["tag"] == f"vsRandom:{c}"]
        if not g:
            print(f"{c:10}  —  (无数据/崩溃)")
            continue
        n = len(g)
        wr = sum(1 for r in g if r["winner"] == 0) / n
        lens = [r["length"] for r in g]
        gate = sum(r["gate"] for r in g) / n
        army = statistics.mean(r["army"][0] for r in g)
        vsr_stats[c] = dict(n=n, wr=wr, army=army, gate=gate)
        print(f"{c:10} {n:3d} {wr:5.2f} {statistics.mean(lens):7.0f} {gate:5.2f} "
              f"{army:7.1f} {min(lens):5d} {max(lens):5d}")
    below = [c for c in vsr_stats if vsr_stats[c]["wr"] < 0.90]
    print(f"vs-random wr<0.90 (fail): {below or 'NONE — 全 >=0.90'}")

    # ================= round-robin(座位均衡)=================
    print("\n===== round-robin 相对强度(座位均衡后)=====")
    appear = defaultdict(int)
    wins = defaultdict(int)
    win_lens = defaultdict(list)
    win_gate = defaultdict(int)
    app_army = defaultdict(list)
    seat_appear = defaultdict(lambda: defaultdict(int))   # name -> seat -> count
    for r in rr:
        names = r["names"]
        w = r["winner"]
        for seat, name in enumerate(names):
            appear[name] += 1
            app_army[name].append(r["army"][seat])
            seat_appear[name][seat] += 1
        if w >= 0:
            wname = names[w]
            wins[wname] += 1
            win_lens[wname].append(r["length"])
            if r["gate"] == 1:
                win_gate[wname] += 1
    print(f"{'cmd':10} {'appear':>6} {'wins':>4} {'rr_wr':>5} {'avgWinLen':>9} "
          f"{'winGate':>7} {'avgArmy':>7}")
    rr_stats = {}
    for c in ROSTER:
        a, w = appear[c], wins[c]
        wr = w / a if a else float("nan")
        awl = statistics.mean(win_lens[c]) if win_lens[c] else float("nan")
        wgf = (win_gate[c] / w) if w else float("nan")
        aa = statistics.mean(app_army[c]) if app_army[c] else float("nan")
        rr_stats[c] = dict(a=a, w=w, wr=wr, awl=awl, wgf=wgf, aa=aa)
        print(f"{c:10} {a:6d} {w:4d} {wr:5.2f} {awl:9.1f} {wgf:7.2f} {aa:7.2f}")

    # ---- 座位分布检查(cyclic 轮转是否均衡)----
    print("\n-- 座位出场分布(每指挥官在各座位的次数;均衡=各列接近)--")
    print(f"{'cmd':10} " + " ".join(f"seat{p}" for p in range(P)))
    for c in ROSTER:
        if appear[c]:
            print(f"{c:10} " + " ".join(f"{seat_appear[c][p]:5d}" for p in range(P)))
    # 各座位总胜场(座位偏置的直接证据)
    seat_wins = defaultdict(int)
    for r in rr:
        if r["winner"] >= 0:
            seat_wins[r["winner"]] += 1
    print(f"各座位总胜场(应接近均匀 → 座位无偏): "
          f"{ {p: seat_wins[p] for p in range(P)} }")

    # ---- 难度排名 = 干净分层 ----
    print("\n-- 难度排名(round-robin 胜率 强→弱)= 课程分层 --")
    rank = sorted([c for c in ROSTER if appear[c]],
                  key=lambda c: (-rr_stats[c]["wr"], -rr_stats[c]["w"]))
    for i, c in enumerate(rank, 1):
        s = rr_stats[c]
        print(f"{i:2d}. {c:10} rr_wr={s['wr']:.2f} ({s['w']}/{s['a']})  末军均值={s['aa']:.1f}")
    dom = [c for c in ROSTER if appear[c] and rr_stats[c]["wr"] > 0.85]
    weak = [c for c in ROSTER if appear[c] and rr_stats[c]["wr"] < 0.10]
    never = [c for c in ROSTER if appear[c] and rr_stats[c]["w"] == 0]
    print(f"\nrr_wr>0.85 (统治): {dom or 'NONE'}")
    print(f"rr_wr<0.10 (太弱→改/删): {weak or 'NONE'}")
    print(f"从不胜: {never or 'NONE'}")

    # ---- 弱尾专项 ----
    print("\n===== 弱尾专项(turtle/timing/airtech)=====")
    for c in WEAK_TAIL:
        if c not in rr_stats or not appear[c]:
            print(f"{c}: 无 rr 数据")
            continue
        s = rr_stats[c]
        vs = vsr_stats.get(c, {})
        passive = (s["aa"] < 1.0)   # 出场平均末军 <1 → 「0 军被动」嫌疑
        print(f"{c:10} rr_wr={s['wr']:.2f} 出场末军均值={s['aa']:.2f} "
              f"胜时gate占比={s['wgf']:.2f} | vsRandom wr={vs.get('wr', float('nan')):.2f} "
              f"末军={vs.get('army', float('nan')):.1f} gate={vs.get('gate', float('nan')):.2f}"
              f"  {'⚠️0军被动嫌疑' if passive else ''}")

    # ================= 非退化 =================
    print("\n===== 非退化 / 质量(全部对局)=====")
    alllen = [r["length"] for r in games]
    ep = max(alllen)  # 观测最大 = episode_len 上界代理
    print(f"length min={min(alllen)} median={statistics.median(alllen):.0f} "
          f"max={max(alllen)} mean={statistics.mean(alllen):.0f}")
    print(f"秒杀 <100: {sum(1 for L in alllen if L < 100)}")
    print(f"到 episode_len 硬帽 (=max观测 {ep}): {sum(1 for L in alllen if L >= ep)}")
    print(f"draws (winner<0): {sum(1 for r in games if r['winner'] < 0)}")
    gate_all = sum(r["gate"] for r in games) / len(games)
    print(f"gate 到达率 overall={gate_all:.2f}  vsRandom={sum(r['gate'] for r in vsr)/len(vsr):.2f}"
          f"  rr={sum(r['gate'] for r in rr)/len(rr):.2f}")
    edges = [100, 1000, 2000, 3000, 4000, 5000, 6000]
    labels = ["<100", "100-1k", "1k-2k", "2k-3k", "3k-4k", "4k-5k", "5k-6k", ">=6k"]
    buck = defaultdict(int)
    for L in alllen:
        buck[bisect.bisect_right(edges, L)] += 1
    print("length 直方图:", {labels[k]: buck[k] for k in range(len(labels)) if buck[k]})


if __name__ == "__main__":
    main()
