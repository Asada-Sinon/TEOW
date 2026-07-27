"""独立核验 v2.1 balanced-rr 评测的「座位偏置」假设(只读 games.jsonl,数字全脚本算)。

回答主 agent 的 6 问:
 Q1 rr 段各座位(0-3)总胜场分布;seat0 是否偏高。
 Q2 各指挥官 rr 段每座位出场次数矩阵;turtle/airtech/chaos 的 seat0 是否=0;谁曝光不均。
 Q3 vsRandom 段各指挥官胜率(winner==0);是否全 >=0.90。
 Q4 rr 段各指挥官胜率(wins/appearances);哪些=0;零胜者的 seat0 曝光。
 Q5 非退化:length min/median/max、秒杀<100、硬帽>=6000、draw(winner<0)、gate 率。
 Q6 座位偏置判断:seat0 高胜是「机制强」还是「强指挥官被分到 seat0」的混淆;
    有无指挥官即使有 seat0 曝光也零胜(=更可能真弱)。

用法: .venv/bin/python explorations/seatbias_check_v21.py <games.jsonl>
数据来源: experiments/20260727-v21-balanced-rr/games.jsonl (232 行)
"""
from __future__ import annotations
import json, sys, statistics
from collections import defaultdict

ROSTER = ["balanced","boomer","rusher","turtle","timing",
          "harasser","airtech","tempo","counter","chaos"]

def load(path):
    rows=[]
    with open(path) as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            r=json.loads(line); r["_ln"]=i; rows.append(r)
    return rows

def main():
    rows=load(sys.argv[1])
    games=[r for r in rows if "winner" in r]
    errs=[r for r in rows if "error" in r]
    vsr=[r for r in games if r["tag"].startswith("vsRandom:")]
    rr =[r for r in games if r["tag"].startswith("rr:")]
    print(f"[sanity] rows={len(rows)} games={len(games)} vsRandom={len(vsr)} rr={len(rr)} errors={len(errs)}")
    print(f"[sanity] rr winner value set = {sorted(set(r['winner'] for r in rr))}")
    print(f"[sanity] vsr winner value set = {sorted(set(r['winner'] for r in vsr))}")

    # ---------- Q1: rr 座位总胜场 ----------
    print("\n=== Q1  rr 段各座位总胜场 ===")
    seat_wins=defaultdict(int); seat_app=defaultdict(int); draws=0
    for r in rr:
        for s in range(4): seat_app[s]+=1
        w=r["winner"]
        if w>=0: seat_wins[w]+=1
        else: draws+=1
    tot=sum(seat_wins.values())
    for s in range(4):
        print(f"  seat{s}: wins={seat_wins[s]:3d}  app={seat_app[s]:3d}  seat_wr={seat_wins[s]/seat_app[s]:.3f}")
    print(f"  rr 决胜局数(非draw)={tot}  draws={draws}  均匀期望/座位={tot/4:.1f}")

    # ---------- Q2: 各指挥官 rr 座位出场矩阵 ----------
    print("\n=== Q2  rr 各指挥官 x 座位 出场次数 ===")
    seat_appear=defaultdict(lambda: defaultdict(int))
    appear=defaultdict(int)
    for r in rr:
        for s,name in enumerate(r["names"]):
            seat_appear[name][s]+=1; appear[name]+=1
    print(f"  {'cmd':10} {'seat0':>5} {'seat1':>5} {'seat2':>5} {'seat3':>5} {'total':>6}")
    for c in ROSTER:
        if appear[c]:
            print(f"  {c:10} " + " ".join(f"{seat_appear[c][s]:5d}" for s in range(4)) + f" {appear[c]:6d}")
    zero_seat0=[c for c in ROSTER if appear[c] and seat_appear[c][0]==0]
    print(f"  seat0 出场=0 的指挥官: {zero_seat0}")
    # 曝光不均度: 每指挥官座位分布的极差
    print("  座位曝光极差(max-min 座位次数;0=完全均衡):")
    for c in ROSTER:
        if appear[c]:
            counts=[seat_appear[c][s] for s in range(4)]
            print(f"    {c:10} counts={counts} 极差={max(counts)-min(counts)}")

    # ---------- Q3: vsRandom 胜率 ----------
    print("\n=== Q3  vsRandom 各指挥官胜率(winner==0 占比) ===")
    vsr_wr={}
    for c in ROSTER:
        g=[r for r in vsr if r["tag"]==f"vsRandom:{c}"]
        if not g:
            print(f"  {c:10} 无数据"); continue
        n=len(g); w=sum(1 for r in g if r["winner"]==0)
        vsr_wr[c]=w/n
        print(f"  {c:10} n={n:2d} wins={w:2d} wr={w/n:.3f}")
    below=[c for c in vsr_wr if vsr_wr[c]<0.90]
    print(f"  vsRandom wr<0.90: {below or 'NONE — 全部 >=0.90'}")

    # ---------- Q4: rr 胜率 + 零胜 seat0 曝光 ----------
    print("\n=== Q4  rr 各指挥官胜率(wins/appearances) ===")
    wins=defaultdict(int)
    for r in rr:
        w=r["winner"]
        if w>=0: wins[r["names"][w]]+=1
    rr_wr={}
    print(f"  {'cmd':10} {'app':>4} {'wins':>4} {'rr_wr':>6} {'seat0_app':>9}")
    for c in ROSTER:
        if appear[c]:
            rr_wr[c]=wins[c]/appear[c]
            print(f"  {c:10} {appear[c]:4d} {wins[c]:4d} {wins[c]/appear[c]:6.3f} {seat_appear[c][0]:9d}")
    zero_win=[c for c in ROSTER if appear[c] and wins[c]==0]
    print(f"  rr 零胜指挥官: {zero_win}")
    for c in zero_win:
        print(f"    {c}: seat0出场={seat_appear[c][0]} 座位分布={[seat_appear[c][s] for s in range(4)]}")

    # ---------- Q5: 非退化 ----------
    print("\n=== Q5  非退化 / 质量(全 232 局) ===")
    L=[r["length"] for r in games]
    print(f"  length: min={min(L)} median={statistics.median(L):.0f} max={max(L)} mean={statistics.mean(L):.1f}")
    print(f"  秒杀 <100 tick: {sum(1 for x in L if x<100)}")
    print(f"  撞硬帽 >=6000 tick: {sum(1 for x in L if x>=6000)}")
    print(f"  和局 winner<0: {sum(1 for r in games if r['winner']<0)}  (vsr={sum(1 for r in vsr if r['winner']<0)} rr={sum(1 for r in rr if r['winner']<0)})")
    print(f"  gate 到达率: overall={sum(r['gate'] for r in games)/len(games):.3f} "
          f"vsRandom={sum(r['gate'] for r in vsr)/len(vsr):.3f} rr={sum(r['gate'] for r in rr)/len(rr):.3f}")

    # ---------- Q6: 座位偏置解混淆 ----------
    print("\n=== Q6  座位偏置解混淆 ===")
    # (a) seat0 的胜场由谁贡献(是否集中在个别强指挥官)
    print("  (a) rr seat0 胜场按指挥官拆解(谁坐 seat0 时赢了):")
    seat0_winner_by_cmd=defaultdict(int); seat0_occupant=defaultdict(int)
    for r in rr:
        occ=r["names"][0]; seat0_occupant[occ]+=1
        if r["winner"]==0: seat0_winner_by_cmd[occ]+=1
    for c in ROSTER:
        if seat0_occupant[c]:
            print(f"    {c:10} 坐seat0 {seat0_occupant[c]:2d} 局, 其中赢 {seat0_winner_by_cmd[c]:2d} "
                  f"(seat0胜率={seat0_winner_by_cmd[c]/seat0_occupant[c]:.2f})")
    # (b) 各指挥官 x 座位 的胜场(看同一指挥官换座位表现是否翻转)
    print("  (b) 各指挥官在每座位的 wins/app(解 seat vs cmd 混淆):")
    seatwin=defaultdict(lambda: defaultdict(int))
    for r in rr:
        w=r["winner"]
        if w>=0: seatwin[r["names"][w]][w]+=1
    print(f"    {'cmd':10} " + " ".join(f"s{s}(w/a)" for s in range(4)))
    for c in ROSTER:
        if appear[c]:
            cells=[]
            for s in range(4):
                a=seat_appear[c][s]; w=seatwin[c][s]
                cells.append(f"{w}/{a}" if a else "-/-")
            print(f"    {c:10} " + "  ".join(f"{x:>7}" for x in cells))
    # (c) 有 seat0 曝光却零胜的指挥官 = 更可能真弱
    print("  (c) 有 seat0 曝光(>0)却 rr 零胜的指挥官(更可能真弱,非座位剥夺):")
    real_weak=[c for c in ROSTER if appear[c] and wins[c]==0 and seat_appear[c][0]>0]
    print(f"    {real_weak or 'NONE'}")
    # (d) combo 级别: 胜者座位跨 8 seed 是否一致(反映 seat/matchup 决定 vs 噪声)
    print("  (d) 每 combo 胜者座位跨 seed 分布(看是否某座位垄断该 combo):")
    combo_seatwins=defaultdict(lambda: defaultdict(int)); combo_names={}
    for r in rr:
        combo_seatwins[r["tag"]][r["winner"]]+=1; combo_names[r["tag"]]=r["names"]
    for tag in sorted(combo_seatwins):
        dist={s:combo_seatwins[tag][s] for s in range(4) if combo_seatwins[tag][s]}
        nm=combo_names[tag]
        # 找出该 combo 的主胜座位
        print(f"    {tag:40} names={nm} 胜者座位分布={dist}")

if __name__=="__main__":
    main()
