# Aggregates v1.9 commander round-robin eval.
# Reads ONLY experiments/20260727-v19-roundrobin/games.jsonl (read-only product).
# Answers: (1) vs-random win-rates/length/gate, (2) round-robin win-rate ranking,
# (3) non-degeneracy (length dist, 秒杀<100, hard-cap 6000, gate fraction),
# (4) style fingerprint, (5/6) filter + difficulty tiers.
# All numbers printed here are computed from the file; nothing hand-derived.
import json, statistics
from collections import defaultdict

P = "/home/michael/workspace/pi05/temp/TEOW/experiments/20260727-v19-roundrobin/games.jsonl"
EPISODE_LEN = 6000  # from design.json / resolved_config.json
ROSTER = ['balanced','boomer','rusher','turtle','timing','harasser','airtech','tempo','counter','chaos']

rows=[]
with open(P) as f:
    for i,line in enumerate(f,1):
        line=line.strip()
        if not line: continue
        r=json.loads(line); r["_ln"]=i; rows.append(r)

vsr=[r for r in rows if r["tag"].startswith("vsRandom:")]
rr =[r for r in rows if r["tag"].startswith("rr:")]
print(f"total rows={len(rows)} vsRandom={len(vsr)} rr={len(rr)}")

# ---- winner value sanity ----
wv_vsr=sorted(set(r["winner"] for r in vsr)); wv_rr=sorted(set(r["winner"] for r in rr))
print(f"winner values vsRandom={wv_vsr} rr={wv_rr}")

# ================= PHASE A: vs random =================
print("\n===== PHASE A: vs random x3 (commander at seat0) =====")
print(f"{'cmd':10} {'n':>2} {'wins':>4} {'wr':>4} {'avglen':>7} {'gateReach':>9} {'avgArmy':>7} {'len_min':>7} {'len_max':>7}")
vsr_stats={}
for c in ROSTER:
    g=[r for r in vsr if r["tag"]==f"vsRandom:{c}"]
    n=len(g); wins=sum(1 for r in g if r["winner"]==0)  # seat0 = commander
    wr=wins/n if n else float('nan')
    lens=[r["length"] for r in g]
    gate=sum(r["gate"] for r in g)/n
    army=statistics.mean(r["army"][0] for r in g)  # seat0 army
    vsr_stats[c]=dict(n=n,wins=wins,wr=wr,avglen=statistics.mean(lens),gate=gate,army=army,lmin=min(lens),lmax=max(lens))
    print(f"{c:10} {n:2d} {wins:4d} {wr:4.2f} {statistics.mean(lens):7.0f} {gate:9.2f} {army:7.1f} {min(lens):7d} {max(lens):7d}")
below=[c for c in ROSTER if vsr_stats[c]["wr"]<0.9]
print(f"vsRandom wr<0.90 (fail criterion): {below if below else 'NONE — all >=0.90'}")

# ================= PHASE B: round robin =================
print("\n===== PHASE B: round-robin relative strength =====")
appear=defaultdict(int); winsB=defaultdict(int)
win_lens=defaultdict(list); appear_gate=defaultdict(int); appear_army=defaultdict(list)
for r in rr:
    seat_names=r["names"]
    w=r["winner"]
    for seat,name in enumerate(seat_names):
        appear[name]+=1
        appear_army[name].append(r["army"][seat])
    if w>=0:
        wname=seat_names[w]
        winsB[wname]+=1
        win_lens[wname].append(r["length"])
        if r["gate"]==1: appear_gate[wname]+=1

print(f"{'cmd':10} {'appear':>6} {'wins':>4} {'rr_wr':>5} {'avgWinLen':>9} {'winGateFrac':>11} {'avgArmyAllApp':>13}")
rr_stats={}
for c in ROSTER:
    a=appear[c]; w=winsB[c]; wr=w/a if a else float('nan')
    awl=statistics.mean(win_lens[c]) if win_lens[c] else float('nan')
    wgf=(appear_gate[c]/w) if w else float('nan')
    aa=statistics.mean(appear_army[c]) if appear_army[c] else float('nan')
    rr_stats[c]=dict(a=a,w=w,wr=wr,awl=awl,wgf=wgf,aa=aa)
    print(f"{c:10} {a:6d} {w:4d} {wr:5.2f} {awl:9.1f} {wgf:11.2f} {aa:13.2f}")

print("\n-- DIFFICULTY RANKING (round-robin win-rate, strong->weak) --")
rank=sorted(ROSTER,key=lambda c:(-rr_stats[c]["wr"], -rr_stats[c]["w"]))
for i,c in enumerate(rank,1):
    s=rr_stats[c]
    print(f"{i:2d}. {c:10} rr_wr={s['wr']:.2f} ({s['w']}/{s['a']})")

# fail criteria
dom=[c for c in ROSTER if rr_stats[c]["wr"]>0.85]
weak=[c for c in ROSTER if rr_stats[c]["wr"]<0.10]
print(f"\nrr_wr>0.85 (dominates -> balance flag): {dom if dom else 'NONE'}")
print(f"rr_wr<0.10 (too weak -> fix/delete flag): {weak if weak else 'NONE'}")
neverwin=[c for c in ROSTER if rr_stats[c]["w"]==0]
print(f"never wins any rr game: {neverwin if neverwin else 'NONE'}")

# ================= NON-DEGENERACY =================
print("\n===== NON-DEGENERACY / QUALITY (all 80 games) =====")
alllen=[r["length"] for r in rows]
print(f"length min={min(alllen)} median={statistics.median(alllen):.0f} max={max(alllen)} mean={statistics.mean(alllen):.0f}")
kill=[r for r in rows if r["length"]<100]
cap =[r for r in rows if r["length"]>=EPISODE_LEN]
near=[r for r in rows if r["length"]>=5000]
print(f"秒杀 <100 ticks: {len(kill)}")
print(f"hard-cap >=6000: {len(cap)}")
print(f">=5000 ticks (near cap): {len(near)}  -> tags: {sorted(set(r['tag'] for r in near))}")
print(f"gate reached overall: {sum(r['gate'] for r in rows)}/{len(rows)} = {sum(r['gate'] for r in rows)/len(rows):.2f}")
print(f"  vsRandom gate frac: {sum(r['gate'] for r in vsr)/len(vsr):.2f}   rr gate frac: {sum(r['gate'] for r in rr)/len(rr):.2f}")
# winner==-1 draws?
draws=[r for r in rows if r["winner"]<0]
print(f"games with no winner (winner<0): {len(draws)}")

# length buckets
import bisect
edges=[100,1000,2000,3000,4000,5000,6000]
buck=defaultdict(int)
for L in alllen:
    b=bisect.bisect_right(edges,L); buck[b]+=1
labels=["<100","100-1k","1k-2k","2k-3k","3k-4k","4k-5k","5k-6k",">=6k"]
print("length histogram:", {labels[k]:buck[k] for k in range(len(labels)) if buck[k]})
