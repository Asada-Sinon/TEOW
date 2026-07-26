# Follow-up: (a) determinism of length=4182, (b) rusher head-to-head,
# (c) opponent sets for thin-sample commanders, (d) macro-commander loss targets.
import json
from collections import defaultdict, Counter
P="/home/michael/workspace/pi05/temp/TEOW/experiments/20260727-v19-roundrobin/games.jsonl"
rows=[json.loads(l) for l in open(P) if l.strip()]
rr=[r for r in rows if r["tag"].startswith("rr:")]
vsr=[r for r in rows if r["tag"].startswith("vsRandom:")]

print("=== length value frequency (all 80) ===")
c=Counter(r["length"] for r in rows)
for L,n in sorted(c.items(),key=lambda x:-x[1])[:8]:
    print(f"  len={L}: {n} games")
print(f"  exactly 4182: {sum(1 for r in rows if r['length']==4182)} games")

print("\n=== vsRandom: winner army==0 (passive win?) ===")
for r in vsr:
    if r['army'][0]==0:
        print(f"  {r['names'][0]:9} seed{r['seed']} len={r['length']} gate={r['gate']} army={r['army']}")

print("\n=== rusher rr matchups (opponent -> did rusher win?) ===")
for r in rr:
    if "rusher" in r["names"]:
        seat=r["names"].index("rusher"); won = r["winner"]==seat
        print(f"  {r['tag'][3:]:35} seed{r['seed']} winner={r['winner_name']:9} rusherWon={won}")

print("\n=== opponent set (distinct matchups) per commander ===")
mset=defaultdict(set)
for r in rr:
    key=r["tag"]
    for n in r["names"]: mset[n].add(key)
for cmd in ['balanced','boomer','rusher','turtle','timing','harasser','airtech','tempo','counter','chaos']:
    print(f"  {cmd:9} in {len(mset[cmd])} matchups, {len(mset[cmd])*4} games")

print("\n=== who did turtle/timing/airtech LOSE to? (winner of their games) ===")
for cmd in ['turtle','timing','airtech']:
    winners=Counter()
    for r in rr:
        if cmd in r["names"]:
            winners[r["winner_name"]]+=1
    print(f"  {cmd:8}: winners of its games -> {dict(winners)}")
