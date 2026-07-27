"""交叉校验(不同代码路径,纯 stdlib)+ 座位混淆的决定性证据。只读 games.jsonl。
 (1) 数据完整性: winner_name 是否恒等于 names[winner]。
 (2) 独立重算 rr 座位胜场(用 Counter,与 seatbias_check 的 defaultdict 循环互证)。
 (3) 决定性证据: 每指挥官坐 seat0 时的对手集 + seat0 胜率(证明是对手弱还是座位强)。
 (4) 弱尾/boomer 在 rr 全部出场的对手并集(它们输给谁)。
"""
import json, sys
from collections import Counter, defaultdict
rows=[json.loads(l) for i,l in enumerate(open(sys.argv[1]),1) if l.strip()]
for i,r in enumerate(rows,1): r["_ln"]=i
rr=[r for r in rows if r["tag"].startswith("rr:")]

# (1) 完整性
bad=[r["_ln"] for r in rows if r["winner"]>=0 and r["winner_name"]!=r["names"][r["winner"]]]
print(f"(1) winner_name != names[winner] 的行数 = {len(bad)} (应为0)")

# (2) Counter 重算 rr 座位胜场
sw=Counter(r["winner"] for r in rr if r["winner"]>=0)
print(f"(2) Counter rr 各座位胜场 = {dict(sorted(sw.items()))}  sum={sum(sw.values())} rr局数={len(rr)}")

# (3) 每指挥官坐 seat0 的对手集 + seat0 胜率
print("(3) 每指挥官坐 seat0 时: 局数 / seat0胜 / 对手集:")
occ_games=defaultdict(list)
for r in rr: occ_games[r["names"][0]].append(r)
for occ in sorted(occ_games):
    g=occ_games[occ]
    won=sum(1 for r in g if r["winner"]==0)
    oppsets=sorted(set(tuple(r["names"][1:]) for r in g))
    print(f"   {occ:10} 局数={len(g):2d} seat0胜={won:2d}/{len(g):2d} 对手集={oppsets}")

# (4) 弱尾/boomer 对手并集
print("(4) 弱尾/boomer 在 rr 全部出场的对手并集:")
for c in ["turtle","airtech","chaos","boomer"]:
    sub=[r for r in rr if c in r["names"]]
    opps=sorted(set(x for r in sub for x in r["names"] if x!=c))
    print(f"   {c:10} 出场={len(sub)} 对手并集={opps}")
