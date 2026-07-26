"""v1.8 审计续:细分 monster_hp/alive 不一致方向 + 从 metrics 核 winner。"""
import numpy as np, json
d = np.load('experiments/20260727-v18-audit-cover/trajectory.npz')
P, E = 4, 64
ma = d['monster_alive']; mhp = d['monster_hp']; tick = d['tick']
alive = d['alive']
hq_alive = np.stack([alive[:, p*E] for p in range(P)], axis=1)  # [T,P]

# 方向 A(危险):alive=True 但 hp<=0 —— 0 血僵尸怪仍活
zombie = ma & (mhp <= 0)
# 方向 B(良性):alive=False 但 hp>0 —— despawn 后 hp 残留(cleanup 未清 hp)
stale = (~ma) & (mhp > 0)
print("方向A 危险(alive&hp<=0)僵尸怪 计数:", int(zombie.sum()),
      " 出现帧数:", int(zombie.any(axis=(1,2)).sum()))
print("方向B 良性(~alive&hp>0)残留 计数:", int(stale.sum()),
      " 出现帧数:", int(stale.any(axis=(1,2)).sum()))

# 残留只出现在死亡玩家行?
stale_rows = np.where(stale.any(axis=2))  # (帧, 玩家)
bad_alive_player = 0
for t, p in zip(*stale_rows):
    if hq_alive[t, p]:          # 若该玩家还活着却有 ~alive&hp>0 → 反常
        bad_alive_player += 1
print("  残留发生在【存活】玩家行的次数(应为0):", bad_alive_player)

# 每玩家死亡帧(hq 首次死)与 gate 关系
for p in range(P):
    dead_idx = np.where(~hq_alive[:, p])[0]
    if len(dead_idx):
        print(f"  玩家{p} hq 首次死于记录 tick={int(tick[dead_idx[0]])}")
    else:
        print(f"  玩家{p} 全程存活")

# metrics winner 核验(轨迹因 record_every=2 漏掉奇数 done 帧)
rows = [json.loads(l) for l in open('experiments/20260727-v18-audit-cover/metrics.jsonl')]
wv = set(r['winner'] for r in rows if r['done'])
print("metrics done 帧 winner 集合:", wv, " 绝不==4(和局哨兵):", all(w!=4 for w in wv),
      " ∈[0,3]:", all(0<=w<=3 for w in wv))
last = rows[-1]
print("末行:", {k: last[k] for k in ('tick','done','winner')})
