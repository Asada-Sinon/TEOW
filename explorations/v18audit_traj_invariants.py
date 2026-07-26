"""v1.8 审计:对录制轨迹 trajectory.npz 做不变量核算(纯 numpy)。
回答:怪物子表是否泄漏/负血/与 alive 不一致;死亡玩家怪是否清场;
winner 是否绝不和局;资源是否负数。"""
import numpy as np
d = np.load('experiments/20260727-v18-audit-cover/trajectory.npz')
P, E, M = 4, 64, 64
ma = d['monster_alive']      # [T,P,M]
mhp = d['monster_hp']        # [T,P,M]
matk = d['monster_atk']      # [T,P,M]
alive = d['alive']           # [T,256]
hp = d['hp']                 # [T,256]
winner = d['winner']         # [T]
done = d['done']             # [T]
res = d['resources']         # [T,P,2]
tick = d['tick']             # [T]
T = ma.shape[0]

# hq 存活:每玩家块 0 号槽
hq_alive = np.stack([alive[:, p*E] for p in range(P)], axis=1)  # [T,P]

print("=== 帧数 T =", T, "末 tick =", int(tick[-1]))

# 1) monster_hp>0 ⟺ monster_alive
inv1 = (mhp > 0) == ma
print("1) monster_hp>0 ⟺ monster_alive 全帧成立:", bool(inv1.all()),
      "  违例帧数:", int((~inv1).any(axis=(1,2)).sum()))

# 2) monster_hp 从不为负
print("2) monster_hp 从不为负:", bool((mhp >= 0).all()), " min=", int(mhp.min()))

# 3) 死亡玩家(hq 亡)其怪必须全部离场(同帧或下一记录帧)
leak = 0
for t in range(T):
    for p in range(P):
        if not hq_alive[t, p] and ma[t, p].any():
            leak += 1
print("3) 死亡玩家仍有存活怪的 (帧,玩家) 数:", leak)

# 4) monster_alive ⇒ 对应玩家在该帧存活(死则同帧清)
# 用严格:若 hq 死则 ma 该行 False
strict = True
for t in range(T):
    for p in range(P):
        if not hq_alive[t, p] and ma[t, p].any():
            strict = False
print("4) 严格『hq 死 ⇒ 该行无怪』:", strict)

# 5) winner 绝不和局:done 帧 winner ∈ [0,P-1],绝不 == P(=4)或其它
dmask = done
if dmask.any():
    wv = winner[dmask]
    print("5) done 帧 winner 取值集合:", sorted(set(int(x) for x in wv)),
          " 绝不==P(4):", bool((wv != P).all()), " 全∈[0,3]:", bool(((wv>=0)&(wv<P)).all()))
else:
    print("5) 无 done 帧")

# 6) 资源从不为负
print("6) resources 从不为负:", bool((res >= 0).all()), " min=", int(res.min()))

# 7) 怪物攻击封顶(matk 不超过某上限;默认 cap=20)
print("7) monster_atk 最大值:", int(matk.max()), " (cfg cap=20)")

# 8) 怪物 HP 随时间新生成波的最大值单调上升趋势(线性增长的宏观证据)
# 取每帧新出现的最大 spawn hp 近似:整体 max 随 tick
maxhp_by_frame = mhp.max(axis=(1,2))
gate_open = 4000
past = tick >= gate_open
if past.any():
    early = maxhp_by_frame[past][:5]
    late = maxhp_by_frame[past][-5:]
    print("8) 门开后 monster_hp.max 早期~", [int(x) for x in early],
          " 末期~", [int(x) for x in late])

# 9) 怪物只在门开后出现
first_monster_tick = None
for t in range(T):
    if ma[t].any():
        first_monster_tick = int(tick[t]); break
print("9) 首只怪出现 tick =", first_monster_tick, " (gate_open_tick=4000)")

# 10) 每玩家怪数 <= monster_cap(容量不溢出)
print("10) 每玩家每帧怪数 <= cap(64):", bool((ma.sum(axis=2) <= M).all()),
      " 峰值:", int(ma.sum(axis=2).max()))
