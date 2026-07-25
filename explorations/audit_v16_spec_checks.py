"""v1.6 终审规格专项(读 trajectory.npz 离线核对,不复跑引擎):
回答:①七个 v1.6 新实体是否在覆盖局出现;②对空表反证——空军掉血帧必须存在
射程内(+两步余量)可对空敌方攻击者;③数量上限:法师塔/喷火器/激光炮/迫击炮
每玩家≤1、地雷≤5(逐帧);④解锁门:各新建筑首现帧其玩家 HQ 等级≥解锁级,
投石车/飞艇首现帧其玩家兵营最高级≥6、龙≥7;⑤地雷一次性:雷亡帧触发圈内
(+余量)须有敌方地面单位(或该家被淘汰);⑥aboard 全程状态统计(scripted
不登艇,预期恒 -1);⑦资源逐帧非负。
用法:.venv/bin/python explorations/audit_v16_spec_checks.py <run_dir>
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from teow.config import Config

run = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run / "resolved_config.json").read_text()))
tr = np.load(run / "trajectory.npz")
P, E = cfg.n_players, cfg.e_max
N = P * E
owner = np.repeat(np.arange(P), E)
F = tr["tick"].shape[0]
rec = int(tr["record_every"])
print(f"run={run.name} frames={F} record_every={rec}")

MT, LM, FL, LS, CAT, AIR, DRG, MOR = 22, 23, 24, 25, 26, 27, 28, 18
names = {MT: "法师塔", LM: "地雷", FL: "喷火器", LS: "激光炮",
         CAT: "投石车", AIR: "飞艇", DRG: "龙"}
alive = tr["alive"]; etype = tr["etype"]; hp = tr["hp"]; pos = tr["pos"]
level = tr["level"]; res = tr["resources"]; aboard = tr["aboard"]

# ① 七新实体出现
print("① 新实体出现帧(首帧, 峰值数量):")
for t, nm in names.items():
    cnt = (alive & (etype == t)).sum(axis=1)
    first = int(np.argmax(cnt > 0)) if (cnt > 0).any() else -1
    print(f"   {nm}: 首帧={first} 峰值={int(cnt.max())}")

# ② 空军掉血帧须有可对空敌方攻击者(帧间隔 rec tick,余量按双方 rec+1 步速度)
can_aa = np.zeros(32, bool)
for t in (12, 15, 9, 22, 25, 28):  # 弓/法/塔/法师塔/激光炮/龙
    can_aa[t] = True
rng_t = np.asarray(cfg.atk_range_by_type)
is_air_t = np.asarray(cfg.is_air_by_type).astype(bool)
spd_t = np.asarray(cfg.speed_by_type)
bad_air_hits = 0
for f in range(1, F):
    a0, a1 = alive[f - 1], alive[f]
    airu = a0 & a1 & is_air_t[np.clip(etype[f], 0, 31)]
    drop = airu & (hp[f] < hp[f - 1])
    if not drop.any():
        continue
    et0 = np.clip(etype[f - 1], 0, 31)
    att = a0 & can_aa[et0]
    d = np.linalg.norm(pos[f - 1][:, None, :] - pos[f - 1][None, :, :], axis=-1)
    margin = (spd_t[et0][None, :] + spd_t[np.clip(etype[f], 0, 31)][:, None]) * (rec + 1)
    ok = (att[None, :] & (owner[:, None] != owner[None, :])
          & (d <= rng_t[et0][None, :] + margin))
    miss = drop & ~ok.any(axis=1)
    if miss.any():
        bad_air_hits += 1
        if bad_air_hits <= 5:
            print(f"   [!] 帧{f} 空军掉血无可对空来源 slots={np.nonzero(miss)[0].tolist()}")
print(f"② 空军掉血无对空来源的帧数: {bad_air_hits}")

# ③ 数量上限逐帧
cap_bad = 0
for t, cap in ((MT, 1), (FL, 1), (LS, 1), (MOR, 1), (LM, 5)):
    cnt = np.zeros((F, P), int)
    for p in range(P):
        cnt[:, p] = (alive[:, p * E:(p + 1) * E]
                     & (etype[:, p * E:(p + 1) * E] == t)).sum(axis=1)
    if (cnt > cap).any():
        cap_bad += 1
        print(f"   [!] {names.get(t, t)} 超上限 max={cnt.max()}")
print(f"③ 数量上限违例种数: {cap_bad}")

# ④ 解锁门(首现帧口径;帧间隔内 HQ 不会降级,安全)
unlock = {MT: 3, LM: 4, FL: 6, LS: 7}
gate_bad = 0
for t, need in unlock.items():
    for p in range(P):
        blk = slice(p * E, (p + 1) * E)
        has = (alive[:, blk] & (etype[:, blk] == t)).any(axis=1)
        if has.any():
            f0 = int(np.argmax(has))
            hq_lv = int(level[f0, p * E])
            if hq_lv < need:
                gate_bad += 1
                print(f"   [!] p{p} {names[t]} 首现帧{f0} HQ级{hq_lv} < {need}")
for t, need in ((CAT, 6), (AIR, 6), (DRG, 7)):
    for p in range(P):
        blk = slice(p * E, (p + 1) * E)
        has = (alive[:, blk] & (etype[:, blk] == t)).any(axis=1)
        if has.any():
            f0 = int(np.argmax(has))
            bar = (alive[f0, blk] & (etype[f0, blk] == 7))
            bar_lv = int(level[f0, blk][bar].max()) if bar.any() else 0
            if bar_lv < need:
                gate_bad += 1
                print(f"   [!] p{p} {names[t]} 首现帧{f0} 兵营最高级{bar_lv} < {need}")
print(f"④ 解锁门违例数: {gate_bad}")

# ⑤ 地雷一次性:雷亡帧(非淘汰)触发圈+余量内须有敌方地面单位
mine_bad = 0; mine_deaths = 0
for f in range(1, F):
    died = alive[f - 1] & ~alive[f] & (etype[f - 1] == LM)
    if not died.any():
        continue
    for i in np.nonzero(died)[0]:
        p = owner[i]
        if not alive[f, p * E]:      # 淘汰清场
            continue
        mine_deaths += 1
        et0 = np.clip(etype[f - 1], 0, 31)
        gu = (alive[f - 1] & (spd_t[et0] > 0) & ~is_air_t[et0]
              & (owner != p))
        d = np.linalg.norm(pos[f - 1][gu] - pos[f - 1][i], axis=-1)
        margin = cfg.landmine_trigger_radius + spd_t[et0][gu].max(initial=0) * (rec + 1)
        if not (d <= margin).any():
            mine_bad += 1
            print(f"   [!] 帧{f} 雷{i} 亡但触发圈内无敌地面单位")
print(f"⑤ 雷爆(非淘汰)次数={mine_deaths} 无触发者违例={mine_bad}")

# ⑥⑦
print(f"⑥ aboard>=0 帧实体数: {int((aboard >= 0).sum())}(scripted 预期 0)")
print(f"⑦ 资源最小值: {int(res.min())}(须 >=0)")
