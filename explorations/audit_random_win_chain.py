"""v1.0 审计:random(p0) 为何打赢 scripted(p1)——重放 seed 2 取证据链。
逐 tick 记录:p1 HQ hp、贴脸攻击者(类型/归属)、p1 步兵数(是否达到进攻阈值 6)、
p1 是否发过 ATTACK、双方卸货停摆时刻。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_random_win_chain.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import jax
import numpy as np

from teow.config import TYPE_INFANTRY, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.state import ORDER_ATTACK, ORDER_HARVEST, owner_of_slots
from teow.step import new_world

cfg = Config(seed=2)
state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller("random", "scripted", cfg, m))
owner = np.asarray(owner_of_slots(cfg))
HQ1 = cfg.e_max

hq1_hp_hist = []
max_inf_p1 = 0
p1_ever_attack = False
adj_ticks_by_type = {TYPE_WORKER: 0, TYPE_INFANTRY: 0}
dmg_by_type = {TYPE_WORKER: 0, TYPE_INFANTRY: 0}
last_dep = [-1, -1]
prev = {k: np.asarray(v) for k, v in state._asdict().items()}

for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}

    hq1_hp_hist.append(int(cur["hp"][HQ1]))
    n_inf1 = int(((owner == 1) & cur["alive"] & (cur["etype"] == TYPE_INFANTRY)).sum())
    max_inf_p1 = max(max_inf_p1, n_inf1)
    if (((owner == 1) & cur["alive"] & (cur["order"] == ORDER_ATTACK)).any()):
        p1_ever_attack = True
    # p0 单位贴脸 p1 HQ(Chebyshev<=1)
    hq_pos = cur["pos"][HQ1]
    cheb = np.abs(cur["pos"] - hq_pos).max(axis=1)
    adj = (owner == 0) & cur["alive"] & ~cur["inside"] & (cheb <= 1)
    for ty, atk in ((TYPE_WORKER, cfg.worker_atk), (TYPE_INFANTRY, cfg.infantry_atk)):
        k = int((adj & (cur["etype"] == ty)).sum())
        adj_ticks_by_type[ty] += k
        dmg_by_type[ty] += k * atk
    # 卸货侦测
    alive_both = prev["alive"] & cur["alive"]
    drop = np.where(alive_both, prev["cargo"].astype(int) - cur["cargo"], 0)
    for p in (0, 1):
        if drop[(owner == p)].clip(0).sum() > 0:
            last_dep[p] = t
    prev = cur
    if bool(state.done):
        break

hp = np.asarray(hq1_hp_hist)
first_dmg = int(np.argmax(hp < cfg.hq_hp)) if (hp < cfg.hq_hp).any() else -1
print(f"终局 tick={int(state.tick)} winner={int(state.winner)}")
print(f"p1 HQ 首次掉血 tick={first_dmg}; hp 里程碑: "
      f"{[(i, int(hp[i])) for i in range(0, len(hp), 100)]}")
print(f"p1 步兵数峰值={max_inf_p1}(进攻阈值 {cfg.ai_attack_threshold}); "
      f"p1 是否下过 ATTACK={p1_ever_attack}")
print(f"p0 贴 HQ 的单位-tick 数: {adj_ticks_by_type} → 理论伤害 {dmg_by_type} "
      f"(合计 {sum(dmg_by_type.values())},HQ 血量 {cfg.hq_hp})")
print(f"最后卸货 tick: p0(random)={last_dep[0]} p1(scripted)={last_dep[1]}")
st = {k: np.asarray(v) for k, v in state._asdict().items()}
harv_stuck = ((owner == 1) & st["alive"] & (st["order"] == ORDER_HARVEST)).sum()
print(f"末态 p1 HARVEST 工人数={int(harv_stuck)} res_p1={st['resources'][1].tolist()}")
