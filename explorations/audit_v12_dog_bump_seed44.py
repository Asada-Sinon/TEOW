"""v1.2 复审定向核验(audit2, seed44):步兵线研发完成拍,存量狗/步兵 hp 恰 +表差额,
工人 hp 不动;工人线完成拍工人 +表差额,狗/步兵不动。
方法:同 seed 逐 tick 重放(与录制同后端 cpu),抓 upgrades 跳变的精确 tick,
对比前后拍每个存活单位 hp 增量;当拍受敌方伤害的单位以 (Δhp - 期望差额) 是否为
非正伤害量判定(满血→满血的未受伤单位必须精确相等)。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v12_dog_bump_seed44.py <run_dir>
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import numpy as np
from teow.config import TYPE_DOG, TYPE_INFANTRY, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.state import owner_of_slots
from teow.step import new_world

run_dir = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
cmd = (run_dir / "command.txt").read_text().split()
p0 = cmd[cmd.index("--p0") + 1]; p1 = cmd[cmd.index("--p1") + 1]
print(f"run={run_dir.name} seed={cfg.seed} backend={jax.default_backend()}")
state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(p0, p1, cfg, m))
owner = np.asarray(owner_of_slots(cfg))
tables = {TYPE_INFANTRY: (np.asarray(cfg.inf_hp_by_level), 0),
          TYPE_DOG: (np.asarray(cfg.dog_hp_by_level), 0),
          TYPE_WORKER: (np.asarray(cfg.worker_hp_by_level), 1)}
names = {TYPE_INFANTRY: "步兵", TYPE_DOG: "狗", TYPE_WORKER: "工人"}
bad = 0; events = 0
for t in range(int(cfg.episode_len)):
    prev = state
    key, ka, ks = jax.random.split(key, 3)  # 与 run.py 录制一致的 key 流
    state = step_fn(state, joint(state, ka), ks)
    up0 = np.asarray(prev.upgrades); up1 = np.asarray(state.upgrades)
    if (up1 != up0).any():
        for p in (0, 1):
            for line in (0, 1):
                o, n = int(up0[p, line]), int(up1[p, line])
                if o == n: continue
                events += 1
                assert n == o + 1, f"线级步进违例 t={t} p={p} line={line} {o}->{n}"
                al0 = np.asarray(prev.alive); al1 = np.asarray(state.alive)
                et = np.asarray(prev.etype); hp0 = np.asarray(prev.hp); hp1 = np.asarray(state.hp)
                for ut, (tab, tline) in tables.items():
                    exp = int(tab[n] - tab[o]) if tline == line else 0
                    idx = np.where(al0 & al1 & (owner == p) & (et == ut))[0]
                    full = hp0[idx] == tab[o if tline == line else int(up0[p, tline])]
                    d = hp1[idx] - hp0[idx]
                    exact = int((d[full] == exp).sum()); n_full = int(full.sum())
                    over = int((d > exp).sum())  # 任何单位增量超过差额 = 凭空加血
                    print(f"t={t+1} p{p} 线{line} {o}->{n}: {names[ut]}存量{len(idx)} "
                          f"满血{n_full} 满血精确+{exp}: {exact}/{n_full} 超差额:{over}")
                    if over or exact != n_full: bad += 1
print(f"研发完成事件={events} 违例={bad}")
