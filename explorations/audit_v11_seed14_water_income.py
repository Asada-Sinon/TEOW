"""复审 seed14(random vs scripted):scripted(p1)是否复现 seed12 式
「水收入长期归零」活锁(rich_for_node 修复效果验证)。
逐 tick 重放,统计 p1 分资源类型的卸货时间线,输出最大水卸货间隔;
顺带归因 t=1010 守恒违例(p0 +10 矿无卸货记账)。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v11_seed14_water_income.py <run_dir>
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax
import numpy as np
from teow.config import Config
from teow.controller import make_joint_controller
from teow.state import owner_of_slots
from teow.step import new_world

run_dir = pathlib.Path(sys.argv[1])
cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
cmd = (run_dir / "command.txt").read_text().split()
p0 = cmd[cmd.index("--p0") + 1] if "--p0" in cmd else "scripted"
p1 = cmd[cmd.index("--p1") + 1] if "--p1" in cmd else "scripted"
state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(p0, p1, cfg, m))
owner = np.asarray(owner_of_slots(cfg))
prev = {k: np.asarray(v) for k, v in state._asdict().items()}
last = {(p, r): 0 for p in (0, 1) for r in (0, 1)}
gap = {(p, r): (0, -1) for p in (0, 1) for r in (0, 1)}
for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}
    ab = prev["alive"] & cur["alive"]
    drop = np.maximum(np.where(ab, prev["cargo"].astype(int) - cur["cargo"], 0), 0)
    for i in np.nonzero(drop)[0]:
        p, r = int(owner[i]), int(prev["cargo_type"][i])
        if t - last[(p, r)] > gap[(p, r)][0]:
            gap[(p, r)] = (t - last[(p, r)], last[(p, r)])
        last[(p, r)] = t
    if t == 1010:
        died = prev["alive"] & ~cur["alive"]
        for i in np.nonzero(died)[0]:
            print(f"t=1010 死亡 slot={i} p{owner[i]} etype={int(prev['etype'][i])} "
                  f"cargo={int(prev['cargo'][i])}(type{int(prev['cargo_type'][i])}) pos={prev['pos'][i].tolist()}")
    prev = cur
    if bool(state.done): break
end_t = int(state.tick)
for (p, r), (g, s) in sorted(gap.items()):
    tail = end_t - last[(p, r)]
    print(f"p{p} {'矿' if r==0 else '水'}: 最大卸货间隔={g} (起点t={s}) 末次卸货t={last[(p,r)]} 距终局={tail}")
