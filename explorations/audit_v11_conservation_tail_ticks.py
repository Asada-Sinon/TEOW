"""复审 v1.1 audit2 seed13:守恒违例 t=1054/1057/1081 的实体级归因。
问题:违例是引擎错账,还是审计脚本 ab(两帧均存活)门控漏记了
「同 tick 下单即被摧毁 / 卸货即被击杀」?逐 tick 重放,对 t in 焦点集
打印 死亡实体的 btimer/btype/cargo 变化 与 node_build_timer 变化。
用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/audit_v11_conservation_tail_ticks.py <run_dir>
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
FOCUS = {1054, 1057, 1081}

state, key, step_fn, m = new_world(cfg)
joint = jax.jit(make_joint_controller(p0, p1, cfg, m))
owner = np.asarray(owner_of_slots(cfg))
prev = {k: np.asarray(v) for k, v in state._asdict().items()}
for t in range(cfg.episode_len):
    key, ka, ks = jax.random.split(key, 3)
    state = step_fn(state, joint(state, ka), ks)
    cur = {k: np.asarray(v) for k, v in state._asdict().items()}
    if t in FOCUS:
        print(f"--- t={t} Δres={ (cur['resources'].astype(int)-prev['resources']).tolist() }")
        died = prev["alive"] & ~cur["alive"]
        for i in np.nonzero(died)[0]:
            print(f"  死亡 slot={i} p{owner[i]} etype={int(prev['etype'][i])} "
                  f"btype {int(prev['btype'][i])}->{int(cur['btype'][i])} "
                  f"btimer {int(prev['btimer'][i])}->{int(cur['btimer'][i])} "
                  f"cargo {int(prev['cargo'][i])}({int(prev['cargo_type'][i])})->{int(cur['cargo'][i])}")
        nbt = cur["node_build_timer"].astype(int) - prev["node_build_timer"]
        for k in np.nonzero(nbt != 0)[0]:
            print(f"  node{k} owner {int(prev['node_owner'][k])}->{int(cur['node_owner'][k])} "
                  f"build_timer {int(prev['node_build_timer'][k])}->{int(cur['node_build_timer'][k])} "
                  f"ntype={int(np.asarray(m.node_kind)[k]) if hasattr(m,'node_kind') else '?'}")
        # 存活但 btimer 上升的(正常记账路径)也打出来对照
        inc = prev["alive"] & cur["alive"] & (cur["btimer"] > prev["btimer"])
        for i in np.nonzero(inc)[0]:
            print(f"  开单 slot={i} p{owner[i]} etype={int(cur['etype'][i])} btype={int(cur['btype'][i])} btimer->{int(cur['btimer'][i])}")
        # 卸货(含死亡者)
        dropped = (prev["cargo"].astype(int) - np.where(cur["alive"], cur["cargo"], 0))
        for i in np.nonzero(prev["cargo"] > 0)[0]:
            if dropped[i] != 0:
                print(f"  cargo变动 slot={i} p{owner[i]} alive {bool(prev['alive'][i])}->{bool(cur['alive'][i])} "
                      f"cargo {int(prev['cargo'][i])}(type{int(prev['cargo_type'][i])})->{int(cur['cargo'][i]) if cur['alive'][i] else 'dead'} pos={prev['pos'][i].tolist()}")
    prev = cur
    if bool(state.done): break
print("done tick=", int(state.tick))
