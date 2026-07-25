"""v1.4 审计辅助:trajectory.npz 中各实体类型是否出现过(按玩家),
回答「覆盖局是否真覆盖九新兵种+迫击炮开火」。
用法: .venv/bin/python explorations/audit_v14_type_coverage.py <run_dir>"""
import pathlib
import sys

import numpy as np

NAMES = {1: "HQ", 2: "矿", 3: "泵", 4: "工人", 5: "步兵", 6: "营", 7: "兵营",
         8: "狗", 9: "哨塔", 10: "大力士", 11: "马车", 12: "弓箭手", 13: "轻骑",
         14: "重甲", 15: "法师", 16: "奶妈", 17: "攻城车", 18: "迫击炮"}

run = pathlib.Path(sys.argv[1])
raw = np.load(run / "trajectory.npz")
alive, etype = raw["alive"], raw["etype"]        # [T,N]
half = alive.shape[1] // 2
for p, sl in ((0, slice(0, half)), (1, slice(half, None))):
    seen = sorted(set(etype[:, sl][alive[:, sl]].tolist()))
    print(f"p{p} 出现过的类型:", [f"{t}:{NAMES.get(t, '?')}" for t in seen])
shot = (raw["shell_timer"] > 0).any(axis=0)
print("有过在途弹的槽:", np.nonzero(shot)[0].tolist(),
      "类型:", etype[-1, np.nonzero(shot)[0]].tolist())
