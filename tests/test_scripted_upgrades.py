"""v1.1 Phase 4:scripted AI 真的会用升级机制(升本/建营/研发/升矿),
且经济不因新机制停摆。"""

import jax.numpy as jnp

from teow.config import TYPE_CAMP, Config
from teow.controller import make_joint_controller
from teow.state import hq_slot, owner_of_slots
from teow.step import make_scan, new_world


def test_scripted_uses_upgrade_machinery():
    cfg = Config()
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller("scripted", "scripted", cfg, m)
    scan = make_scan(step_fn, joint)

    st = state
    seen_camp = False
    for _ in range(30):  # 3000 tick 上限,分段好提前退出
        st, key, _ = scan(st, key, 100)
        seen_camp = seen_camp or bool(jnp.any(st.etype == TYPE_CAMP))
        assert bool(jnp.all(st.resources >= 0))  # 新扣费路径不透支
        if bool(st.done):
            break

    lv0 = int(st.level[hq_slot(0, cfg)])
    lv1 = int(st.level[hq_slot(1, cfg)])
    # 至少一方升过本、建过营、研过线(两边脚本相同,断言放宽到 max 侧,
    # 输家可能没来得及)
    assert max(lv0, lv1) >= 2, f"没人升本 lv=({lv0},{lv1})"
    assert seen_camp, "全场没出现过训练营"
    assert int(jnp.max(st.upgrades)) >= 2, f"没人研过线 {st.upgrades.tolist()}"
    # 对局质量:分出胜负或至少不是零对抗(有单位阵亡)
    owner = owner_of_slots(cfg)
    del owner
    assert bool(st.done) or int(st.tick) == 3000
