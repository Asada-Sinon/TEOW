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
    joint = make_joint_controller(*(["scripted"] * cfg.n_players), cfg=cfg, mapdata=m)
    scan = make_scan(step_fn, joint)

    st = state
    seen_camp = False
    seen_lv2 = False        # v1.5:淘汰者 HQ 槽 level 停泊回 1,必须过程中采样
    seen_res2 = False
    hq_slots = [hq_slot(p, cfg) for p in range(cfg.n_players)]
    for _ in range(cfg.episode_len // 100):
        st, key, _ = scan(st, key, 100)
        seen_camp = seen_camp or bool(jnp.any(st.etype == TYPE_CAMP))
        seen_lv2 = seen_lv2 or bool(
            jnp.max(st.level[jnp.asarray(hq_slots)]) >= 2)
        seen_res2 = seen_res2 or bool(jnp.max(st.upgrades) >= 2)
        assert bool(jnp.all(st.resources >= 0))  # 新扣费路径不透支
        if bool(st.done):
            break

    assert seen_lv2, "整局没人升过本"
    assert seen_camp, "全场没出现过训练营"
    assert seen_res2, "整局没人研过线"
    owner = owner_of_slots(cfg)
    del owner
    assert bool(st.done) or int(st.tick) == cfg.episode_len
