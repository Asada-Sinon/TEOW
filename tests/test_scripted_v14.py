"""v1.4 涌现验证:富开局覆盖局里 scripted AI 真的用上整棵兵种树。

配置=覆盖局标定值(explorations 同款;默认 config 的对局节奏测试仍在
test_scripted_v13):富开局 + 基地目标 5 + 攻击阈值 14,seed 0 实测时间线
mortar 391 → bar5 1242 → ram 1622 → mortar 开火 1636 → 1952 分胜负。
断言全部走事件旗(any over scan),不做时间线断言。"""

import jax
import jax.numpy as jnp

from teow.config import (
    TYPE_ARCHER,
    TYPE_HEALER,
    TYPE_HEAVY,
    TYPE_MAGE,
    TYPE_MORTAR,
    TYPE_RAM,
    TYPE_STRONGMAN,
    TYPE_WAGON,
    Config,
)
from teow.controller import make_joint_controller
from teow.state import ORDER_GARRISON
from teow.step import new_world

COVER_CFG = dict(start_ore=3500, start_water=2200, ai_base_level_target=5,
                 ai_worker_target=6, ai_attack_threshold=14)


def test_scripted_exercises_full_unit_tree():
    cfg = Config(seed=0, **COVER_CFG)
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller("scripted", "scripted", cfg=cfg, mapdata=m)

    watch_types = (TYPE_STRONGMAN, TYPE_WAGON, TYPE_ARCHER, TYPE_HEAVY,
                   TYPE_MAGE, TYPE_HEALER, TYPE_RAM, TYPE_MORTAR)
    is_harv = ((state.etype == 4) | (state.etype == TYPE_STRONGMAN)
               | (state.etype == TYPE_WAGON))
    del is_harv  # 采集单位驻守检查在 body 内按当帧 etype 现算

    def body(carry, _):
        st, k = carry
        k, k_act, k_step = jax.random.split(k, 3)
        st = step_fn(st, joint(st, k_act), k_step)
        al = st.alive
        seen_types = jnp.stack(
            [jnp.any(al & (st.etype == t)) for t in watch_types])
        fired = jnp.any(al & (st.etype == TYPE_MORTAR) & (st.shell_timer > 0))
        harv_gar = jnp.any(
            al & (st.order == ORDER_GARRISON)
            & ((st.etype == 4) | (st.etype == TYPE_STRONGMAN)
               | (st.etype == TYPE_WAGON)))
        bar5 = jnp.any(al & (st.etype == 7) & (st.level >= 5))
        return (st, k), (seen_types, fired, harv_gar, bar5, st.winner)

    (fin, _), (seen, fired, harv_gar, bar5, winners) = jax.lax.scan(
        body, (state, key), None, length=cfg.episode_len)

    seen_any = jnp.any(seen, axis=0)
    names = ("大力士", "马车", "弓箭手", "重甲", "法师", "奶妈", "攻城车", "迫击炮")
    for i, nm in enumerate(names):
        assert bool(seen_any[i]), f"整局未出现 {nm}"
    assert bool(jnp.any(fired)), "迫击炮整局没开过火"
    assert bool(jnp.any(bar5)), "兵营整局没到过 5 级"
    assert not bool(jnp.any(harv_gar)), "采集单位出现在驻守状态(规格违例)"
    assert int(fin.winner) != -1, "覆盖局应分出结果(胜或和)"
