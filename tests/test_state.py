"""初始状态:形状、HQ 槽位约定、起始单位与资源。"""

import jax.numpy as jnp

from teow.config import TYPE_HQ, TYPE_WORKER, Config
from teow.map import build_map
from teow.state import hq_slot, init_state, owner_of_slots


def test_init_state():
    cfg = Config()
    m = build_map(cfg)
    st = init_state(cfg, m)
    n = cfg.n_total

    assert st.alive.shape == (n,)
    assert st.pos.shape == (n, 2)
    assert st.node_owner.shape == (cfg.n_nodes,)

    for p in (0, 1):
        base = hq_slot(p, cfg)
        assert st.etype[base] == TYPE_HQ
        assert st.hp[base] == cfg.hq_hp
        assert tuple(st.pos[base]) == tuple(m.hq_pos[p])
        workers = (st.alive & (st.etype == TYPE_WORKER)
                   & (owner_of_slots(cfg) == p))
        assert int(jnp.sum(workers)) == cfg.start_workers

    assert int(jnp.sum(st.alive)) == 2 * (1 + cfg.start_workers)
    assert st.resources.tolist() == [[cfg.start_ore, cfg.start_water]] * 2
    assert int(st.tick) == 0 and not bool(st.done) and int(st.winner) == -1
    # 所有资源点无主、未建、无施工
    assert jnp.all(st.node_owner == -1) and jnp.all(st.node_ent == -1)
    assert jnp.all(st.node_build_timer == 0)
