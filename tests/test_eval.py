"""v1.9 评测 rollout 原语 src/teow/eval.py 冒烟(小局:短 episode + 早 gate + 强怪,快分胜负)。"""

import jax.numpy as jnp

from teow.config import Config
from teow.eval import matchup_runner, run_matchup

_FAST = dict(episode_len=400, gate_open_tick=100, monster_atk_base=120,
             monster_hp_base=200, monster_speed=0.8)


def test_matchup_runner_metrics_shape_and_winner():
    cfg = Config(**_FAST)
    run = matchup_runner(cfg, ("rusher", "random", "boomer", "random"))
    out = run(jnp.arange(3, dtype=jnp.int32))
    assert out["winner"].shape == (3,)
    assert out["army"].shape == (3, cfg.n_players)
    assert out["hq_alive"].shape == (3, cfg.n_players)
    w = out["winner"]
    assert bool(jnp.all((w >= 0) & (w < cfg.n_players))), "v1.8 必分胜负:winner∈[0,P-1]"


def test_run_matchup_rows():
    cfg = Config(**_FAST)
    rows = run_matchup(cfg, ("rusher", "random", "turtle", "random"), [0, 1])
    assert len(rows) == 2
    for r in rows:
        assert r["winner_name"] in r["names"]
        assert 0 < r["length"] <= cfg.episode_len
