"""对局评测 / 批量 rollout(v1.9 提升自 explorations/eval_commanders_v18.py,DECISIONS
2026-07-26;v2.0 RL rollout 复用此批量骨架)。

核心 = `matchup_runner`:一个 matchup(P 家指挥官名)`build_step` 编译一次、`jax.vmap`
over seeds 批量跑整局(GPU B≈64 ~4000 env-tick/s,见 research-log 20260726-v18-bench)。
v2.0 的 PPO rollout 可在此 vmap 批量骨架上扩为「每 tick 收集 (obs, action, reward, done)」——
把 `one()` 里的 `make_scan` 换成收集轨迹的 scan body 即可,批量/决定论/终局冻结语义不变。

聚合(胜率矩阵/质量指标)见 explorations/eval_commanders_v18.py 的 CLI 包装;本模块只出
可复用、可测的 rollout 原语,不含 provenance/argparse 副作用。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .controller import make_joint_controller
from .map import build_map
from .state import init_state
from .step import build_step, make_scan


def matchup_runner(cfg: Config, names: tuple[str, ...]):
    """返回 `run(seeds: int32[S]) -> dict`:对固定 matchup(names,长度 = cfg.n_players)
    vmap over seeds 批量跑整局,取每局末态指标。

    指标(均 [S] 或 [S,P]):
      winner        int32  胜者玩家 id(v1.8 必分胜负,∈[0,P-1])
      length        int32  终局 tick
      gate_reached  int32  是否进异界之门 overtime(tick >= gate_open_tick)
      army          int32[P] 每家末存活战斗单位数
      hq_alive      int32[P] 每家 HQ 末存活(唯一胜者=1、其余 0)
    """
    if len(names) != cfg.n_players:
        raise ValueError(f"需要 {cfg.n_players} 家指挥官,收到 {len(names)}")
    mapdata = build_map(cfg)
    step_fn = build_step(cfg, mapdata)
    joint = make_joint_controller(*names, cfg=cfg, mapdata=mapdata)
    scan = make_scan(step_fn, joint)
    state0 = init_state(cfg, mapdata)
    P, e = cfg.n_players, cfg.e_max
    is_combat = jnp.asarray(cfg.is_combat_by_type, bool)

    def one(seed):
        st, _, _ = scan(state0, jax.random.PRNGKey(seed), cfg.episode_len)
        et = jnp.clip(st.etype.astype(jnp.int32), 0, 31)
        army = (st.alive & is_combat[et]).reshape(P, e).sum(axis=1)
        hq_alive = st.alive.reshape(P, e)[:, 0].astype(jnp.int32)
        return dict(winner=st.winner.astype(jnp.int32), length=st.tick,
                    gate_reached=(st.tick >= cfg.gate_open_tick).astype(jnp.int32),
                    army=army, hq_alive=hq_alive)

    return jax.jit(jax.vmap(one))


def run_matchup(cfg: Config, names: tuple[str, ...], seeds) -> list[dict]:
    """便捷:对 names 跑 len(seeds) 局,返回 per-game dict 列表(winner/length/…/winner_name)。"""
    runner = matchup_runner(cfg, names)
    out = jax.tree.map(lambda x: x.tolist(), runner(jnp.asarray(seeds, jnp.int32)))
    rows = []
    for i in range(len(seeds)):
        w = out["winner"][i]
        rows.append(dict(names=list(names), seed=int(seeds[i]), winner=w,
                         winner_name=(names[w] if 0 <= w < len(names) else "none"),
                         length=out["length"][i], gate_reached=out["gate_reached"][i],
                         army=[out["army"][i][p] for p in range(cfg.n_players)]))
    return rows
