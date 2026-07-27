"""v2.1 Phase B5:RL-vs-脚本 eval 谐波(策略当 controller,整局跑,出胜率/名次/行为诊断)。

为什么不能直接用 `src/teow/eval.matchup_runner`:它经 `make_joint_controller` 只认**脚本名**,
无法把 RL 策略(一份 params)塞进 0 座。本模块镜像 matchup_runner 的末态归约,但 0 座换成
**策略贪心**(argmax masked logits,测真实强度),1..P-1 座 = 脚本;整局 `lax.scan`(全长 6000,
与用户「全长局」一致),vmap over eval seeds。

产出(供训练循环 B5 监控 + Phase D 诊断,数字全脚本算不手估):
  winner/length/gate/army/hq_alive(同 matchup_runner)+ rank_p0(FFA 名次,逐 tick elim 推)
  + econ_p0(投入价值,复用骨架 _invested_value)+ act_hist[A](己方 actable 动作频率,查退化)
  + policy_entropy(己方 actable 平均熵,查熵塌缩)。

复用骨架 `rl_skeleton_v20` 的已验证原语,不建第二真源。
"""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from teow.actions import n_actions  # noqa: E402
from teow.controller import make_controller, merge_actions  # noqa: E402
from teow.state import hq_slot, init_state, owner_of_slots  # noqa: E402
from teow.step import build_step  # noqa: E402

# 复用骨架原语(同目录 import)
from rl_skeleton_v20 import (  # noqa: E402
    RLConfig,
    _derived_norms,
    _invested_value,
    actor_critic,
    encode_obs,
    terminal_rank_reward,
    type_cost_table,
)


def build_policy_eval(cfg, mapdata, opp_names: tuple[str, ...], rlcfg: RLConfig):
    """返回 `eval_fn(params, seeds:int32[S]) -> dict`(各字段 [S] 或 [S,·])。
    0 座 = 策略贪心,其余 = opp_names(长度 P-1)。整局 scan;done 后冻结不计入诊断累计。"""
    if len(opp_names) != cfg.n_players - 1:
        raise ValueError(f"需要 {cfg.n_players - 1} 个对手脚本名,收到 {len(opp_names)}")
    owner = owner_of_slots(cfg)
    step_fn = build_step(cfg, mapdata)
    P, e = cfg.n_players, cfg.e_max
    player = 0
    opp_ctrls = [make_controller(nm, q, cfg, mapdata)
                 for q, nm in zip(range(1, P), opp_names, strict=True)]
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = (
        rlcfg.res_scale, rlcfg.pot_scale, rlcfg.mon_hp_scale)
    cost_tbl = type_cost_table(cfg)
    state0 = init_state(cfg, mapdata)
    is_combat = jnp.asarray(cfg.is_combat_by_type, bool)
    A = n_actions(cfg)
    big_t = jnp.int32(cfg.episode_len + 1)

    def one(params, seed):
        hist0 = jnp.zeros(A, jnp.float32)

        def body(carry, _):
            state, key, elim, hist, ent_sum, cnt = carry
            key, k_opp, k_step = jax.random.split(key, 3)
            obs = encode_obs(state, cfg, mapdata, player, norms, cost_tbl)
            logits, _ = actor_critic(params, obs)
            action = jnp.argmax(logits, axis=-1).astype(jnp.int32)   # 贪心测强度
            opp_keys = jax.random.split(k_opp, P - 1)
            opp_acts = [c(state, key=kk) for c, kk in zip(opp_ctrls, opp_keys, strict=True)]
            joint = merge_actions(owner, [action] + opp_acts)
            nstate = step_fn(state, joint, k_step)
            hq_alive_now = jnp.stack([nstate.alive[hq_slot(q, cfg)] for q in range(P)])
            elim = jnp.minimum(elim, jnp.where(hq_alive_now, big_t, nstate.tick))
            # 行为诊断累计(仅未终局 tick;done 后冻结不计入)
            active = (~state.done).astype(jnp.float32)
            m = obs.loss_mask.astype(jnp.float32)                    # 己方 actable
            onehot = jax.nn.one_hot(action, A)                       # [N,A]
            hist = hist + jnp.sum(onehot * m[:, None], axis=0) * active
            logsm = jax.nn.log_softmax(logits, axis=-1)
            ent_i = -jnp.sum(jnp.exp(logsm) * logsm, axis=-1)        # [N]
            denom = jnp.maximum(jnp.sum(m), 1.0)
            ent_sum = ent_sum + (jnp.sum(ent_i * m) / denom) * active
            return (nstate, key, elim, hist, ent_sum, cnt + active), None

        elim0 = jnp.full(P, big_t, jnp.int32)
        carry0 = (state0, jax.random.PRNGKey(seed), elim0, hist0,
                  jnp.float32(0.0), jnp.float32(0.0))
        (fstate, _, felim, hist, ent_sum, cnt), _ = jax.lax.scan(
            body, carry0, None, length=cfg.episode_len)
        et = jnp.clip(fstate.etype.astype(jnp.int32), 0, 31)
        army = (fstate.alive & is_combat[et]).reshape(P, e).sum(axis=1)
        hq_alive = fstate.alive.reshape(P, e)[:, 0].astype(jnp.int32)
        # FFA 名次(elim→rank;terminal_rank_reward 内含 rank 逻辑,这里直接算 rank）
        et2 = felim.astype(jnp.int32)
        idx = jnp.arange(P)
        better = ((et2[None, :] > et2[:, None])
                  | ((et2[None, :] == et2[:, None]) & (idx[None, :] < idx[:, None])))
        rank = 1 + jnp.sum(better, axis=1)                           # [P] ∈ 1..P
        econ = _invested_value(fstate, cfg, owner, cost_tbl)         # [P]
        cnt_safe = jnp.maximum(cnt, 1.0)
        return dict(
            winner=fstate.winner.astype(jnp.int32), length=fstate.tick,
            gate=(fstate.tick >= cfg.gate_open_tick).astype(jnp.int32),
            army=army, hq_alive=hq_alive, rank_p0=rank[player].astype(jnp.int32),
            econ_p0=econ[player], act_hist=hist / jnp.sum(jnp.maximum(hist, 1e-6)),
            policy_entropy=ent_sum / cnt_safe, active_ticks=cnt)

    return jax.jit(jax.vmap(one, in_axes=(None, 0)))


def summarize_eval(out: dict, n_seeds: int) -> dict:
    """把 build_policy_eval 的批量输出(jax 数组)归约成 host 端 python 标量摘要(诊断/日志用)。"""
    import numpy as np
    o = {k: np.asarray(v) for k, v in out.items()}
    winner = o["winner"]
    return dict(
        n=int(n_seeds),
        winrate_p0=float((winner == 0).mean()),          # 主学习信号
        avg_rank_p0=float(o["rank_p0"].mean()),          # FFA 平均名次(越小越强)
        avg_length=float(o["length"].mean()),
        gate_rate=float(o["gate"].mean()),
        avg_army_p0=float(o["army"][:, 0].mean()),
        avg_econ_p0=float(o["econ_p0"].mean()),
        avg_policy_entropy=float(o["policy_entropy"].mean()),
        act_hist_mean=o["act_hist"].mean(axis=0).tolist(),   # [A] 平均动作分布
    )
