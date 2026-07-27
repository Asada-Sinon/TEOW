"""v2.1 Phase B:PPO 训练主循环(全长局 + 持续环境分段 rollout)。**真训练**(区别 v2.0 骨架)。

对齐 plan docs/plans + v2.0 设计 §4/§6/§7。结构:Python 外循环 + 内层全 lax.scan。
- B1 持续环境 rollout:每段 T tick,环境跨 update carry,done→reset 开新局(打破 step 的 done
  冻结),跨段 value bootstrap。全长 episode(6000)靠分段避免 141GB 显存。
- B2/B3 update:epoch×minibatch 全 scan;lr/ent/β 线性退火(traced 标量传入免重编译)。
- B4 checkpoint:np.savez(params/opt)+ JSON 边车(provenance)。
- B6 课程:CurriculumScheduler 按 vs-当前档胜率升档。

复用骨架 rl_skeleton_v20 已验证原语(encode_obs/actor_critic/potential/terminal_rank_reward/
init_params/compute_gae/ppo_loss/adam_*),不建第二真源。数字全脚本落盘不手估(CLAUDE.md #1)。

用法:
  smoke(CPU 秒级验管线):JAX_PLATFORMS=cpu .venv/bin/python explorations/rl_train_v21.py --smoke --out-root /tmp/xxx
  小批(GPU 看学习信号):.venv/bin/python explorations/rl_train_v21.py --slug v21-train-small --num-envs 64 --num-steps 128 --total-updates 400
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import shlex
import subprocess
import sys
from collections import deque

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from teow.actions import n_actions  # noqa: E402
from teow.config import Config  # noqa: E402
from teow.controller import make_controller, merge_actions  # noqa: E402
from teow.map import build_map  # noqa: E402
from teow.state import hq_slot, init_state, owner_of_slots  # noqa: E402
from teow.step import build_step  # noqa: E402

from rl_eval_v21 import build_policy_eval, summarize_eval  # noqa: E402
from rl_skeleton_v20 import (  # noqa: E402
    RLConfig,
    _derived_norms,
    _flatten_time_batch,
    actor_critic,
    adam_init,
    adam_step,
    compute_gae,
    encode_obs,
    init_params,
    potential,
    ppo_loss,
    terminal_rank_reward,
    type_cost_table,
)

# ppo_loss 消费的 batch 字段(GAE 用的 reward/done 不进 batch)
_BATCH_KEYS = ("tokens", "etype_idx", "entity_mask", "loss_mask", "global_vec",
               "action_mask", "action", "logp", "value")


# =====================================================================================
# B1) 持续环境分段 rollout(carry 跨 update;done→reset 开新局)
# =====================================================================================
def build_collect_rollout(cfg, mapdata, opp_names: tuple[str, ...], rlcfg: RLConfig):
    """返回 (roll_fn, state0, elim0)。roll_fn(params, carry, beta, key)→(traj, last_value, new_carry)。
    carry=(state[B], key[B], elim[B]);policy 占 0 座,opp_names 占 1..P-1 座。"""
    if len(opp_names) != cfg.n_players - 1:
        raise ValueError(f"需要 {cfg.n_players - 1} 个对手,收到 {len(opp_names)}")
    owner = owner_of_slots(cfg)
    step_fn = build_step(cfg, mapdata)
    P = cfg.n_players
    player = 0
    opp_ctrls = [make_controller(nm, q, cfg, mapdata)
                 for q, nm in zip(range(1, P), opp_names, strict=True)]
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = (
        rlcfg.res_scale, rlcfg.pot_scale, rlcfg.mon_hp_scale)
    cost_tbl = type_cost_table(cfg)
    state0 = init_state(cfg, mapdata)
    big_t = jnp.int32(cfg.episode_len + 1)
    elim0 = jnp.full(P, big_t, jnp.int32)

    @jax.jit
    def roll(params, carry, beta, key):
        def one(state, key_env, elim):
            def body(c, _):
                state, key, elim = c
                key, k_pol, k_opp, k_step = jax.random.split(key, 4)
                obs = encode_obs(state, cfg, mapdata, player, norms, cost_tbl)
                logits, value = actor_critic(params, obs)
                action = jax.random.categorical(k_pol, logits, axis=-1).astype(jnp.int32)
                logsm = jax.nn.log_softmax(logits, axis=-1)
                lp = jnp.take_along_axis(logsm, action[:, None], axis=-1)[:, 0]
                logp = jnp.sum(lp * obs.loss_mask.astype(jnp.float32))
                opp_keys = jax.random.split(k_opp, P - 1)
                opp_acts = [c(state, key=kk) for c, kk in zip(opp_ctrls, opp_keys, strict=True)]
                joint = merge_actions(owner, [action] + opp_acts)
                nstate = step_fn(state, joint, k_step)
                prev_done = state.done
                phi_s = potential(state, cfg, owner, cost_tbl, rlcfg.pot_scale)
                phi_n = potential(nstate, cfg, owner, cost_tbl, rlcfg.pot_scale)
                shaping = beta * (rlcfg.gamma * phi_n - phi_s) * (~prev_done)
                hq_alive = jnp.stack([nstate.alive[hq_slot(q, cfg)] for q in range(P)])
                elim = jnp.minimum(elim, jnp.where(hq_alive, big_t, nstate.tick))
                done_trans = nstate.done & (~prev_done)
                r_all = shaping + terminal_rank_reward(elim, cfg) * done_trans
                out = dict(
                    tokens=obs.tokens, etype_idx=obs.etype_idx,
                    entity_mask=obs.entity_mask, loss_mask=obs.loss_mask,
                    global_vec=obs.global_vec, action_mask=obs.action_mask,
                    action=action, logp=logp, value=value,
                    reward=r_all[player], done=nstate.done)
                # autoreset:done 那拍记完奖励后回 state0 开新局(持续环境保证进入 body 时非 done)
                next_state = jax.tree.map(
                    lambda o, n: jnp.where(nstate.done, o, n), state0, nstate)
                next_elim = jnp.where(nstate.done, elim0, elim)
                return (next_state, key, next_elim), out

            (fstate, fkey, felim), traj = jax.lax.scan(
                body, (state, key_env, elim), None, length=rlcfg.num_steps)
            obs_T = encode_obs(fstate, cfg, mapdata, player, norms, cost_tbl)
            _, last_value = actor_critic(params, obs_T)
            return traj, last_value, (fstate, fkey, felim)

        state_b, key_b, elim_b = carry
        return jax.vmap(one)(state_b, key_b, elim_b)

    return roll, state0, elim0


def init_env_carry(state0, elim0, B, key):
    state_b = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), state0)
    key_b = jax.random.split(key, B)
    elim_b = jnp.broadcast_to(elim0, (B,) + elim0.shape)
    return (state_b, key_b, elim_b)


# =====================================================================================
# B3) 退火(返回 jnp 标量,传进 jit 免重编译)
# =====================================================================================
def anneal(u: int, rlcfg: RLConfig):
    p = u / max(rlcfg.total_updates, 1)
    lr = rlcfg.lr * (1.0 - p) if rlcfg.anneal_lr else rlcfg.lr
    if rlcfg.anneal_ent:
        ent = rlcfg.ent_coef + (rlcfg.ent_coef_final - rlcfg.ent_coef) * p
    else:
        ent = rlcfg.ent_coef
    if rlcfg.anneal_shaping:
        fb = max(0.0, 1.0 - p / max(rlcfg.shaping_anneal_frac, 1e-9))
        beta = rlcfg.shaping_beta_final + (rlcfg.shaping_beta - rlcfg.shaping_beta_final) * fb
    else:
        beta = rlcfg.shaping_beta
    return jnp.float32(lr), jnp.float32(ent), jnp.float32(beta)


# =====================================================================================
# B2) update:epoch×minibatch 全 scan
# =====================================================================================
def build_update(cfg, rlcfg: RLConfig):
    M = rlcfg.num_envs * rlcfg.num_steps
    nmb = rlcfg.num_minibatches
    if M % nmb != 0:
        raise ValueError(f"M=B·T={M} 不整除 num_minibatches={nmb}")
    mb = M // nmb

    @jax.jit
    def update(params, opt, batch, lr, ent, key):
        def epoch(carry, _):
            params, opt, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, M)
            sh = jax.tree.map(lambda x: x[perm], batch)
            mbs = jax.tree.map(lambda x: x.reshape((nmb, mb) + x.shape[1:]), sh)

            def mb_step(c, b):
                params, opt = c
                (loss, aux), g = jax.value_and_grad(ppo_loss, has_aux=True)(
                    params, b, rlcfg, ent)
                params, opt, gn = adam_step(params, g, opt, rlcfg, lr)
                return (params, opt), {"loss": loss, "gnorm": gn, **aux}

            (params, opt), st = jax.lax.scan(mb_step, (params, opt), mbs)
            return (params, opt, key), st

        (params, opt, _), st = jax.lax.scan(
            epoch, (params, opt, key), None, length=rlcfg.update_epochs)
        return params, opt, jax.tree.map(lambda x: jnp.mean(x), st)

    return update


def build_batch(traj, adv, ret):
    b = {k: _flatten_time_batch(traj[k]) for k in _BATCH_KEYS}
    b["adv"] = _flatten_time_batch(adv)
    b["ret"] = _flatten_time_batch(ret)
    return b


def explained_var(value: np.ndarray, ret: np.ndarray) -> float:
    var = float(np.var(ret))
    return float("nan") if var == 0 else float(1.0 - np.var(ret - value) / var)


# =====================================================================================
# B4) checkpoint(np.savez params/opt + JSON 边车)
# =====================================================================================
def save_ckpt(run_dir: pathlib.Path, u: int, params, opt, key, sched_state, rlcfg, git):
    ckdir = run_dir / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    d = {f"p__{k}": np.asarray(v) for k, v in params.items()}
    d.update({f"m__{k}": np.asarray(v) for k, v in opt["m"].items()})
    d.update({f"v__{k}": np.asarray(v) for k, v in opt["v"].items()})
    np.savez(ckdir / f"ckpt_{u:07d}.npz", **d)
    (ckdir / f"ckpt_{u:07d}.json").write_text(json.dumps(dict(
        update_step=u, adam_t=int(opt["t"]), rng=np.asarray(key).tolist(),
        curriculum=sched_state, rlcfg=dataclasses.asdict(rlcfg), git=git),
        indent=2, ensure_ascii=False))


def load_ckpt(npz_path, json_path):
    z = np.load(npz_path)
    meta = json.loads(pathlib.Path(json_path).read_text())
    params = {k[3:]: jnp.asarray(z[k]) for k in z.files if k.startswith("p__")}
    m = {k[3:]: jnp.asarray(z[k]) for k in z.files if k.startswith("m__")}
    v = {k[3:]: jnp.asarray(z[k]) for k in z.files if k.startswith("v__")}
    key = jnp.asarray(meta["rng"], jnp.uint32)
    return params, dict(m=m, v=v, t=meta["adam_t"]), key, meta


# =====================================================================================
# B6) 课程调度器(纯 Python host 端)
# =====================================================================================
class CurriculumScheduler:
    """按对当前档 eval 胜率的滑窗均值升档。tiers 有序,每档 = P-1 个对手脚本名。"""

    def __init__(self, tiers, promote_threshold=0.6, window=3):
        self.tiers = [tuple(t) for t in tiers]
        self.stage = 0
        self.win = deque(maxlen=window)
        self.threshold = promote_threshold

    def opp_names(self):
        return self.tiers[self.stage]

    def step(self, winrate) -> bool:
        self.win.append(float(winrate))
        if (len(self.win) == self.win.maxlen
                and sum(self.win) / len(self.win) >= self.threshold
                and self.stage < len(self.tiers) - 1):
            self.stage += 1
            self.win.clear()
            return True
        return False

    def state(self):
        return dict(stage=self.stage, tier=list(self.opp_names()), window=list(self.win))


# =====================================================================================
# provenance + 训练主循环
# =====================================================================================
def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        porc = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
        return f"{h}{' (dirty)' if porc.strip() else ''}"
    except Exception as e:  # noqa: BLE001
        return f"git 不可用: {e}"


def new_run_dir(out_root: pathlib.Path, slug: str) -> pathlib.Path:
    day = datetime.date.today().strftime("%Y%m%d")
    d, i = out_root / f"{day}-{slug}", 2
    while d.exists():
        d = out_root / f"{day}-{slug}-{i}"
        i += 1
    d.mkdir(parents=True)
    return d


def write_provenance(run_dir, cfg, rlcfg, seed, tiers):
    (run_dir / "resolved_config.json").write_text(json.dumps(dict(
        engine=dataclasses.asdict(cfg), rl=dataclasses.asdict(rlcfg),
        seed=seed, curriculum_tiers=[list(t) for t in tiers]),
        indent=2, ensure_ascii=False))
    (run_dir / "git.txt").write_text(git_hash())
    (run_dir / "backend.txt").write_text(jax.default_backend())
    (run_dir / "command.txt").write_text(shlex.join(sys.argv))
    try:
        porc = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
        if porc.strip():
            (run_dir / "git.diff").write_text(
                subprocess.check_output(["git", "diff"], cwd=ROOT).decode())
    except Exception:  # noqa: BLE001
        pass


def train(cfg, rlcfg, run_dir, tiers, seed, eval_seeds):
    mapdata = build_map(cfg)
    key = jax.random.PRNGKey(seed)
    key, ki, ke = jax.random.split(key, 3)
    A = n_actions(cfg)
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = (
        rlcfg.res_scale, rlcfg.pot_scale, rlcfg.mon_hp_scale)
    cost_tbl = type_cost_table(cfg)
    obs0 = encode_obs(init_state(cfg, mapdata), cfg, mapdata, 0, norms, cost_tbl)
    F, G = obs0.tokens.shape[1], obs0.global_vec.shape[0]
    params = init_params(F, G, A, rlcfg, ki)
    n_param = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    opt = adam_init(params)
    sched = CurriculumScheduler(tiers)
    update_fn = build_update(cfg, rlcfg)
    git = git_hash()
    eval_cache = {}

    def get_eval(opp):
        if opp not in eval_cache:
            eval_cache[opp] = build_policy_eval(cfg, mapdata, opp, rlcfg)
        return eval_cache[opp]

    roll, state0, elim0 = build_collect_rollout(cfg, mapdata, sched.opp_names(), rlcfg)
    carry = init_env_carry(state0, elim0, rlcfg.num_envs, ke)
    eval_seeds_j = jnp.arange(eval_seeds, dtype=jnp.int32)
    print(f"F={F} G={G} A={A} params={n_param:,} | B={rlcfg.num_envs} T={rlcfg.num_steps} "
          f"updates={rlcfg.total_updates} tiers={[list(t) for t in tiers]}", flush=True)

    metrics_f = (run_dir / "metrics.jsonl").open("w")
    eval_f = (run_dir / "eval.jsonl").open("w")
    for u in range(rlcfg.total_updates):
        lr, ent, beta = anneal(u, rlcfg)
        key, kr, ku = jax.random.split(key, 3)
        traj, last_value, carry = roll(params, carry, beta, kr)
        adv, ret = jax.vmap(compute_gae, in_axes=(0, 0, 0, 0, None, None))(
            traj["reward"], traj["value"], traj["done"], last_value,
            rlcfg.gamma, rlcfg.gae_lambda)
        batch = build_batch(traj, adv, ret)
        params, opt, stats = update_fn(params, opt, batch, lr, ent, ku)
        stats = {k: float(v) for k, v in stats.items()}
        ev = explained_var(np.asarray(batch["value"]), np.asarray(batch["ret"]))
        rec = dict(update=u, lr=float(lr), ent_coef=float(ent), beta=float(beta),
                   stage=sched.stage, explained_var=ev,
                   mean_reward=float(np.asarray(traj["reward"]).mean()),
                   done_count=int(np.asarray(traj["done"]).sum()), **stats)
        metrics_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        metrics_f.flush()
        if u % max(rlcfg.eval_every, 1) == 0 or u == rlcfg.total_updates - 1:
            out = get_eval(sched.opp_names())(params, eval_seeds_j)
            summ = summarize_eval(out, eval_seeds)
            summ.update(update=u, stage=sched.stage, opp=list(sched.opp_names()))
            eval_f.write(json.dumps(summ, ensure_ascii=False) + "\n")
            eval_f.flush()
            print(f"[u{u}] wr_vs_{sched.opp_names()[0]}={summ['winrate_p0']:.2f} "
                  f"rank={summ['avg_rank_p0']:.2f} ent={summ['avg_policy_entropy']:.2f} "
                  f"gate={summ['gate_rate']:.2f} army={summ['avg_army_p0']:.1f} "
                  f"loss={stats['loss']:+.3f} kl={stats['approx_kl']:+.3f} ev={ev:.2f}",
                  flush=True)
            if sched.step(summ["winrate_p0"]):
                key, kc = jax.random.split(key)
                roll, state0, elim0 = build_collect_rollout(
                    cfg, mapdata, sched.opp_names(), rlcfg)
                carry = init_env_carry(state0, elim0, rlcfg.num_envs, kc)
                print(f"  ↑ 升档 → stage {sched.stage} {sched.opp_names()}", flush=True)
        if u % max(rlcfg.ckpt_every, 1) == 0 or u == rlcfg.total_updates - 1:
            save_ckpt(run_dir, u, params, opt, key, sched.state(), rlcfg, git)
    metrics_f.close()
    eval_f.close()
    print(f"训练完成,产物 → {run_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="v21-train")
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=128)
    ap.add_argument("--total-updates", type=int, default=400)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--eval-seeds", type=int, default=32)
    ap.add_argument("--episode-len", type=int, default=6000)
    ap.add_argument("--gate-open", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--shaping-beta", type=float, default=0.1)
    ap.add_argument("--no-anneal-shaping", action="store_true",
                    help="试训期关 β 退火(保密集塑形信号)")
    # 课程:逗号分隔档,档内 | 分隔 P-1 个对手;默认单档 random(起步验证学习信号)
    ap.add_argument("--curriculum", default="random|random|random")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        (args.num_envs, args.num_steps, args.total_updates, args.update_epochs,
         args.num_minibatches, args.eval_every, args.ckpt_every, args.eval_seeds,
         args.episode_len, args.gate_open) = 4, 32, 3, 2, 2, 1, 1, 2, 400, 100

    tiers = [tuple(tier.split("|")) for tier in args.curriculum.split(",")]
    cfg = dataclasses.replace(Config(), episode_len=args.episode_len,
                              gate_open_tick=args.gate_open)
    rlcfg = RLConfig(
        seed=args.seed, num_envs=args.num_envs, num_steps=args.num_steps,
        total_updates=args.total_updates, update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every, gamma=args.gamma, shaping_beta=args.shaping_beta,
        anneal_shaping=not args.no_anneal_shaping)
    run_dir = new_run_dir(pathlib.Path(args.out_root), args.slug)
    write_provenance(run_dir, cfg, rlcfg, args.seed, tiers)
    print(f"=== v2.1 训练 run → {run_dir}  backend={jax.default_backend()} ===", flush=True)
    train(cfg, rlcfg, run_dir, tiers, args.seed, args.eval_seeds)


if __name__ == "__main__":
    main()
