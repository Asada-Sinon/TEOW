"""v2.1 Phase C:真实训练吞吐 bench(含异界之门 + 网在环 + 更新 → games/day)。

区别于旧 `bench_batch_v18`(纯引擎、dirty 无 gate、4025 env-tick/s **不可当训练吞吐**):本 bench
在**真训练链路**(build_collect_rollout 网前向+step / build_update epoch×mb 前向反向)上计墙钟,并
**分 gate 前/后两制式**——pre-gate 段(tick<4000)不碰怪物 spawn/monster_combat,overtime 段(先把
carry 推过 gate_open 再计)含怪物逐 tick 开销,更贵。games/day 按「一局 = 4000 pre + 2000 overtime」
加权算(硬约束 #1:由测得墙钟脚本算出,不口算)。

用法(需 GPU,不设 JAX_PLATFORMS):
  .venv/bin/python explorations/bench_train_v21.py --slug v21-throughput
  smoke(CPU 验脚本):JAX_PLATFORMS=cpu … --smoke --out-root /tmp/xxx
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import pathlib
import shlex
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from teow.actions import n_actions  # noqa: E402
from teow.config import Config  # noqa: E402
from teow.map import build_map  # noqa: E402
from teow.state import init_state  # noqa: E402

from rl_skeleton_v20 import (  # noqa: E402
    RLConfig,
    _derived_norms,
    adam_init,
    compute_gae,
    encode_obs,
    init_params,
    type_cost_table,
)
from rl_train_v21 import (  # noqa: E402
    build_batch,
    build_collect_rollout,
    build_update,
    init_env_carry,
)


def time_call(fn, reps: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn())
    t0 = perf_counter()
    r = None
    for _ in range(reps):
        r = fn()
    jax.block_until_ready(r)
    return (perf_counter() - t0) / reps


def peak_mem_bytes():
    try:
        dev = jax.local_devices()[0]
        return int(dev.memory_stats().get("peak_bytes_in_use", 0))
    except Exception:  # noqa: BLE001  CPU 后端无 memory_stats
        return 0


def bench_one(cfg, mapdata, params, opt, rlcfg, key):
    """对给定 (B,T)=(rlcfg.num_envs,num_steps) 测 pre-gate/overtime rollout + update 墙钟,
    算加权 games/day。返回 dict。"""
    B, T = rlcfg.num_envs, rlcfg.num_steps
    P = cfg.n_players
    beta = jnp.float32(rlcfg.shaping_beta)
    lr, ent = jnp.float32(rlcfg.lr), jnp.float32(rlcfg.ent_coef)
    roll, state0, elim0 = build_collect_rollout(cfg, mapdata, ("random",) * (P - 1), rlcfg)
    update_fn = build_update(cfg, rlcfg)
    k1, k2 = jax.random.split(key)

    # ---- pre-gate rollout(fresh carry;reps*T 应 < gate_open 才全在门前)----
    carry = init_env_carry(state0, elim0, B, k1)
    box = {"c": carry}

    def do_roll():
        traj, lv, box["c"] = roll(params, box["c"], beta, k2)
        return lv
    reps = max(1, min(3, cfg.gate_open_tick // T - 1))
    roll_pre = time_call(do_roll, reps)
    pre_tick = int(np.asarray(box["c"][0].tick).reshape(-1)[0])

    # ---- 一次 update 墙钟(用 pre-gate 段的真实数据)----
    traj, lv, _ = roll(params, init_env_carry(state0, elim0, B, k1), beta, k2)
    adv, ret = jax.vmap(compute_gae, in_axes=(0, 0, 0, 0, None, None))(
        traj["reward"], traj["value"], traj["done"], lv, rlcfg.gamma, rlcfg.gae_lambda)
    batch = build_batch(traj, adv, ret)
    upd_box = {"o": opt}

    def do_update():
        _, upd_box["o"], st = update_fn(params, opt, batch, lr, ent, k2)
        return st
    upd_wall = time_call(do_update, reps)

    # ---- 推进到 overtime(不计时),再测 overtime rollout ----
    n_push = math.ceil(cfg.gate_open_tick / T) + 1
    box2 = {"c": init_env_carry(state0, elim0, B, k1)}

    def push():
        traj, lv, box2["c"] = roll(params, box2["c"], beta, k2)
        return lv
    for _ in range(n_push):
        jax.block_until_ready(push())
    ot_tick = int(np.asarray(box2["c"][0].tick).reshape(-1)[0])
    roll_ot = time_call(push, reps)

    # ---- 加权 games/day(一局 = gate_open pre + (episode_len-gate_open) overtime)----
    pre_per_tick = roll_pre / T
    ot_per_tick = roll_ot / T
    ep, gate = cfg.episode_len, cfg.gate_open_tick
    game_roll = gate * pre_per_tick + (ep - gate) * ot_per_tick   # 一局 rollout 墙钟(秒)
    game_update = (ep / T) * upd_wall                             # 一局 update 摊销
    game_wall = game_roll + game_update
    games_day = B * 86400.0 / game_wall
    return dict(
        B=B, T=T, reps=reps,
        roll_pre_s=roll_pre, roll_ot_s=roll_ot, update_s=upd_wall,
        pre_env_tick_s=B * T / roll_pre, ot_env_tick_s=B * T / roll_ot,
        eff_env_tick_s=B * ep / game_wall,
        game_wall_s=game_wall, games_day=games_day,
        pre_tick_reached=pre_tick, ot_tick_reached=ot_tick,
        peak_mem_gb=round(peak_mem_bytes() / 1e9, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="v21-throughput")
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envs", default="32,64,128,256")
    ap.add_argument("--steps", default="128,256")
    ap.add_argument("--episode-len", type=int, default=6000)
    ap.add_argument("--gate-open", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.envs, args.steps, args.episode_len, args.gate_open = "4", "32", 400, 200

    cfg = dataclasses.replace(Config(), episode_len=args.episode_len,
                              gate_open_tick=args.gate_open)
    mapdata = build_map(cfg)
    key = jax.random.PRNGKey(args.seed)
    # 一套 params 复用(bench 只测吞吐,不训练;维度用 base RLConfig 量)
    base = RLConfig()
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = base.res_scale, base.pot_scale, base.mon_hp_scale
    cost_tbl = type_cost_table(cfg)
    obs0 = encode_obs(init_state(cfg, mapdata), cfg, mapdata, 0, norms, cost_tbl)
    F, G, A = obs0.tokens.shape[1], obs0.global_vec.shape[0], n_actions(cfg)
    key, ki = jax.random.split(key)
    params = init_params(F, G, A, base, ki)
    opt = adam_init(params)

    envs = [int(x) for x in args.envs.split(",")]
    steps = [int(x) for x in args.steps.split(",")]
    results = []
    for T in steps:
        for B in envs:
            rlcfg = dataclasses.replace(base, num_envs=B, num_steps=T)
            # 让 gate_open/T 有余量给 pre-gate reps
            key, kb = jax.random.split(key)
            print(f"--- bench B={B} T={T} (编译+计时中…) ---", flush=True)
            try:
                r = bench_one(dataclasses.replace(cfg), mapdata, params, opt, rlcfg, kb)
            except Exception as e:  # noqa: BLE001
                r = dict(B=B, T=T, error=f"{type(e).__name__}: {str(e)[:200]}")
                print(f"  [FAIL] {r['error']}", flush=True)
            results.append(r)
            if "error" not in r:
                print(f"  pre={r['pre_env_tick_s']:.0f} ot={r['ot_env_tick_s']:.0f} "
                      f"eff={r['eff_env_tick_s']:.0f} env-tick/s | games/day={r['games_day']:.0f} "
                      f"| peakGB={r['peak_mem_gb']}", flush=True)

    # ---- 落盘 ----
    day = datetime.date.today().strftime("%Y%m%d")
    out_root = pathlib.Path(args.out_root)
    d, i = out_root / f"{day}-{args.slug}", 2
    while d.exists():
        d = out_root / f"{day}-{args.slug}-{i}"
        i += 1
    d.mkdir(parents=True)
    (d / "resolved_config.json").write_text(json.dumps(dict(
        engine=dataclasses.asdict(cfg), rl_base=dataclasses.asdict(base),
        seed=args.seed, F=F, G=G, A=A), indent=2, ensure_ascii=False))
    (d / "backend.txt").write_text(jax.default_backend())
    (d / "command.txt").write_text(shlex.join(sys.argv))
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        porc = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
        (d / "git.txt").write_text(f"{h}\ndirty: {bool(porc.strip())}")
    except Exception as e:  # noqa: BLE001
        (d / "git.txt").write_text(f"git 不可用: {e}")
    (d / "bench_train.json").write_text(json.dumps(dict(
        backend=jax.default_backend(), episode_len=args.episode_len,
        gate_open=args.gate_open, F=F, G=G, A=A, results=results),
        indent=2, ensure_ascii=False))
    print(f"\n=== 写入 {d}/bench_train.json ===", flush=True)


if __name__ == "__main__":
    main()
