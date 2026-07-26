"""v1.8 P0 基准:64×64/4p 引擎吞吐——单环境 CPU / 单环境 GPU / vmap 批量 GPU。

背景:v1.9 海量评测 + v2.0 rollout 完全依赖「vmap 批量 GPU」把单环境的慢摊薄。距离场
`_relax_fields`(movement.py:39-58,64² × ~36 场 × ~128 迭代/tick)是热点,单环境 64×64
可能极慢(v1.3 @24×24 仅 57-66 tick/s);唯一出路是批量。**先测再定**——不臆断。

口径:同一初始局(init_state 确定性,不吃 key)复制 B 份 + B 个不同 key 做 B 条独立 rollout
(scripted×scripted 对 key 不敏感,但**吞吐**与是否分化无关,计时准确)。度量:
- world_tick_s = ticks / wall(单条 rollout 的等效 tick/s)
- env_tick_s  = B*ticks / wall(总环境-tick/s,批量真吞吐;v1.9/v2.0 看这个)

用法:
  # 需要 GPU,故**不**强制 JAX_PLATFORMS=cpu(要 CPU+GPU 都可见)
  .venv/bin/python explorations/bench_batch_v18.py --slug v18-bench
  烟测:… --smoke --out-root /tmp/xxx
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
import time

# 注意:本脚本要同时用 CPU 与 GPU,**不**设 JAX_PLATFORMS(与门禁脚本相反,已在文档说明)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from teow.config import Config
from teow.controller import make_joint_controller
from teow.step import build_step, make_scan
from teow.map import build_map
from teow.state import init_state


def new_run_dir(out_root: pathlib.Path, slug: str) -> pathlib.Path:
    day = datetime.date.today().strftime("%Y%m%d")  # 允许:非数值结果的目录名
    base = out_root / f"{day}-{slug}"
    d, i = base, 2
    while d.exists():
        d = out_root / f"{day}-{slug}-{i}"
        i += 1
    d.mkdir(parents=True)
    return d


def write_provenance(d: pathlib.Path, cfg: Config, seeds, extra: dict) -> None:
    (d / "resolved_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    (d / "seeds.txt").write_text("\n".join(str(s) for s in seeds))
    (d / "backend.txt").write_text(jax.default_backend())
    (d / "command.txt").write_text(shlex.join(sys.argv))
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        porc = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
        (d / "git.txt").write_text(f"{h}\ndirty: {bool(porc.strip())}\n{porc}")
        if porc.strip():
            (d / "git.diff").write_text(
                subprocess.check_output(["git", "diff"], cwd=ROOT).decode())
    except Exception as e:  # noqa: BLE001
        (d / "git.txt").write_text(f"git 不可用: {e}")
    (d / "design.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False))


def replicate(tree, B):
    return jax.tree.map(lambda x: jnp.broadcast_to(x[None], (B,) + x.shape), tree)


def time_run(fn, *args, repeats=1):
    """warmup 1 次(触发 jit + block),再计时 repeats 次取墙钟总和/次数。"""
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="v18-bench")
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-ticks", type=int, default=60)
    ap.add_argument("--batch-ticks", type=int, default=100)
    ap.add_argument("--batches", default="64,256,1024")
    ap.add_argument("--cpu-batches", default="32,128")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.single_ticks, args.batch_ticks = 5, 5
        args.batches, args.cpu_batches = "8,16", "8"

    cfg = dataclasses.replace(Config(), seed=args.seed)
    mapdata = build_map(cfg)
    step_fn = build_step(cfg, mapdata)
    names = tuple(["scripted"] * cfg.n_players)
    joint = make_joint_controller(*names, cfg=cfg, mapdata=mapdata)
    scan_steps = make_scan(step_fn, joint)

    state0 = init_state(cfg, mapdata)
    key0 = jax.random.PRNGKey(args.seed)

    devs = {d.platform: d for d in jax.devices()}
    # jax.devices() 只返回默认后端;显式取 cpu / gpu
    try:
        cpu_dev = jax.devices("cpu")[0]
    except RuntimeError:
        cpu_dev = None
    try:
        gpu_dev = jax.devices("gpu")[0]
    except RuntimeError:
        gpu_dev = None

    results = []

    def bench_single(dev, label, ticks):
        if dev is None:
            print(f"[skip] {label}: 设备不可用")
            return
        s = jax.device_put(state0, dev)
        k = jax.device_put(key0, dev)
        print(f"[run ] {label}: 编译+跑 {ticks} tick …", flush=True)
        try:
            dt = time_run(lambda: scan_steps(s, k, ticks))
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {label}: {type(e).__name__}: {str(e)[:200]}")
            results.append(dict(mode=label, error=f"{type(e).__name__}: {str(e)[:200]}"))
            return
        r = dict(mode=label, device=str(dev), B=1, ticks=ticks, wall_s=round(dt, 4),
                 world_tick_s=round(ticks / dt, 2), env_tick_s=round(ticks / dt, 2))
        results.append(r)
        print(f"[ok  ] {label}: {r['world_tick_s']} tick/s  ({dt:.3f}s)")

    def bench_batch(dev, B, ticks, plat="gpu"):
        if dev is None:
            print(f"[skip] batch-{plat} B={B}: 设备不可用")
            return
        try:
            sB = jax.device_put(replicate(state0, B), dev)
            kB = jax.device_put(jax.random.split(key0, B), dev)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] batch B={B} device_put: {type(e).__name__}: {str(e)[:160]}")
            results.append(dict(mode=f"batch-{plat}-B{B}", error=str(e)[:160]))
            return
        f = jax.jit(jax.vmap(lambda s, k: scan_steps(s, k, ticks)))
        print(f"[run ] batch-{plat} B={B}: 编译+跑 {ticks} tick × {B} env …", flush=True)
        try:
            dt = time_run(lambda: f(sB, kB))
        except Exception as e:  # noqa: BLE001
            print(f"[fail] batch B={B}: {type(e).__name__}: {str(e)[:200]}")
            results.append(dict(mode=f"batch-{plat}-B{B}", error=f"{type(e).__name__}: {str(e)[:200]}"))
            return
        r = dict(mode=f"batch-{plat}-B{B}", device=str(dev), B=B, ticks=ticks,
                 wall_s=round(dt, 4), world_tick_s=round(ticks / dt, 2),
                 env_tick_s=round(B * ticks / dt, 1))
        results.append(r)
        print(f"[ok  ] batch B={B}: env {r['env_tick_s']} tick/s, per-world {r['world_tick_s']} tick/s ({dt:.2f}s)")

    print(f"=== devices: {list(devs)} | cpu={cpu_dev} gpu={gpu_dev} ===", flush=True)
    bench_single(cpu_dev, "single-cpu", args.single_ticks)
    for B in [int(x) for x in args.cpu_batches.split(",")]:
        bench_batch(cpu_dev, B, args.batch_ticks, plat="cpu")
    for B in [int(x) for x in args.batches.split(",")]:
        bench_batch(gpu_dev, B, args.batch_ticks, plat="gpu")

    out_root = pathlib.Path(args.out_root)
    d = new_run_dir(out_root, args.slug)
    write_provenance(d, cfg, [args.seed], dict(
        note="v1.8 P0 吞吐基准", grid=f"{cfg.grid_h}x{cfg.grid_w}",
        n_players=cfg.n_players, e_max=cfg.e_max,
        single_ticks=args.single_ticks, batch_ticks=args.batch_ticks))
    (d / "bench.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n=== 写入 {d}/bench.json ===")
    for r in results:
        print(" ", r)


if __name__ == "__main__":
    main()
