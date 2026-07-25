"""TEOW 入口:play(跑对局)/ replay(回放渲染)/ bench(吞吐)。

用法(务必用 .venv/bin/python,见 CLAUDE.md):
  .venv/bin/python src/run.py play --p0 scripted --p1 scripted --seed 0
  .venv/bin/python src/run.py play --p0 scripted --p1 random --slug smoke --record
  .venv/bin/python src/run.py replay experiments/<run_id>
  .venv/bin/python src/run.py bench --ticks 2000

每次 play 落一个 experiments/<YYYYMMDD>-<slug>/ run 目录,写齐六项 provenance
(resolved config / git / seed / 命令 / 日志 / metrics),写完即只读
(.claude/rules/experiments.md)。日志用进程内 Tee,不用 shell 重定向。
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

from teow import Config, new_world
from teow.controller import make_joint_controller
from teow.state import WorldState
from teow.step import make_scan


class Tee:
    """stdout/stderr 同时写终端与文件(六项 provenance 之五)。"""

    def __init__(self, stream, path: pathlib.Path):
        self._stream = stream
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, s):
        self._stream.write(s)
        self._fh.write(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()


def parse_overrides(pairs) -> dict:
    """`--set field=value` → 按 Config 字段声明类型强转(照 alicization 惯用法)。"""
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs or ():
        name, _, raw = p.partition("=")
        if name not in types:
            raise SystemExit(f"--set: Config 没有字段 {name!r}")
        t = types[name]
        t = {"int": int, "float": float, "bool": bool}.get(t, t) if isinstance(t, str) else t
        out[name] = (raw not in ("0", "False", "false")) if t is bool else t(raw)
    return out


def make_run_dir(slug: str) -> pathlib.Path:
    date = datetime.date.today().strftime("%Y%m%d")
    root = pathlib.Path(__file__).resolve().parent.parent / "experiments"
    run_dir = root / f"{date}-{slug}"
    i = 1
    while run_dir.exists():
        i += 1
        run_dir = root / f"{date}-{slug}-{i}"
    run_dir.mkdir(parents=True)
    return run_dir


def write_provenance(run_dir: pathlib.Path, cfg: Config) -> None:
    (run_dir / "resolved_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    (run_dir / "seed.txt").write_text(str(cfg.seed) + "\n")
    (run_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    head = subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    (run_dir / "git.txt").write_text(f"{head}\ndirty: {bool(dirty)}\n{dirty}")
    if dirty:
        diff = subprocess.run(["git", "diff"], capture_output=True, text=True).stdout
        (run_dir / "git.diff").write_text(diff)


def state_to_numpy(st: WorldState) -> dict:
    return {k: np.asarray(v) for k, v in st._asdict().items()}


def cmd_play(args) -> None:
    cfg = dataclasses.replace(Config(), seed=args.seed, **parse_overrides(args.set))
    run_dir = make_run_dir(args.slug)
    write_provenance(run_dir, cfg)
    sys.stdout = Tee(sys.__stdout__, run_dir / "stdout.log")
    sys.stderr = Tee(sys.__stderr__, run_dir / "stderr.log")

    print(f"run_dir: {run_dir}")
    print(f"device: {jax.devices()[0]}  p0={args.p0} p1={args.p1} seed={cfg.seed}")

    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller(args.p0, args.p1, cfg, m)
    joint = jax.jit(joint)

    traj: list[dict] = []
    metrics_fh = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    record_every = max(1, args.record_every)
    for t in range(cfg.episode_len):
        key, k_act, k_step = jax.random.split(key, 3)
        actions = joint(state, k_act)
        state = step_fn(state, actions, k_step)
        if args.record and t % record_every == 0:
            traj.append(state_to_numpy(state))
        if t % 100 == 0 or bool(state.done):
            row = {
                "tick": int(state.tick),
                "res_p0": state.resources[0].tolist(),
                "res_p1": state.resources[1].tolist(),
                "alive_p0": int(jnp.sum(state.alive[:cfg.e_max])),
                "alive_p1": int(jnp.sum(state.alive[cfg.e_max:])),
                "base_level": [int(state.level[0]), int(state.level[cfg.e_max])],
                "upgrades_p0": state.upgrades[0].tolist(),
                "upgrades_p1": state.upgrades[1].tolist(),
                "done": bool(state.done),
                "winner": int(state.winner),
            }
            metrics_fh.write(json.dumps(row) + "\n")
            metrics_fh.flush()
        if bool(state.done):
            break
    dt = time.time() - t0
    metrics_fh.close()

    outcome = {-1: "未分胜负(到达循环上限)", 0: "玩家0胜", 1: "玩家1胜", 2: "和局"}
    print(f"结束 tick={int(state.tick)} winner={int(state.winner)}"
          f"({outcome[int(state.winner)]})  {int(state.tick) / max(dt, 1e-9):.0f} tick/s")
    if args.record:
        stacked = {k: np.stack([f[k] for f in traj]) for k in traj[0]}
        np.savez_compressed(run_dir / "trajectory.npz",
                            record_every=record_every, **stacked)
        print(f"轨迹已存 {run_dir / 'trajectory.npz'}(replay 子命令可回放)")


def cmd_replay(args) -> None:
    from teow.render import replay_run
    replay_run(pathlib.Path(args.run_dir), fps=args.fps, save=args.save)


def cmd_bench(args) -> None:
    cfg = dataclasses.replace(Config(), seed=args.seed, **parse_overrides(args.set))
    state, key, step_fn, m = new_world(cfg)
    joint = make_joint_controller("scripted", "scripted", cfg, m)
    scan = make_scan(step_fn, joint)
    # 预热编译
    st, key, _ = scan(state, key, 10)
    jax.block_until_ready(st)
    t0 = time.time()
    st, key, dones = scan(state, key, args.ticks)
    jax.block_until_ready(st)
    dt = time.time() - t0
    print(f"{args.ticks} ticks in {dt:.3f}s = {args.ticks / dt:.0f} tick/s (单环境)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="teow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="跑一场对局并落 run 目录")
    p.add_argument("--p0", default="scripted", help="玩家0控制器 random|scripted")
    p.add_argument("--p1", default="scripted", help="玩家1控制器")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slug", default="play", help="run 目录名后缀")
    p.add_argument("--record", action="store_true", help="录制轨迹供 replay")
    p.add_argument("--record-every", type=int, default=2, help="每几 tick 录一帧")
    p.add_argument("--set", nargs="*", metavar="FIELD=VALUE")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("replay", help="回放 run 目录里的轨迹")
    p.add_argument("run_dir")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--save", default=None, help="存成 gif/mp4 路径(不弹窗)")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("bench", help="单环境吞吐基准")
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", metavar="FIELD=VALUE")
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
