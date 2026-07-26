# ruff: noqa: E402  (JAX_PLATFORMS 须先于 import jax 设定 + sys.path 注入 src,
# 故 teow 导入不在文件顶部;与 explorations/ 其余脚本同一惯例)
"""v1.8 多风格指挥官烟测:每个 profile 打 3 个 random,查建军 sanity + 胜负。

问题:base.commander_actions 的每个 profile 能否 (a) 不崩、(b) 建兵营+出军、
(c) 多数情况碾压 3 个 random?附带核验 balanced≈scripted(共享宏观管线逐算子等价)
与 chaos 的 stochastic 决定论(同 seed 复现、异 seed 分化)。

口径:CPU 单环境 make_scan 短局(episode_len=1200 / gate_open_tick=800);seed 显式;
provenance(git hash + resolved config + seeds)打印到 stdout。degenerate(never 建军 /
输 random)只标记不阻断——留 P4 调参。

用法:JAX_PLATFORMS=cpu .venv/bin/python explorations/smoke_commanders_v18.py
      仅测一个:… --only balanced   自定 seeds/局长:… --seeds 0,1,2 --episode 1200
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # setup:CPU 单环境(烟测,非门禁)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from teow.commanders import PROFILES
from teow.config import TYPE_BARRACKS, Config
from teow.controller import make_joint_controller
from teow.map import build_map
from teow.state import init_state
from teow.step import build_step


def provenance(cfg, seeds):
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        porc = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT).decode().strip()
        dirty = bool(porc)
    except Exception as e:  # noqa: BLE001
        h, dirty = f"git 不可用: {e}", "?"
    print(f"# git={h} dirty={dirty} backend={jax.default_backend()}")
    print(f"# seeds={seeds}")
    print(f"# resolved_config={dataclasses.asdict(cfg)}")


def make_runner(cfg, name):
    """返回 run(seeds[int32 S]) -> dict[S]:profile `name` 在 seat0 打 random×3。"""
    mapdata = build_map(cfg)
    step_fn = build_step(cfg, mapdata)
    joint = make_joint_controller(name, "random", "random", "random",
                                  cfg=cfg, mapdata=mapdata)
    state0 = init_state(cfg, mapdata)
    P, e = cfg.n_players, cfg.e_max
    is_combat = jnp.asarray(cfg.is_combat_by_type, bool)

    def body(carry, _):
        st, k = carry
        k, ka, ks = jax.random.split(k, 3)
        st = step_fn(st, joint(st, ka), ks)
        et = jnp.clip(st.etype.astype(jnp.int32), 0, 31)
        bar0 = jnp.any(st.alive[:e] & (st.etype[:e] == TYPE_BARRACKS))
        army0 = jnp.sum(st.alive[:e] & is_combat[et[:e]])
        # 事件旗:曾建兵营 / 曾出军(累积 max)
        return (st, k), jnp.stack([bar0.astype(jnp.int32),
                                   (army0 > 0).astype(jnp.int32)])

    def one(seed):
        (fin, _), ev = jax.lax.scan(
            body, (state0, jax.random.PRNGKey(seed)), None, length=cfg.episode_len)
        et = jnp.clip(fin.etype.astype(jnp.int32), 0, 31)
        army = (fin.alive & is_combat[et]).reshape(P, e).sum(axis=1)      # [P]
        alive0 = fin.alive[:e].sum()
        return dict(winner=fin.winner.astype(jnp.int32), length=fin.tick,
                    ever_bar=ev[:, 0].max(), ever_army=ev[:, 1].max(),
                    army0=army[0], army_field=army[1:].sum(), alive0=alive0)

    return jax.jit(jax.vmap(one))


def sweep(cfg, seeds, names):
    seeds_a = jnp.asarray(seeds, jnp.int32)
    print(f"\n{'profile':<10} {'win/N':>6} {'ever_bar':>8} {'ever_army':>9} "
          f"{'army0~':>7} {'field~':>7} {'len~':>6}  flags", flush=True)
    for name in names:
        t0 = time.time()
        try:
            out = jax.tree.map(lambda x: x.tolist(), make_runner(cfg, name)(seeds_a))
        except Exception as e:  # noqa: BLE001
            print(f"{name:<10}  CRASH: {type(e).__name__}: {e}", flush=True)
            continue
        S = len(seeds)
        wins = sum(1 for w in out["winner"] if w == 0)
        ever_bar = min(out["ever_bar"])          # 全 seed 都建了兵营?取最保守
        ever_army = min(out["ever_army"])
        army0 = sum(out["army0"]) / S
        field = sum(out["army_field"]) / S
        alen = sum(out["length"]) / S
        flags = []
        if ever_army == 0:
            flags.append("NEVER-ARMY")
        if wins < (S + 1) // 2:
            flags.append("LOSES-RANDOM")
        if ever_bar == 0:
            flags.append("NO-BARRACKS")
        dt = time.time() - t0
        print(f"{name:<10} {wins:>4}/{S} {ever_bar:>8} {ever_army:>9} "
              f"{army0:>7.1f} {field:>7.1f} {alen:>6.0f}  "
              f"{'|'.join(flags) or 'ok':<12} ({dt:.0f}s)", flush=True)


def _rollout_state(cfg, names, seed, length):
    mapdata = build_map(cfg)
    step_fn = build_step(cfg, mapdata)
    joint = make_joint_controller(*names, cfg=cfg, mapdata=mapdata)
    state0 = init_state(cfg, mapdata)

    def body(carry, _):
        st, k = carry
        k, ka, ks = jax.random.split(k, 3)
        st = step_fn(st, joint(st, ka), ks)
        return (st, k), None

    (fin, _), _ = jax.lax.scan(body, (state0, jax.random.PRNGKey(seed)), None,
                               length=length)
    return fin


def equiv_balanced_vs_scripted():
    """balanced 走 commander_actions,scripted 走原函数;默认 config 下、早期窗口
    (gate_open//3=1333)之前逐 tick 必等价(adaptive 未触发)。跑 600 tick 比末态。"""
    cfg = Config()                                # 默认:gate_open=4000 → early_win 1333
    L = 600
    a = _rollout_state(cfg, ["scripted"] * 4, 7, L)
    b = _rollout_state(cfg, ["balanced"] * 4, 7, L)
    fields = ("alive", "etype", "hp", "pos", "resources", "level", "upgrades",
              "order", "btype", "btimer", "tick", "winner")
    diffs = [f for f in fields
             if not bool(jnp.array_equal(getattr(a, f), getattr(b, f)))]
    print(f"\n[equiv] balanced vs scripted @tick{L}(默认config,<early_win）: "
          f"{'IDENTICAL' if not diffs else 'DIVERGE ' + str(diffs)}")
    return not diffs


def determinism_chaos():
    """chaos(stochastic=1):同 seed 两跑必等(key 决定论);seed0 vs seed1 应分化。"""
    cfg = Config(episode_len=400, gate_open_tick=300)
    a1 = _rollout_state(cfg, ["chaos", "random", "random", "random"], 3, 400)
    a2 = _rollout_state(cfg, ["chaos", "random", "random", "random"], 3, 400)
    b = _rollout_state(cfg, ["chaos", "random", "random", "random"], 4, 400)
    same_seed = bool(jnp.array_equal(a1.resources, a2.resources)
                     and jnp.array_equal(a1.alive, a2.alive))
    diff_seed = not (bool(jnp.array_equal(a1.resources, b.resources))
                     and bool(jnp.array_equal(a1.alive, b.alive)))
    print(f"[determinism] chaos same-seed identical={same_seed}  "
          f"diff-seed diverges={diff_seed}")
    return same_seed and diff_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只测一个 profile 名")
    ap.add_argument("--profiles", default=None,
                    help="逗号分隔的 profile 子集(默认全测);配合 --skip-checks 分批跑")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episode", type=int, default=1200)
    ap.add_argument("--gate", type=int, default=800)
    # 默认给中等启动资源:默认经济(100/50)在 1200 tick 短局里到不了兵营
    # (balanced==scripted 均停在经济期),military 杠杆不被检验;适度加资源让
    # 兵营/军队在短局内可靠出现(同 test_scripted_v16 富开局定标先例)。
    ap.add_argument("--start-ore", type=int, default=600)
    ap.add_argument("--start-water", type=int, default=400)
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = Config(episode_len=args.episode, gate_open_tick=args.gate,
                 start_ore=args.start_ore, start_water=args.start_water)
    provenance(cfg, seeds)

    if not args.skip_checks:
        equiv_balanced_vs_scripted()
        determinism_chaos()

    if args.only:
        seeds_a = jnp.asarray(seeds, jnp.int32)
        t0 = time.time()
        out = jax.tree.map(lambda x: x.tolist(), make_runner(cfg, args.only)(seeds_a))
        print(f"\n[{args.only}] {out}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        names = args.profiles.split(",") if args.profiles else list(PROFILES)
        sweep(cfg, seeds, names)


if __name__ == "__main__":
    main()
