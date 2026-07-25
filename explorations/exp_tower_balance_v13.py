"""v1.3 Phase 6 哨塔平衡实验:config-only 杠杆(攻/血/造价)扫描,数据交用户定案。

问题:v1.2 哨塔(造价 50/30,L1 攻 6 血 120)在狗 rush 与全局对局里是否过强/过弱,
config-only 杠杆是否足够调平(不够才考虑「攻击间隔」新机制)。

场景 A「狗 rush 微操局」:手工摆位——p0 基地防线(HQ + 1 座 L1 塔 + 3 个工人,
双方默认出生工人撤场保持无菌)对 N∈{2,3,4,5} 只 ORDER_ATTACK 狗。塔 (5,5) 卡在
敌军攻 HQ(3,3) 的对角进军路上,工人在塔东北一层作业位,接战有保证。
记录:塔存活/余血、狗存活数、工人伤亡、分出结果的 tick(狗灭/塔毁/HQ 陷/超时)。

场景 B「全局对局敏感性」:scripted vs scripted × 8 seeds,记录胜负与终局 tick
(scripted 双方各建至多 1 座塔,见 controller.py)。

变体只改等级 1 的表项:两场景都只出现 1 级塔(场景 A 手术摆位即 L1;场景 B
scripted 只建塔不升塔,controller.py 无塔升级分支)。cost 变体在场景 A 无效
(塔为手术放置、不走扣费),其行预期与 base 相同,留作对照。

产物:experiments/20260725-tower-balance-<variant>/ 每变体一个 run 目录
(resolved config / git / seeds / backend / command + scenario_a.jsonl +
scenario_b.jsonl),另加 …-tower-balance-summary/ 放跨变体汇总表 summary.md。
scenario_b.jsonl 的 tower_seen_p* 是「块边界(每 250 tick)抽样时该玩家有存活塔」,
非逐 tick 精确。

用法(务必 .venv/bin/python,CPU 跑):
  JAX_PLATFORMS=cpu .venv/bin/python explorations/exp_tower_balance_v13.py
  烟测(不污染 experiments/):… --smoke --out-root /tmp/xxx
"""

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 实验/门禁一律 CPU(MEMORY LEARN:env)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from teow.actions import A_NOOP
from teow.config import TYPE_DOG, TYPE_TOWER, TYPE_WORKER, Config
from teow.controller import make_joint_controller
from teow.state import ORDER_ATTACK, hq_slot
from teow.step import make_scan, new_world


def _set1(tbl: tuple, v: int) -> tuple:
    """只改 *_by_level 表的等级 1 项(本实验两场景都只出现 1 级塔)。"""
    t = list(tbl)
    t[1] = v
    return tuple(t)


_BASE = Config()
# 变体清单来自 plan Phase 6 规格:{现值, atk 6→4, atk 6→3, cost 50/30→80/50,
# hp 120→90};数值是实验设计的一部分,随本脚本进版本控制,并 dump 进各 run 目录
# 的 resolved_config.json。
VARIANTS = {
    "base": {},
    "atk4": dict(tower_atk_by_level=_set1(_BASE.tower_atk_by_level, 4)),
    "atk3": dict(tower_atk_by_level=_set1(_BASE.tower_atk_by_level, 3)),
    "cost80-50": dict(tower_cost_ore=80, tower_cost_water=50),
    "hp90": dict(tower_hp_by_level=_set1(_BASE.tower_hp_by_level, 90)),
}

SCEN_A_SEED = 0             # 场景 A 无控制器随机源,seed 仅驱动 step 内仲裁并记档
SCEN_A_DOG_COUNTS = (2, 3, 4, 5)
SCEN_A_MAX_TICKS = 400
SCEN_B_SEEDS = tuple(range(8))
SCEN_B_CHUNK = 250          # scan 分块步长,块间查终局早停(done 后 step 恒等,安全)

# 场景 A 摆位(24×24 图;全部避开 HQ 格 (3,3)/(20,20) 与 8 个资源点格,
# 资源点格永久不可通行,见 map.py)
A_TOWER_POS = (5.0, 5.0)
A_WORKER_POS = ((6.0, 6.0), (5.0, 6.0), (6.0, 5.0))
A_DOG_POS = ((10.0, 10.0), (9.0, 10.0), (10.0, 9.0), (11.0, 10.0), (10.0, 11.0))


def new_run_dir(out_root: pathlib.Path, slug: str) -> pathlib.Path:
    date = datetime.date.today().strftime("%Y%m%d")
    d = out_root / f"{date}-{slug}"
    i = 1
    while d.exists():
        i += 1
        d = out_root / f"{date}-{slug}-{i}"
    d.mkdir(parents=True)
    return d


def write_provenance(run_dir: pathlib.Path, cfg, seeds) -> None:
    """三件套(git hash / resolved config / seed)+ backend + 命令行。"""
    if cfg is not None:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    (run_dir / "seeds.txt").write_text("\n".join(str(s) for s in seeds) + "\n")
    (run_dir / "backend.txt").write_text(jax.default_backend() + "\n")
    (run_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=ROOT).stdout
    (run_dir / "git.txt").write_text(f"{head}\ndirty: {bool(dirty)}\n{dirty}")
    if dirty:
        diff = subprocess.run(["git", "diff"], capture_output=True, text=True,
                              cwd=ROOT).stdout
        (run_dir / "git.diff").write_text(diff)


def _spawn(st, slot, etype, pos, hp, order=0):
    return st._replace(
        alive=st.alive.at[slot].set(True),
        etype=st.etype.at[slot].set(etype),
        pos=st.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
        hp=st.hp.at[slot].set(hp),
        order=st.order.at[slot].set(order),
    )


def run_scene_a(cfg, step_fn, state0, n_dogs: int) -> dict:
    st = state0
    # 无菌化:撤掉双方默认出生工人,场景 = HQ + 塔 + 3 工人 vs N 狗
    for p in (0, 1):
        for i in range(cfg.start_workers):
            st = st._replace(
                alive=st.alive.at[hq_slot(p, cfg) + 1 + i].set(False))
    tw = hq_slot(0, cfg) + 30
    st = _spawn(st, tw, TYPE_TOWER, A_TOWER_POS, cfg.tower_hp_by_level[1])
    wslots = [hq_slot(0, cfg) + 31 + i for i in range(len(A_WORKER_POS))]
    for s, p in zip(wslots, A_WORKER_POS):
        st = _spawn(st, s, TYPE_WORKER, p, cfg.worker_hp_by_level[1])
    dslots = jnp.asarray(
        [hq_slot(1, cfg) + 40 + i for i in range(n_dogs)])
    for i in range(n_dogs):
        st = _spawn(st, int(dslots[i]), TYPE_DOG, A_DOG_POS[i],
                    cfg.dog_hp_by_level[1], order=ORDER_ATTACK)

    key = jax.random.PRNGKey(SCEN_A_SEED)
    noop = jnp.full(cfg.n_total, A_NOOP, jnp.int32)
    outcome, tick = "timeout", SCEN_A_MAX_TICKS
    for t in range(SCEN_A_MAX_TICKS):
        key, sub = jax.random.split(key)
        st = step_fn(st, noop, sub)
        tower_alive = bool(st.alive[tw])
        dogs_left = int(jnp.sum(st.alive[dslots]))
        if not tower_alive or dogs_left == 0 or bool(st.done):
            tick = t + 1
            outcome = ("dogs_wiped" if dogs_left == 0 else
                       "tower_down" if not tower_alive else "hq_down")
            break
    return {
        "n_dogs": n_dogs, "outcome": outcome, "resolved_tick": tick,
        "tower_alive": bool(st.alive[tw]), "tower_hp": int(st.hp[tw]),
        "dogs_alive": int(jnp.sum(st.alive[dslots])),
        "workers_lost": sum(1 for s in wslots if not bool(st.alive[s])),
        "hq_hp": int(st.hp[hq_slot(0, cfg)]),
    }


def run_scene_b(cfg, step_fn, mapdata, state0, seeds) -> list[dict]:
    joint = make_joint_controller("scripted", "scripted", cfg, mapdata)
    scan = make_scan(step_fn, joint)
    rows = []
    for seed in seeds:
        st, key = state0, jax.random.PRNGKey(seed)
        tower_seen = [False, False]
        t0 = time.time()
        n_chunks = -(-cfg.episode_len // SCEN_B_CHUNK)
        for _ in range(n_chunks):
            st, key, _ = scan(st, key, SCEN_B_CHUNK)
            for p in (0, 1):
                sl = slice(p * cfg.e_max, (p + 1) * cfg.e_max)
                tower_seen[p] |= bool(jnp.any(
                    st.alive[sl] & (st.etype[sl] == TYPE_TOWER)))
            if bool(st.done):
                break
        rows.append({
            "seed": seed, "winner": int(st.winner), "end_tick": int(st.tick),
            "tower_seen_p0": tower_seen[0], "tower_seen_p1": tower_seen[1],
            "wall_s": round(time.time() - t0, 1),
        })
    return rows


def summarize(name: str, overrides: dict, a_rows, b_rows) -> str:
    """汇总表一行(markdown)。"""
    a_cells = "; ".join(
        f"N={r['n_dogs']}:{r['outcome']}@{r['resolved_tick']}"
        f"(狗余{r['dogs_alive']},工亡{r['workers_lost']},塔血{r['tower_hp']})"
        for r in a_rows)
    wins = [sum(1 for r in b_rows if r["winner"] == w) for w in (0, 1, 2)]
    med = statistics.median(r["end_tick"] for r in b_rows)
    towers = sum(1 for r in b_rows
                 if r["tower_seen_p0"] or r["tower_seen_p1"])
    ov = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "现值"
    return (f"| {name} | {ov} | {a_cells} | "
            f"p0胜{wins[0]}/p1胜{wins[1]}/和{wins[2]} | {med:.0f} | "
            f"{towers}/{len(b_rows)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "experiments"),
                    help="run 目录根(烟测时指到临时目录,别污染 experiments/)")
    ap.add_argument("--smoke", action="store_true",
                    help="烟测:只跑 base 变体、场景 A N=2、场景 B seed 0")
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)

    variants = {"base": VARIANTS["base"]} if args.smoke else VARIANTS
    a_counts = (2,) if args.smoke else SCEN_A_DOG_COUNTS
    b_seeds = (0,) if args.smoke else SCEN_B_SEEDS

    t0 = time.time()
    table_rows, run_ids = [], []
    for name, overrides in variants.items():
        vcfg = dataclasses.replace(Config(), **overrides)
        run_dir = new_run_dir(out_root, f"tower-balance-{name}")
        write_provenance(run_dir, vcfg, (SCEN_A_SEED,) + tuple(b_seeds))
        state0, _, step_fn, mapdata = new_world(vcfg)
        print(f"== 变体 {name} -> {run_dir.name}", flush=True)

        a_rows = []
        with open(run_dir / "scenario_a.jsonl", "w", encoding="utf-8") as fh:
            for n in a_counts:
                row = {"variant": name, **run_scene_a(vcfg, step_fn, state0, n)}
                a_rows.append(row)
                fh.write(json.dumps(row) + "\n")
                print(f"  A {row}", flush=True)

        b_rows = []
        with open(run_dir / "scenario_b.jsonl", "w", encoding="utf-8") as fh:
            for row in run_scene_b(vcfg, step_fn, mapdata, state0, b_seeds):
                row = {"variant": name, **row}
                b_rows.append(row)
                fh.write(json.dumps(row) + "\n")
                print(f"  B {row}", flush=True)

        run_ids.append(run_dir.name)
        table_rows.append(summarize(name, overrides, a_rows, b_rows))

    sum_dir = new_run_dir(out_root, "tower-balance-summary")
    write_provenance(sum_dir, None, (SCEN_A_SEED,) + tuple(b_seeds))
    header = (
        "# v1.3 哨塔平衡扫描汇总(Phase 6)\n\n"
        f"变体 run 目录:{', '.join(run_ids)}\n\n"
        "场景 A = 手术局(HQ+L1塔+3工人 vs N 狗,无菌);"
        "场景 B = scripted vs scripted × seeds(胜负/终局 tick;"
        "tower_seen 为 250-tick 块边界抽样)。cost 变体在场景 A 无效(塔手术放置"
        "不扣费),其 A 行为 base 对照。\n\n"
        "| 变体 | 改动 | 场景 A(逐 N) | 场景 B 胜负 | B 终局中位 tick "
        "| B 出现过塔的局 |\n|---|---|---|---|---|---|")
    md = header + "\n" + "\n".join(table_rows) + "\n"
    (sum_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"总耗时 {time.time() - t0:.0f}s;汇总 -> {sum_dir}")


if __name__ == "__main__":
    main()
