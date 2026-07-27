"""v2.1 Phase D:读 run 的 metrics.jsonl + eval.jsonl → training_curves.png 诊断板。

6 面板:①winrate+avg rank ②policy entropy ③loss/pg/vf ④PPO health(kl/explained_var/grad_norm)
⑤mean_reward + gate rate(叠 winrate 查 reward-hacking:代理奖励升但胜率不升即警) ⑥末次 eval 动作
分布(查单一动作垄断)。标签用英文(matplotlib 默认字体无中文 glyph);数字全读自 jsonl。

用法:.venv/bin/python explorations/plot_training_v21.py <experiments/run_dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_jsonl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: plot_training_v21.py <run_dir>")
    run = pathlib.Path(sys.argv[1])
    m = load_jsonl(run / "metrics.jsonl")
    e = load_jsonl(run / "eval.jsonl")
    mu = [r["update"] for r in m]
    eu = [r["update"] for r in e]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"v2.1 training diagnostics  {run.name}", fontsize=13)

    # (1) winrate + avg rank
    a = ax[0, 0]
    a.plot(eu, [r["winrate_p0"] for r in e], "-o", color="C0", label="winrate vs tier")
    a.axhline(0.25, ls=":", color="gray", lw=1, label="FFA random 0.25")
    a.set_ylim(0, 1); a.set_xlabel("update"); a.set_ylabel("winrate", color="C0")
    a.set_title("(1) learning signal: winrate + rank")
    a2 = a.twinx()
    a2.plot(eu, [r["avg_rank_p0"] for r in e], "--s", color="C3", label="avg rank (lower=stronger)")
    a2.set_ylabel("avg rank", color="C3"); a2.invert_yaxis()
    a.legend(loc="upper left", fontsize=8); a2.legend(loc="lower right", fontsize=8)

    # (2) policy entropy
    a = ax[0, 1]
    a.plot(eu, [r["avg_policy_entropy"] for r in e], "-o", color="C2")
    a.set_xlabel("update"); a.set_ylabel("policy entropy")
    a.set_title("(2) policy entropy (->0 = degenerate)")

    # (3) loss breakdown
    a = ax[0, 2]
    for k, c in [("loss", "C0"), ("pg", "C1"), ("vf", "C2"), ("ent", "C4")]:
        if k in m[0]:
            a.plot(mu, [r[k] for r in m], color=c, label=k, lw=1)
    a.set_xlabel("update"); a.set_title("(3) loss breakdown"); a.legend(fontsize=8)

    # (4) PPO health
    a = ax[1, 0]
    a.plot(mu, [r["approx_kl"] for r in m], color="C1", label="approx_kl", lw=1)
    a.plot(mu, [r.get("explained_var", float("nan")) for r in m],
           color="C2", label="explained_var", lw=1)
    a.plot(mu, [r["gnorm"] for r in m], color="C5", label="grad_norm", lw=1)
    a.axhline(0.3, ls=":", color="C1", lw=1)
    a.set_xlabel("update"); a.set_title("(4) PPO health (kl<0.3, ev>0)"); a.legend(fontsize=8)

    # (5) reward-hacking check: mean_reward + gate rate vs winrate
    a = ax[1, 1]
    a.plot(mu, [r["mean_reward"] for r in m], color="C6", label="mean_reward", lw=1)
    a.set_xlabel("update"); a.set_ylabel("mean_reward", color="C6")
    a.set_title("(5) reward-hacking check (reward up but winrate flat = warn)")
    a5 = a.twinx()
    a5.plot(eu, [r["winrate_p0"] for r in e], "-o", color="C0", label="winrate", ms=3)
    a5.plot(eu, [r["gate_rate"] for r in e], "--", color="C3", label="gate rate", lw=1)
    a5.set_ylabel("winrate / gate rate"); a5.set_ylim(0, 1)
    a.legend(loc="upper left", fontsize=8); a5.legend(loc="lower right", fontsize=8)

    # (6) action distribution (last eval)
    a = ax[1, 2]
    hist = e[-1]["act_hist_mean"]
    a.bar(range(len(hist)), hist, color="C0", width=1.0)
    mx = max(hist)
    a.axhline(0.95, ls=":", color="C3", lw=1, label="degen threshold 0.95")
    a.set_xlabel("action id"); a.set_ylabel("freq")
    a.set_title(f"(6) action dist last eval (peak={mx:.2f})"); a.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = run / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"=== wrote {out} ===")


if __name__ == "__main__":
    main()
