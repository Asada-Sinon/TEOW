"""v2.1 Phase D:读 run 的 metrics.jsonl + eval.jsonl → verdict.md [AI-DRAFT]「没训废」四门判定。

四门(plan §D;阈值 [AI-DRAFT],以**实测未训基线**=eval update0 为锚):
  门1 学习信号:vs 对手胜率明显高于未训基线且随训练升(Δ≥+0.2)。
  门2 行为不退化:动作分布非单一垄断(max<0.95)、熵不塌(未跌破基线的零头)。
  门3 不摆烂等门:不是 100% 靠异界之门耗死(有军队/有主动作战)——直接回应用户否决短局的理由。
  门4 PPO 健康:loss 有限、approx_kl 不长期失控、explained_var>0、grad_norm 收敛。
判定是 [AI-DRAFT],须人工/result-analyst 读原始输出后定夺。数字全读自落盘文件,不手估。

用法:.venv/bin/python explorations/diagnose_train_v21.py <experiments/run_dir>
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

import numpy as np


def load_jsonl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: diagnose_train_v21.py <run_dir>")
    run = pathlib.Path(sys.argv[1])
    metrics = load_jsonl(run / "metrics.jsonl")
    evals = load_jsonl(run / "eval.jsonl")
    if not evals or not metrics:
        sys.exit("metrics.jsonl / eval.jsonl 为空,无法诊断")

    out = [f"# v2.1 试训诊断 verdict [AI-DRAFT]", f"", f"- run: `{run}`",
           f"- updates: {len(metrics)}  evals: {len(evals)}",
           f"- 对手档序列: {[e.get('opp') for e in evals]}", ""]

    # ---- 门1:学习信号 ----
    base_wr = evals[0]["winrate_p0"]
    last_wr = evals[-1]["winrate_p0"]
    best_wr = max(e["winrate_p0"] for e in evals)
    delta = last_wr - base_wr
    base_rank = evals[0]["avg_rank_p0"]
    last_rank = evals[-1]["avg_rank_p0"]
    g1 = "PASS" if (delta >= 0.20 and last_wr > base_wr) else (
        "WARN" if delta > 0.05 else "FAIL")
    out += [f"## 门1 学习信号: **{g1}**",
            f"- 未训基线 winrate={base_wr:.3f} → 末 winrate={last_wr:.3f} "
            f"(Δ={delta:+.3f}),峰值={best_wr:.3f}",
            f"- 平均名次 {base_rank:.2f} → {last_rank:.2f}(越小越强)",
            f"- 注:胜率须与实测未训基线比,而非绝对值(随机初始网在 FFA 可能靠运气赢一些)", ""]

    # ---- 门2:行为不退化 ----
    hist = np.asarray(evals[-1]["act_hist_mean"], float)
    mx = float(hist.max()) if hist.size else float("nan")
    mx_id = int(hist.argmax()) if hist.size else -1
    noop = float(hist[0]) if hist.size else float("nan")
    ent = evals[-1]["avg_policy_entropy"]
    base_ent = evals[0]["avg_policy_entropy"]
    g2 = "FAIL" if (mx > 0.95 or ent < 0.05) else (
        "WARN" if (mx > 0.80 or ent < 0.2 * max(base_ent, 1e-6)) else "PASS")
    out += [f"## 门2 行为不退化: **{g2}**",
            f"- 动作分布峰值={mx:.3f}(动作 id {mx_id});NOOP 占比={noop:.3f}",
            f"- 策略熵 基线 {base_ent:.3f} → 末 {ent:.3f}(塌到 0 = 退化)",
            f"- 判据:峰值>0.95 或 熵<0.05 → 单一动作垄断/熵塌(FAIL)", ""]

    # ---- 门3:不摆烂等门 ----
    g_rate = evals[-1]["gate_rate"]
    army = evals[-1]["avg_army_p0"]
    econ = evals[-1].get("avg_econ_p0", float("nan"))
    g3 = "PASS" if (army >= 1.0 or g_rate < 0.95) else "WARN"
    out += [f"## 门3 不摆烂等门: **{g3}**",
            f"- 末军均值={army:.2f};gate 到达率={g_rate:.2f};投入价值={econ:.1f}",
            f"- 用户否决短局的理由:防 RL 只龟缩耗死脚本。army≥1(有作战单位)或 gate<0.95 "
            f"(不总靠门)→ 非纯摆烂;army≈0 且 gate≈1 → ⚠️ 疑似只等门", ""]

    # ---- 门4:PPO 健康 ----
    losses = [m["loss"] for m in metrics]
    kls = [m["approx_kl"] for m in metrics]
    gnorms = [m["gnorm"] for m in metrics]
    evs = [m["explained_var"] for m in metrics
           if m.get("explained_var") is not None and np.isfinite(m["explained_var"])]
    finite = all(np.isfinite(x) for x in losses)
    kl_tail = statistics.mean(kls[-max(1, len(kls) // 5):])
    ev_last = evs[-1] if evs else float("nan")
    kl_ok = kl_tail < 0.3
    ev_ok = (ev_last > 0) if evs else False
    gn_ok = np.isfinite(gnorms[-1])
    g4 = "PASS" if (finite and kl_ok and ev_ok and gn_ok) else (
        "FAIL" if not finite else "WARN")
    out += [f"## 门4 PPO 健康: **{g4}**",
            f"- loss 全有限={finite};尾段 approx_kl 均值={kl_tail:.4f}(目标 <0.3)",
            f"- explained_var 末={ev_last:.3f}(>0 才比均值强);grad_norm 末={gnorms[-1]:.3f}",
            f"- reward-hacking 警戒:若 mean_reward/Φ 升但门1 胜率不升 → 查(见 plot Φ-vs-胜率叠图)",
            ""]

    verdicts = [g1, g2, g3, g4]
    overall = ("训废/管线坏 → 停下查" if "FAIL" in verdicts else
               "需人工复核" if "WARN" in verdicts else "四门全过(初步「在学、没训废」)")
    out += ["---", f"## 综合 [AI-DRAFT]: **{overall}**",
            f"- 门1={g1} 门2={g2} 门3={g3} 门4={g4}",
            "- 本判定为 [AI-DRAFT],须人工/result-analyst 读原始 metrics/eval jsonl 后定夺;"
            "任一 FAIL 即停下排查,勿放大 run。"]
    txt = "\n".join(out) + "\n"
    (run / "verdict.md").write_text(txt)
    print(txt)
    print(f"=== 写入 {run / 'verdict.md'} ===")


if __name__ == "__main__":
    main()
