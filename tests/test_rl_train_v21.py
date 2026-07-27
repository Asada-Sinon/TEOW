"""v2.1 训练管线 slow smoke:CPU 微训整环路 + ckpt 往返 + 课程升档 + 持续环境 reset 正确性。

标 @slow(verify.sh/门禁排除;真训练前的正确性门)。用 _FAST 小局(短 episode + 早开门 + 强怪)
让整局 scan 在 CPU 秒级跑完。持续环境 reset 是最易静默出 bug 处(终局 obs 泄进新局 / elim 未
重置 → 名次污染),这里用「done 后必须恢复非 done」钉死。
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_skeleton_v20 import (
    RLConfig,
    _derived_norms,
    adam_init,
    encode_obs,
    init_params,
    type_cost_table,
)
from rl_train_v21 import (
    CurriculumScheduler,
    build_collect_rollout,
    init_env_carry,
    load_ckpt,
    save_ckpt,
    train,
)

from teow.actions import n_actions
from teow.config import Config
from teow.map import build_map
from teow.state import init_state

pytestmark = pytest.mark.slow

# 短局 + 早开门 + 强怪 → CPU 秒级必分胜负
_FAST = dict(episode_len=300, gate_open_tick=80, monster_atk_base=120,
             monster_hp_base=200, monster_speed=0.8)


def _tiny_rlcfg(**kw):
    base = RLConfig(num_envs=4, num_steps=32, total_updates=2, update_epochs=2,
                    num_minibatches=2, eval_every=1, ckpt_every=1)
    return dataclasses.replace(base, **kw)


def _mk_params(cfg, mapdata, rlcfg, key):
    norms = _derived_norms(cfg)
    norms["res"], norms["pot"], norms["mon_hp"] = (
        rlcfg.res_scale, rlcfg.pot_scale, rlcfg.mon_hp_scale)
    cost_tbl = type_cost_table(cfg)
    obs0 = encode_obs(init_state(cfg, mapdata), cfg, mapdata, 0, norms, cost_tbl)
    F, G, A = obs0.tokens.shape[1], obs0.global_vec.shape[0], n_actions(cfg)
    return init_params(F, G, A, rlcfg, key)


def test_train_smoke_runs(tmp_path):
    """微训 2 update 跑通,落盘 metrics/eval/ckpt,loss 有限、参数在动(gnorm>0)。"""
    cfg = dataclasses.replace(Config(), **_FAST)
    rlcfg = _tiny_rlcfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, rlcfg, run_dir, [("random", "random", "random")], seed=0, eval_seeds=2)

    m = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(m) == 2, "应有 2 条 update 记录"
    assert all(np.isfinite(r["loss"]) for r in m), "loss 必须有限"
    assert any(r["gnorm"] > 0 for r in m), "梯度范数应 >0(参数在动)"
    e = [json.loads(x) for x in (run_dir / "eval.jsonl").read_text().splitlines()]
    assert len(e) >= 1 and 0.0 <= e[0]["winrate_p0"] <= 1.0
    assert (run_dir / "checkpoints").exists()


def test_ckpt_roundtrip(tmp_path):
    """save→load 后 params/opt 逐叶 bit 相等、adam_t/rng/meta 一致。"""
    cfg = dataclasses.replace(Config(), **_FAST)
    mapdata = build_map(cfg)
    rlcfg = _tiny_rlcfg()
    key = jax.random.PRNGKey(1)
    params = _mk_params(cfg, mapdata, rlcfg, key)
    opt = adam_init(params)
    opt = dict(opt, t=7)  # 非零 adam step 计数
    run_dir = tmp_path
    (run_dir / "checkpoints").mkdir()
    save_ckpt(run_dir, 5, params, opt, key, {"stage": 1}, rlcfg, "testhash")
    p2, o2, k2, meta = load_ckpt(
        run_dir / "checkpoints" / "ckpt_0000005.npz",
        run_dir / "checkpoints" / "ckpt_0000005.json")
    assert params.keys() == p2.keys()
    for k in params:
        assert np.array_equal(np.asarray(params[k]), np.asarray(p2[k])), f"params[{k}] 不一致"
    for k in opt["m"]:
        assert np.array_equal(np.asarray(opt["m"][k]), np.asarray(o2["m"][k]))
        assert np.array_equal(np.asarray(opt["v"][k]), np.asarray(o2["v"][k]))
    assert o2["t"] == 7
    assert np.array_equal(np.asarray(key), np.asarray(k2))
    assert meta["update_step"] == 5 and meta["curriculum"]["stage"] == 1


def test_curriculum_promote():
    """窗口满且滑窗均值≥阈值 → 升档;不满/低于阈值不升。"""
    tiers = [("random", "random", "random"), ("turtle", "turtle", "turtle")]
    s = CurriculumScheduler(tiers, promote_threshold=0.6, window=3)
    assert s.stage == 0
    assert not s.step(0.9)          # 窗口未满(1/3)
    assert not s.step(0.9)          # 2/3
    assert s.step(0.9)              # 3/3 且均值 0.9≥0.6 → 升
    assert s.stage == 1
    # 已到最后一档,不再升
    for _ in range(5):
        s.step(0.9)
    assert s.stage == 1
    # 低胜率不升
    s2 = CurriculumScheduler(tiers, promote_threshold=0.6, window=3)
    for _ in range(4):
        assert not s2.step(0.3)
    assert s2.stage == 0


def test_persistent_env_reset():
    """持续环境正确性:num_steps 跨多局时,done 必须能恢复非 done(reset 生效),
    否则 step 的 done 冻结会让「一旦 done 永久 done」。同时验 reward/traj 无 NaN。"""
    cfg = dataclasses.replace(Config(), **_FAST)   # 局长 ~300(gate 80 后强怪速杀)
    mapdata = build_map(cfg)
    rlcfg = _tiny_rlcfg(num_envs=2, num_steps=700)  # >2 局,必多次终局
    key = jax.random.PRNGKey(2)
    key, ki = jax.random.split(key)
    params = _mk_params(cfg, mapdata, rlcfg, ki)
    roll, s0, e0 = build_collect_rollout(
        cfg, mapdata, ("random", "random", "random"), rlcfg)
    carry = init_env_carry(s0, e0, rlcfg.num_envs, key)
    traj, lv, carry = roll(params, carry, jnp.float32(0.1), key)

    done = np.asarray(traj["done"])          # [B, T] bool
    reward = np.asarray(traj["reward"])
    assert np.isfinite(reward).all(), "reward 含 NaN/Inf"
    assert np.isfinite(np.asarray(lv)).all()
    for b in range(rlcfg.num_envs):
        assert done[b].sum() >= 1, f"env{b} 700 tick 内应至少终局一次"
        first = int(np.argmax(done[b]))      # 第一次 done 的 tick
        if first < rlcfg.num_steps - 1:
            # reset 生效:第一次 done 之后不应「全程 True」(否则=永久冻结未重置)
            assert not done[b][first + 1:].all(), \
                f"env{b} done 后未 reset(持续环境 autoreset 失效)"
    # carry 末态 tick 应 < episode_len(reset 后重新计,而非停在终局硬冻结)
    final_tick = int(np.asarray(carry[0].tick).reshape(-1)[0])
    assert final_tick < cfg.episode_len, "carry 未 reset,tick 卡在终局"
