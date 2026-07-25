"""单 tick 步进:各阶段的组合与结算顺序(本文件头注释是顺序的唯一权威说明)。

tick 结算顺序(设计依据见 docs/plans/ 下两份调研报告与 plan):
  1. production_tick     上一拍订的训练倒计时/落地(先算旧账,保证「训练耗时 T」
  2. construction_tick   与「建造耗时 T」精确成立:本 tick 新下的单不掉计时)
  3. harvest_tick        入驻/开采/出矿/卸货
  4. apply_orders        本 tick 新指令(合法性掩码;训练在此扣费)
  5. start_constructions 到场工人开工(抢点仲裁 + 扣费对账)
  6. movement_tick       距离场下降 + 抢格仲裁
  7. combat_tick         相邻同时结算
  8. cleanup_deaths      翻 alive + 连锁清理
  9. _end_tick           冷却/计时/胜负判定
终局冻结:done 之后再 step 是恒等映射(scan 安全)。

随机性:step(state, actions, key)——key 只用于两处仲裁(抢格优先级、抢点先手),
每 tick split,决定论 = f(init_state, 指令序列, key 序列)。
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .actions import apply_orders
from .combat import cleanup_deaths, combat_tick
from .config import Config
from .economy import (
    construction_tick,
    harvest_tick,
    paid_orders_pass,
    production_tick,
    special_tasks_tick,
    start_constructions,
)
from .map import MapData, build_map
from .movement import movement_tick
from .state import WorldState, hq_slot, init_state, owner_of_slots


def _end_tick(state: WorldState, cfg: Config) -> WorldState:
    st = state
    tick = st.tick + 1
    hq_dead = jnp.stack([st.hp[hq_slot(0, cfg)] <= 0,
                         st.hp[hq_slot(1, cfg)] <= 0])
    winner = jnp.where(hq_dead[0] & hq_dead[1], 2,
                       jnp.where(hq_dead[0], 1,
                                 jnp.where(hq_dead[1], 0, st.winner)))
    winner = jnp.where((winner == -1) & (tick >= cfg.episode_len), 2, winner)
    done = winner != -1
    return st._replace(tick=tick, done=done, winner=winner.astype(jnp.int8))


def build_step(cfg: Config, mapdata: MapData):
    """返回 jit 后的 step(state, actions, key) -> state。
    静态地图闭包在此,不进 state、不进 scan carry。"""
    owner = owner_of_slots(cfg)

    def step(state: WorldState, actions: jax.Array, key: jax.Array) -> WorldState:
        k_claim, k_move = jax.random.split(key)
        st = production_tick(state, cfg, mapdata)
        st = special_tasks_tick(st, cfg, owner)  # 负数 btype 任务(升级/研发/建营)
        st = construction_tick(st, cfg, mapdata, owner)
        st = harvest_tick(st, cfg, mapdata, owner)
        st, act = apply_orders(st, actions, cfg, mapdata, owner)
        st = paid_orders_pass(st, act, cfg, mapdata, owner)  # 付费指令顺序对账(B-1)
        st = start_constructions(st, cfg, mapdata, owner, k_claim)
        st = movement_tick(st, cfg, mapdata, owner, k_move)
        st = combat_tick(st, cfg, owner)
        st = cleanup_deaths(st, cfg, owner)
        st = _end_tick(st, cfg)
        # 终局冻结:done 的局再 step 恒等(tick 也不再走)
        return jax.tree.map(
            lambda old, new: jnp.where(state.done, old, new), state, st)

    return jax.jit(step)


def make_scan(step_fn, controller_fn):
    """把「控制器出招 + step」打包成 scan body,跑 n 步(用于 bench/批量对局)。
    controller_fn(state, key) -> actions[N](两家动作已合并)。"""

    def body(carry, _):
        state, key = carry
        key, k_act, k_step = jax.random.split(key, 3)
        actions = controller_fn(state, k_act)
        state = step_fn(state, actions, k_step)
        return (state, key), state.done

    @functools.partial(jax.jit, static_argnums=(2,))
    def scan_steps(state, key, n_steps):
        (state, key), dones = jax.lax.scan(body, (state, key), None, length=n_steps)
        return state, key, dones

    return scan_steps


def new_world(cfg: Config):
    """便捷入口:地图 + 初始状态 + jit step + 主 PRNGKey。"""
    mapdata = build_map(cfg)
    state = init_state(cfg, mapdata)
    step_fn = build_step(cfg, mapdata)
    key = jax.random.PRNGKey(cfg.seed)
    return state, key, step_fn, mapdata
