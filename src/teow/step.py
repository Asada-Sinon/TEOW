"""单 tick 步进:各阶段的组合与结算顺序(本文件头注释是顺序的唯一权威说明)。

tick 结算顺序(设计依据见 docs/plans/ 下两份调研报告与 plan):
  1. production_tick     上一拍订的训练倒计时/落地(先算旧账,保证「训练耗时 T」
  2. construction_tick   与「建造耗时 T」精确成立:本 tick 新下的单不掉计时)
  3. harvest_tick        入驻/开采/出矿/卸货
  4. apply_orders        本 tick 新指令(合法性掩码;训练在此扣费)
  5. start_constructions 到场工人开工(抢点仲裁 + 扣费对账)
  6. movement_tick       距离场下降 + 抢格仲裁
  6.5 gate_tick          异界之门(v1.8):门开(tick>=gate_open_tick)向每个存活玩家
                         中心出阵营隔离怪 + 怪物沿目标玩家 HQ 场慢速 descent
  7. combat_tick         相邻同时结算
  7.5 monster_combat_tick 怪物近战 pass(v1.8):阵营隔离(行 p 的怪↔owner==p 实体),
                         晚正常近战一个子阶段——不能并入 combat 的 incoming(已消费),
                         同一 pre-pass 快照算两侧、允许互杀
  8. cleanup_deaths      翻 alive + 连锁清理(玩家亡→其怪同帧离场)
  9. _end_tick           冷却/计时/胜负判定(v1.8:异界之门必分胜负,不再判和局)
终局冻结:done 之后再 step 是恒等映射(scan 安全)。

GARRISON 语义(v1.3):驻守单位在 movement 阶段走向锚点(target_cell),入
garrison_hold_radius 圈即站住,被互推挤出圈下 tick 自动回岗;永不自转 IDLE,
只在 cleanup_deaths 里因目标 node 被拆/易主而清除(战斗照常,驻守不影响互砍)。

迫击炮子顺序(v1.4,均在阶段 7 combat 内):先递减在途弹并结算落地爆炸,
再判定开火放新弹——弹恰飞 mortar_flight_time 拍,落地伤害与同 tick 近战
同帧并入 incoming(同时结算语义不破);炮死弹消(cleanup 停泊 shell_timer)。

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
from .gate import gate_tick, monster_combat_tick
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
    """胜负判定(v1.8:异界之门必分胜负,不再有和局)。HQ 亡即淘汰(清场在
    cleanup_deaths);存活 1 家→该家胜;0 家(同帧齐灭,极罕见)→argmax 定序给 0
    (确定性非和局);≥2 家存活继续打(异界之门保证终将只剩 1 家)。防御硬帽:到
    episode_len(应因怪物无上限升级而不可达)仍 ≥2 家→残余总血打分兜底,绝不判和。"""
    st = state
    tick = st.tick + 1
    P = cfg.n_players
    hq_alive = jnp.stack([st.alive[hq_slot(p, cfg)] for p in range(P)])
    n_alive = jnp.sum(hq_alive.astype(jnp.int32))
    survivor = jnp.argmax(hq_alive).astype(jnp.int8)         # 唯一存活者;全灭则 0
    winner = jnp.where(n_alive <= 1, survivor, st.winner.astype(jnp.int8))
    # 防御硬帽兜底:残余总血最高者胜(正常局异界之门早分出胜负,不会走到这)
    hp_by_p = (st.hp * st.alive).reshape(P, cfg.e_max).sum(axis=1)   # [P]
    cap_winner = jnp.argmax(hp_by_p).astype(jnp.int8)
    winner = jnp.where((winner == -1) & (tick >= cfg.episode_len), cap_winner, winner)
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
        st = gate_tick(st, cfg, mapdata)          # 异界之门:门开出怪 + 怪物慢速 descent
        st = combat_tick(st, cfg, owner)
        st = monster_combat_tick(st, cfg, owner)  # 怪物近战(阵营隔离,晚正常近战一子阶段)
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
