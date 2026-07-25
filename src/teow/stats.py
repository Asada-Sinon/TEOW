"""实体属性查表集中处(v1.4):血量/攻击/护甲/射程/目标能力的唯一消费入口。

设计(plan D3):max-HP 不入 state,由「类型 × 有效等级」现算——引擎已维护
「未受伤实体 = 表值」不变量(出生查表、研发完成补差、建筑升级补差),再存一份
就是第二真源。本模块只 import config/state,无环。

有效等级:战斗单位 = upgrades[owner, line_of_type];建筑 = state.level;
采集单位 = 1(固定基线,行内各级同值,gather 结果与等级无关)。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    N_TYPES,
    TYPE_AIRSHIP,
    TYPE_ARCHER,
    TYPE_BARRACKS,
    TYPE_CAMP,
    TYPE_CATAPULT,
    TYPE_DOG,
    TYPE_DRAGON,
    TYPE_FENCE_IRON,
    TYPE_FENCE_STONE,
    TYPE_FENCE_WOOD,
    TYPE_FLAMER,
    TYPE_HEALER,
    TYPE_HEAVY,
    TYPE_HQ,
    TYPE_INFANTRY,
    TYPE_LANDMINE,
    TYPE_LASER,
    TYPE_LCAV,
    TYPE_MAGE,
    TYPE_MAGETOWER,
    TYPE_MINE,
    TYPE_MORTAR,
    TYPE_PUMP,
    TYPE_RAM,
    TYPE_STRONGMAN,
    TYPE_TOWER,
    TYPE_WAGON,
    TYPE_WORKER,
    Config,
)
from .state import WorldState


def _flat(v) -> tuple:
    """标量 → 8 位等级表(各级同值;0 位废位同填,gather 免特判)。"""
    return (v,) * 8


def hp_table(cfg: Config) -> jax.Array:
    """int32 [N_TYPES, 8]:类型 × 等级 → 满血。未定义类型行全 0。"""
    rows = {
        TYPE_HQ: _flat(cfg.hq_hp),
        TYPE_MINE: _flat(cfg.node_struct_hp),
        TYPE_PUMP: _flat(cfg.node_struct_hp),
        TYPE_WORKER: _flat(cfg.worker_hp),
        TYPE_INFANTRY: cfg.inf_hp_by_level,
        TYPE_CAMP: cfg.camp_hp_by_level,
        TYPE_BARRACKS: cfg.barracks_hp_by_level,
        TYPE_DOG: cfg.dog_hp_by_level,
        TYPE_TOWER: cfg.tower_hp_by_level,
        TYPE_STRONGMAN: _flat(cfg.strongman_hp),
        TYPE_WAGON: _flat(cfg.wagon_hp),
        TYPE_ARCHER: cfg.archer_hp_by_level,
        TYPE_LCAV: cfg.lcav_hp_by_level,
        TYPE_HEAVY: cfg.heavy_hp_by_level,
        TYPE_MAGE: cfg.mage_hp_by_level,
        TYPE_HEALER: cfg.healer_hp_by_level,
        TYPE_RAM: cfg.ram_hp_by_level,
        TYPE_MORTAR: _flat(cfg.mortar_hp),
        TYPE_FENCE_WOOD: _flat(cfg.fence_wood_hp),
        TYPE_FENCE_STONE: _flat(cfg.fence_stone_hp),
        TYPE_FENCE_IRON: _flat(cfg.fence_iron_hp),
        TYPE_MAGETOWER: _flat(cfg.magetower_hp),
        TYPE_LANDMINE: _flat(cfg.landmine_hp),
        TYPE_FLAMER: _flat(cfg.flamer_hp),
        TYPE_LASER: _flat(cfg.laser_hp),
        TYPE_CATAPULT: _flat(cfg.catapult_hp),
        TYPE_AIRSHIP: _flat(cfg.airship_hp),
        TYPE_DRAGON: _flat(cfg.dragon_hp),
    }
    return jnp.asarray([rows.get(t, (0,) * 8) for t in range(N_TYPES)], jnp.int32)


def atk_table(cfg: Config) -> jax.Array:
    """int32 [N_TYPES, 8]:类型 × 等级 → 攻击力(奶妈行 0,治疗走 heal 表)。"""
    rows = {
        TYPE_INFANTRY: cfg.inf_atk_by_level,
        TYPE_DOG: cfg.dog_atk_by_level,
        TYPE_TOWER: cfg.tower_atk_by_level,
        TYPE_ARCHER: cfg.archer_atk_by_level,
        TYPE_LCAV: cfg.lcav_atk_by_level,
        TYPE_HEAVY: cfg.heavy_atk_by_level,
        TYPE_MAGE: cfg.mage_atk_by_level,
        TYPE_RAM: cfg.ram_atk_by_level,
        TYPE_MORTAR: _flat(cfg.mortar_atk),
        TYPE_FLAMER: _flat(cfg.flamer_atk),      # 自心圆取值;rng=0 不进单体路径
        TYPE_MAGETOWER: _flat(cfg.magetower_atk),
        TYPE_LANDMINE: _flat(cfg.landmine_atk),   # 爆炸伤害取值;rng=0 不当攻击者
        TYPE_LASER: _flat(cfg.laser_atk),
        TYPE_CATAPULT: _flat(cfg.catapult_atk),
        TYPE_DRAGON: _flat(cfg.dragon_air_atk),   # 单体路径=对空;喷火走自心圆表
    }
    return jnp.asarray([rows.get(t, (0,) * 8) for t in range(N_TYPES)], jnp.int32)


def effective_level(state: WorldState, cfg: Config, owner: jax.Array) -> jax.Array:
    """int32 [N]:属性查表用的有效等级(战斗单位=线级,建筑=自身级,采集=1)。"""
    et = state.etype.astype(jnp.int32)
    line = jnp.asarray(cfg.line_of_type, jnp.int32)[jnp.clip(et, 0, N_TYPES - 1)]
    line_lv = state.upgrades[owner.astype(jnp.int32),
                             jnp.clip(line, 0, state.upgrades.shape[1] - 1)]
    lv = jnp.where(line >= 0, line_lv.astype(jnp.int32),
                   state.level.astype(jnp.int32))
    return jnp.clip(lv, 0, 7)


def max_hp_of(state: WorldState, cfg: Config, owner: jax.Array) -> jax.Array:
    """int32 [N]:各实体当前满血上限(在建建筑也按建成满血算——治疗只作用单位,
    消费方自行门控)。"""
    et = jnp.clip(state.etype.astype(jnp.int32), 0, N_TYPES - 1)
    return hp_table(cfg)[et, effective_level(state, cfg, owner)]


def atk_of(state: WorldState, cfg: Config, owner: jax.Array) -> jax.Array:
    """int32 [N]:各实体当前攻击力(原始值,未过护甲)。"""
    et = jnp.clip(state.etype.astype(jnp.int32), 0, N_TYPES - 1)
    return atk_table(cfg)[et, effective_level(state, cfg, owner)]


def physical_damage(atk: jax.Array, armor: jax.Array) -> jax.Array:
    """护甲公式唯一定义处(用户定案:百分比减伤,向上取整,至少 1;
    仅对 atk>0 有意义,调用方门控)。纯整数,无浮点无 RNG。"""
    return jnp.maximum(1, (atk * (100 - armor) + 99) // 100)


def type_tables(cfg: Config) -> dict:
    """combat/movement 共用的 per-type jnp 表包(避免两处各自 asarray 漂移)。"""
    return dict(
        speed=jnp.asarray(cfg.speed_by_type, jnp.float32),
        rng=jnp.asarray(cfg.atk_range_by_type, jnp.float32),
        min_rng=jnp.asarray(cfg.atk_min_range_by_type, jnp.float32),
        hit_u=jnp.asarray(cfg.can_hit_units_by_type, bool),
        hit_b=jnp.asarray(cfg.can_hit_buildings_by_type, bool),
        armor=jnp.asarray(cfg.armor_by_type, jnp.int32),
        magic=jnp.asarray(cfg.dmg_magic_by_type, bool),
        period=jnp.asarray(cfg.atk_period_by_type, jnp.int32),
        aoe=jnp.asarray(cfg.aoe_radius_by_type, jnp.float32),
        line=jnp.asarray(cfg.line_of_type, jnp.int32),
        air=jnp.asarray(cfg.is_air_by_type, bool),
        hit_air=jnp.asarray(cfg.can_hit_air_by_type, bool),
        combat=jnp.asarray(cfg.is_combat_by_type, bool),
        flight=jnp.asarray(cfg.shell_flight_by_type, jnp.int32),
        self_aoe=jnp.asarray(cfg.self_aoe_radius_by_type, jnp.float32),
    )


def can_target(tt: dict, et_att: jax.Array, tgt_is_air: jax.Array,
               tgt_is_unit: jax.Array) -> jax.Array:
    """bool [N,N] 目标合法矩阵(行=攻击者 i,列=目标 j)。唯一公式
    (v1.6 plan D1;combat 选靶与 movement 停步共用,防两处手写漂移):
    空中目标看 can_hit_air;地面目标按 单位/建筑 查表。"""
    return jnp.where(tgt_is_air[None, :], tt["hit_air"][et_att][:, None],
                     jnp.where(tgt_is_unit[None, :],
                               tt["hit_u"][et_att][:, None],
                               tt["hit_b"][et_att][:, None]))


def etype_idx(state: WorldState) -> jax.Array:
    """int32 [N]:etype 安全下标(clip 到表长;新类型进表自动生效)。"""
    return jnp.clip(state.etype.astype(jnp.int32), 0, N_TYPES - 1)
