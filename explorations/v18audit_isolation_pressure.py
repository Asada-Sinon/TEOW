"""v1.8 审计:全 4 玩家的阵营隔离(双向)+ 各阵营压力一致(直接调子系统)。"""
import jax, jax.numpy as jnp
from teow.config import Config, TYPE_INFANTRY, TYPE_TOWER
from teow.gate import gate_tick, monster_combat_tick
from teow.map import build_map
from teow.state import hq_slot, init_state, owner_of_slots

cfg = Config()
m = build_map(cfg); owner = owner_of_slots(cfg)
st = init_state(cfg, m)

# 四家各放一个步兵在同一点 (30,30);再只给玩家 2 放一只怪
POS = (30.0, 30.0)
inf_hp = int(cfg.inf_hp_by_level[1]); inf_atk = int(cfg.inf_atk_by_level[1])
units = {}
for p in range(4):
    s = hq_slot(p, cfg) + 7
    units[p] = s
    st = st._replace(alive=st.alive.at[s].set(True),
                     etype=st.etype.at[s].set(TYPE_INFANTRY),
                     hp=st.hp.at[s].set(inf_hp),
                     pos=st.pos.at[s].set(jnp.asarray(POS, jnp.float32)))
# 只给玩家2一只怪(100血,atk 10)
st = st._replace(
    monster_alive=st.monster_alive.at[2, 0].set(True),
    monster_hp=st.monster_hp.at[2, 0].set(100),
    monster_atk=st.monster_atk.at[2, 0].set(10),
    monster_pos=st.monster_pos.at[2, 0].set(jnp.asarray(POS, jnp.float32)))
out = monster_combat_tick(st, cfg, owner)
dmg = {p: inf_hp - int(out.hp[units[p]]) for p in range(4)}
print("各玩家步兵受怪伤:", dmg, " → 仅玩家2应>0")
print("玩家2怪受伤:", 100 - int(out.monster_hp[2,0]), " → 应=玩家2步兵atk =", inf_atk,
      "(若跨阵营也打会=4*atk)")
iso_ok = (dmg[0]==0 and dmg[1]==0 and dmg[3]==0 and dmg[2]>0
          and (100-int(out.monster_hp[2,0]))==inf_atk)
print("阵营隔离(双向)PASS:", iso_ok)

# 各阵营压力一致:门开首波,四家全存活 → 四行 spawn 数/hp/atk 完全相同
cfg2 = Config(gate_open_tick=0, monster_spawn_interval=1, monster_wave_count=3,
              monster_hp_base=50, monster_hp_slope=0.7,
              monster_atk_base=5, monster_atk_slope=0.01)
m2 = build_map(cfg2); base = init_state(cfg2, m2)
g = gate_tick(base._replace(tick=jnp.asarray(120, jnp.int32)), cfg2, m2)
cnts = [int(g.monster_alive[p].sum()) for p in range(4)]
hps = [sorted(int(x) for x in g.monster_hp[p][g.monster_alive[p]]) for p in range(4)]
atks = [sorted(int(x) for x in g.monster_atk[p][g.monster_alive[p]]) for p in range(4)]
print("四家首波怪数:", cnts, " 全相等:", len(set(cnts))==1)
print("四家怪HP集合全相同:", all(h==hps[0] for h in hps), hps[0])
print("四家怪ATK集合全相同:", all(a==atks[0] for a in atks), atks[0])
# 中心生成核验:怪应在 (h/2,w/2) 附近
ctr = jnp.asarray([cfg2.grid_h/2, cfg2.grid_w/2])
maxd = float(jnp.max(jnp.linalg.norm(g.monster_pos[g.monster_alive] - ctr, axis=-1)))
print("怪距地图中心最大距离:", round(maxd,3), " (应≈生成偏移<1.5)")
