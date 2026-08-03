rows=160 games=160 vsRandom=160 rr=0 failed_matchups=0 P=4
winner values: vsRandom=[0] rr=[]

===== vs random (指挥官@seat0) =====
cmd          n    wr  avglen  gate avgArmy  lmin  lmax
balanced    16  1.00    2253  0.12     8.9  1868  4183
boomer      16  1.00    4182  1.00     0.9  4182  4182
rusher      16  1.00    1076  0.00     5.8  1028  1125
turtle      16  1.00    4182  1.00     1.1  4182  4183
timing      16  1.00    4081  0.69     9.8  3734  4183
harasser    16  1.00    2916  0.19     6.2  2319  4182
airtech     16  1.00    4182  1.00     1.1  4182  4183
tempo       16  1.00    4177  1.00     1.0  4104  4182
counter     16  1.00    4182  1.00     2.0  4182  4183
chaos       16  1.00    3918  0.62     6.6  3294  4183
vs-random wr<0.90 (fail): NONE — 全 >=0.90

===== round-robin 相对强度(座位均衡后)=====
cmd        appear wins rr_wr avgWinLen winGate avgArmy
balanced        0    0   nan       nan     nan     nan
boomer          0    0   nan       nan     nan     nan
rusher          0    0   nan       nan     nan     nan
turtle          0    0   nan       nan     nan     nan
timing          0    0   nan       nan     nan     nan
harasser        0    0   nan       nan     nan     nan
airtech         0    0   nan       nan     nan     nan
tempo           0    0   nan       nan     nan     nan
counter         0    0   nan       nan     nan     nan
chaos           0    0   nan       nan     nan     nan

-- 座位出场分布(每指挥官在各座位的次数;均衡=各列接近)--
cmd        seat0 seat1 seat2 seat3
各座位总胜场(应接近均匀 → 座位无偏): {0: 0, 1: 0, 2: 0, 3: 0}

-- 难度排名(round-robin 胜率 强→弱)= 课程分层 --

rr_wr>0.85 (统治): NONE
rr_wr<0.10 (太弱→改/删): NONE
从不胜: NONE

===== 弱尾专项(turtle/timing/airtech)=====
turtle: 无 rr 数据
timing: 无 rr 数据
airtech: 无 rr 数据

===== 非退化 / 质量(全部对局)=====
length min=1028 median=4182 max=4183 mean=3515
秒杀 <100: 0
到 episode_len 硬帽 (=max观测 4183): 6
draws (winner<0): 0
Traceback (most recent call last):
  File "/home/xrl/intern/TEOW/explorations/agg_v21_balanced.py", line 173, in <module>
    main()
  File "/home/xrl/intern/TEOW/explorations/agg_v21_balanced.py", line 163, in main
    f"  rr={sum(r['gate'] for r in rr)/len(rr):.2f}")
            ~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~
ZeroDivisionError: division by zero
