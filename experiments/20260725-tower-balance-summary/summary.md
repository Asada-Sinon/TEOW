# v1.3 哨塔平衡扫描汇总(Phase 6)

变体 run 目录:20260725-tower-balance-base, 20260725-tower-balance-atk4, 20260725-tower-balance-atk3, 20260725-tower-balance-cost80-50, 20260725-tower-balance-hp90

场景 A = 手术局(HQ+L1塔+3工人 vs N 狗,无菌);场景 B = scripted vs scripted × seeds(胜负/终局 tick;tower_seen 为 250-tick 块边界抽样)。cost 变体在场景 A 无效(塔手术放置不扣费),其 A 行为 base 对照。

| 变体 | 改动 | 场景 A(逐 N) | 场景 B 胜负 | B 终局中位 tick | B 出现过塔的局 |
|---|---|---|---|---|---|
| base | 现值 | N=2:dogs_wiped@10(狗余0,工亡1,塔血120); N=3:dogs_wiped@13(狗余0,工亡1,塔血120); N=4:dogs_wiped@17(狗余0,工亡1,塔血120); N=5:dogs_wiped@21(狗余0,工亡3,塔血105) | p0胜8/p1胜0/和0 | 1519 | 8/8 |
| atk4 | tower_atk_by_level=(0, 4, 8, 10, 13, 16, 20, 24) | N=2:dogs_wiped@12(狗余0,工亡1,塔血120); N=3:dogs_wiped@16(狗余0,工亡2,塔血120); N=4:dogs_wiped@22(狗余0,工亡3,塔血120); N=5:dogs_wiped@29(狗余0,工亡3,塔血117) | p0胜8/p1胜0/和0 | 1498 | 8/8 |
| atk3 | tower_atk_by_level=(0, 3, 8, 10, 13, 16, 20, 24) | N=2:dogs_wiped@14(狗余0,工亡1,塔血120); N=3:dogs_wiped@20(狗余0,工亡3,塔血120); N=4:dogs_wiped@28(狗余0,工亡3,塔血120); N=5:dogs_wiped@36(狗余0,工亡3,塔血111) | p0胜8/p1胜0/和0 | 1584 | 8/8 |
| cost80-50 | tower_cost_ore=80, tower_cost_water=50 | N=2:dogs_wiped@10(狗余0,工亡1,塔血120); N=3:dogs_wiped@13(狗余0,工亡1,塔血120); N=4:dogs_wiped@17(狗余0,工亡1,塔血120); N=5:dogs_wiped@21(狗余0,工亡3,塔血105) | p0胜8/p1胜0/和0 | 1147 | 8/8 |
| hp90 | tower_hp_by_level=(0, 90, 150, 180, 220, 260, 300, 350) | N=2:dogs_wiped@10(狗余0,工亡1,塔血90); N=3:dogs_wiped@13(狗余0,工亡1,塔血90); N=4:dogs_wiped@17(狗余0,工亡1,塔血90); N=5:dogs_wiped@21(狗余0,工亡3,塔血75) | p0胜8/p1胜0/和0 | 1519 | 8/8 |
