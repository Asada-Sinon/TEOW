# v1.7 对决矩阵汇总

成本归一:count=budget/(ore+w·water);胜负=对方全灭且己方有余,超时按余血占比。
「疑似超模」= 全灭对方且余兵≥50%,或余血占比差>0.7。

| 对决 | A×n | B×n | 结果 | tick | A余/B余 | A血比/B血比 | 超模 |
|---|---|---|---|---|---|---|---|
| melee_inf_vs_dog | infantry6 | dog10 | A_win | 81 | 4/0 | 0.429/0.2 | A≫B? |
| melee_heavy_vs_lcav | heavy3 | lcav4 | B_win | 68 | 0/2 | 0.133/0.189 | B≫A? |
| melee_heavy_vs_inf | heavy3 | infantry6 | A_win | 41 | 2/0 | 0.411/0.0 | A≫B? |
| melee_lcav_vs_dog | lcav4 | dog10 | A_win | 19 | 2/0 | 0.183/0.0 | A≫B? |
| melee_inf_vs_lcav | infantry6 | lcav4 | B_win | 61 | 0/2 | 0.325/0.256 | B≫A? |
| ranged_archer_vs_inf | archer5 | infantry6 | A_win | 14 | 5/0 | 0.813/0.0 | A≫B? |
| ranged_archer_vs_dog | archer5 | dog10 | A_win | 12 | 5/0 | 0.76/0.0 | A≫B? |
| ranged_mage_vs_heavy | mage3 | heavy3 | A_win | 16 | 3/0 | 1.0/0.0 | A≫B? |
| ranged_mage_vs_inf | mage3 | infantry6 | A_win | 16 | 3/0 | 0.68/0.0 | A≫B? |
| siege_ram_vs_tower | ram2 | tower1 | A_win | 16 | 2/0 | 0.867/0.0 | A≫B? |
| siege_ram_vs_barracks | ram2 | barracks1 | A_win | 18 | 2/0 | 1.0/0.0 | A≫B? |
| siege_ram_vs_hq | ram2 | hq1 | A_win | 28 | 2/0 | 1.0/0.0 | A≫B? |
| def_tower_vs_dogwave | tower1 | dog10 | B_win | 49 | 0/7 | 0.225/0.688 | B≫A? |
| def_mortar_vs_infwave | mortar1 | infantry6 | B_win | 30 | 0/6 | 0.0/0.8 | B≫A? |
| def_magetower_vs_infwave | magetower1 | infantry6 | B_win | 29 | 0/5 | 0.0/0.658 | B≫A? |
| def_flamer_vs_dogwave | flamer1 | dog10 | B_win | 68 | 0/3 | 0.044/0.263 |  |
| def_laser_vs_heavy | laser1 | heavy1 | A_win | 12 | 1/0 | 0.893/0.0 | A≫B? |
| dragon_vs_airship | dragon2 | airship3 | A_win | 363 | 2/0 | 1.0/0.61 | A≫B? |
| dragon_vs_infwave | dragon1 | infantry6 | B_win | 61 | 0/6 | 1.0/0.542 | B≫A? |
| dragon_vs_tower | dragon1 | tower1 | B_win | 74 | 0/1 | 0.0/0.667 | B≫A? |
| dragon_vs_barracks | dragon1 | barracks1 | A_win | 332 | 1/0 | 1.0/0.0 | A≫B? |
| support_infhealer_vs_inf | infantry6,healer3 | infantry6 | A_win | 36 | 9/0 | 0.8/0.0 | A≫B? |
