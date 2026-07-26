# v1.7 对决矩阵汇总

成本归一:count=budget/(ore+w·water);胜负=对方全灭且己方有余,超时按余血占比。
「疑似超模」= 全灭对方且余兵≥50%,或余血占比差>0.7。
模式:brawl=原地接战(单位vs单位) / assault=攻防局(建筑防御)。

| 对决 | 模式 | A×n | B×n | 投入A/B | 结果 | tick | A余/B余 | A血比/B血比 | 超模 |
|---|---|---|---|---|---|---|---|---|---|
| melee_inf_vs_dog | brawl | infantry6 | dog10 | 240/250 | even | 800 | 5/3 | 0.446/0.3 |  |
| melee_heavy_vs_lcav | brawl | heavy3 | lcav4 | 225/260 | even | 800 | 1/2 | 0.233/0.278 |  |
| melee_heavy_vs_inf | brawl | heavy3 | infantry6 | 225/240 | A_ahead | 800 | 3/1 | 0.378/0.167 |  |
| melee_lcav_vs_dog | brawl | lcav4 | dog10 | 260/250 | even | 800 | 3/6 | 0.6/0.575 |  |
| melee_inf_vs_lcav | brawl | infantry6 | lcav4 | 240/260 | even | 800 | 2/2 | 0.333/0.3 |  |
| ranged_archer_vs_inf | brawl | archer5 | infantry6 | 225/240 | B_win | 11 | 0/2 | 0.0/0.271 |  |
| ranged_archer_vs_dog | brawl | archer5 | dog10 | 225/250 | A_win | 22 | 2/0 | 0.26/0.0 |  |
| ranged_mage_vs_heavy | brawl | mage3 | heavy3 | 240/225 | B_win | 8 | 0/3 | 0.0/0.417 | B≫A? |
| ranged_mage_vs_inf | brawl | mage3 | infantry6 | 240/240 | B_win | 5 | 0/6 | 0.0/0.621 | B≫A? |
| siege_ram_vs_tower | assault | ram1 | tower1 | 120/80 | A_win | 21 | 1/0 | 0.75/0.0 | A≫B? |
| siege_ram_vs_barracks | assault | ram1 | barracks1 | 120/120 | A_win | 23 | 1/0 | 1.0/0.0 | A≫B? |
| siege_ram_vs_hq | assault | ram2 | hq1 | 240/400 | A_win | 31 | 2/0 | 1.0/0.0 | A≫B? |
| def_tower_vs_dog_1x | assault | tower1 | dog3 | 80/75 | A_win | 25 | 1/0 | 0.55/0.0 | A≫B? |
| def_tower_vs_dog_2x | assault | tower1 | dog6 | 80/150 | B_win | 24 | 0/4 | 0.0/0.521 | B≫A? |
| def_mortar_vs_inf_1x | assault | mortar1 | infantry4 | 140/160 | B_win | 21 | 0/4 | 0.0/0.944 | B≫A? |
| def_mortar_vs_inf_2x | assault | mortar1 | infantry7 | 140/280 | B_win | 21 | 0/7 | 0.0/0.968 | B≫A? |
| def_magetower_vs_inf_1x | assault | magetower1 | infantry3 | 120/120 | B_win | 25 | 0/2 | 0.0/0.433 | B≫A? |
| def_magetower_vs_inf_2x | assault | magetower1 | infantry6 | 120/240 | B_win | 20 | 0/6 | 0.0/0.767 | B≫A? |
| def_flamer_vs_dog_1x | assault | flamer1 | dog5 | 130/125 | A_win | 25 | 1/0 | 0.475/0.0 | A≫B? |
| def_flamer_vs_dog_2x | assault | flamer1 | dog10 | 130/250 | B_win | 25 | 0/3 | 0.0/0.138 |  |
| def_laser_vs_heavy_1x | assault | laser1 | heavy3 | 210/225 | A_win | 36 | 1/0 | 0.6/0.0 | A≫B? |
| def_laser_vs_heavy_2x | assault | laser1 | heavy6 | 210/450 | B_win | 34 | 0/4 | 0.0/0.528 | B≫A? |
| dragon_vs_airship | brawl | dragon2 | airship3 | 640/600 | A_win | 51 | 2/0 | 1.0/0.0 | A≫B? |
| dragon_vs_infwave | brawl | dragon1 | infantry8 | 320/320 | A_ahead | 800 | 1/1 | 1.0/0.125 | A>B? |
| dragon_vs_tower | assault | dragon1 | tower1 | 320/80 | B_win | 76 | 0/1 | 0.0/0.667 | B≫A? |
| dragon_vs_barracks | assault | dragon1 | barracks1 | 320/120 | A_win | 335 | 1/0 | 1.0/0.0 | A≫B? |
| support_infhealer_vs_inf | brawl | infantry3,healer2 | infantry6 | 260/240 | B_win | 14 | 0/4 | 0.0/0.5 | B≫A? |
