# v1.7 迫击炮/法师塔补强扫参

标准:1×造价该 A_win(建筑守住)、2×造价该 B_win(被攻破)。

| 建筑 | 候选 | override | 1× | 1×塔血 | 2× | 2×波余 | 达标 |
|---|---|---|---|---|---|---|---|
| mortar | base | 现值 | B_win | 0.0 | B_win | 7 |  |
| mortar | period25 | mortar_atk_period=25 | B_win | 0.0 | B_win | 7 |  |
| mortar | period15 | mortar_atk_period=15 | B_win | 0.0 | B_win | 7 |  |
| mortar | minr1.0 | mortar_min_range=1.0 | B_win | 0.0 | B_win | 7 |  |
| mortar | minr1.0_period25 | mortar_min_range=1.0, mortar_atk_period=25 | B_win | 0.0 | B_win | 7 |  |
| mortar | hp250_period25 | mortar_hp=250, mortar_atk_period=25 | B_win | 0.0 | B_win | 7 |  |
| mortar | atk50_period20 | mortar_atk=50, mortar_atk_period=20 | B_win | 0.0 | B_win | 7 |  |
| magetower | base | 现值 | B_win | 0.0 | B_win | 6 |  |
| magetower | atk18 | magetower_atk=18 | B_win | 0.0 | B_win | 6 |  |
| magetower | atk20 | magetower_atk=20 | A_win | 0.2 | B_win | 3 | ✓ |
| magetower | atk24 | magetower_atk=24 | A_win | 0.2 | B_win | 3 | ✓ |
| magetower | atk28 | magetower_atk=28 | A_win | 0.2 | B_win | 3 | ✓ |
| dragon | base | 现值 | A_ahead | 1.0 | - | - |  |
| dragon | r3.5 | dragon_breath_radius=3.5 | A_win | 1.0 | - | - | ✓ |
| dragon | r4.0 | dragon_breath_radius=4.0 | A_win | 1.0 | - | - | ✓ |
| dragon | r4.5 | dragon_breath_radius=4.5 | A_win | 1.0 | - | - | ✓ |
| dragon | r5.0 | dragon_breath_radius=5.0 | A_win | 1.0 | - | - | ✓ |
