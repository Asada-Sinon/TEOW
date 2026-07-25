# TEOW 贴图生成提示词包(PNG 替换槽)

前端默认用 `web/sprites.js` 的矢量图案;想换 AI 生成的精美贴图时,把生成的 PNG
放进 `web/assets/`,**文件名对上就自动生效**(热替换,不用改代码):

| 文件名 | 实体 | 建议画面 |
|---|---|---|
| `hq.png` / `hq_p0.png` / `hq_p1.png` | 大本营 | 石制堡垒/指挥部 |
| `mine.png` | 矿 | 矿井口+矿石堆+镐 |
| `pump.png` | 水泵 | 抽水站+水滴 |
| `worker.png` | 工人 | 戴安全帽的小人 |
| `infantry.png` | 步兵 | 持盾矛的士兵 |
| `camp.png` | 技能训练营 | 训练帐篷+旗帜 |
| `barracks.png` | 兵营 | 营房+交叉武器 |
| `dog.png` | 狗子 | 冲锋的战犬 |
| `tower.png` | 哨塔 | 瞭望塔/炮塔 |

带 `_p0`/`_p1` 后缀 = 阵营专属版(蓝/红);不带后缀 = 双方共用同一张
(注意:用共用贴图时阵营只能靠血条/位置区分,建议生成两套或留描边区域)。

## 通用规格(每张都要遵守,保证一套图风格一致)

- **尺寸**:256×256,PNG,**透明背景**(必须,画布会直接叠加)
- **视角**:俯视 30°(oblique top-down,类星际/帝国时代单位视角)
- **构图**:主体居中,占画面 80%,底部轻微投影,无边框无文字
- **风格**:统一写实卡通(stylized painterly),描边清晰,小尺寸(24-48px)缩放后轮廓仍可辨识

## 提示词模板(把 <主体描述> 换掉;蓝方把 blue 换 red 即红方)

```
stylized RTS game sprite, oblique top-down view, <主体描述>,
blue faction color accents, painterly with clean silhouette,
centered, isolated on transparent background, no text, no border,
soft ambient occlusion shadow at base, 256x256
```

各实体的 <主体描述>:

- hq: `a sturdy stone command fortress with battlements and a banner`
- mine: `a mineshaft entrance with ore rocks and a pickaxe`
- pump: `a small industrial water pump station with a pipe and water drop`
- worker: `a small worker character wearing a yellow hard hat carrying a pouch`
- infantry: `a foot soldier with a kite shield and spear`
- camp: `a military training tent with a flag and target dummy`
- barracks: `a wooden barracks building with crossed swords sign`
- dog: `a lean armored war hound in mid-sprint`
- tower: `a tapered stone watchtower with crenellations and an arrow slit`

## 验收清单(生成后自查)

1. 透明背景(放在深色页面上无白边)
2. 九张风格一致(同一次会话/同一风格参数生成)
3. 缩到 32×32 仍能认出是什么
4. 命名精确匹配上表(小写,含下划线)
