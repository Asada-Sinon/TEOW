// TEOW 矢量图案(v1.2):每种实体一个可辨识轮廓,阵营色参数化。
// PNG 替换槽:web/assets/<name>.png 存在则优先绘制贴图(AI 生图可热替换,
// 提示词包见 docs/sprite-prompts.md)。命名:hq/mine/pump/worker/infantry/
// camp/barracks/dog/tower + "_p0"/"_p1"(不带后缀则双方共用,按阵营染色失效)。

const TYPE_NAMES = {1: "hq", 2: "mine", 3: "pump", 4: "worker", 5: "infantry",
                    6: "camp", 7: "barracks", 8: "dog", 9: "tower",
                    // v1.4 兵种树(PNG 槽命名与 fig/ 中文贴图映射:
                    // strongman=大力士工人 wagon=运输马车 archer=弓箭手
                    // cavalry=骑兵 heavy=重装刀斧手 mage=法师 healer=奶妈神官
                    // ram=攻城车(无贴图) mortar=迫击炮;infantry=普通刀斧手)
                    10: "strongman", 11: "wagon", 12: "archer", 13: "cavalry",
                    14: "heavy", 15: "mage", 16: "healer", 17: "ram",
                    18: "mortar",
                    // v1.5 栅栏三档(无贴图,矢量)
                    19: "fence_wood", 20: "fence_stone", 21: "fence_iron",
                    // v1.6 防御建筑群+空军(矢量;PNG 槽同名可热替换)
                    22: "magetower", 23: "landmine", 24: "flamer",
                    25: "laser", 26: "catapult", 27: "airship", 28: "dragon"};
const P_COLOR = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"];  // 蓝/红/绿/琥珀
const P_DARK  = ["#1d4ed8", "#b91c1c", "#15803d", "#b45309"];  // (v1.5 四人)

const _png = {};   // name -> Image|null(null=已探测不存在)
function pngFor(name, owner) {
  for (const key of [`${name}_p${owner}`, name]) {
    if (!(key in _png)) {
      const img = new Image();
      img.src = `assets/${key}.png`;
      img.onload = () => { _png[key] = img; };
      img.onerror = () => { _png[key] = null; };
      _png[key] = undefined;      // 探测中
    }
    if (_png[key] instanceof Image && _png[key].complete && _png[key].naturalWidth)
      return _png[key];
  }
  return null;
}

// s = 一格的像素边长;(x,y) 是实体中心像素坐标
export function drawSprite(ctx, type, owner, x, y, s, opts = {}) {
  const name = TYPE_NAMES[type] || "unknown";
  const img = pngFor(name, owner);
  const alpha = opts.building ? 0.45 : (opts.inside ? 0.4 : 1.0);
  ctx.save();
  ctx.globalAlpha = alpha;
  if (img) {                       // PNG 替换槽命中
    const isBld = opts.bld !== undefined ? opts.bld
                                         : (type <= 3 || (type >= 6 && type <= 9));
    const d = s * (isBld ? 1.0 : 0.7);
    ctx.drawImage(img, x - d / 2, y - d / 2, d, d);
    ctx.restore();
    return;
  }
  const C = P_COLOR[owner], D = P_DARK[owner];
  ctx.lineWidth = Math.max(1, s * 0.06);
  ctx.strokeStyle = D;
  ctx.fillStyle = C;
  const u = s / 24;                // 24 单位设计网格
  ctx.translate(x, y);
  switch (name) {
    case "hq": {                   // 堡垒:主体+三垛口+门
      ctx.fillRect(-9 * u, -6 * u, 18 * u, 13 * u);
      for (const dx of [-9, -3, 3]) ctx.fillRect(dx * u, -9 * u, 6 * u * 0.7, 4 * u);
      ctx.strokeRect(-9 * u, -6 * u, 18 * u, 13 * u);
      ctx.fillStyle = D;
      ctx.fillRect(-2.5 * u, 0, 5 * u, 7 * u);   // 门
      break;
    }
    case "mine": {                 // 矿:岩堆+镐柄
      ctx.beginPath();
      ctx.moveTo(-8 * u, 7 * u); ctx.lineTo(-3 * u, -4 * u); ctx.lineTo(1 * u, 2 * u);
      ctx.lineTo(5 * u, -6 * u); ctx.lineTo(9 * u, 7 * u); ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#8b5e34"; ctx.lineWidth = 2 * u;
      ctx.beginPath(); ctx.moveTo(2 * u, -8 * u); ctx.lineTo(8 * u, -2 * u); ctx.stroke();
      break;
    }
    case "pump": {                 // 泵:水滴+立管
      ctx.fillRect(-1.5 * u, -8 * u, 3 * u, 8 * u);
      ctx.beginPath();
      ctx.moveTo(0, -2 * u);
      ctx.bezierCurveTo(7 * u, 3 * u, 5 * u, 9 * u, 0, 9 * u);
      ctx.bezierCurveTo(-5 * u, 9 * u, -7 * u, 3 * u, 0, -2 * u);
      ctx.fill(); ctx.stroke();
      break;
    }
    case "worker": {               // 工人:圆脸+安全帽
      ctx.beginPath(); ctx.arc(0, 1 * u, 6 * u, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#facc15";
      ctx.beginPath(); ctx.arc(0, -1 * u, 6.2 * u, Math.PI, 2 * Math.PI); ctx.fill();
      ctx.fillRect(-7 * u, -1.5 * u, 14 * u, 1.8 * u);
      break;
    }
    case "infantry": {             // 步兵:盾+矛
      ctx.beginPath();
      ctx.moveTo(-5 * u, -6 * u); ctx.lineTo(5 * u, -6 * u);
      ctx.lineTo(5 * u, 2 * u); ctx.lineTo(0, 8 * u); ctx.lineTo(-5 * u, 2 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#d1d5db"; ctx.lineWidth = 1.6 * u;
      ctx.beginPath(); ctx.moveTo(6 * u, 8 * u); ctx.lineTo(6 * u, -8 * u); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(6 * u, -8 * u); ctx.lineTo(4.5 * u, -5 * u);
      ctx.lineTo(7.5 * u, -5 * u); ctx.closePath();
      ctx.fillStyle = "#d1d5db"; ctx.fill();
      break;
    }
    case "dog": {                  // 狗:体+头+四腿+尾
      ctx.beginPath(); ctx.ellipse(-1 * u, 0, 6 * u, 3.5 * u, 0, 0, 7);
      ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(6 * u, -2 * u, 3 * u, 0, 7); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(7.5 * u, -5 * u); ctx.lineTo(9 * u, -7 * u);
      ctx.lineTo(9.5 * u, -4.5 * u); ctx.closePath(); ctx.fill();   // 耳
      ctx.lineWidth = 1.6 * u; ctx.strokeStyle = D;
      for (const dx of [-5, -2, 1, 4])
        { ctx.beginPath(); ctx.moveTo(dx * u, 3 * u); ctx.lineTo(dx * u, 6.5 * u); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(-7 * u, -1 * u); ctx.lineTo(-10 * u, -4 * u); ctx.stroke();
      break;
    }
    case "camp": {                 // 训练营:帐篷+旗
      ctx.beginPath();
      ctx.moveTo(0, -9 * u); ctx.lineTo(9 * u, 7 * u); ctx.lineTo(-9 * u, 7 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = D;
      ctx.beginPath(); ctx.moveTo(0, -1 * u); ctx.lineTo(3.5 * u, 7 * u);
      ctx.lineTo(-3.5 * u, 7 * u); ctx.closePath(); ctx.fill();     // 门
      ctx.strokeStyle = D; ctx.lineWidth = 1.4 * u;
      ctx.beginPath(); ctx.moveTo(0, -9 * u); ctx.lineTo(0, -13 * u); ctx.stroke();
      ctx.fillStyle = "#facc15"; ctx.fillRect(0, -13 * u, 5 * u, 3 * u);
      break;
    }
    case "barracks": {             // 兵营:营房+人字顶+交叉剑
      ctx.fillRect(-9 * u, -3 * u, 18 * u, 10 * u);
      ctx.strokeRect(-9 * u, -3 * u, 18 * u, 10 * u);
      ctx.beginPath(); ctx.moveTo(-10 * u, -3 * u); ctx.lineTo(0, -10 * u);
      ctx.lineTo(10 * u, -3 * u); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#d1d5db"; ctx.lineWidth = 1.6 * u;
      ctx.beginPath(); ctx.moveTo(-4 * u, 6 * u); ctx.lineTo(4 * u, -1 * u); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(4 * u, 6 * u); ctx.lineTo(-4 * u, -1 * u); ctx.stroke();
      break;
    }
    case "tower": {                // 哨塔:锥台+垛口+瞭望窗
      ctx.beginPath();
      ctx.moveTo(-5 * u, -8 * u); ctx.lineTo(5 * u, -8 * u);
      ctx.lineTo(8 * u, 9 * u); ctx.lineTo(-8 * u, 9 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      for (const dx of [-6, -1.5, 3]) ctx.fillRect(dx * u, -11 * u, 3 * u, 3.2 * u);
      ctx.fillStyle = "#0f172a";
      ctx.beginPath(); ctx.arc(0, -3 * u, 2.2 * u, 0, 7); ctx.fill();
      break;
    }
    case "strongman": {            // 大力士:圆脸+杠铃横杆
      ctx.beginPath(); ctx.arc(0, 1 * u, 6.5 * u, 0, 7); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 2 * u;
      ctx.beginPath(); ctx.moveTo(-9 * u, -6 * u); ctx.lineTo(9 * u, -6 * u); ctx.stroke();
      ctx.fillStyle = "#374151";
      ctx.fillRect(-11 * u, -8.5 * u, 3 * u, 5 * u);
      ctx.fillRect(8 * u, -8.5 * u, 3 * u, 5 * u);
      break;
    }
    case "wagon": {                // 马车:车斗+双轮
      ctx.fillRect(-8 * u, -6 * u, 16 * u, 8 * u);
      ctx.strokeRect(-8 * u, -6 * u, 16 * u, 8 * u);
      ctx.fillStyle = "#374151";
      ctx.beginPath(); ctx.arc(-4.5 * u, 5 * u, 3.2 * u, 0, 7); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(4.5 * u, 5 * u, 3.2 * u, 0, 7); ctx.fill(); ctx.stroke();
      break;
    }
    case "archer": {               // 弓箭手:弓弧+箭
      ctx.lineWidth = 1.8 * u;
      ctx.beginPath(); ctx.arc(0, 0, 7 * u, -Math.PI / 2.6, Math.PI / 2.6); ctx.stroke();
      ctx.strokeStyle = "#d1d5db";
      ctx.beginPath(); ctx.moveTo(-6 * u, 0); ctx.lineTo(7 * u, 0); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(7 * u, 0); ctx.lineTo(4 * u, -2 * u);
      ctx.moveTo(7 * u, 0); ctx.lineTo(4 * u, 2 * u); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(6.3 * u, -5.5 * u); ctx.lineTo(6.3 * u, 5.5 * u);
      ctx.strokeStyle = D; ctx.stroke();
      break;
    }
    case "cavalry": {              // 轻骑兵:马体+骑手
      ctx.beginPath(); ctx.ellipse(0, 2 * u, 8 * u, 4 * u, 0, 0, 7);
      ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(7.5 * u, -1 * u, 2.6 * u, 0, 7); ctx.fill(); ctx.stroke();
      ctx.lineWidth = 1.6 * u;
      for (const dx of [-5, -2, 2, 5])
        { ctx.beginPath(); ctx.moveTo(dx * u, 5.5 * u); ctx.lineTo(dx * u, 9 * u); ctx.stroke(); }
      ctx.beginPath();               // 骑手
      ctx.moveTo(-1 * u, -3 * u); ctx.lineTo(2 * u, -8 * u); ctx.lineTo(4 * u, -3 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      break;
    }
    case "heavy": {                // 重盔甲战士:全身大盾+铆钉
      ctx.beginPath();
      ctx.moveTo(-7 * u, -8 * u); ctx.lineTo(7 * u, -8 * u);
      ctx.lineTo(7 * u, 3 * u); ctx.lineTo(0, 9 * u); ctx.lineTo(-7 * u, 3 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = D;
      for (const [dx, dy] of [[-4, -5], [4, -5], [-4, 0], [4, 0], [0, 4]])
        { ctx.beginPath(); ctx.arc(dx * u, dy * u, 1.1 * u, 0, 7); ctx.fill(); }
      break;
    }
    case "mage": {                 // 法师:尖帽+法杖+杖头珠
      ctx.beginPath();
      ctx.moveTo(0, -10 * u); ctx.lineTo(6 * u, 0); ctx.lineTo(-6 * u, 0);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillRect(-5 * u, 0, 10 * u, 7 * u);
      ctx.strokeStyle = "#8b5e34"; ctx.lineWidth = 1.8 * u;
      ctx.beginPath(); ctx.moveTo(8 * u, 8 * u); ctx.lineTo(8 * u, -6 * u); ctx.stroke();
      ctx.fillStyle = "#a78bfa";
      ctx.beginPath(); ctx.arc(8 * u, -7.5 * u, 2 * u, 0, 7); ctx.fill();
      break;
    }
    case "healer": {               // 奶妈神官:圆袍+十字
      ctx.beginPath(); ctx.arc(0, 0, 7.5 * u, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(-1.5 * u, -5 * u, 3 * u, 10 * u);
      ctx.fillRect(-5 * u, -1.5 * u, 10 * u, 3 * u);
      break;
    }
    case "ram": {                  // 攻城车:棚车+前伸撞木
      ctx.fillRect(-8 * u, -5 * u, 13 * u, 8 * u);
      ctx.strokeRect(-8 * u, -5 * u, 13 * u, 8 * u);
      ctx.beginPath(); ctx.moveTo(-9 * u, -5 * u); ctx.lineTo(-1.5 * u, -9 * u);
      ctx.lineTo(6 * u, -5 * u); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#8b5e34"; ctx.lineWidth = 2.4 * u;
      ctx.beginPath(); ctx.moveTo(3 * u, 0); ctx.lineTo(11 * u, 0); ctx.stroke();
      ctx.fillStyle = "#374151";
      ctx.beginPath(); ctx.arc(-5 * u, 4.5 * u, 2.4 * u, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(1 * u, 4.5 * u, 2.4 * u, 0, 7); ctx.fill();
      break;
    }
    case "mortar": {               // 迫击炮:斜炮管+底座
      ctx.fillRect(-8 * u, 4 * u, 16 * u, 4 * u);
      ctx.strokeRect(-8 * u, 4 * u, 16 * u, 4 * u);
      ctx.save();
      ctx.rotate(-0.7);
      ctx.fillStyle = D;
      ctx.fillRect(-2 * u, -9 * u, 4 * u, 12 * u);
      ctx.restore();
      ctx.beginPath(); ctx.arc(0, 3 * u, 2.6 * u, 0, 7); ctx.fill(); ctx.stroke();
      break;
    }
    case "fence_wood": {           // 木栅栏:三根竖桩+横梁
      ctx.strokeStyle = "#8b5e34"; ctx.lineWidth = 2.2 * u;
      for (const dx of [-6, 0, 6])
        { ctx.beginPath(); ctx.moveTo(dx * u, -8 * u); ctx.lineTo(dx * u, 8 * u); ctx.stroke(); }
      ctx.lineWidth = 1.8 * u;
      ctx.beginPath(); ctx.moveTo(-9 * u, -3 * u); ctx.lineTo(9 * u, -3 * u); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-9 * u, 3 * u); ctx.lineTo(9 * u, 3 * u); ctx.stroke();
      break;
    }
    case "fence_stone": {          // 石栅栏:砌石墙
      ctx.fillStyle = "#94a3b8"; ctx.strokeStyle = "#475569";
      ctx.fillRect(-9 * u, -6 * u, 18 * u, 12 * u);
      ctx.strokeRect(-9 * u, -6 * u, 18 * u, 12 * u);
      ctx.lineWidth = 1 * u;
      ctx.beginPath(); ctx.moveTo(-9 * u, 0); ctx.lineTo(9 * u, 0); ctx.stroke();
      for (const dx of [-3, 3])
        { ctx.beginPath(); ctx.moveTo(dx * u, -6 * u); ctx.lineTo(dx * u, 0); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, 6 * u); ctx.stroke();
      break;
    }
    case "fence_iron": {           // 铁栅栏:栏杆+尖头
      ctx.strokeStyle = "#64748b"; ctx.lineWidth = 1.8 * u;
      ctx.beginPath(); ctx.moveTo(-9 * u, 5 * u); ctx.lineTo(9 * u, 5 * u); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-9 * u, -2 * u); ctx.lineTo(9 * u, -2 * u); ctx.stroke();
      for (const dx of [-7, -2.5, 2.5, 7]) {
        ctx.beginPath(); ctx.moveTo(dx * u, 8 * u); ctx.lineTo(dx * u, -6 * u); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo((dx - 1) * u, -5 * u); ctx.lineTo(dx * u, -9 * u);
        ctx.lineTo((dx + 1) * u, -5 * u); ctx.closePath();
        ctx.fillStyle = "#64748b"; ctx.fill();
      }
      break;
    }
    case "magetower": {            // 法师塔:细塔+顶珠
      ctx.beginPath();
      ctx.moveTo(-4 * u, 9 * u); ctx.lineTo(-2.5 * u, -6 * u);
      ctx.lineTo(2.5 * u, -6 * u); ctx.lineTo(4 * u, 9 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#a78bfa";
      ctx.beginPath(); ctx.arc(0, -8.5 * u, 3 * u, 0, 7); ctx.fill(); ctx.stroke();
      break;
    }
    case "landmine": {             // 地雷:半球+触角
      ctx.beginPath(); ctx.arc(0, 3 * u, 6 * u, Math.PI, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
      ctx.fillRect(-6 * u, 3 * u, 12 * u, 1.6 * u);
      ctx.strokeStyle = D; ctx.lineWidth = 1.4 * u;
      for (const a of [-0.9, -0.45, 0, 0.45, 0.9]) {
        ctx.beginPath();
        ctx.moveTo(Math.sin(a) * 5 * u, 3 * u - Math.cos(a) * 5 * u);
        ctx.lineTo(Math.sin(a) * 8 * u, 3 * u - Math.cos(a) * 8 * u);
        ctx.stroke();
      }
      break;
    }
    case "flamer": {               // 喷火器:罐体+喷口火舌
      ctx.fillRect(-6 * u, -2 * u, 9 * u, 10 * u);
      ctx.strokeRect(-6 * u, -2 * u, 9 * u, 10 * u);
      ctx.fillStyle = "#f97316";
      ctx.beginPath();
      ctx.moveTo(3 * u, 0); ctx.lineTo(10 * u, -4 * u);
      ctx.lineTo(8 * u, 1 * u); ctx.lineTo(11 * u, 3 * u); ctx.lineTo(3 * u, 4 * u);
      ctx.closePath(); ctx.fill();
      break;
    }
    case "laser": {                // 激光炮:基座+双轨炮管+光束点
      ctx.fillRect(-8 * u, 3 * u, 16 * u, 5 * u);
      ctx.strokeRect(-8 * u, 3 * u, 16 * u, 5 * u);
      ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2 * u;
      ctx.beginPath(); ctx.moveTo(-2 * u, 3 * u); ctx.lineTo(6 * u, -8 * u); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-5 * u, 3 * u); ctx.lineTo(3 * u, -8 * u); ctx.stroke();
      ctx.fillStyle = "#22d3ee";
      ctx.beginPath(); ctx.arc(5 * u, -8.5 * u, 1.8 * u, 0, 7); ctx.fill();
      break;
    }
    case "catapult": {             // 投石车:框架+抛臂+石弹
      ctx.fillRect(-8 * u, 2 * u, 16 * u, 5 * u);
      ctx.strokeRect(-8 * u, 2 * u, 16 * u, 5 * u);
      ctx.strokeStyle = "#8b5e34"; ctx.lineWidth = 2.2 * u;
      ctx.beginPath(); ctx.moveTo(-5 * u, 2 * u); ctx.lineTo(6 * u, -8 * u); ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.beginPath(); ctx.arc(7 * u, -8.5 * u, 2.2 * u, 0, 7); ctx.fill();
      ctx.fillStyle = "#374151";
      ctx.beginPath(); ctx.arc(-5 * u, 8 * u, 2.2 * u, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(5 * u, 8 * u, 2.2 * u, 0, 7); ctx.fill();
      break;
    }
    case "airship": {              // 飞艇:气囊+吊舱(阴影由 render 层画)
      ctx.beginPath(); ctx.ellipse(0, -2 * u, 9 * u, 4.5 * u, 0, 0, 7);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = D;
      ctx.fillRect(-3.5 * u, 3.5 * u, 7 * u, 3.5 * u);
      ctx.beginPath(); ctx.moveTo(-3 * u, 2.5 * u); ctx.lineTo(-3 * u, 3.5 * u);
      ctx.moveTo(3 * u, 2.5 * u); ctx.lineTo(3 * u, 3.5 * u); ctx.stroke();
      break;
    }
    case "dragon": {               // 龙骑兵:双翼+长身+吐息
      ctx.beginPath();
      ctx.moveTo(-9 * u, -1 * u); ctx.lineTo(-2 * u, -5 * u);
      ctx.lineTo(0, -1 * u); ctx.lineTo(2 * u, -5 * u); ctx.lineTo(9 * u, -1 * u);
      ctx.lineTo(2 * u, 1 * u); ctx.lineTo(0, 5 * u); ctx.lineTo(-2 * u, 1 * u);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#f97316";
      ctx.beginPath(); ctx.moveTo(0, 5 * u); ctx.lineTo(2 * u, 9 * u);
      ctx.lineTo(-2 * u, 9 * u); ctx.closePath(); ctx.fill();
      break;
    }
    default: {
      ctx.beginPath(); ctx.arc(0, 0, 6 * u, 0, 7); ctx.fill(); ctx.stroke();
    }
  }
  ctx.restore();
}

// 军旗(v1.3):非实体,不进 TYPE_NAMES/drawSprite 分发,render.js 按帧 flags 数组直调
// v1.5:同样吃 PNG 替换槽(assets/flag_p<owner>.png,fig/军旗 贴图接线)
export function drawFlag(ctx, owner, x, y, s) {
  const img = pngFor("flag", owner);
  if (img) {
    ctx.save();
    ctx.drawImage(img, x - s * 0.5, y - s * 0.6, s, s);
    ctx.restore();
    return;
  }
  const u = s / 24;
  ctx.save();
  ctx.translate(x, y);
  ctx.strokeStyle = "#d1d5db";                 // 旗杆
  ctx.lineWidth = Math.max(1, 1.6 * u);
  ctx.beginPath(); ctx.moveTo(0, 9 * u); ctx.lineTo(0, -9 * u); ctx.stroke();
  ctx.fillStyle = P_COLOR[owner];              // 三角旗面
  ctx.strokeStyle = P_DARK[owner];
  ctx.lineWidth = Math.max(1, 1.2 * u);
  ctx.beginPath();
  ctx.moveTo(0, -9 * u); ctx.lineTo(9 * u, -5.5 * u); ctx.lineTo(0, -2 * u);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.restore();
}

export { P_COLOR, P_DARK, TYPE_NAMES };
