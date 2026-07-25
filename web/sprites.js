// TEOW 矢量图案(v1.2):每种实体一个可辨识轮廓,阵营色参数化。
// PNG 替换槽:web/assets/<name>.png 存在则优先绘制贴图(AI 生图可热替换,
// 提示词包见 docs/sprite-prompts.md)。命名:hq/mine/pump/worker/infantry/
// camp/barracks/dog/tower + "_p0"/"_p1"(不带后缀则双方共用,按阵营染色失效)。

const TYPE_NAMES = {1: "hq", 2: "mine", 3: "pump", 4: "worker", 5: "infantry",
                    6: "camp", 7: "barracks", 8: "dog", 9: "tower"};
const P_COLOR = ["#3b82f6", "#ef4444"];          // 蓝 P0 / 红 P1
const P_DARK  = ["#1d4ed8", "#b91c1c"];

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
    const d = s * (type <= 3 || type >= 6 ? 1.0 : 0.7);
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
    default: {
      ctx.beginPath(); ctx.arc(0, 0, 6 * u, 0, 7); ctx.fill(); ctx.stroke();
    }
  }
  ctx.restore();
}

// 军旗(v1.3):非实体,不进 TYPE_NAMES/drawSprite 分发,render.js 按帧 flags 数组直调
export function drawFlag(ctx, owner, x, y, s) {
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
