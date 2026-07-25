// TEOW 观战渲染(v1.2):Canvas2D,WS 收帧,帧间线性插值平滑连续移动。
import { drawFlag, drawSprite, P_COLOR } from "./sprites.js";

const canvas = document.getElementById("map");
const hud = document.getElementById("hud");
const ctx = canvas.getContext("2d");

let meta = null;
let prev = null, cur = null;      // 相邻两帧,插值用
let curAt = 0;                    // cur 到达的墙钟毫秒
let frameInterval = 50;           // 由服务端 fps 推断(相邻帧到达间隔的滑动均值)
let ended = false;

const NODE_COLOR = { "-1": "#94a3b8", 0: P_COLOR[0], 1: P_COLOR[1] };

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.kind === "meta") { meta = msg; fit(); return; }
    if (msg.kind === "end") { ended = true; return; }
    const now = performance.now();
    if (cur) frameInterval = 0.8 * frameInterval + 0.2 * (now - curAt);
    prev = cur; cur = msg; curAt = now;
  };
  ws.onclose = () => { if (!ended) setTimeout(connect, 1000); };
}

function fit() {
  const s = Math.min(window.innerWidth - 40, window.innerHeight - 90);
  canvas.width = canvas.height = s;
}
window.addEventListener("resize", fit);

function cellPx() { return canvas.width / meta.grid[1]; }
function px(rc) {  // (row,col) -> 像素中心
  const s = cellPx();
  return [rc[1] * s + s / 2, rc[0] * s + s / 2];
}

function lerpEnts(a, b, t) {
  if (!a) return b.ents;
  const prevBy = Object.fromEntries(a.ents.map(e => [e.s, e]));
  return b.ents.map(e => {
    const p0 = prevBy[e.s];
    if (!p0) return e;
    return { ...e, p: [p0.p[0] + (e.p[0] - p0.p[0]) * t,
                       p0.p[1] + (e.p[1] - p0.p[1]) * t] };
  });
}

// v1.4:满血由服务端按 config 查表下发(e.mx),前端不再硬编码数值;
// 血条只画建筑(meta.building_types),单位血条按比例意义不大

function draw() {
  requestAnimationFrame(draw);
  if (!meta || !cur) return;
  const s = cellPx();
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 网格淡线
  ctx.strokeStyle = "rgba(148,163,184,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= meta.grid[0]; i++) {
    ctx.beginPath(); ctx.moveTo(0, i * s); ctx.lineTo(canvas.width, i * s); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(i * s, 0); ctx.lineTo(i * s, canvas.height); ctx.stroke();
  }

  // 资源点(归属描边:◇矿 ○水)
  meta.node_pos.forEach((rc, k) => {
    const [x, y] = px(rc);
    ctx.strokeStyle = NODE_COLOR[String(cur.node_owner[k])];
    ctx.lineWidth = Math.max(2, s * 0.09);
    ctx.beginPath();
    if (meta.node_type[k] === 0) {
      ctx.moveTo(x, y - s * 0.42); ctx.lineTo(x + s * 0.42, y);
      ctx.lineTo(x, y + s * 0.42); ctx.lineTo(x - s * 0.42, y); ctx.closePath();
    } else {
      ctx.arc(x, y, s * 0.4, 0, 7);
    }
    ctx.stroke();
  });

  // 军旗(v1.3;服务端只下发 active 旗:[p, r, c]。旧 run 无此字段则跳过)
  (cur.flags || []).forEach(([p, fr, fc]) => {
    const [x, y] = px([fr, fc]);
    drawFlag(ctx, p, x, y, s);
  });

  // 实体(帧间插值)
  const t = Math.min(1, (performance.now() - curAt) / frameInterval);
  const ents = lerpEnts(prev, cur, t);
  const bldTypes = meta.building_types || [1, 2, 3, 6, 7, 9];
  for (const e of ents) {
    const owner = e.s < meta.e_max ? 0 : 1;
    const [x, y] = px(e.p);
    const isBldT = bldTypes.includes(e.t);
    drawSprite(ctx, e.t, owner, x, y, s * (isBldT ? 1.5 : 1.0),
               { building: e.bt < 0, inside: e.in, bld: isBldT });
    if (e.lv > 1) {               // 等级角标
      ctx.fillStyle = "#facc15";
      ctx.font = `bold ${Math.max(9, s * 0.38)}px sans-serif`;
      ctx.fillText(String(e.lv), x + s * 0.45, y - s * 0.35);
    }
    if (e.cg > 0) {               // 载荷点
      ctx.fillStyle = "#facc15";
      ctx.beginPath(); ctx.arc(x, y - s * 0.5, s * 0.1, 0, 7); ctx.fill();
    }
    const mx = isBldT ? e.mx : 0;  // 旧回放无 mx 字段 → 不画条
    if (mx && e.hp < mx) {        // 建筑血条(受损才显示)
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(x - s * 0.5, y + s * 0.55, s, s * 0.12);
      ctx.fillStyle = e.hp / mx > 0.4 ? "#22c55e" : "#ef4444";
      ctx.fillRect(x - s * 0.5, y + s * 0.55, s * e.hp / mx, s * 0.12);
    }
  }

  // HUD(v1.4 八线:逐线太长,显示已研线数/最高线级)
  const r = cur.res, u = cur.upgrades;
  const lineTag = (lv) => {
    const up = lv.filter(x => x > 1).length;
    return `研${up}线/最高${Math.max(...lv)}`;
  };
  const win = cur.winner >= 0 ?
    `   ⚑ ${cur.winner === 2 ? "和局" : "玩家" + cur.winner + " 获胜"}` : "";
  hud.innerHTML =
    `<span style="color:${P_COLOR[0]}">P0 矿${r[0][0]} 水${r[0][1]} ${lineTag(u[0])}</span>` +
    `&nbsp; tick ${cur.tick} &nbsp;` +
    `<span style="color:${P_COLOR[1]}">P1 矿${r[1][0]} 水${r[1][1]} ${lineTag(u[1])}</span>` +
    win + (ended ? "  (回放结束,刷新重播)" : "");
}

connect();
requestAnimationFrame(draw);
