"use strict";

const get = id => document.getElementById(id);
const data = JSON.parse(get("data").textContent);
const arms = [["ordinary", "普通计时", "#315f95"], ["toppquad", "自动调整 TOPPQuad", "#097e69"]];
const value = (number, digits = 2) => number == null ? "—" : number.toFixed(digits);
const signed = (number, digits = 2) => number == null ? "—" : (number > 0 ? "+" : "") + value(number, digits);
const clock = get("time");
let selected;
let tracks = [];
let scene;
let playback;
let lastFrame;
let current = 0;

function cell(tag, content) {
  const node = document.createElement(tag);
  if (content instanceof Node) node.append(content);
  else node.textContent = content;
  return node;
}

function table(id, headings, rows) {
  const head = document.createElement("thead");
  const body = document.createElement("tbody");
  head.insertRow().append(...headings.map(text => cell("th", text)));
  for (const row of rows) body.insertRow().append(...row.map(entry => cell("td", entry)));
  get(id).replaceChildren(head, body);
}

function options(id, key, label) {
  const entries = new Map(data.pairs.map(pair => [pair[key], pair[label]]));
  for (const [value, text] of entries) get(id).add(new Option(text, value));
}

function links(files = {}) {
  const list = document.createElement("div");
  list.className = "links";
  for (const [name, path] of Object.entries(files)) {
    const link = document.createElement("a");
    link.textContent = name;
    link.href = path;
    list.append(link);
  }
  return list;
}

function status(trial) {
  return trial.status + (trial.landed ? " · 已落地" : "");
}

function comparisonTime(trial) {
  if (!trial.passed && trial.tracking_s != null) return "中止于 " + value(trial.tracking_s, 3);
  return value(trial.tracking_s, 3);
}

function adjustedComparisonRow(pair, suffix = "") {
  const ordinary = pair.ordinary;
  const auto = pair.toppquad;
  const extraPlanning = ordinary.planning_s == null || auto.planning_s == null
    ? null : auto.planning_s - ordinary.planning_s;
  const saved = pair.tracking_delta_s == null ? null : -pair.tracking_delta_s;
  const savingRate = pair.tracking_delta_pct == null ? "—" : value(-pair.tracking_delta_pct, 1) + "%";
  const efficiency = saved != null && extraPlanning > 0 ? saved / extraPlanning : null;
  return [
    pair.route_name + "自动调整 TOPPQuad" + suffix,
    value(auto.planning_s, 3), comparisonTime(auto), value(saved, 3), savingRate, value(efficiency, 3)
  ];
}

function renderComparisons() {
  const geometries = new Map(data.pairs.map(pair => [pair.geometry, pair.geometry_name]));
  for (const [geometry, name] of geometries) {
    const group = document.createElement("section");
    const scroller = document.createElement("div");
    const grid = document.createElement("table");
    group.className = "comparison-group";
    group.id = "comparison-" + geometry;
    scroller.className = "scroll";
    grid.id = "matrix-" + geometry;
    grid.setAttribute("aria-label", name + "对照");
    scroller.append(grid);
    group.append(scroller);
    get("comparison-tables").append(group);
    const rows = [];
    for (const pair of data.pairs.filter(p => p.geometry === geometry)) {
      const baseline = pair.ordinary;
      rows.push([pair.route_name + "基线", value(baseline.planning_s, 3), comparisonTime(baseline), "—", "—", "—"]);
      rows.push(adjustedComparisonRow(pair, pair.retry ? "（首测）" : ""));
      if (pair.retry) rows.push(adjustedComparisonRow({...pair, ...pair.retry}, "（复测）"));
    }
    table(grid.id, [name, "规划时间 (s)", "运行时间 (s)", "TOPPQuad 减少时间 (s)",
      "运行降幅", "优化效率 (s/s)"], rows);
  }
}

function renderTables() {
  const pair = selected;
  get("pair-label").textContent = pair.route_name + " / " + pair.geometry_name;
  const issues = arms.flatMap(([key, label]) => {
    const trial = pair[key];
    return trial.error ? [label + " — " + trial.error] : [];
  });
  if (pair.same_tracking_conditions === false) issues.push("两次跟踪条件指纹不同");
  if (pair.toppquad.planning_feasible === false && pair.segments.length) {
    issues.push("下方逐段数据来自预检拒绝后保留的参考");
  }
  get("selection-note").textContent = issues.join("。");
  table("flights", ["计时方式", "结果", "参考 (s)", "跟踪 (s)", "规划 (s)", "峰速 (m/s)",
    "峰值误差 (m)", "最大倾角 (°)", "最大采样间隔 (s)", "原始数据"], arms.map(([key, label]) => {
    const t = pair[key];
    return [label, status(t), value(t.reference_s), value(t.tracking_s), value(t.planning_s),
      value(t.speed_mps), value(t.peak_error_m, 3), value(t.tilt_deg, 1), value(t.gap_s, 4), links(t.links)];
  }));
  table("waypoints", ["计时方式", "航点", "最小距离 (m)", "判定时距离 (m)", "判定时速度 (m/s)", "条件"],
    arms.flatMap(([key, label]) => {
      const t = pair[key];
      const rows = (t.waypoints ?? []).map(w => [label, "W" + (w.index + 1), value(w.minimum_distance_m, 3),
        value(w.hit_distance_m, 3), value(w.hit_speed_mps, 3), w.passed ? "通过" : "未通过"]);
      const end = t.terminal;
      if (end) rows.push([label, "E", value(end.minimum_distance_m, 3), value(end.end_position_error_m, 3),
        value(end.end_speed_mps, 3), (end.passed ? "通过" : "未通过") + " · 停稳 " + value(end.continuous_s, 3) + " s"]);
      return rows;
    }));
  const legs = pair.segments;
  get("segment-total").textContent = legs.length ?
    "参考总用时差 " + signed(legs.reduce((sum, row) => sum + row.delta_s, 0), 3) + " s" : "暂无分段回执";
  table("segments", ["航段", "普通计时 (s)", "自动调整 (s)", "用时差 (s)", "变化率", "起点 α", "终点 α"],
    legs.map(row => [legName(row.segment, legs.length), value(row.baseline_s, 3), value(row.adjusted_s, 3),
      signed(row.delta_s, 3), signed(row.delta_pct, 1) + "%", value(row.alpha_start, 4), value(row.alpha_end, 4)]));
}

function legName(index, count) {
  const left = index === 1 ? "S" : "W" + (index - 1);
  const right = index === count ? "E" : "W" + index;
  return left + " → " + right;
}

function plot(id) {
  const canvas = get(id);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return null;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {
    canvas.width = Math.round(w*dpr);
    canvas.height = Math.round(h*dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '12px "Microsoft YaHei UI", sans-serif';
  ctx.fillStyle = "#536a6d";
  return {ctx, w, h};
}

function line(ctx, points, color, width = 1.7) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(...points[i]);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function dot(ctx, x, y, color, r = 4) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2*Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function grid(chart, xMax, yMin, yMax, xticks, yticks) {
  const {ctx, w, h} = chart;
  const left = 51, right = w - 17, top = 16, bottom = h - 30;
  const x = v => left + v / xMax * (right-left);
  const y = v => bottom - (v - yMin) / (yMax-yMin) * (bottom-top);
  for (const [val, text] of yticks) {
    line(ctx, [[left,y(val)],[right,y(val)]], val === 0 ? "#a9bfbb" : "#e1eae8", 1);
    ctx.textAlign = "right";
    ctx.fillStyle = "#526a6c";
    ctx.fillText(text, left-6, y(val)+4);
  }
  for (const [val, text] of xticks) {
    ctx.textAlign = "center";
    ctx.fillStyle = "#526a6c";
    ctx.fillText(text, x(val), bottom+17);
  }
  return {x, y, top, bottom};
}

function empty(chart, message) {
  chart.ctx.textAlign = "center";
  chart.ctx.fillText(message, chart.w / 2, chart.h / 2);
}

function timingCharts() {
  const delta = plot("delta"), alpha = plot("alpha");
  if (!delta || !alpha) return;
  const legs = selected.segments;
  if (!legs.length) {
    for (const chart of [delta,alpha]) empty(chart, "暂无分段规划数据");
    return;
  }
  const n = legs.length;
  const limit = Math.max(0.05, Math.ceil(Math.max(...legs.map(r => Math.abs(r.delta_s)))*12)/10);
  const ticks = legs.map((r,i) => [i+0.5, legName(r.segment,n)]);
  const bars = grid(delta,n,-limit,limit,ticks,[-limit,-limit/2,0,limit/2,limit].map(t => [t,value(t,2)+" s"]));
  const barWidth = (bars.x(1)-bars.x(0)) * 0.49;
  for (const [i, row] of legs.entries()) {
    const {ctx} = delta;
    const xp = bars.x(i+.5), yp = bars.y(row.delta_s), zero = bars.y(0);
    ctx.fillStyle = row.delta_s <= 0 ? "#097e69" : "#a05028";
    ctx.fillRect(xp-barWidth/2, Math.min(zero,yp), barWidth, Math.max(1,Math.abs(zero-yp)));
    ctx.fillStyle = "#243c44";
    ctx.textAlign = "center";
    ctx.fillText(signed(row.delta_s,3),xp,yp+(row.delta_s <= 0 ? 14 : -5));
  }
  const nodes = Array.from({length:n+1}, (_,i) => [i, i === 0 ? "S" : i === n ? "E" : "W"+i]);
  const a = grid(alpha,n,0,1,nodes,[0,.25,.5,.75,1].map(v => [v,value(v,2)]));
  for (const [i] of nodes) line(alpha.ctx,[[a.x(i),a.top],[a.x(i),a.bottom]],"#e1eae8",1);
  line(alpha.ctx,selected.alpha.map(([x,y]) => [a.x(x),a.y(y)]),"#097e69",2.4);
  for (const [i] of nodes) dot(alpha.ctx,a.x(i),a.y(selected.alpha[i*40][1]),"#097e69");
}

function telemetryChart(id, index, unit) {
  const chart = plot(id);
  if (!chart) return;
  const available = tracks.filter(t => t.trajectory.length);
  if (!available.length) return empty(chart,"暂无跟踪样本");
  const maxTime = Number(clock.max);
  const maxValue = Math.max(.001, ...available.flatMap(t => t.trajectory.map(p => p[index]))) * 1.14;
  const axes = grid(chart,maxTime,0,maxValue,
    [0,.25,.5,.75,1].map(f => [maxTime*f,value(maxTime*f,1)+" s"]),
    [0,.5,1].map(f => [maxValue*f,value(maxValue*f,unit === "m" ? 3 : 2)]));
  chart.ctx.textAlign = "left";
  chart.ctx.fillText(unit,5,13);
  for (const {trajectory,color} of available) {
    line(chart.ctx,trajectory.map(p => [axes.x(p[0]),axes.y(p[index])]),color);
  }
  line(chart.ctx,[[axes.x(current),axes.top],[axes.x(current),axes.bottom]],"#687c7c",1);
}

function rightIndex(rows,time) {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (rows[mid][0] <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function interpolate(rows,time) {
  const i = rightIndex(rows,time);
  if (i === 0) return rows[0];
  if (i === rows.length) return rows.at(-1);
  const a = rows[i-1], b = rows[i];
  const fraction = (time-a[0])/(b[0]-a[0]);
  return a.map((value,k) => value + fraction*(b[k]-value));
}

function plane() {
  const chart = plot("xy");
  if (!chart) return;
  const {ctx,w,h} = chart;
  const side = Math.min(w,h) - 52;
  const left = (w-side)/2, bottom = (h+side)/2;
  const x = pos => left + pos * side / 8;
  const y = pos => bottom - pos * side / 8;
  for (let i = 0; i <= 8; i++) {
    line(ctx,[[x(i),y(0)],[x(i),y(8)]],"#d4e4e0",1);
    line(ctx,[[x(0),y(i)],[x(8),y(i)]],"#d4e4e0",1);
    ctx.fillStyle = "#536a6d";
    ctx.textAlign = "center";
    if (i % 2 === 0) ctx.fillText(i,x(i),y(0)+16);
    ctx.textAlign = "right";
    if (i % 2 === 0) ctx.fillText(i,x(0)-6,y(i)+4);
  }
  ctx.fillText("X (m)",x(8),y(0)+25);
  ctx.fillText("Y (m)",x(0)+23,y(8)-8);
  for (const {trajectory,color} of tracks) {
    line(ctx,trajectory.map(p => [x(p[1]),y(p[2])]),color,1.4);
    if (trajectory.length && current <= trajectory.at(-1)[0]) {
      const p = interpolate(trajectory,current);
      dot(ctx,x(p[1]),y(p[2]),color,5);
    }
  }
  selected.points.forEach((p,i) => {
    dot(ctx,x(p[0]),y(p[1]),"#1a3c43",2.6);
    ctx.fillStyle = "#1a3c43";
    ctx.textAlign = "left";
    ctx.fillText(i === 0 ? "S" : i === selected.points.length-1 ? "E" : "W"+i,x(p[0])+5,y(p[1])-6);
  });
}

function draw() {
  clock.value = current;
  get("clock").textContent = value(current) + " s";
  if (scene && !get("scene3d").hidden) scene.draw(current,new Set(tracks.map(t => t.id)),interpolate,rightIndex);
  else plane();
  telemetryChart("speed",7,"m/s");
  telemetryChart("error",8,"m");
}

function pause() {
  if (playback != null) cancelAnimationFrame(playback);
  playback = null;
  lastFrame = null;
  get("play").textContent = "播放";
}

function step(timestamp) {
  if (lastFrame != null) current = Math.min(Number(clock.max),current+(timestamp-lastFrame)/1000);
  lastFrame = timestamp;
  draw();
  if (current >= Number(clock.max)) pause();
  else playback = requestAnimationFrame(step);
}

function choose() {
  pause();
  const pair = data.pairs.find(p => p.route === get("route").value && p.geometry === get("geometry").value);
  get("retry-control").hidden = !pair.retry;
  if (!pair.retry) get("attempt").value = "initial";
  selected = pair.retry && get("attempt").value === "retry" ? {...pair, ...pair.retry} : pair;
  tracks = arms.map(([id,,color]) => ({id,color,trajectory:selected[id].samples}));
  current = 0;
  clock.max = Math.max(0,...tracks.map(t => t.trajectory.at(-1)?.[0] ?? 0));
  clock.disabled = get("play").disabled = Number(clock.max) === 0;
  if (scene) scene.setRoute(selected.points,tracks);
  renderTables();
  timingCharts();
  draw();
}

function showView(threeD) {
  get("scene3d").hidden = !threeD;
  get("scene2d").hidden = threeD;
  get("view").textContent = threeD ? "切换 XY" : "切换三维";
  get("reset").hidden = !threeD;
  if (threeD) scene.resize();
  if (selected) draw();
}

options("route","route","route_name");
options("geometry","geometry","geometry_name");
const stats = data.stats;
const retrySummary = stats.retries ? ` · 复测 ${stats.retries_passed}/${stats.retries} 通过` : "";
get("summary").textContent = `${stats.recorded}/${stats.requested} 项已记录 · ${stats.passed} 项验收通过 · ${stats.matched} 组有效配对${retrySummary}`;
renderComparisons();
if (window.makeRouteScene) {
  try { scene = window.makeRouteScene(get("scene3d")); }
  catch { scene = null; }
}
get("view").hidden = !scene;
showView(Boolean(scene));
get("view").addEventListener("click",() => showView(get("scene3d").hidden));
get("reset").addEventListener("click",() => scene?.reset());
get("route").addEventListener("change",choose);
get("geometry").addEventListener("change",choose);
get("attempt").addEventListener("change",choose);
clock.addEventListener("input",() => {pause();current=Number(clock.value);draw();});
get("play").addEventListener("click",() => {
  if (playback != null) return pause();
  if (current >= Number(clock.max)) current=0;
  get("play").textContent="暂停";
  playback=requestAnimationFrame(step);
});
new ResizeObserver(() => {timingCharts();draw();}).observe(get("delta"));
choose();
