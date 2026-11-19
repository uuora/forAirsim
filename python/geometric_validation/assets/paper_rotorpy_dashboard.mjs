import {BODY_MESHES, NOSE_LABEL, drawBodyPreview, nearestSample, parseSimulation, worldAxes, worldVertices} from './paper_rotorpy_pose.mjs';
import {AttitudeChart} from './paper_rotorpy_attitude.mjs';

const $ = id => document.getElementById(id);
const GEOMETRY = {bezier:'Bézier', hermite:'Hermite', bspline:'B 样条'};
const RAD_TO_DEG = 180 / Math.PI;
const LIMIT_N = .14375;
const POSE_VERTICES = [...BODY_MESHES.map(mesh => mesh.vertices), NOSE_LABEL];
const POSE_TRACE_IDS = POSE_VERTICES.map((_, index) => index + 3);
const GATES = {
  reference_speed:'参考限速', reference_rotor:'参考电机推力', waypoint:'实际过点',
  position:'位置误差', attitude:'姿态误差', tilt:'实际倾角', terminal:'终点停稳', heading:'机头跟踪',
};
const STATUS = {
  completed:'正常停稳', terminal_timeout:'终点超时',
  nonfinite_or_flyaway_or_ground:'仿真提前停止', error:'规划或记录失败',
  planning_rejected:'规划约束未通过',
};
let design, results, diagnostics, series = [], drawRequest, currentIndex = -1, modelScale = 1;
let animationId = 0, sceneReady = false, waitingPose = null, poseUpdate = null, cameraRevision = 0;
let sceneUpdate = Promise.resolve();
let selectedRun;
let qualification = false;
let headingCampaign = false;
let paperHeading = false;
let planningPairs = [];
let sourceDiagnostics;
let geometryRows = new Map();

function finite(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

async function read(path, {type = 'json', signal} = {}) {
  const response = await fetch(path, {cache:'no-store',signal});
  if (!response.ok) throw new Error('HTTP ' + response.status);
  return response[type]();
}

function setOptions(id, keys, labels = keys) {
  $(id).replaceChildren(...keys.map((key,index) => new Option(labels[index],key)));
}

function table(id, columns, rows) {
  const result = document.createElement('table');
  const header = result.createTHead().insertRow();
  for (const name of columns) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = name;
    header.append(th);
  }
  const body = result.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) tr.insertCell().append(value);
  }
  $(id).replaceChildren(result);
}

function median(rows, key) {
  const values = rows.map(row => row[key]).filter(Number.isFinite).sort((a,b) => a-b);
  if (!values.length) return null;
  return (values[Math.floor((values.length-1)/2)] + values[Math.ceil((values.length-1)/2)]) / 2;
}

function label(arm) {
  return design.labels[design.arms.indexOf(arm)];
}

function paperMetrics() {
  return paperHeading && design.independent_random;
}

function chartMethod() {
  return $(paperMetrics() ? 'comparisonMethod' : 'method').value;
}

function renderSummary(rows) {
  const metrics = paperMetrics();
  const columns = ['路径','方法','已记录', ...(!metrics ? ['任务通过'] : []), '正常完成','参考推力合规'];
  if (paperHeading) columns.push('参考限速达标');
  columns.push('参考时间中位 s','完成时间中位 s','位置 RMSE 中位 m');
  if (!metrics) columns.push('姿态峰值 °');
  table('groups',columns,
    design.methods.flatMap(method => design.arms.map(arm => {
      const cohort = rows.filter(row => row.method === method && row.arm === arm);
      const count = predicate => {
        const n = cohort.filter(predicate).length;
        return metrics ? n + '/' + cohort.length : n;
      };
      return [
        GEOMETRY[method], label(arm), cohort.length, ...(!metrics ? [count(row => row.passed) + '/' + design.routes.length] : []),
        count(row => row.status === 'completed'), count(row => row.reference_rotor_feasible),
        ...(paperHeading ? [count(row => row.reference_speed_passed)] : []),
        finite(median(cohort,'reference_s')), finite(median(cohort,'complete_s')),
        finite(median(cohort,'position_rmse_m'),3), ...(!metrics ? [finite(median(cohort,'attitude_peak_deg'))] : []),
      ];
    })));
}

function renderPairs(rows) {
  const columns = ['路径','比较方法','配对子集','样本数','参考时间差 s','完成时间差 s','RMSE 差 m','峰值差 m','姿态峰值差 °'];
  if (!paperHeading) columns.push('离散规划时间差 s');
  table('pairs',columns,
    rows.map(row => [
      GEOMETRY[row.method], label(row.comparator), row.cohort, row.n, finite(row.reference_s), finite(row.complete_s),
      finite(row.position_rmse_m,4), finite(row.position_peak_m,4), finite(row.attitude_peak_deg),
      ...(!paperHeading ? [finite(row.raw_reference_delta_s)] : []),
    ]));
}

function renderPaperTimes() {
  const method = chartMethod();
  const selected = [...results.values()].filter(row => row.method === method);
  const alpha = design.arms.filter(arm => arm.endsWith('_alpha'));
  const comparators = alpha.length ? alpha : design.arms.filter(arm => arm !== 'toppquad');
  const arms = [...comparators, 'toppquad'];
  const layout = unit => ({
    height:310, margin:{t:14,b:75,l:64,r:20}, showlegend:false, yaxis:{title:{text:unit},zeroline:true},
  });
  const box = (arm, y) => ({type:'box',name:label(arm),y,boxpoints:'outliers'});
  const options = {responsive:true,displaylogo:false};
  Plotly.react('referenceTimes',arms.map(arm =>
    box(arm,selected.filter(row => row.arm === arm && row.reference_rotor_feasible).map(row => row.reference_s))),
    layout('参考飞行时间 s'),options);
  Plotly.react('pairedTimeReduction',comparators.map(arm =>
    box(arm,planningPairs.filter(row => row.method === method && row.comparator === arm).map(row => row.pct))),
    layout('成对缩短率 %'),options);
  if (paperMetrics()) {
    $('completedTimes').hidden = false;
    Plotly.react('completedTimes',design.arms.map(arm =>
      box(arm,selected.filter(row => row.arm === arm && row.status === 'completed').map(row => row.complete_s))),
      layout('闭环完成时间 s'),options);
  }
}

async function loadPaperTimes() {
  const csv = await read('paired.csv',{type:'text'});
  const [header,...rows] = csv.trim().replace(/^\uFEFF/,'').split(/\r?\n/);
  const names = header.split(',');
  const [method,comparator,pct] = ['method','comparator','reference_reduction_pct'].map(key => names.indexOf(key));
  planningPairs = rows.map(row => row.split(',')).filter(row => row[pct] !== '')
    .map(row => ({method:row[method],comparator:row[comparator],pct:Number(row[pct])}));
  $('timeSection').hidden = false;
  renderPaperTimes();
  if (location.hash === '#timeSection') $('timeSection').scrollIntoView();
}

function renderHeadingPairs(groups, csv) {
  const lines = csv.trim().replace(/^\uFEFF/,'').split(/\r?\n/);
  const fields = lines.shift().split(',');
  const column = field => fields.indexOf(field);
  const rows = lines.map(line => line.split(','));
  table('pairs', ['路径','重优化首次通过','重优化最终通过','额外闭环次数','双方通过数','完成时间差 s','位置 RMSE 差 mm'],
    design.methods.map(method => {
      const group = groups.find(row => row.method === method && row.arm === 'heading_toppquad');
      const paired = rows.filter(row => row[column('method')] === method &&
        row[column('baseline_passed')] === 'True' && row[column('toppquad_passed')] === 'True');
      const deltas = paired.map(row => ({
        time: Number(row[column('delta_complete_s')]),
        rmse: 1000*Number(row[column('delta_position_rmse_m')]),
      }));
      return [GEOMETRY[method],group.first_passed+'/'+group.attempts,group.passed+'/'+group.attempts,
        group.retries,paired.length,finite(median(deltas,'time')),finite(median(deltas,'rmse'))];
    }));
}

function renderThrust() {
  const rows = [...results.values()];
  if (rows.length !== Number($('progress').max)) return;
  const source = rows.filter(row => row.method === chartMethod());
  const traces = design.arms.flatMap(arm => [
    ['reference_thrust_min_n','x','y'], ['reference_thrust_max_n','x2','y2'],
  ].map(([key,xaxis,yaxis]) => ({
    type:'box', name:label(arm), xaxis, yaxis, showlegend:false,
    y:source.filter(row => row.arm === arm).map(row => row[key]).filter(Number.isFinite),
  })));
  Plotly.react('force',traces,{
    height:490, margin:{t:16,b:76,l:60,r:14}, grid:{rows:2,columns:1,pattern:'independent'},
    yaxis:{title:{text:'最小电机推力 N'}}, yaxis2:{title:{text:'最大电机推力 N'}},
    shapes:[
      {type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{dash:'dot'}},
      {type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:LIMIT_N,y1:LIMIT_N,line:{dash:'dot'}},
    ],
  },{responsive:true,displaylogo:false});
}

function renderFailures(data) {
  diagnostics = data;
  const metrics = paperMetrics();
  const gates = metrics ? ['reference_speed','waypoint','position','attitude','tilt','heading','terminal'] : Object.keys(data.gates);
  const names = metrics ? ['参考超速','中间点未达标','位置误差超限','整体姿态误差超限','实际倾角超限','机头误差超限','末点未停稳'] :
    Object.values(data.gates);
  table('failures',['路径','方法',metrics ? '样本数' : '通过',...names,'指令推力超限'],
    data.groups.map(row => [
      GEOMETRY[row.method], label(row.arm), metrics ? row.attempts : row.passed + '/' + row.attempts,
      ...gates.map(gate => row.failures[gate] ?? 0), row.command_outside,
    ]));
  if (selectedRun) renderRunFailures(selectedRun.row,selectedRun.path,selectedRun.knotSpeeds);
}

function failedGates(row) {
  if (row.error || !row.samples) return [];
  const limits = design.limits;
  const flags = {
    reference_speed:row.reference_speed_passed === false,
    reference_rotor:row.reference_rotor_feasible === false,
    waypoint:row.waypoint_passed === false,
    position:row.position_peak_m > limits.position_error_m,
    attitude:row.attitude_peak_deg > limits.attitude_error_rad * RAD_TO_DEG,
    tilt:row.tilt_peak_deg > limits.safety_tilt_rad * RAD_TO_DEG,
    terminal:row.status !== 'completed',
    heading:row.heading_passed === false,
  };
  return Object.keys(flags).filter(key => flags[key]).map(key => diagnostics?.gates[key] ?? GATES[key]);
}

function playbackTime(time) {
  if (!series.length || !Number.isFinite(time)) return finite(time);
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = finite(time);
  button.addEventListener('click',() => seekTo(time));
  return button;
}

function renderRunFailures(row, path, knotSpeeds = []) {
  const evidence = qualification && row?.arm === 'flat' ? sourceDiagnostics : diagnostics;
  const detail = evidence?.runs?.[path];
  const flags = detail ? detail.failed_gates.map(key => evidence.gates[key]) : row ? failedGates(row) : [];
  const stop = detail ? evidence.statuses[detail.status] : null;
  const over = paperMetrics() ? '超限指标 · ' : '未通过判据 · ';
  const inRange = paperMetrics() ? '本次记录未发现指标超限。' : '该运行通过全部门槛。';
  $('currentFailure').textContent = !row ? '此项暂无运行判据。' :
    flags.length ? over + flags.join('、') + (stop ? ' ｜ 结束状态 · ' + stop : '') :
    (paperMetrics() ? row.samples : row.passed) ? inRange : row.error || '该运行未生成完整判据。';
  $('currentFailure').classList.toggle('bad', Boolean(flags.length));
  const times = $('failureTimes');
  times.replaceChildren();
  if (detail && series.length) {
    for (const [key,name] of [
      ['first_position_over_s','首次位置超限'], ['first_attitude_over_s','首次姿态超限'],
      ['first_tilt_over_s','首次倾角超限'], ['first_command_over_s','首次指令推力超限'],
    ]) {
      if (!Number.isFinite(detail[key])) continue;
      const button = document.createElement('button');
      button.textContent = name + ' · ' + finite(detail[key]) + ' s';
      button.addEventListener('click',() => seekTo(detail[key]));
      times.append(button);
    }
  }
  const passages = row?.passages ?? [];
  if (passages.length) table('passages',[
    '航点','结果','参考过点时刻 s','参考过点速度 m/s','合格采样时刻 s','合格采样速度 m/s',
    '最近距离 m','最近点实测速度 m/s',
  ],
    passages.map((item,index) => [
      '中间点 ' + item.waypoint_index,
      evidence?.waypoint_states[detail?.waypoint_states[index]] ?? (item.passed ? '通过' : '未通过'),
      playbackTime(item.reference_time_s), finite(knotSpeeds[index],3),
      playbackTime(item.actual_time_s), finite(item.actual_speed_mps,3),
      finite(item.nearest_distance_m,3), finite(item.nearest_speed_mps,3),
    ]));
  else $('passages').replaceChildren();
}

function describeRun(row, samples, issue, path, knotSpeeds) {
  $('result').textContent = row ? [
    ...(!paperMetrics() ? [row.passed ? '全任务通过' : '全任务未通过'] : []), STATUS[row.status] ?? row.status,
    '参考 ' + finite(row.reference_s) + ' s', '完成 ' + finite(row.complete_s) + ' s',
    '位置峰值 ' + finite(row.position_peak_m,3) + ' m',
    row.closed_loop_retiming_attempts ? '倾角复核后重定时 ' + row.closed_loop_retiming_attempts + ' 次' : '',
    row.retries ? '重试 '+row.retries+' 次 · 时长倍率 '+finite(row.dilation) : '',
    row.error, issue,
  ].filter(Boolean).join(' · ') : issue || '此项尚无记录。';
  const runKey = [row?.route,row?.method,row?.arm].join('/');
  selectedRun = {row,path:runKey,knotSpeeds};
  renderRunFailures(row,runKey,knotSpeeds);
  $('links').replaceChildren();
  if (!row) return;
  for (const [name,file,available] of [
    ['运行判据','result.json',true], ['解析参考','reference.json',row.reference_s != null],
    ['参考时序 CSV','reference_metrics.csv',row.reference_s != null], ['50 Hz 实测时序','simulation.csv.gz',row.samples],
    ['逐电机参考推力 CSV','actuator_reference.csv',row.reference_s != null &&
      (headingCampaign || (paperHeading && ['heading_baseline','toppquad'].includes(row.arm)))],
    ['首次飞行判据','timing_attempts/attempt_000/result.json',row.closed_loop_retiming_attempts || row.retries],
    ['航向时序','heading.csv',(headingCampaign || paperHeading) && row.samples],
    ['规划回执','planning.json',(headingCampaign || paperHeading) && row.planning_status],
  ]) {
    if (!available) continue;
    const link = document.createElement('a');
    link.href = path + '/' + file;
    link.textContent = name;
    $('links').append(link,'　');
  }
  if (row.samples && !samples.length && !issue) $('result').append(' · 无可用姿态采样');
}

function coords(points) {
  return ['x','y','z'].map((_,axis) => points.map(point => point[axis]));
}

function lineTrace(points, name, color, dash = 'solid') {
  const [x,y,z] = coords(points);
  return {type:'scatter3d', mode:'lines', name, x, y, z, line:{color,dash,width:5}, hoverinfo:'skip'};
}

function meshTrace(mesh, index, sample, scale) {
  // Hold fixed topology and update only the vertex positions during playback.
  const [x,y,z] = coords(worldVertices(mesh.vertices,sample?.position,sample?.rotation,scale));
  const [i,j,k] = coords(mesh.triangles);
  return {
    type:'mesh3d', x,y,z,i,j,k, facecolor:mesh.colors, hoverinfo:'skip', showlegend:false,
    flatshading:true, opacity:index === 1 ? .82 : 1,
    lighting:{ambient:.87,diffuse:.28,specular:.12,roughness:.65,fresnel:.04},
    lightposition:{x:25,y:35,z:50},
  };
}

function renderScene(waypoints, reference, samples) {
  const points = samples.map(sample => sample.position);
  const first = samples[0];
  const engine = worldAxes([...waypoints,...reference,...points]);
  modelScale = engine.scale;
  const markers = lineTrace(waypoints,'航点','#1e384e');
  markers.mode = 'markers+text';
  markers.text = ['起点',...waypoints.slice(1,-1).map((_,i) => '中间点 '+(i+1)),'终点'];
  markers.textposition = 'top center';
  markers.marker = {size:4,color:'#1e384e'};
  const [xaxis,yaxis,zaxis] = engine.ranges.map((range,i) => ({
    title:{text:['东 E (m)','北 N (m)','上 U (m)'][i]}, range, autorange:false, backgroundcolor:'#f2f6f9', showbackground:true,
  }));
  const [x,y,z] = coords(worldVertices(NOSE_LABEL,first?.position,first?.rotation,modelScale));
  const headLabel = {
    type:'scatter3d', mode:'text', x, y, z, text:['+X'], textposition:'top center',
    textfont:{color:'#9a3c04',size:14}, hoverinfo:'skip', showlegend:false,
  };
  const chord = lineTrace(waypoints,'航点连线','#748493','dash');
  chord.line.width = 3;
  return Plotly.react('plot',[
    markers, lineTrace(reference,'参考轨迹','#218075','dot'), lineTrace(points,'实测轨迹','#673bae'),
    ...BODY_MESHES.map((mesh,index) => meshTrace(mesh,index,first,modelScale)), headLabel, chord,
  ],{
    margin:{l:0,r:0,t:8,b:30}, legend:{orientation:'h'}, paper_bgcolor:'#fff',
    scene:{xaxis,yaxis,zaxis,aspectmode:'data',uirevision:++cameraRevision},
  },{responsive:true,displaylogo:false});
}

async function positionModel(sample) {
  waitingPose = sample;
  if (poseUpdate) return;
  poseUpdate = updateModel();
  try {
    await poseUpdate;
  } finally {
    poseUpdate = null;
  }
}

async function updateModel() {
  while (sceneReady && waitingPose) {
    const pose = waitingPose;
    waitingPose = null;
    const vertices = POSE_VERTICES.map(points => coords(worldVertices(points,pose.position,pose.rotation,modelScale)));
    await Plotly.restyle('plot',{
      x:vertices.map(mesh => mesh[0]), y:vertices.map(mesh => mesh[1]), z:vertices.map(mesh => mesh[2]),
    },POSE_TRACE_IDS);
  }
}

function showSample(index, moveModel = true) {
  const sample = series[index];
  if (!sample || index === currentIndex) return;
  currentIndex = index;
  $('seek').value = index;
  $('clock').textContent = finite(sample.time) + ' s';
  for (const key of ['roll','pitch','yaw','attitudeError','tilt']) {
    $(key).textContent = Number.isFinite(sample[key]) ? finite(sample[key],1) + '°' : '—';
  }
  $('speed').textContent = finite(sample.speed) + ' m/s';
  $('headingReadout').hidden = !sample.heading;
  if (sample.heading) {
    const [yaw, error] = sample.heading;
    $('headingReadout').textContent = '航向命令 '+finite(yaw,1)+'° · 实测机头偏差 '+finite(error,1)+'°';
    $('headingReadout').classList.toggle('bad',Math.abs(error)>design.heading_policy.heading_error_limit_deg);
  }
  $('attitudeError').classList.toggle('bad', sample.attitudeError > design.limits.attitude_error_rad*RAD_TO_DEG);
  $('tilt').classList.toggle('bad', sample.tilt > design.limits.safety_tilt_rad*RAD_TO_DEG);
  drawBodyPreview($('bodyView'),sample.rotation);
  attitudeChart.setTime(sample.time);
  if (moveModel) positionModel(sample).catch(() => { $('result').append(' · 机体刷新失败'); });
}

function pause() {
  cancelAnimationFrame(animationId);
  animationId = 0;
  $('play').textContent = '播放';
}

function seekTo(time) {
  if (!series.length) return;
  pause();
  showSample(nearestSample(series,time));
}

const attitudeChart = new AttitudeChart($('attitude'), seekTo);

function setPlayback(samples, row) {
  series = samples;
  currentIndex = -1;
  $('headingReadout').hidden = true;
  $('seek').max = Math.max(0,series.length-1);
  $('seek').value = 0;
  for (const id of ['seek','play','attitudePeak','tiltPeak']) $(id).disabled = !series.length;
  for (const id of ['clock','roll','pitch','yaw','attitudeError','tilt','speed']) {
    $(id).textContent = '—';
    $(id).classList.remove('bad');
  }
  attitudeChart.setSeries(series,
    [design.limits.attitude_error_rad*RAD_TO_DEG,design.limits.safety_tilt_rad*RAD_TO_DEG],
    [...(row?.passages ?? []).map(passage => passage.reference_time_s),row?.reference_s]);
  if (series.length) showSample(0,false);
  else drawBodyPreview($('bodyView'),null);
}

function parseReference(csv, passages = []) {
  const [header, ...rows] = csv.trim().split(/\r?\n/);
  const columns = header.split(',');
  const timeIndex = columns.indexOf('time_s');
  const speedIndex = columns.indexOf('speed_mps');
  const samples = rows.map(row => row.split(',').map(field => field === '' ? NaN : Number(field)));
  const points = samples.map(sample => sample.slice(1,4))
    .filter(point => point.length === 3 && point.every(Number.isFinite));
  const knotSpeeds = passages.map(passage =>
    samples.find(sample => Math.abs(sample[timeIndex] - passage.reference_time_s) < 1e-7)?.[speedIndex]);
  return {points,knotSpeeds};
}

async function readSimulation(path, signal) {
  const response = await fetch(path,{cache:'no-store',signal});
  if (!response.ok) throw new Error('HTTP '+response.status);
  const text = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).text();
  return parseSimulation(text);
}

function mergeHeading(samples,csv) {
  const angles = new Map(csv.trim().split(/\r?\n/).slice(1).map(line => {
    const [time,command,,,error] = line.split(',').map(Number);
    const yaw = ((command + 180) % 360 + 360) % 360 - 180;
    return [time,[yaw,error]];
  }));
  for (const sample of samples) sample.heading = angles.get(sample.time);
}

function renderGeometry() {
  const detail = geometryRows.get($('route').value + '/' + $('method').value);
  const note = $('geometryNote');
  note.hidden = !detail;
  note.textContent = detail ?
    '参考偏离对应航点线段最多 ' + finite(detail.distance,2) + ' m · 参考弧长 / 航点连线长度 ' +
    finite(detail.ratio,2) + ' · 航段最大转角 ' + finite(detail.turn,1) + '°' : '';
}

function loadGeometry(path) {
  return read(path,{type:'text'}).then(csv => {
    const [header,...lines] = csv.trim().split(/\r?\n/);
    const names = header.replace(/^\uFEFF/,'').split(',');
    for (const line of lines) {
      const cols = line.split(',');
      const field = key => cols[names.indexOf(key)];
      if (field('route') && field('method')) {
        geometryRows.set(field('route') + '/' + field('method'),{
          distance:Number(field('maximum_chord_distance_m')), ratio:Number(field('arc_to_chords_ratio')),
          turn:Number(field('maximum_turn_deg')),
        });
      }
    }
    renderGeometry();
  }).catch(() => {});
}

async function draw() {
  pause();
  drawRequest?.abort();
  const request = new AbortController();
  drawRequest = request;
  const {signal} = request;
  sceneReady = false;
  waitingPose = null;
  selectedRun = null;
  renderRunFailures(null);
  setPlayback([]);
  $('result').textContent = '正在读取记录…';
  const task = design.routes.find(item => item.id === $('route').value);
  renderGeometry();
  const waypoints = task.points_ned.map(([north,east,down]) => [east,north,-down]);
  const arm = $('arm').value;
  const prefix = qualification && arm === 'flat' ? [design.source_path] : [];
  const path = [...prefix,task.id,$('method').value,arm].join('/');
  let row, samples = [], reference = [], knotSpeeds = [], issue;
  try {
    row = await read(path + '/result.json',{signal});
    const reads = await Promise.allSettled([
      row.reference_s != null ? read(path+'/reference_metrics.csv',{type:'text',signal})
        .then(csv => parseReference(csv,row.passages)) : null,
      row.samples ? readSimulation(path+'/simulation.csv.gz',signal) : [],
      row.samples && (headingCampaign || paperHeading) ? read(path+'/heading.csv',{type:'text',signal}) : null,
    ]);
    if (reads[0].status === 'fulfilled' && reads[0].value) {
      ({points:reference,knotSpeeds} = reads[0].value);
    }
    if (reads[1].status === 'fulfilled') samples = reads[1].value;
    if (reads[2].status === 'fulfilled' && reads[2].value) mergeHeading(samples,reads[2].value);
    if (reads.some(result => result.status === 'rejected')) issue = '部分轨迹数据读取失败';
  } catch (error) {
    if (error.message !== 'HTTP 404') issue = '运行判据读取失败';
  }
  if (signal.aborted) return;
  const ready = sceneUpdate.then(async () => {
    if (poseUpdate) await poseUpdate.catch(() => {});
    if (!signal.aborted) await renderScene(waypoints,reference,samples);
  });
  sceneUpdate = ready.catch(() => {});
  await ready;
  if (signal.aborted) return;
  sceneReady = true;
  setPlayback(samples,row);
  describeRun(row,samples,issue,path,knotSpeeds);
}

window.addEventListener('pagehide',pause);

function drawSelection() {
  draw().catch(error => { $('result').textContent = '三维回放载入失败 · ' + error.message; });
}

function jumpToPeak(key) {
  if (!series.length) return;
  const sample = series.reduce((max,item) => item[key] > max[key] ? item : max);
  seekTo(sample.time);
}

function play() {
  if (animationId) { pause(); return; }
  if (!series.length) return;
  if (currentIndex === series.length - 1) showSample(0);
  const start = series[currentIndex].time;
  const end = series.at(-1).time;
  let started;
  const frame = now => {
    started ??= now;
    const time = start + (now-started)/1000;
    showSample(nearestSample(series,Math.min(time,end)));
    if (time >= end) pause();
    else animationId = requestAnimationFrame(frame);
  };
  $('play').textContent = '暂停';
  animationId = requestAnimationFrame(frame);
}

async function refresh() {
  try {
    const records = await read('progress.json');
    results = new Map(records.map(row => [[row.route,row.method,row.arm].join('/'),row]));
    $('progress').value = records.length;
    const progress = qualification ?
      '稳健计时 '+records.filter(row => row.arm === design.default_arm).length+' / '+Number($('progress').max)/design.arms.length+' 项' :
      records.length+' / '+$('progress').max+' 项';
    const batch = design.independent_random ? '新随机批次' : '种子 '+design.seed;
    $('status').textContent = batch+' · '+design.routes.length+' 条随机航路 · '+progress;
    renderSummary(records);
    if (records.length === Number($('progress').max)) {
      const summary = await read('summary.json');
      if (!qualification && !headingCampaign) renderPairs(summary.paired);
      if (headingCampaign) renderHeadingPairs(summary.groups,await read('paired.csv',{type:'text'}));
      if (paperHeading) await loadPaperTimes();
      renderThrust();
      read('failure_diagnostics.json').then(renderFailures).catch(() => {
        $('failures').textContent = '完成实验后可生成逐组诊断。';
      });
      return;
    }
  } catch {
    if (!results.size) $('status').textContent = '等待实验记录';
  }
  setTimeout(refresh,4000);
}

function showCampaignLink(href, caption) {
  const link = document.createElement('a');
  link.href = href;
  link.textContent = caption;
  const container = $('campaignLink');
  if (container.childElementCount) container.append(' · ');
  container.append(link);
  container.hidden = false;
}

async function initialize() {
  design = await read('design.json');
  qualification = design.experiment_type === 'baseline_qualification';
  headingCampaign = design.experiment_type === 'heading_toppquad';
  paperHeading = design.experiment_type === 'paper_heading';
  if (paperHeading) {
    document.title = design.independent_random ? 'RotorPy 稳健基线与论文对照' : 'RotorPy 论文七组对照 · 统一航向';
    document.querySelector('h1').textContent = document.title;
    $('headingRule').textContent = '各方法共用同一航向函数，机头误差按水平投影计算。';
    $('experimentDescription').append(' 机头误差 ≤10°。各组统计首次闭环。');
    if (design.arms.some(arm => arm.endsWith('_alpha'))) {
      $('experimentDescription').append(' 三种 α 方法以逐电机参考推力合规为目标做统一减速。');
    }
    document.querySelector('#pairSection h2').textContent = 'TOPPQuad 与各对照组的同航路配对';
    document.querySelector('#pairSection p').textContent = '差值为 TOPPQuad 减比较组。参考时间同时报告逐电机推力合规及附加参考限速的结果。正常完成和任务通过配对同时报告完成时间和跟踪误差。';
    document.querySelector('a[href="failure_analysis.md"]').href = 'conclusion.md';
    const diagnosticsLink = document.querySelector('a[href="failure_diagnostics.csv"]');
    diagnosticsLink.href = 'failure_diagnostics.json';
    diagnosticsLink.textContent = '逐次判据';
    if (design.independent_random) {
      $('experimentDescription').append(' 主基线规划参考峰速 ≤4.2 m/s、加速度 ≤4.9 m/s²、参考倾角 ≤30°，逐电机预留 0.01 N。');
      $('replaySection').before($('comparisonSection'),$('timeSection'),$('pairSection'));
      $('comparisonSection').querySelector('h2').textContent = '逐电机推力与各组完成情况';
      $('comparisonScope').hidden = false;
      $('comparisonFilters').hidden = false;
      $('timeSection').querySelector('h2').textContent = '参考飞行时间和闭环完成时间';
      $('timeSection').querySelector('p').textContent = '从上至下分别为参考推力合规样本的参考时间、同路线参考时间缩短率、正常停稳样本的闭环完成时间。';
      $('pairSection').querySelector('p').textContent = '差值为 TOPPQuad 减比较组。参考时间按逐电机推力合规筛选，并列出附加限速后的样本。闭环完成时间及跟踪误差取双方正常完成的路线。';
      $('diagnosticSection').querySelector('h2').textContent = '逐项验收参考';
      $('diagnosticSection').querySelector('p.note').textContent = '每列独立统计超限次数。同一项飞行可能触发多列。';
      $('currentFailure').textContent = '选择一条记录查看逐项指标。';
      showCampaignLink('#comparisonSection','论文式对比');
      read('paper_figures/manifest.json').then(() =>
        showCampaignLink('paper_figures/','论文式图集与下载')).catch(() => {});
      showCampaignLink('#designSection','实验设计与数据');
      showCampaignLink('#diagnosticSection','逐项验收参考');
    } else {
      showCampaignLink('../','原始固定航向 2,100 项');
      showCampaignLink('../heading_toppquad_full/','两组航向开发对照与重跑');
    }
  } else if (headingCampaign) {
    document.querySelector('h1').textContent = 'RotorPy TOPPQuad 航向跟踪核验';
    $('headingRule').textContent = '航向命令沿前视路径平滑转向，机头误差按水平投影计算。';
    document.querySelector('#pairSection h2').textContent = '同航路配对结果';
    document.querySelector('#pairSection p').textContent = '时间与 RMSE 差值为重优化减去同航向基线，仅统计双方通过的路线。';
    showCampaignLink('../baseline_verified_v2/', '查看稳健计时基线');
    document.querySelector('a[href="failure_analysis.md"]').href = 'conclusion.md';
    const diagnosticsLink = document.querySelector('a[href="failure_diagnostics.csv"]');
    diagnosticsLink.href = 'failure_diagnostics.json';
    diagnosticsLink.textContent = '逐次判据';
    loadGeometry('../baseline_verified_v2/path_geometry.csv');
  } else if (qualification) {
    document.querySelector('h1').textContent = 'RotorPy 稳健计时基线核验';
    showCampaignLink(design.source_path + '/', '返回论文七组对照');
    $('pairSection').hidden = true;
    document.querySelector('a[href="failure_analysis.md"]').href = design.source_path + '/failure_analysis.md';
    const diagnosticsLink = document.querySelector('a[href="failure_diagnostics.csv"]');
    diagnosticsLink.href = 'qualification_diagnostics.csv';
    diagnosticsLink.textContent = '逐次核验 CSV';
    read(design.source_path + '/failure_diagnostics.json').then(data => {
      sourceDiagnostics = data;
      if (selectedRun) renderRunFailures(selectedRun.row,selectedRun.path,selectedRun.knotSpeeds);
    }).catch(() => {});
    const p = design.reference_policy;
    $('experimentDescription').prepend(
      '在同一批航点和固定几何上只重新分配时间。新基线规划峰速 ' + p.speed_cap_mps +
      ' m/s、峰加速度 ' + p.acceleration_cap_mps2 + ' m/s²、参考倾角 ' + p.reference_tilt_deg +
      '°，每台电机留 ' + p.motor_margin_n + ' N 余量。原任务门槛一致。'
    );
  } else {
    read('baseline_verified_v2/audit.json').then(audit => {
      if (audit.verified_new_flights === audit.attempts) {
        showCampaignLink('baseline_verified_v2/', '查看稳健计时基线核验');
        return loadGeometry('baseline_verified_v2/path_geometry.csv');
      }
    }).catch(() => {});
  }
  results = new Map();
  setOptions('route',design.routes.map(route => route.id));
  setOptions('method',design.methods,design.methods.map(method => GEOMETRY[method]));
  setOptions('arm',design.arms,design.labels);
  if (design.default_arm) $('arm').value = design.default_arm;
  const query = new URLSearchParams(location.search);
  for (const id of ['route','method','arm']) {
    const selector = $(id);
    const requested = query.get(id);
    if (Array.from(selector.options).some(option => option.value === requested)) selector.value = requested;
  }
  if (paperMetrics()) {
    setOptions('comparisonMethod',design.methods,design.methods.map(method => GEOMETRY[method]));
    $('comparisonMethod').value = $('method').value;
    $('comparisonMethod').onchange = () => {
      renderThrust();
      if (!$('timeSection').hidden) renderPaperTimes();
    };
  }
  if (qualification) loadGeometry('path_geometry.csv');
  if (qualification) {
    read('../heading_toppquad_full/design.json').then(() =>
      showCampaignLink('../heading_toppquad_full/','查看 TOPPQuad 航向跟踪对照')).catch(() => {});
  }
  if (!paperHeading && !qualification) {
    const path = (headingCampaign ? '../' : '') + 'paper_heading_full/';
    read(path + 'design.json').then(() => showCampaignLink(path,'查看统一航向的论文七组对照')).catch(() => {});
  }
  const protocolPath = (paperHeading || headingCampaign ? '../' : qualification ? design.source_path + '/' : '') +
    'stable_baseline/';
  if (!design.independent_random) {
    read(protocolPath + 'audit.json').then(() => showCampaignLink(protocolPath,'主基线 · 实验步骤与配对结果')).catch(() => {});
  }
  $('progress').max = design.routes.length*design.methods.length*design.arms.length;
  for (const id of ['route','method','arm']) $(id).onchange = () => {
    drawSelection();
    if (id === 'method') {
      renderThrust();
      if (paperHeading && !$('timeSection').hidden) renderPaperTimes();
    }
  };
  $('seek').oninput = () => { pause(); showSample(Number($('seek').value)); };
  $('play').onclick = play;
  $('attitudePeak').onclick = () => jumpToPeak('attitudeError');
  $('tiltPeak').onclick = () => jumpToPeak('tilt');
  $('latest').onclick = () => {
    const record = [...results.values()].findLast(row => row.samples && (!qualification || row.arm === design.default_arm));
    if (!record) return;
    for (const key of ['route','method','arm']) $(key).value = record[key];
    drawSelection();
    renderThrust();
  };
  refresh();
  drawSelection();
}

initialize().catch(error => { $('status').textContent = '请通过本地 HTTP 服务打开此页 · ' + error.message; });
