const degrees = 180 / Math.PI;
const ROTOR_CENTERS = [[.58, .58], [.58, -.58], [-.58, -.58], [-.58, .58]];

export function poseFromQuaternion(quaternion) {
  const norm = Math.hypot(...quaternion);
  if (!Number.isFinite(norm) || norm < 1e-12) return null;
  const [x, y, z, w] = quaternion.map(value => value / norm);
  const rotation = [
    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
  ];
  const [east, north, up] = rotation;
  const projection = Math.hypot(east[0], north[0]);
  const pitch = Math.atan2(-up[0], projection) * degrees;
  // ZYX roll and yaw are not separately defined at the singularity.
  const roll = projection < 1e-8 ? null : Math.atan2(up[1], up[2]) * degrees;
  const yaw = projection < 1e-8 ? null : Math.atan2(north[0], east[0]) * degrees;
  return { rotation, roll, pitch, yaw };
}

export function parseSimulation(csv) {
  const [header, ...rows] = csv.trim().split(/\r?\n/);
  const names = header.split(',');
  const fields = [
    'time_s', 'position_e_m', 'position_n_m', 'position_u_m',
    'velocity_e_mps', 'velocity_n_mps', 'velocity_u_mps', 'qx', 'qy', 'qz', 'qw',
    'attitude_error_deg', 'actual_tilt_deg',
  ];
  const indices = fields.map(field => names.indexOf(field));
  if (indices.some(index => index < 0)) throw new Error('缺少 RotorPy 位置或姿态列');
  return rows.flatMap(row => {
    const cells = row.split(',');
    const values = indices.map(index => Number(cells[index] || NaN));
    if (!values.every(Number.isFinite)) return [];
    const [time, east, north, up, ve, vn, vu, x, y, z, w, attitudeError, tilt] = values;
    const pose = poseFromQuaternion([x, y, z, w]);
    if (!pose) return [];
    return [{time, position:[east,north,up], speed:Math.hypot(ve,vn,vu), attitudeError, tilt, ...pose}];
  });
}

function newMesh(color) {
  return {vertices:[], triangles:[], colors:[], color};
}

function append(mesh, vertices, triangles, color) {
  const offset = mesh.vertices.length;
  mesh.vertices.push(...vertices);
  mesh.triangles.push(...triangles.map(face => face.map(index => index + offset)));
  mesh.colors.push(...triangles.map(() => color || mesh.color));
}

function box(mesh, x0, y0, x1, y1, halfWidth, bottom, top, color) {
  const length = Math.hypot(x1-x0,y1-y0);
  const dx = -(y1-y0) * halfWidth / length;
  const dy = (x1-x0) * halfWidth / length;
  const corners = [[x0-dx,y0-dy],[x1-dx,y1-dy],[x1+dx,y1+dy],[x0+dx,y0+dy]];
  append(mesh, [bottom,top].flatMap(z => corners.map(([x,y]) => [x,y,z])),
    [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], color);
}

function disc(mesh, x, y, z, r, color) {
  const sides = 18;
  const vertices = [[x,y,z], ...Array.from({length:sides}, (_, k) => {
    const angle = k * 2*Math.PI / sides;
    return [x+r*Math.cos(angle),y+r*Math.sin(angle),z];
  })];
  append(mesh, vertices, Array.from({length:sides}, (_, k) => [0,k+1,(k+1)%sides+1]), color);
}

const airframe = newMesh('#2a4a64');
const propellers = newMesh('#528096');
const front = newMesh('#f18a15');
const nose = [[0,0,.21],[1.39,0,.21],[.98,.22,.21],[.98,-.22,.21]];
append(front, nose, [[1,2,3]]);
box(front, .02,0,1.02,0,.05,.195,.225);
box(airframe, -.27,0,.33,0,.23,-.13,.14,'#28455b');
box(airframe, -.20,0,.23,0,.17,.14,.21,'#92bbc4');
for (const [x,y] of ROTOR_CENTERS) {
  box(airframe, 0,0,x,y,.075,-.075,.075,'#3a5d72');
  box(airframe, x-.077,y,x+.077,y,.077,-.072,.13,x > 0 ? '#d48226' : '#274b62');
  disc(propellers,x,y,.127,.263,x > 0 ? '#cb8a45' : '#578393');
}

export const BODY_MESHES = [airframe, propellers, front];
export const NOSE_LABEL = [[1.46,0,.21]];

export function worldVertices(vertices, position, rotation, scale) {
  if (!position || !rotation) return [];
  return vertices.map(vertex => rotation.map((axis, i) =>
    position[i] + scale * (axis[0]*vertex[0] + axis[1]*vertex[1] + axis[2]*vertex[2])));
}

export function drawBodyPreview(canvas, rotation) {
  const context = canvas.getContext('2d');
  const {width,height} = canvas;
  context.clearRect(0,0,width,height);
  if (!rotation) return;
  const units = .26*Math.min(width,height);
  const faces = BODY_MESHES.flatMap(mesh => {
    const enu = worldVertices(mesh.vertices,[0,0,0],rotation,1);
    const points = enu.map(([east,north,up]) => [
      width/2 + units*(east-north)*.7071,
      height/2 + units*((east+north)*.3665 - up*.8552),
      (east+north)*.6047 + up*.5184,
    ]);
    return mesh.triangles.map((face,index) => {
      const vertices = face.map(i => points[i]);
      return {vertices, color:mesh.colors[index], depth:vertices.reduce((sum,p) => sum+p[2],0)};
    });
  });
  faces.sort((a,b) => a.depth-b.depth);
  for (const face of faces) {
    context.fillStyle = face.color;
    context.beginPath();
    const [first,...rest] = face.vertices;
    context.moveTo(first[0],first[1]);
    for (const point of rest) context.lineTo(point[0],point[1]);
    context.closePath();
    context.fill();
  }
}

export function worldAxes(points) {
  const min = [Infinity,Infinity,Infinity], max = [-Infinity,-Infinity,-Infinity];
  for (const point of points) {
    point.forEach((value, axis) => {
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    });
  }
  const span = Math.max(1, ...max.map((value, i) => value-min[i]));
  const scale = Math.min(.92, Math.max(.34, .052*span));
  const pad = 1.55*scale;
  return {scale, ranges:min.map((value, axis) => [value-pad,max[axis]+pad])};
}

export function nearestSample(samples, time) {
  let left = 0, right = samples.length - 1;
  while (left < right) {
    const middle = Math.floor((left+right)/2);
    if (samples[middle].time < time) left = middle + 1;
    else right = middle;
  }
  return left > 0 && time - samples[left-1].time < samples[left].time - time ? left-1 : left;
}
