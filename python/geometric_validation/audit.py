"""Read-only numerical audit; no AirSim connection and no controller imports.

Recomputes errors using independent quaternion/Euler formulas, validates CSV,
hashes, clock, command bounds, release/terminal hold and outcome separately.
"""
from __future__ import annotations
import csv
import os
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def rotation(q):
    q=np.asarray(q,dtype=float);n=float(q@q)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(n-1)>.01:raise ValueError('Invalid state quaternion')
    w,x,y,z=q/math.sqrt(n)
    # Rodrigues/quaternion vector formula, independent of controller's entries.
    u=np.array([x,y,z]);K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (w*w-u@u)*np.eye(3)+2*np.outer(u,u)+2*w*K


def angle(R,D):
    return math.acos(float(np.clip((np.trace(np.asarray(D).T@R)-1)/2,-1,1)))


def matrix_euler(roll,pitch,yaw):
    sr,cr=math.sin(roll),math.cos(roll);sp,cp=math.sin(pitch),math.cos(pitch);sy,cy=math.sin(yaw),math.cos(yaw)
    X=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]);Y=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]);Z=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Z@Y@X


def calculate(rows,cfg):
    data=[];issues=[];l=cfg['limits'];cc=cfg['controller']
    for r in rows:
        if r.get('phase')!='TRACKING':continue
        try:
            p=np.array(r['position']);v=np.array(r['velocity']);ref=r['reference'];cmd=r['command'];R=rotation(r['quaternion_wxyz'])
            D=np.asarray(r['R_reference']);C=np.asarray(cmd['R_des'])
            for M in (D,C):
                if M.shape!=(3,3) or not np.allclose(M.T@M,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(M),1,atol=1e-7):raise ValueError('Invalid desired SO3')
            if not np.allclose(matrix_euler(cmd['roll'],cmd['pitch'],cmd['yaw']),C,atol=1e-8):raise ValueError('Command Euler/R mismatch')
            # Reference attitude comes solely from acceleration + heading (unsaturated if feasible).
            f=np.array([0,0,cc['gravity']])-ref['a'];b3=f/np.linalg.norm(f);heading=np.array([math.cos(ref['yaw']),math.sin(ref['yaw']),0])
            b2=np.cross(b3,heading);b2/=np.linalg.norm(b2);nominal=np.column_stack([np.cross(b2,b3),b2,b3])
            if not np.allclose(nominal,D,atol=1e-7):raise ValueError('Nominal reference was saturated/altered; task not feasible')
            e=float(np.linalg.norm(p-ref['p']));ev=float(np.linalg.norm(v-ref['v']));att=angle(R,D);att_cmd=angle(R,C)
            if abs(att-r['attitude_reference_error_rad'])>1e-6:raise ValueError('Logged reference attitude error differs')
            throttle=float(cmd['throttle']);tilt=math.acos(float(np.clip(R[2,2],-1,1)));rate=float(np.linalg.norm(r['angular_velocity']))
            command_bound=cc['min_throttle']-1e-9<=throttle<=cc['max_throttle']+1e-9 and math.acos(float(np.clip(C[2,2],-1,1)))<=cc['max_tilt_rad']+1e-8
            if not command_bound:issues.append('Command limit violated')
            data.append(dict(timestamp_ns=r['timestamp_ns'],t=r['t'],position_error_m=e,velocity_error_mps=ev,
                attitude_reference_error_rad=att,attitude_command_error_rad=att_cmd,actual_tilt_rad=tilt,
                actual_rate_radps=rate,actual_speed_mps=float(np.linalg.norm(v)),throttle=throttle,
                saturated=any(cmd['saturation'].values()),command_sent=bool(r.get('command_sent')),failure=r.get('failure','')))
        except Exception as ex:issues.append('Invalid sample: '+repr(ex))
    if not data:return {},False,data,issues+['No tracking samples']
    t=np.array([d['t'] for d in data]);stamps=np.array([d['timestamp_ns'] for d in data],dtype=np.int64)
    delta=np.r_[t[0],np.diff(stamps)/1e9]
    if not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<=0) or np.any(delta<=0):issues.append('Tracking clock not strictly increasing')
    # Right-endpoint sample-hold convention; first interval from release t=0.
    weights=np.diff(np.r_[0.,t]);duration=float(t[-1]);denom=max(duration,1e-12)
    def values(k):return np.array([d[k] for d in data])
    def stats(k,threshold=None):
        val=values(k);res=dict(rmse=float(math.sqrt(float(weights@(val*val))/denom)),peak=float(val.max()))
        if threshold is not None:res['exceedance_s']=float(weights@(val>threshold))
        return res
    metrics=dict(duration_observed_s=duration,samples=len(data),position_error_m=stats('position_error_m',l['position_error_m']),
        velocity_error_mps=stats('velocity_error_mps'),attitude_reference_error_rad=stats('attitude_reference_error_rad',l['attitude_error_rad']),
        attitude_command_error_rad=stats('attitude_command_error_rad',l['attitude_error_rad']),
        actual_tilt_rad=stats('actual_tilt_rad',l['safety_tilt_rad']),actual_rate_radps=stats('actual_rate_radps',l['safety_rate_radps']),
        actual_speed_mps=stats('actual_speed_mps',l['safety_speed_mps']),throttle=dict(min=float(values('throttle').min()),max=float(values('throttle').max())),
        command_saturation_s=float(weights@values('saturated')),max_state_interval_s=float(delta.max()) if len(delta) else None,
        motor_saturation='UNOBSERVED: command clipping is recorded; no per-motor control signal telemetry',
        integration='time-weighted right-endpoint hold from release; no interpolation/smoothing')
    safe=all((np.array(d['position'])>=l['bounds_ned_min']).all() and (np.array(d['position'])<=l['bounds_ned_max']).all() for d in rows if d.get('phase')=='TRACKING')
    passes=bool(not issues and safe and len(delta) and delta.max()<=l['max_state_interval_s'] and not any(d['failure'] for d in data)
        and values('position_error_m').max()<=l['position_error_m'] and values('attitude_reference_error_rad').max()<=l['attitude_error_rad']
        and values('actual_tilt_rad').max()<=l['safety_tilt_rad'] and values('actual_rate_radps').max()<=l['safety_rate_radps']
        and values('actual_speed_mps').max()<=l['safety_speed_mps'] and all(d['command_sent'] for d in data))
    return metrics,passes,data,issues


def _audit(directory):
    directory=Path(directory);cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'));manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (directory/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    metrics,passes,data,issues=calculate(rows,cfg)
    baseline=manifest.get('airborne_collision_baseline_ns',manifest.get('live',{}).get('initial_state',{}).get('collision_timestamp_ns',0))
    if any(r.get('collision') and r.get('collision_timestamp_ns',1)>baseline for r in rows if r.get('phase')=='TRACKING'):passes=False
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(directory/'config.json')!=manifest['config_sha256']:issues.append('Config hash mismatch')
    for name,value in manifest['sources'].items():
        p=directory/'source'/name
        if not p.is_file() or sha(p)!=value:issues.append('Snapshot hash mismatch: '+name)
    # CSV is a derived transparent numerical export; verify existing exports, never overwrite a differing one.
    csvpath=directory/'metrics_samples.csv'
    if data:
        if csvpath.exists():
            with csvpath.open(encoding='utf-8-sig',newline='') as stream:actual=list(csv.DictReader(stream))
            if len(actual)!=len(data) or any(str(row[k])!=a.get(k) for row,a in zip(data,actual) for k in row):issues.append('CSV differs from recomputed JSONL')
        else:
            with csvpath.open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    events=json.loads((directory/'events.json').read_text(encoding='utf-8'));report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    releases=[e for e in events if e['event']=='RELEASE'];ends=[e for e in events if e['event']=='TERMINAL_STABLE']
    # Independent endpoint linear system (no controller/reference implementation imports).
    task={"start":cfg['start'],"yaw":cfg.get('yaw',0),**cfg['cases'][manifest['case']]}
    start=np.asarray(task['start'],dtype=float)
    knots=np.asarray([start]+[np.asarray(p) for p in task.get('points',[])]) if 'points' in task else np.asarray([start]+[start+np.asarray(o) for o in task['offsets']])
    durations=np.asarray(task['segment_durations']);times=np.r_[0,np.cumsum(durations)]
    matrix=[]
    for u in (0.,1.):
        for d in range(4):matrix.append([0 if k<d else math.factorial(k)/math.factorial(k-d)*u**(k-d) for k in range(8)])
    deriv=[knots]+[np.asarray(task.get(key,np.zeros_like(knots))) for key in ['velocities','accelerations','jerks']]
    coefs=[]
    for i,T in enumerate(durations):
        b=np.asarray([deriv[d][j]*T**d for j in (i,i+1) for d in range(4)])
        coefs.append(np.linalg.solve(matrix,b))
    for row in rows:
        if row.get('phase') not in ('TRACKING','GEOMETRIC_SETTLING'):continue
        t=row['t']
        if t>times[-1]:expected=[knots[-1]]+[np.zeros(3)]*4
        else:
            i=min(max(0,int(np.searchsorted(times,t,side='right'))-1),len(durations)-1);u=(t-times[i])/durations[i]
            expected=[sum((math.factorial(k)/math.factorial(k-d)*coefs[i][k]*u**(k-d)/durations[i]**d for k in range(d,8)),start=np.zeros(3)) for d in range(5)]
        if any(not np.allclose(row['reference'][k],v,atol=1e-7,rtol=1e-8) for k,v in zip(['p','v','a','jerk','snap'],expected)) or abs(row['reference']['yaw']-task['yaw'])>1e-10:
            issues.append('Reference differs from frozen task');break
    if releases:
        settling=[r for r in rows if r.get('phase')=='GEOMETRIC_SETTLING' and r['timestamp_ns']<=releases[0]['timestamp_ns']]
        stable=[]
        for r in reversed(settling):
            if np.linalg.norm(np.array(r['position'])-start)>cfg['execution']['start_position_m'] or np.linalg.norm(r['velocity'])>cfg['execution']['start_speed_mps'] or angle(rotation(r['quaternion_wxyz']),np.eye(3) if task['yaw']==0 else matrix_euler(0,0,task['yaw']))>cfg['execution']['start_attitude_rad']:break
            stable.append(r)
        if not stable or (releases[0]['timestamp_ns']-stable[-1]['timestamp_ns'])/1e9<cfg['execution']['start_hold_s']:
            issues.append('Unified start hold not independently verified')
    if report['normal_completion']:
        if len(releases)!=1 or len(ends)!=1:issues.append('Missing unique release/completion')
        if data:
            if not releases or any(abs((d['timestamp_ns']-releases[0]['timestamp_ns'])/1e9-d['t'])>1e-8 for d in data):issues.append('Synchronized reference clock mismatch')
            end=data[-1]['t'];tail=[]
            for row in reversed(data):
                if row['t']<times[-1] or row['position_error_m']>cfg['limits']['terminal_position_m'] or row['actual_speed_mps']>cfg['limits']['terminal_speed_mps'] or row['attitude_reference_error_rad']>cfg['limits']['terminal_attitude_rad']:break
                tail.append(row)
            if not tail or end-tail[-1]['t']<cfg['execution']['terminal_hold_s']:issues.append('Terminal hold not independently verified')
            if report['actual_completion_time_s'] is None or abs(end-report['actual_completion_time_s'])>1e-8:issues.append('Completion time mismatch')
            if ends and abs(ends[0]['t']-end)>1e-8:issues.append('Terminal event time mismatch')
    # Failure outcomes are data, not audit failures; no-flight intake failures are explicitly incomplete.
    if 'live' not in manifest:issues.append('Live runtime provenance missing')
    output=dict(protocol=cfg['protocol'],data_audit_passed=not issues,control_constraints_passed=bool(passes),issues=issues,metrics=metrics,
        normal_completion=report['normal_completion'],raw_jsonl_sha256=sha(directory/'samples.jsonl'),
        audit_scope='Recorded state/command arithmetic, timing, snapshot and CSV consistency; not an independent sensor, binary-build proof, or flight success')
    (directory/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return output


def audit(directory):
    directory=Path(directory).resolve()
    if os.name=='nt' and not str(directory).startswith('\\\\?\\'):directory=Path('\\\\?\\'+str(directory))
    try:return _audit(directory)
    except Exception as ex:
        output=dict(data_audit_passed=False,control_constraints_passed=False,metrics={},issues=['Incomplete/malformed evidence: '+repr(ex)])
        (Path(directory)/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        return output
