"""Independent read-only checks of protocol13/14 saved execution evidence."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from audit_airsim_curve_replay_pilot import independent_metrics
from run_airsim_curve_replay_pilot import plan_curve
from trajectory_research.curves import PolynomialSegment
from .reproduction_paths import workspace_record_path,leaf_name


def independent_route_metrics(curve,states,commands):
    # Reuse the timestamp/error calculation, then integrate jerk separately on
    # each polynomial. One global quadrature would miss piece boundaries.
    config=dict(execution=dict(terminal_interpolation_max_gap_s=.1),
        control_error_gates=dict(sampled_peak_position_m=.5,terminal_position_m=.25,terminal_velocity_mps=.5))
    values=independent_metrics(curve,states,commands,config)
    nodes,weights=np.polynomial.legendre.leggauss(8)
    energy=0.
    for segment in curve.segments:
        jerk=segment.sample(segment.duration_s*(nodes+1)/2,derivative=3)
        energy+=segment.duration_s/2*float(weights@np.sum(jerk*jerk,axis=1))
    times=np.array([s['reference_elapsed_s'] for s in states])
    positions=np.array([s['position_zup_m'] for s in states]);velocities=np.array([s['velocity_zup_mps'] for s in states])
    rows=[];spec=curve.spec
    for event in curve.waypoint_times:
        stamp=event['evaluation_time_s'];index=event['index']
        assert times[0]<=stamp<=times[-1]
        position=np.array([np.interp(stamp,times,positions[:,axis]) for axis in range(3)])
        velocity=np.array([np.interp(stamp,times,velocities[:,axis]) for axis in range(3)])
        distance=float(np.linalg.norm(position-event['point']));speed=float(np.linalg.norm(velocity))
        last=index==len(curve.waypoint_times)-1
        speed_ok=speed<=.5 if last else speed<=spec['stop_speed_limit_mps'] if spec['policy']=='stop' else speed>=spec['minimum_pass_speed_mps']
        rows.append(dict(index=index,status='MEASURED',reference_time_s=stamp,position_error_m=distance,actual_speed_mps=speed,
            passed=bool(distance<=spec['waypoint_radius_m'] and speed_ok),rule='terminal_stop' if last else spec['policy']))
    values['control_error_checks']['waypoints']=all(r['passed'] for r in rows)
    values['control_error_gates_passed']=all(values['control_error_checks'].values())
    n=len(rows);T=curve.duration_s;rmse=values['position_rmse_m']
    values.update(jerk_energy_m2_s5=energy,waypoint_results=rows,task_success=values['control_error_gates_passed'],
        quality_term=.7*T/(T+5*n)+.1*energy/(energy+30*n)+.2*rmse/(rmse+.25))
    return values


def audit_run(folder):
    folder=Path(folder).resolve()
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    manifest=read('live_manifest.json');summary=read('summary.json');trace=read('trace.json')
    revision=manifest['protocol_revision']
    if revision not in (13,14,15) or summary['protocol_revision']!=revision:raise ValueError('不是实时协议13/14/15')
    sources=manifest['source_sha256']
    for path,digest in sources.items():
        archived=folder/'curve_source'/leaf_name(path)
        if hashlib.sha256(archived.read_bytes()).hexdigest()!=digest:raise ValueError('源码快照损坏：'+path)
    if summary.get('sources_unchanged') is False:raise ValueError('执行期间源码发生变化')
    for name in ('curves.py','parameterizations.py')+(('route_curve.py',) if revision in (14,15) else ()):
        original=next(p for p in sources if leaf_name(p)==name);path=workspace_record_path(original)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sources[original]:raise ValueError('当前曲线解码器与冻结版本不同，需使用冻结版本审计')
    states,commands=trace['states'],trace['commands'];metrics=summary.get('metrics')
    receipt=dict(status='NO_VALID_SCORE',mode=summary['mode'],outcome=summary['outcome'],states=len(states),method=summary['method'],
        condition_fingerprint=manifest['condition_fingerprint'],scope='Saved evidence consistency, not a new flight or physical safety guarantee')
    if metrics is None:
        receipt['reason']=summary.get('metric_error') or summary.get('error');return receipt
    if revision==15:
        prep=read('preparation.json');events=prep['events'];stages=[e['stage'] for e in events]
        order=['MOVING_TO_START_ARRIVED','PLANNING_STARTED','PLANNING_FINISHED','SETTLING_ARRIVED','REFERENCE_READY','REFERENCE_STARTED']
        selected=[next(e for e in events if e['stage']==stage) for stage in order]
        assert all(a['host_monotonic_s']<b['host_monotonic_s'] for a,b in zip(selected,selected[1:]))
        for event in (selected[0],selected[3]):
            np.testing.assert_allclose(event['position'],manifest['plan']['route']['start'],atol=.020001)
            assert np.linalg.norm(event['velocity'])<=.030001
        request=read('planning_request.json')
        assert request['method']==summary['method']
        assert request['task']['start']==manifest['plan']['route']['start']
        elapsed=selected[2]['planning_wall_s']
        assert elapsed>0 and abs(elapsed-summary['planning_evidence']['parent_total_planning_wall_s'])<1e-8
        assert summary['airsim_search_calls']==0
        receipt.update(preparation_order_verified=True,planning_after_measured_arrival=True,planning_wall_s=elapsed)
    t=np.array([s['reference_elapsed_s'] for s in states]);stamps=np.array([s['sim_timestamp_ns'] for s in states],dtype=np.int64)
    assert len(states)==len(commands) and len(states)>=2
    np.testing.assert_allclose(t,(stamps-stamps[0])*1e-9,rtol=0,atol=2e-12)
    assert t[0]==0 and np.all(np.diff(t)>0) and np.diff(t).max()<=.100001
    if revision in (14,15):
        from .route_curve import decode_route
        curve=decode_route(manifest['plan'],summary['config']['curve_experiment']['time_scale'])
        assert abs(curve.duration_s-manifest['effective_duration_s'])<1e-10
    else:
        original=plan_curve(manifest['plan']);curve=PolynomialSegment(original.coefficients.copy(),manifest['effective_duration_s'])
        assert abs(curve.duration_s-original.duration_s/summary['config']['curve_experiment']['time_scale'])<1e-10
    conditions=dict(protocol_revision=13,execution=dict(terminal_interpolation_max_gap_s=.1),
        control_error_gates=dict(sampled_peak_position_m=.5,terminal_position_m=.25,terminal_velocity_mps=.5))
    recomputed=independent_route_metrics(curve,states,commands[:-1]) if revision in (14,15) else independent_metrics(curve,states,commands[:-1],conditions)
    if 'task_success' in metrics:recomputed['task_success']=bool(summary['outcome']=='COMPLETED' and recomputed['control_error_gates_passed'])
    for key,value in recomputed.items():
        if revision in (14,15) and key not in metrics:continue
        if key=='waypoint_results':
            assert len(value)==len(metrics[key])
            for expected,actual in zip(value,metrics[key]):
                for k,v in expected.items():
                    if isinstance(v,(str,bool)):assert v==actual[k],k
                    else:np.testing.assert_allclose(v,actual[k],rtol=2e-8,atol=2e-8,err_msg=k)
            continue
        if isinstance(value,dict):assert value==metrics[key],key
        else:np.testing.assert_allclose(value,metrics[key],rtol=2e-8,atol=2e-8,err_msg=key)
    limits=summary['config']['control'];previous=np.zeros(3)
    for index,(state,command) in enumerate(zip(states,commands)):
        sampled=min(float(t[index]),curve.duration_s)
        q,qv,qa=[curve.sample(sampled,derivative=d) for d in (0,1,2)]
        if t[index]>curve.duration_s:qv=np.zeros(3);qa=np.zeros(3)
        np.testing.assert_allclose(q,state['reference_position_zup_m'],atol=1e-8)
        raw=qv+2*(q-np.array(state['position_zup_m']))+.8*(qv-np.array(state['velocity_zup_mps']))+.9*qa
        speed=raw*min(1.,limits['max_speed']/max(np.linalg.norm(raw),1e-300))
        dt=max(float(t[index]-t[index-1]) if index else 0.,.001)
        delta=speed-previous;expected=previous+delta*min(1.,limits['max_acceleration']*dt/max(np.linalg.norm(delta),1e-300))
        np.testing.assert_allclose(expected,command['limited'],rtol=1e-8,atol=1e-8)
        assert bool(np.linalg.norm(raw)>limits['max_speed'])==command['speed_clipped']
        assert bool(np.linalg.norm(delta)>limits['max_acceleration']*dt)==command['slew_clipped']
        previous=expected
    with (folder/'states.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==len(states)
    for row,state in zip(rows,states):
        np.testing.assert_allclose([float(row[k]) for k in ['t_s','px','py','pz']],
            [state['reference_elapsed_s'],*state['position_zup_m']],atol=1e-9)
    marker=summary.get('marker_receipts',[])
    for item in marker:np.testing.assert_allclose(item['requested_ned'],item['observed_ned'],atol=.002)
    hidden=any(m['name']=='DC_TargetRed' and not m['visible'] for m in marker)
    view=summary.get('scene_view',{})
    following=any('SetFollow true' in r and 'succeeded' in r for r in view.get('scene_receipts') or [])
    if summary['outcome']=='COMPLETED' and summary['mode']=='airsim' and 'marker_receipts' in summary:assert hidden,'到达目标未隐藏'
    receipt.update(status='PASS',metrics_recomputed=True,commands_reconstructed=True,csv_verified=True,
        control_passed=recomputed['control_error_gates_passed'],goal_hidden_readback=hidden,native_follow_acknowledged=following,
        protocol_revision=revision,policy=manifest['plan'].get('route',{}).get('policy'),
        peak_actual_speed_mps=max(float(np.linalg.norm(s['velocity_zup_mps'])) for s in states),
        tracking_host_wall_s=summary.get('tracking_host_wall_s'),reference_duration_s=curve.duration_s)
    return receipt
