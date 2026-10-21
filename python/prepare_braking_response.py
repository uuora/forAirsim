"""Freeze low-speed diagnostic configurations; no AirSim I/O or outcomes."""
import argparse,copy,hashlib,json
from datetime import datetime
from pathlib import Path
from braking_response import validate_measurement
ROOT=Path(__file__).resolve().parents[1]


def prepare(output):
    parent=ROOT/'configs/trajectory_timed_pilot_v6p2/center.json'
    base=json.loads(parent.read_text(encoding='utf-8-sig'))
    for name,h in base['flight_source_sha256'].items():
        if hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()!=h:
            raise ValueError('Frozen V6.2 dependency changed: '+name)
    if output.exists():raise ValueError('Preparation refuses to overwrite')
    names=list(base['flight_source_sha256'])+['braking_response.py','braking_response_experiment.py']
    frozen={name:hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest() for name in names}
    direction=[('x_plus',[1.,0.]),('x_minus',[-1.,0.]),('y_plus',[0.,1.]),('y_minus',[0.,-1.])]
    pulses=[{'direction_id':label,'direction_xy':d,'amplitude_m_s':a} for a in (.03,.06) for label,d in direction]
    measurement=dict(mode='pulse_measurement',hold_s=1.,ramp_timeout_s=2.,brake_timeout_s=6.,
        stop_speed_m_s=.003,stop_acceleration_m_s2=.02,stop_stable_s=.5,max_query_gap_s=.25,
        entry_velocity_tolerance_m_s=.006,entry_minimum_amplitude_fraction=.7,
        detect_speed_drop_m_s=.002,pulses=pulses)
    output.mkdir(parents=True)
    rows=[]
    for name,mode,sequence in [('zero_short','zero_hold_gate',[]),('low_gate','pulse_measurement',pulses[:4]),
                               ('session_forward','pulse_measurement',pulses),('session_reverse','pulse_measurement',list(reversed(pulses)))]:
        cfg=copy.deepcopy(base);cfg['protocol_id']='braking_response_v1_20261003'
        cfg['research_stage']='independent_fixed_height_self_state_braking_development'
        cfg['flight_source_sha256']=frozen
        cfg['planning']['execution_reference_mode']='not_used_in_braking_measurement'
        cfg['planning']['image_center_reference_mode']='not_used_in_braking_measurement'
        cfg['braking_measurement']=dict(measurement,mode=mode,pulses=sequence)
        cfg['evidence_limits']=['Independent low-speed self-state pulses, not perception/planning effectiveness',
            'No target pose or RGB/depth read in measurement runner',
            'Original .15m/s, .30m/s2 and workspace/height budgets retained',
            'Request ramp/envelope reuses declared model; sampled stopping does not certify physical safety',
            'Only .03 and .06m/s XY request range; no extrapolation to maximum budget or hardware',
            'Eight pulses within one flight are nested observations, not independent sessions']
        validate_measurement(cfg)
        p=output/(name+'.json');p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows.append({'configuration':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'mode':mode,'planned_pulses':len(sequence)})
    receipt={'kind':'BRAKING_RESPONSE_CONFIG_FREEZE','status':'PREPARED_NOT_EXECUTED',
        'prepared_local':datetime.now().astimezone().isoformat(),'formal_thesis_effect_results':False,
        'configurations':rows,'declared_flight_source_sha256':frozen,'parent_config':str(parent),
        'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'gates':'zero short then separate low-amplitude four-direction then two full sessions',
        'full_planned_flights':2,'full_planned_pulses':16,'ai_assistance':'Preparation assisted by AI; no unexecuted result'}
    (output/'conditions.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();prepare(args.output)
