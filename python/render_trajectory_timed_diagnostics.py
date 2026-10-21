"""Plot saved time-indexed request diagnostics, including failed flights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(receipt_path, output):
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite: {output}')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if receipt['audit_status']!='VERIFIED':
        raise ValueError('Require verified read-only diagnostic inputs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw_inputs=[]
    for row in receipt['runs']:
        path=Path(row['report_path'])
        if digest(path)!=row['report_sha256']:
            raise ValueError(f'Plot input changed: {path}')
        raw_inputs.append((row,json.loads(path.read_text(encoding='utf-8-sig'))))
    output.mkdir(parents=True)
    figures=[];points=[]
    for row,raw in raw_inputs:
        commands=raw['timed_command_records']
        if not commands:
            continue
        origin=commands[0]['dispatch_host_s'];p=raw['configuration']['planning']
        target=[c for c in commands if c['target_used']]
        target_x=[c['dispatch_host_s']-origin for c in target]
        figure,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        a=axes[0,0]
        a.plot(target_x,[c['reference']['reference_time_s'] for c in target],'.',color='#2463ab',markersize=2)
        a.set(title='Plan elapsed reference at each target request',ylabel='Reference time (s)')
        a=axes[0,1]
        age=[c['observation_age_at_dispatch_s'] for c in target]
        total=[v+c['requested_duration_s']+p['execution_envelope']['freshness_reserve_s'] for v,c in zip(age,target)]
        a.plot(target_x,age,'.',color='#2463ab',markersize=2,label='Dispatch age proxy')
        a.plot(target_x,total,'.',color='#d6604d',markersize=2,label='Age + requested lease + reserve')
        a.axhline(raw['configuration']['visual']['frame_max_age_s'],color='#333333',linestyle='--',label='Age budget')
        a.set(title='Request lease audit; not physical actuation age',ylabel='Proxy time (s)');a.legend(fontsize=7)
        a=axes[1,0]
        a.plot([c['dispatch_host_s']-origin for c in commands],[np.linalg.norm(c['command_velocity_xy_m_s']) for c in commands],color='#2463ab',linewidth=.9,label='Requested XY norm')
        observations=[f['planning_input_body_state'] for f in raw['observations']]
        a.plot([f['host_monotonic_s']-origin for f in observations],[np.linalg.norm(f['velocity_ned_m_s']) for f in observations],color='#d6604d',linewidth=.9,label='Actual queried 3D speed')
        a.axhline(p['execution_speed_cap_m_s'],color='#333333',linestyle='--',label='Speed budget')
        a.set(title='Requests and saved actual-state queries',ylabel='Speed (m/s)');a.legend(fontsize=7)
        rate_x=[];rates=[]
        for index,c in enumerate(commands):
            dt=None if index==0 else c['dispatch_host_s']-commands[index-1]['dispatch_host_s']
            delta=None if index==0 else float(np.linalg.norm(np.array(c['command_velocity_xy_m_s'])-np.array(commands[index-1]['command_velocity_xy_m_s'])))
            rate=None if dt is None or dt<=0 else delta/dt
            if dt is not None and dt<0:
                raise ValueError('Request clocks regressed')
            if dt==0 and delta>1e-9:
                raise ValueError('Command changed without elapsed request time')
            if rate is not None:
                rate_x.append(c['dispatch_host_s']-origin);rates.append(rate)
            points.append(dict(condition_id=row['condition_id'],repetition=row['repetition'],policy=row['policy'],
                report_path=row['report_path'],report_sha256=row['report_sha256'],command_sequence=c['command_sequence'],
                target_used=c['target_used'],plan_id=c.get('plan_id'),frame_id=c.get('frame_id'),
                relative_host_s=c['dispatch_host_s']-origin,reference_time_s=(c.get('reference') or {}).get('reference_time_s'),
                dispatch_age_proxy_s=c.get('observation_age_at_dispatch_s'),requested_lease_s=c['requested_duration_s'],
                command_xy_norm_m_s=float(np.linalg.norm(c['command_velocity_xy_m_s'])),request_interval_s=dt,command_change_rate_m_s2=rate))
        a=axes[1,1];a.plot(rate_x,rates,'.',color='#2463ab',markersize=2)
        a.axhline(p['execution_ramp_acceleration_m_s2'],color='#333333',linestyle='--',label='Command change rate budget')
        a.set(title='Command change per actual request interval',ylabel='Command rate (m/s2)');a.legend(fontsize=7)
        for a in axes.flat:
            a.set_xlabel('Host seconds since first approach request');a.grid(alpha=.2)
        status='task met' if row['task_success'] else 'protected / task not met'
        figure.suptitle(f"{row['condition_id']}, repetition {row['repetition']}, {row['policy']}: {status}\nSaved request/query diagnostics; no continuous-motion guarantee",fontsize=11)
        stem=f"{row['condition_id']}_r{row['repetition']}_{row['policy']}"
        for extension in ('png','svg'):
            path=output/(stem+'.'+extension);figure.savefig(path,dpi=160)
            figures.append(dict(path=str(path.resolve()),sha256=digest(path)))
        plt.close(figure)
    csv_path=output/'request_plot_points.csv'
    if points:
        with csv_path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    result=dict(kind='TIME_INDEXED_REQUEST_QUERY_FIGURES',formal_thesis_effect_results=False,
        input_receipt_path=str(receipt_path.resolve()),input_receipt_sha256=digest(receipt_path),
        script_path=str(Path(__file__).resolve()),script_sha256=digest(__file__),figures=figures,
        plotted_requests=len(points),plotted_flights=len(raw_inputs),
        plot_points_sha256=digest(csv_path) if points else None,
        limits=['Each flight retains task failures; adjacent requests are not independent repeats',
                'The plots do not measure exposure, actuation duration, identity or true energy'],
        ai_assistance='Plot implementation assisted by AI; curves reproduce saved raw reports')
    (output/'plot_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),flights=len(raw_inputs),requests=len(points))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.receipt,args.output)
