"""Export measured simulator evidence and calculated metrics, never synthesize flight samples."""
import csv,json,math
import shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from fixed_map_mission import OUT,read,save,Geometry,sha

def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def analyze():
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    task=read(OUT/'fixed_version/task.json');parts=read(OUT/'fixed_version/editor_audit/planning_geometry.json')['primitives'];geometry=Geometry(parts)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    figures,axes=plt.subplots(2,1,figsize=(9,9),constrained_layout=True);series=[];summaries=[]
    runs=sorted(OUT.glob('run_*'))
    arrivals=[r for r in runs if (r/'result_arrived.json').exists() and not (r/'result_aborted.json').exists()]
    aborts=[r for r in runs if (r/'result_aborted.json').exists() and not (r/'result_arrived.json').exists()]
    primary_arrival=arrivals[0] if arrivals else None;primary_abort=aborts[-1] if aborts else None
    save(dest/'primary_evidence_selection.json',dict(arrival=str(primary_arrival),abort=str(primary_abort),
        rule='First completed arrival and latest moving abort are shown; all other runs remain in CSV and summary. No efficacy statistics inferred.'))
    for run in sorted(OUT.glob('run_*')):
        log=run/'timestamped.jsonl'
        if not log.exists():continue
        records=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
        states=[r for r in records if r['kind']=='state']
        if not states:continue
        stem=run.name;states=sorted(states,key=lambda r:r['host_monotonic_s']);base=states[0]['state_timestamp_ns'];host=states[0]['host_monotonic_s']
        points=np.array([r['position_xyz_m'] for r in states]);distance=geometry.distances(points)
        rows=[];clearance=[]
        plan=read(run/'plan.json')['path'] if (run/'plan.json').exists() else []
        def cross_track(p):
            best=1e9
            for a,b in zip(plan,plan[1:]):
                a=np.array(a);d=np.array(b)-a;t=np.clip(np.dot(p-a,d)/np.dot(d,d),0,1);best=min(best,float(np.linalg.norm(p-(a+t*d))))
            return best
        for i,r in enumerate(states):
            p=r['position_xyz_m'];v=r['velocity_xyz_m_s'];q=r['orientation_xyzw'];cmd=r['command_xyz_m_s'];b=r['DroneB_position_xyz_m']
            elapsed=(r['state_timestamp_ns']-base)/1e9
            row=dict(utc_measured=r['utc'],local_time_UTC8=datetime.fromisoformat(r['utc']).astimezone(timezone(timedelta(hours=8))).isoformat(),
                simulator_timestamp_ns_measured=r['state_timestamp_ns'],sim_elapsed_s_calculated=elapsed,host_elapsed_s_calculated=r['host_monotonic_s']-host,
                stage=r['stage'],x_m_measured=p[0],y_m_measured=p[1],z_m_measured=p[2],vx_m_s_measured=v[0],vy_m_s_measured=v[1],vz_m_s_measured=v[2],
                qx_measured=q[0],qy_measured=q[1],qz_measured=q[2],qw_measured=q[3],speed_m_s_measured=r['speed_m_s'],
                goal_error_m_calculated=r['goal_error_m'],cross_track_m_calculated=cross_track(p),
                sphere_envelope_clearance_m_calculated=r['envelope_clearance_calculated_m'],nearest_actor=r['nearest_obstacle'],
                command_vx_m_s_configured=cmd[0],command_vy_m_s_configured=cmd[1],command_vz_m_s_configured=cmd[2],
                envelope_radius_m_configured=task['envelope_radius_config_m'],clearance_threshold_m_configured=task['clearance_stop_m'],
                speed_limit_m_s_configured=task['max_speed_measured_m_s'],arrival_error_m_configured=task['arrival_error_m'],
                DroneB_x_m_measured=b[0],DroneB_y_m_measured=b[1],DroneB_z_m_measured=b[2],DroneB_speed_m_s_measured=r['DroneB_speed_m_s'],
                DroneB_landed_measured=r['DroneB_landed'],DroneB_api_control_measured=r['DroneB_api_control'],stop_reason=r['stop_reason'])
            rows.append(row)
            clearance.append(dict(simulator_timestamp_ns_measured=r['state_timestamp_ns'],sim_elapsed_s_calculated=elapsed,stage=r['stage'],
                                  **{p['id']+'_sphere_clearance_m_calculated':float(distance[i,j]) for j,p in enumerate(parts)}))
        csv_write(dest/(stem+'_telemetry_origin.csv'),rows);csv_write(dest/(stem+'_all_obstacle_clearance_origin.csv'),clearance)
        if plan:csv_write(dest/(stem+'_planned_path_origin.csv'),[dict(waypoint_index_configured=i,x_m_planned=p[0],y_m_planned=p[1],z_m_planned=p[2]) for i,p in enumerate(plan)])
        events=[r for r in records if r['kind']!='state'];save(dest/(stem+'_events.json'),events)
        stops=[r for r in events if r['kind']=='INTERFACE_REQUEST' and r.get('request',{}).get('action')=='stop']
        stop=stops[0] if stops else None
        terminal=next((r for r in events if r['kind']=='HOVER_VERIFIED' and (not stop or (r['stage']=='ABORTED' and r['host_monotonic_s']>=stop['host_monotonic_s']))),None)
        approach=[r for r in states if r['stage'] in ('APPROACH','ARRIVAL_HOVER') and (not terminal or r['host_monotonic_s']<=terminal['host_monotonic_s'])]
        motion=[r for r in states if r['stage'] in ('TAKEOFF','APPROACH','RETURN','LANDING')]
        summary=dict(run=str(run),samples=len(states),log_sha256=sha(log),first_utc=states[0]['utc'],last_utc=states[-1]['utc'],
            final_record_stage=states[-1]['stage'],max_measured_speed_m_s=max(r['speed_m_s'] for r in states),
            max_flight_state_gap_host_s=max((b['host_monotonic_s']-a['host_monotonic_s'] for a,b in zip(states,states[1:]) if a['stage'] in ('TAKEOFF','APPROACH','RETURN','LANDING')),default=0),
            min_approach_sphere_envelope_clearance_m=min((r['envelope_clearance_calculated_m'] for r in approach),default=None),
            max_approach_cross_track_m=max((cross_track(r['position_xyz_m']) for r in approach),default=None),
            DroneB_ground_invariant_pass=all(r['DroneB_landed'] and not r['DroneB_api_control'] and r['DroneB_speed_m_s']<.01 for r in states),
            recorded_results=[read(p)['status'] for p in run.glob('result_*.json')],
            expected_support_contact_exception='negative spherical clearance permitted only during takeoff/landing on audited start platform')
        if terminal:
            last=[r for r in states if r['host_monotonic_s']<=terminal['host_monotonic_s']][-1];end=last['state_timestamp_ns']
            # Reconstruct the whole contiguous qualifying interval; cutting at
            # end-3s would discard the boundary sample that proves >=3s.
            prior=[r for r in states if r['host_monotonic_s']<=terminal['host_monotonic_s']]
            dwell=[];expected='ARRIVAL_HOVER' if terminal['stage']=='ARRIVED' else 'STOP_HOVER'
            for r in reversed(prior):
                good=r['stage']==expected and r['speed_m_s']<=task['arrival_speed_m_s'] and (terminal['stage']!='ARRIVED' or r['goal_error_m']<=task['arrival_error_m'])
                if not good:break
                dwell.append(r)
            dwell.reverse()
            if not dwell:dwell=[last]
            summary.update(terminal_status=terminal['stage'],terminal_goal_error_m=last['goal_error_m'],terminal_speed_m_s=last['speed_m_s'],
                hover_dwell_sampled_span_sim_s=(dwell[-1]['state_timestamp_ns']-dwell[0]['state_timestamp_ns'])/1e9,
                hover_max_speed_m_s=max(r['speed_m_s'] for r in dwell),hover_max_goal_error_m=max(r['goal_error_m'] for r in dwell),
                hover_max_displacement_from_first_sample_m=max(float(np.linalg.norm(np.array(r['position_xyz_m'])-dwell[0]['position_xyz_m'])) for r in dwell))
        if stop:
            following=[r for r in states if r['host_monotonic_s']>=stop['host_monotonic_s']];stable=next((r for r in following if r['speed_m_s']<task['arrival_speed_m_s']),None)
            at=following[0] if following else None
            summary.update(stop_request=stop['request'],stop_to_speed_threshold_host_s=(stable['host_monotonic_s']-stop['host_monotonic_s']) if stable else None,
                stop_request_phase=stop['stage'],stop_first_post_request_measured_speed_m_s=at['speed_m_s'] if at else None,
                stop_to_terminal_host_s=(terminal['host_monotonic_s']-stop['host_monotonic_s']) if terminal else None,
                stop_braking_displacement_m=float(np.linalg.norm(np.array(stable['position_xyz_m'])-at['position_xyz_m'])) if stable and at else None)
        summaries.append(summary)
        label='abort' if stop and stop['stage'] in ('TAKEOFF','APPROACH','RETURN') and 'ABORTED' in summary['recorded_results'] else 'arrival' if not stop and 'ARRIVED' in summary['recorded_results'] else 'additional hover check'
        if label=='additional hover check':continue
        if (label=='arrival' and run!=primary_arrival) or (label=='abort' and run!=primary_abort):continue
        for suffix in ('telemetry_origin.csv','all_obstacle_clearance_origin.csv','planned_path_origin.csv'):
            source=dest/(stem+'_'+suffix)
            if source.exists():shutil.copyfile(source,dest/(label+'_'+suffix))
        if label=='failed attempt':continue
        end_time=terminal['host_monotonic_s'] if terminal else states[-1]['host_monotonic_s']
        main=[r for r in states if r['host_monotonic_s']<=end_time];series.append((label,main))
        ax=axes[0 if label=='arrival' else 1];ax.set_title(label.capitalize()+': planned path and measured trajectory')
        for part in parts:
            if part['group']=='ground':continue
            c=np.array(part['center_xyz_m'])
            if part['shape']=='sphere':ax.add_patch(Circle(c[:2],part['radius_m'],color='#bb8888',alpha=.6))
            else:
                a=np.array(part['orientation_axes']);h=np.array(part['local_size_xyz_m'])/2
                pts=[(c+s*a[0]*h[0]+t*a[1]*h[1])[:2] for s,t in [(-1,-1),(1,-1),(1,1),(-1,1)]]
                ax.add_patch(Polygon(pts,facecolor='#dddddd',edgecolor='#999999',lw=.4))
        planned=np.array(plan);actual=np.array([r['position_xyz_m'] for r in main])
        ax.plot(planned[:,0],planned[:,1],'--',color='#315e91',lw=1.5,label='Planned / calculated')
        ax.plot(actual[:,0],actual[:,1],color='#17805e',lw=1.5,label='Actual / measured')
        ax.scatter(actual[-1,0],actual[-1,1],c='#17805e',s=30);ax.set(xlim=(-6,54),ylim=(-18,18),xlabel='World X / m',ylabel='World Y / m');ax.set_aspect('equal');ax.legend(loc='upper left');ax.grid(alpha=.15)
    if series:
        for ext in ('svg','pdf','png'):figures.savefig(dest/('plan_and_actual.'+ext),dpi=220)
        plt.close(figures)
        fig,axs=plt.subplots(4,1,figsize=(9,10),sharex=True,constrained_layout=True)
        for label,states in series:
            t=np.array([(r['state_timestamp_ns']-states[0]['state_timestamp_ns'])/1e9 for r in states])
            for ax,key in zip(axs,('goal_error_m','speed_m_s','envelope_clearance_calculated_m')):ax.plot(t,[r[key] for r in states],label=label,lw=1.3)
            axs[3].plot(t,[r['position_xyz_m'][2] for r in states],label=label,lw=1.3)
        for ax,title in zip(axs,('Goal error / m\n(calculated)','Speed / (m/s)\n(measured)','Envelope clearance / m\n(calculated)','World Z / m\n(measured)')):ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend()
        axs[0].axhline(task['arrival_error_m'],color='#ad4848',ls='--',label='Configured arrival threshold')
        axs[1].axhline(task['max_speed_measured_m_s'],color='#ad4848',ls='--')
        axs[2].axhline(task['clearance_stop_m'],color='#ad4848',ls='--')
        axs[3].axhline(task['height_world_m'],color='#ad4848',ls='--');axs[-1].set_xlabel('Elapsed simulator time / s')
        for ext in ('svg','pdf','png'):fig.savefig(dest/('error_speed_clearance_height.'+ext),dpi=220)
        plt.close(fig)
    save(dest/'summary.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),runs=summaries,units='metres and seconds',
        limits=['Simulator observations, not hardware measurements; no statistical efficacy inference from these trials.',
                'Clearance is calculated from measured AirSim pose, audited OBB/sphere geometry and an orientation invariant enclosing sphere.',
                'A negative conservative sphere clearance at ground support does not imply mesh penetration.',
                'Trace curves contain only measured samples. No interpolation is used as evidence.'],
        ai_disclosure='AI implemented the executor, performed UI start/return operations and analyzed logs; human stop is separately identified by user confirmation.'))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':analyze()
