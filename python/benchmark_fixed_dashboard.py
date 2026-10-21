"""Read-only ground resource sample; never represents an airborne stress test."""
import argparse,json,subprocess,threading,time,statistics
import numpy as np,psutil,airsim
from fixed_map_mission import OUT,read,save,utc,assert_fixed
from fixed_map_resources import memory_headroom

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--duration',type=float,default=16);args=ap.parse_args()
    assert_fixed();c=airsim.MultirotorClient(port=41454,timeout_value=3)
    engine=psutil.Process(read(OUT/'active_engine.json')['pid']);engine.cpu_percent()
    rows=[];gpu=[];done=threading.Event();psutil.cpu_percent()
    def gpu_worker():
        while not done.is_set():
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                values=[float(x.strip()) for x in p.stdout.splitlines()[0].split(',')]
                gpu.append(dict(utc=utc(),gpu_util_percent=values[0],gpu_memory_used_mib=values[1],gpu_memory_total_mib=values[2]))
            except Exception as e:gpu.append(dict(utc=utc(),unavailable=str(e)))
            done.wait(2)
    thread=threading.Thread(target=gpu_worker,daemon=True);thread.start()
    start=time.monotonic();previous=None
    while time.monotonic()-start<args.duration:
        tick=time.monotonic();s=c.getMultirotorState(vehicle_name='DroneA');b=c.getMultirotorState(vehicle_name='DroneB');latency=time.monotonic()-tick
        assert s.landed_state==b.landed_state==airsim.LandedState.Landed
        row=dict(utc=utc(),host_monotonic_s=tick,state_timestamp_ns=s.timestamp,read_pair_latency_s=latency,
            host_sample_gap_s=tick-previous if previous else None,system_cpu_percent=psutil.cpu_percent(),
            engine_cpu_percent_one_core_units=engine.cpu_percent(),engine_rss_mib=engine.memory_info().rss/2**20,
            system_memory_percent=psutil.virtual_memory().percent,DroneA_speed_m_s=float(np.linalg.norm([s.kinematics_estimated.linear_velocity.x_val,s.kinematics_estimated.linear_velocity.y_val,s.kinematics_estimated.linear_velocity.z_val])),
            DroneB_landed=True)
        row.update(memory_headroom());rows.append(row);previous=tick;time.sleep(max(0,.1-(time.monotonic()-tick)))
    done.set();thread.join(timeout=4)
    summary=dict(utc=utc(),label=args.label,scope='ground read-only; 0.1s requested observation interval; not airborne control robustness',duration_host_s=time.monotonic()-start,
        sample_count=len(rows),read_pair_latency_median_s=statistics.median(r['read_pair_latency_s'] for r in rows),
        read_pair_latency_max_s=max(r['read_pair_latency_s'] for r in rows),sample_gap_max_s=max(r['host_sample_gap_s'] or 0 for r in rows),
        system_cpu_mean_percent=statistics.mean(r['system_cpu_percent'] for r in rows),
        engine_cpu_mean_one_core_units=statistics.mean(r['engine_cpu_percent_one_core_units'] for r in rows),
        engine_rss_max_mib=max(r['engine_rss_mib'] for r in rows),
        gpu_observations=gpu,rows=rows,flight_commands_issued=False)
    save(OUT/'interface_revision_20261009'/('resources_'+args.label+'.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','gpu_observations')},ensure_ascii=False))

if __name__=='__main__':main()
