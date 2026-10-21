"""Separate perception process with atomic bounded latest evidence publication."""
import argparse,copy,json,os,subprocess,sys,threading,time
from pathlib import Path
from fixed_map_cpu_runtime import configure_from_argv
if __name__=='__main__':configure_from_argv()
from fixed_map_perception import PerceptionService,save,read,ROOT,utc

class PerceptionProcess:
    def __init__(self,output,config):
        self.output=Path(output).resolve();self.cfg=copy.deepcopy(config);self.latest={};self.done=threading.Event();self.model_ready=threading.Event();self.warm_ready=threading.Event();self.process_count=0
        self.config_path=self.output.parent/(self.output.name+'_config.json');save(self.config_path,self.cfg);self.publication_metrics=[]
        from fixed_map_perception_execution import static_geometry_v2
        from fixed_map_mission import Geometry
        parts,_=static_geometry_v2(read(ROOT/'reports/fixed_map_flight_20261009/fixed_version/editor_audit/planning_geometry.json')['primitives'])
        self.geometry=Geometry(parts,.75)
    def start(self):
        from fixed_map_resources import memory_headroom
        budget=memory_headroom();save(self.output.parent/(self.output.name+'_prelaunch_memory.json'),budget)
        if self.cfg.get('capture_after_model_load') and min(budget['physical_available_bytes'],budget['commit_available_bytes'])<3*2**30:raise ValueError('MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB')
        log=self.output.parent/(self.output.name+'_worker.log');self.stream=log.open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker',str(self.output),'--config',str(self.config_path)],cwd=str(ROOT),stdout=self.stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.threads=[threading.Thread(target=self._poll,daemon=True,name='perception-publication-poll')];self.threads[0].start()
    def _poll(self):
        last_version=None
        while not self.done.is_set():
            try:
                path=self.output/'publication.json';version=path.stat().st_mtime_ns
                if version==last_version:self.done.wait(.01);continue
                s=read(path);last_version=version;self.latest=s['latest'];self.process_count=s['processed_count']
                if self.latest.get('capture'):
                    now=time.monotonic();metric=dict(frame_id=self.latest.get('frame_id'),received_by_parent_monotonic_s=now,published_monotonic_s=s.get('published_monotonic_s'),source_age_on_parent_receipt_s=now-self.latest['capture']['received_monotonic_s'])
                    self.publication_metrics.append(metric);self.last_publication_metric=metric
                    if self.cfg.get('parent_publication_trace_v18'):
                        with (self.output.parent/(self.output.name+'_publication_trace.jsonl')).open('a',encoding='utf-8') as handle:handle.write(json.dumps(metric)+'\n')
                if s['model_ready']:self.model_ready.set()
                if s['warm_ready']:self.warm_ready.set()
            except (OSError,ValueError):pass
            self.done.wait(.01)
            if self.process.poll() is not None:break
    def stop(self,wait=True):
        if self.output.exists():(self.output/'stop.request').write_text('stop',encoding='utf-8')
        if wait:
            started=time.monotonic();status='WAITING';error=None
            try:
                code=self.process.wait(timeout=15)
                if code!=0:raise RuntimeError('PERCEPTION_PROCESS_EXIT_NONZERO_'+str(code))
                status='PASS'
            except subprocess.TimeoutExpired:
                status='TIMEOUT';error='PERCEPTION_PROCESS_STOP_TIMEOUT'
                raise RuntimeError(error) from None
            except Exception as ex:
                status='ERROR';error=str(ex);raise
            finally:
                self.done.set()
                for thread in self.threads:thread.join(timeout=2)
                if self.process.poll() is not None:self.stream.close()
                save(self.output.parent/(self.output.name+'_publication_metrics.json'),list(self.publication_metrics))
                self.stop_receipt=dict(utc=utc(),status=status,error=error,elapsed_host_s=time.monotonic()-started,exit_code=self.process.poll(),pid=self.process.pid,termination_requested=False)
                save(self.output.parent/(self.output.name+'_stop_receipt.json'),self.stop_receipt)
            return self.stop_receipt
    def reap_after_ground_release(self,ground_probe):
        """Reap only this sensor child, after a new two-vehicle ground check."""
        receipt=ground_probe()
        vehicles=receipt.get('vehicles',{})
        if not receipt.get('passed') or set(vehicles)!={'DroneA','DroneB'} or not all(v.get('landed') is True and v.get('api_control') is False and v.get('fresh') is True for v in vehicles.values()):raise ValueError('SENSOR_REAP_REQUIRES_FRESH_RELEASED_GROUND')
        actions=[]
        if self.process.poll() is None:
            self.process.terminate();actions.append('terminate_sensor_child')
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill();actions.append('kill_sensor_child');self.process.wait(timeout=3)
        self.done.set()
        for thread in self.threads:thread.join(timeout=2)
        self.stream.close()
        result=dict(utc=utc(),actions=actions,exit_code=self.process.poll(),ground_receipt=receipt,pid=self.process.pid,scope='sensor child only; no control owner or UE process terminated')
        save(self.output.parent/(self.output.name+'_reap_receipt.json'),result)
        return result

def worker(output,config):
    publication_lock=threading.Lock()
    def publish(service,closed=False):
        with publication_lock:save(service.output/'publication.json',dict(model_ready=service.model_ready.is_set(),warm_ready=service.warm_ready.is_set(),processed_count=service.process_count,latest=service.latest,published_monotonic_s=time.monotonic(),publication_pipeline_version='direct_frame_publish_v4',closed=closed))
    service=PerceptionService(output,read(config),publication_callback=publish);service.start()
    try:
        while not (service.output/'stop.request').exists():
            if service.process_count==0:publish(service)
            if not service.threads[1].is_alive():break
            time.sleep(.1)
    finally:
        service.stop(wait=True);publish(service,closed=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--config',required=True);a=p.parse_args();worker(a.worker,a.config)
