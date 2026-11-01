"""Owned local planner service client, serialized independently of Tk widgets."""
import hashlib,json,os,subprocess,threading,time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class PlannerClient:
    def __init__(self):
        self.process=None;self.lock=threading.Lock()

    def _start(self):
        if self.process is not None and self.process.poll() is None:return
        log=ROOT/'.runtime/curve_airsim_panel'/f'planner_service_{datetime.now():%Y%m%d_%H%M%S_%f}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(ROOT/'python'))
        with log.open('wb') as stream:
            self.process=subprocess.Popen([str(ROOT/'.venv/python.exe'),str(ROOT/'python/control_target_planner_service.py')],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stream,text=True,encoding='utf-8',bufsize=1,
                cwd=ROOT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        response=self.process.stdout.readline()
        if not response or json.loads(response).get('status')!='READY':raise RuntimeError('planner service did not become ready; '+str(log))

    def generate(self,output,seed,goal):
        return self._generate_request(output,dict(action='generate',output=str(Path(output).resolve()),seed=seed,goal=goal))

    def generate_route(self,output,seed,points,policy,*,pass_speed=1.2,dwell_s=1.,time_scale=1.25,max_speed=6.,max_slew=6.):
        return self._generate_request(output,dict(action='generate_route',output=str(Path(output).resolve()),seed=seed,
            points=points,policy=policy,pass_speed=pass_speed,dwell_s=dwell_s,time_scale=time_scale,max_speed=max_speed,max_slew=max_slew))

    def _generate_request(self,output,request):
        with self.lock:
            started=time.perf_counter();self._start()
            for attempt in range(2):
                self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
                line=self.process.stdout.readline()
                if not line:raise RuntimeError('planner service exited before returning a saved result')
                response=json.loads(line)
                if response['status']=='STALE_SOURCE' and attempt==0:
                    self.close();self._start();continue
                if response['status']!='PASS':raise RuntimeError(response.get('error','planner failed'))
                break
            response['parent_total_planning_wall_s']=time.perf_counter()-started
            response['parent_startup_ipc_and_other_s']=max(0.,response['parent_total_planning_wall_s']-response['request_generation_wall_s'])
            response['client_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            (Path(output)/'planning_service_client_receipt.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
            return response

    def close(self):
        process=self.process;self.process=None
        if process is not None and process.poll() is None:
            try:
                process.stdin.write(json.dumps(dict(action='quit'))+'\n');process.stdin.flush();process.stdin.close()
            except (BrokenPipeError,OSError,ValueError):pass
        # No termination: an active request can finish saving its package.
