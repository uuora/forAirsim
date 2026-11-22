"""Preload-only process startup, followed by explicitly submitted route jobs.

The control owner only queues requests and polls. No simulator API is imported
or used here; process startup and post-arrival planning have separate receipts.
"""
from copy import deepcopy
import hashlib,json,os,queue,subprocess,threading,time
from pathlib import Path
from .simulator import ROOT


def binding(request,output):
    return hashlib.sha256(json.dumps([request,str(Path(output).resolve())],sort_keys=True,
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def source_digest(sources):
    return hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PreparedPlannerClient:
    def __init__(self,folder,method,sources,*,worker_script=None):
        self.folder=Path(folder);self.method=method;self.sources=dict(sources)
        self.worker_script=Path(worker_script or ROOT/'python/prepared_route_planner.py').resolve()
        self.source_sha256=source_digest(self.sources);self.started=time.perf_counter()
        self.requests=queue.Queue();self.replies=queue.Queue();self.lock=threading.Lock()
        self.closed=threading.Event();self.process=None;self.ready=False
        self.submitted=0;self.completed=0;self.error=None;self.startup=None
        self.manifest=self.folder/'planning_worker_sources.json'
        self.manifest.write_text(json.dumps(self.sources,ensure_ascii=False,indent=2),encoding='utf-8')
        self.thread=threading.Thread(target=self._work,name='prepared-planner-io',daemon=True);self.thread.start()

    def submit(self,request,output,*,arrived_host_s):
        with self.lock:
            if self.closed.is_set():raise RuntimeError('规划工作进程已关闭')
            if getattr(self,'error',None):raise RuntimeError('规划预热已经失败：'+str(self.error))
            thread=getattr(self,'thread',None)
            if thread is not None and not thread.is_alive():raise RuntimeError('规划工作线程已经结束')
            if request.get('method')!=self.method:raise ValueError('规划方法与预热方法不一致')
            submitted=time.perf_counter()
            if submitted<arrived_host_s:raise ValueError('不得在准备点到达证据之前提交规划')
            self.submitted+=1
            job=dict(job_id=self.submitted,request=deepcopy(request),output=str(Path(output).resolve()),
                arrived_host_s=arrived_host_s,submitted_host_s=submitted,source_sha256=self.source_sha256)
            job['request_sha256']=binding(job['request'],job['output'])
            self.requests.put(job)
            return deepcopy(job)

    def poll(self):
        rows=[]
        while True:
            try:rows.append(self.replies.get_nowait())
            except queue.Empty:return rows

    def _record(self,row):
        with (self.folder/'prepared_worker_calls.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

    def _work(self):
        active=None
        try:
            env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(ROOT/'python'),
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (self.folder/'planner.log').open('w',encoding='utf-8') as log:
                self.process=subprocess.Popen([str(ROOT/'.venv/python.exe'),'-u',str(self.worker_script),
                    '--serve','--method',self.method,'--source-manifest',str(self.manifest)],cwd=ROOT,env=env,
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if self.closed.is_set():self.process.terminate()
                line=self.process.stdout.readline()
                if not line:raise RuntimeError('规划工作进程预热失败，详见planner.log')
                hello=json.loads(line)
                if hello.get('type')!='ready' or hello.get('source_sha256')!=self.source_sha256 or hello.get('method')!=self.method:
                    raise RuntimeError('规划预热回执未绑定预期方法/源码：'+str(hello))
                ready=time.perf_counter()
                self.startup=dict(started_host_s=self.started,ready_host_s=ready,process_to_ready_s=ready-self.started,
                    worker=hello.get('startup'),scope='Common imports/optimizer warmup only; no route search or objective calls')
                self.replies.put(dict(type='ready',startup=deepcopy(self.startup)));self.ready=True
                while not self.closed.is_set():
                    try:active=self.requests.get(timeout=.05)
                    except queue.Empty:continue
                    sent=time.perf_counter()
                    self.process.stdin.write(json.dumps(active,ensure_ascii=False,allow_nan=False)+'\n');self.process.stdin.flush()
                    line=self.process.stdout.readline()
                    if not line:raise RuntimeError('规划进程在任务中意外退出')
                    reply=json.loads(line);received=time.perf_counter()
                    if reply.get('job_id')!=active['job_id'] or reply.get('request_sha256')!=active['request_sha256'] or reply.get('source_sha256')!=self.source_sha256:
                        raise RuntimeError('规划回执未绑定预期任务/请求/源码')
                    if reply.get('type') not in ('result','error'):raise RuntimeError('未知规划回执')
                    if reply.get('type')=='result' and Path(reply.get('plan_path','')).resolve()!=Path(active['output'])/'airsim_replay_plan.json':
                        raise RuntimeError('规划返回路径与独立任务输出不一致')
                    timing=dict(submitted_host_s=active['submitted_host_s'],sent_host_s=sent,received_host_s=received,
                        queue_wait_s=sent-active['submitted_host_s'],request_to_receive_s=received-active['submitted_host_s'],
                        arrival_to_receive_s=received-active['arrived_host_s'],
                        cold_start_to_receive_s=received-self.started,
                        startup_overlap_before_arrival_s=max(0.,min(self.startup['ready_host_s'],active['arrived_host_s'])-self.started),
                        startup_remaining_at_submit_s=max(0.,self.startup['ready_host_s']-active['submitted_host_s']))
                    reply['parent_timings']=timing
                    self._record(dict(job_id=active['job_id'],request_sha256=active['request_sha256'],
                        source_sha256=self.source_sha256,outcome=reply['type'],timings=timing,worker=reply.get('timings'),error=reply.get('error')))
                    self.completed+=1;self.replies.put(reply);active=None
        except Exception as ex:
            with self.lock:self.error=repr(ex)
            reply=dict(type='error',error=str(ex),job_id=None if active is None else active['job_id'])
            if active is not None:
                try:
                    self._record(dict(job_id=active['job_id'],request_sha256=active['request_sha256'],
                        outcome='aborted' if self.closed.is_set() else 'error',error=repr(ex),
                        request_to_terminal_s=time.perf_counter()-active['submitted_host_s']))
                    self.completed+=1
                except Exception as writing:self.error=repr(writing)+'; prior='+self.error
            self.replies.put(reply)
        finally:
            self.ready=False
            if self.process is not None:
                if self.process.poll() is None:self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
                for stream in (self.process.stdin,self.process.stdout):
                    if stream:stream.close()
            while True:
                try:job=self.requests.get_nowait()
                except queue.Empty:break
                try:
                    self._record(dict(job_id=job['job_id'],outcome='aborted',error='Closed before processing',
                        request_sha256=job['request_sha256'],request_to_terminal_s=time.perf_counter()-job['submitted_host_s']))
                    self.completed+=1
                except Exception as writing:self.error=repr(writing)
                self.replies.put(dict(type='aborted',job_id=job['job_id'],error='Closed before processing'))

    def cancel(self):
        """Stop work without waiting in a control/landing transition."""
        with self.lock:self.closed.set()
        if self.process is not None and self.process.poll() is None:self.process.terminate()

    def close(self):
        self.cancel();self.thread.join(timeout=5.)
        receipt=dict(closed=not self.thread.is_alive(),submitted_jobs=self.submitted,terminal_receipts=self.completed,
            startup=self.startup,error=self.error,close_host_s=time.perf_counter())
        if self.thread.is_alive() or self.completed!=self.submitted:raise RuntimeError('规划线程/逐任务回执未完整关闭：'+str(receipt))
        return receipt
