"""Nonblocking client: control thread only queues/polls, never waits for IPC."""
import json,os,queue,subprocess,threading,time
from .simulator import ROOT


class PlannerClient:
    def __init__(self,log_dir):
        self.requests=queue.Queue();self.replies=queue.Queue();self.closed=threading.Event();self.ready=False;self.sent=0
        self.log_dir=log_dir;self.process=None
        self.thread=threading.Thread(target=self._work,name='online-planner-io',daemon=True);self.thread.start()

    def submit(self,request):
        if not self.ready:raise RuntimeError('planner not ready')
        self.sent+=1;self.requests.put(request)

    def poll(self):
        results=[]
        while True:
            try:results.append(self.replies.get_nowait())
            except queue.Empty:return results

    def _work(self):
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        try:
            with (self.log_dir/'online_worker_stderr.log').open('w',encoding='utf-8') as log:
                self.process=subprocess.Popen([str(ROOT/'.venv/python.exe'),'-u',str(ROOT/'python/online_curve_worker.py')],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                hello=self.process.stdout.readline()
                if not hello:raise RuntimeError('规划进程启动失败，详见 online_worker_stderr.log')
                reply=json.loads(hello);self.ready=reply.get('type')=='ready';self.replies.put(reply)
                while not self.closed.is_set():
                    try:request=self.requests.get(timeout=.05)
                    except queue.Empty:continue
                    self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
                    line=self.process.stdout.readline()
                    if not line:raise RuntimeError('规划进程意外退出')
                    self.replies.put(json.loads(line))
        except Exception as exc:self.replies.put(dict(type='error',error=str(exc)))
        finally:
            self.ready=False
            if self.process is not None:
                if self.process.poll() is None:self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
                for stream in (self.process.stdin,self.process.stdout):
                    if stream:stream.close()

    def close(self):
        self.closed.set()
        if self.process is not None and self.process.poll() is None:self.process.terminate()
        # Windows may retain the stderr file until the I/O thread finishes its
        # process wait and closes all streams. Do not return a false closure.
        self.thread.join(timeout=5.)
        if self.thread.is_alive():raise RuntimeError('规划进程I/O线程未完成关闭')
