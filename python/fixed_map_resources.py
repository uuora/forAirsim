"""Read-only host memory headroom, including Windows commit availability."""
import ctypes,os,json,threading,time
from pathlib import Path
import psutil

def memory_headroom():
    values=dict(physical_available_bytes=psutil.virtual_memory().available)
    if os.name=='nt':
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
                ('total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','available_extended')]
        s=Status();s.length=ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise OSError('memory status unavailable')
        values.update(commit_available_bytes=s.available_commit,commit_total_bytes=s.total_commit)
    else:values['commit_available_bytes']=psutil.virtual_memory().available+psutil.swap_memory().free
    return values


class ResourceTrace:
    """One bounded-rate observer of this trial's named process identities."""
    def __init__(self,path,context,processes,period_s=1.):
        if not .5<=period_s<=5.:raise ValueError('RESOURCE_TRACE_PERIOD_OUTSIDE_BOUNDS')
        self.path=Path(path);self.context=context;self.processes=processes;self.period=period_s
        self.done=threading.Event();self.errors=[];self.samples=0;self.identities={}
    def sample(self):
        row=dict(host_monotonic_s=time.monotonic(),host_time_ns=time.time_ns(),memory=memory_headroom(),context=self.context(),processes={})
        for role,pid in self.processes().items():
            if pid is None:continue
            try:
                process=psutil.Process(pid);identity=(pid,process.create_time())
                if role in self.identities and self.identities[role]!=identity:raise ValueError('RESOURCE_PROCESS_IDENTITY_CHANGED_'+role)
                self.identities[role]=identity
                with process.oneshot():
                    memory=process.memory_info()
                    row['processes'][role]=dict(pid=pid,created=identity[1],rss_bytes=memory.rss,private_bytes=getattr(memory,'private',None),threads=process.num_threads(),cpu_times_s=sum(process.cpu_times()[:2]))
            except psutil.NoSuchProcess:row['processes'][role]=dict(pid=pid,status='EXITED')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        self.samples+=1
        return row
    def _run(self):
        try:
            while not self.done.is_set():self.sample();self.done.wait(self.period)
        except Exception as ex:self.errors.append(str(ex))
    def start(self):
        self.thread=threading.Thread(target=self._run,daemon=True,name='trial-resource-observer');self.thread.start()
    def stop(self):
        self.done.set();self.thread.join(timeout=2)
        if self.thread.is_alive():self.errors.append('RESOURCE_TRACE_JOIN_TIMEOUT')
        return dict(status='PASS' if self.samples and not self.errors else 'FAIL',samples=self.samples,errors=self.errors,path=str(self.path),period_s=self.period,scope='Observer only; no process termination or resource gate relaxation')
