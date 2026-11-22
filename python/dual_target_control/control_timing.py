"""Bounded high-resolution waits and loss-checked asynchronous JSON journals."""
import ctypes,json,os,queue,threading,time
from contextlib import contextmanager
from pathlib import Path


class TimingFrame:
    """Host wall time, including a phase that exits by raising an exception.

    Phases are non-overlapping unless explicitly named as a containing total.
    This is diagnostic instrumentation, not a real-time scheduling guarantee.
    """
    def __init__(self):
        self.started=time.perf_counter();self.timings={};self.failed_stage=None

    @contextmanager
    def measure(self,name):
        started=time.perf_counter()
        try:yield
        except BaseException:
            self.failed_stage=name
            raise
        finally:self.timings[name]=self.timings.get(name,0.)+1000*(time.perf_counter()-started)

    def receipt(self):
        ended=time.perf_counter()
        return dict(host_begin_s=self.started,host_end_s=ended,tick_wall_ms=1000*(ended-self.started),
            timing_ms=dict(self.timings),failed_stage=self.failed_stage)


def diagnostic_value(value):
    """Keep a failed non-finite observation serializable without inventing zeros."""
    import math
    if isinstance(value,dict):return {k:diagnostic_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [diagnostic_value(v) for v in value]
    if hasattr(value,'tolist'):return diagnostic_value(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


class DeadlineWaiter:
    def __init__(self):
        self.handle=None;self.backend='time.sleep';self.error=None
        if os.name=='nt':
            from ctypes import wintypes
            api=ctypes.WinDLL('kernel32',use_last_error=True)
            api.CreateWaitableTimerExW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD];api.CreateWaitableTimerExW.restype=wintypes.HANDLE
            api.SetWaitableTimer.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_longlong),wintypes.LONG,ctypes.c_void_p,ctypes.c_void_p,wintypes.BOOL];api.SetWaitableTimer.restype=wintypes.BOOL
            api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];api.WaitForSingleObject.restype=wintypes.DWORD
            api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
            self.handle=api.CreateWaitableTimerExW(None,None,2,0x1F0003);self.api=api
            if self.handle:self.backend='win32-high-resolution-waitable-timer'
            else:self.error=ctypes.get_last_error()

    def wait(self,seconds):
        if seconds<=0:return
        if self.handle:
            due=ctypes.c_longlong(-max(1,round(seconds*10_000_000)))
            if not self.api.SetWaitableTimer(self.handle,ctypes.byref(due),0,None,None,False):raise ctypes.WinError(ctypes.get_last_error())
            if self.api.WaitForSingleObject(self.handle,max(1000,int(seconds*1000)+100))!=0:raise RuntimeError('高精度等待失败')
        else:time.sleep(seconds)

    def close(self):
        if self.handle:self.api.CloseHandle(self.handle);self.handle=None


class AsyncJournal:
    def __init__(self,folder):
        self.folder=Path(folder);self.items=queue.SimpleQueue();self.error=None;self.closed=False
        self.counts={'states':0,'events':0};self.submitted={'states':0,'events':0}
        self.lock=threading.Lock();self.thread=threading.Thread(target=self._work,name='online-journal',daemon=True);self.thread.start()

    def append(self,channel,row):
        if self.closed:raise RuntimeError('日志已关闭')
        if self.error:raise RuntimeError('后台日志失败：'+self.error)
        with self.lock:self.submitted[channel]+=1;self.items.put((channel,row))

    def _work(self):
        try:
            with (self.folder/'online_states.jsonl').open('w',encoding='utf-8') as states,(self.folder/'online_events.jsonl').open('w',encoding='utf-8') as events:
                streams={'states':states,'events':events}
                while True:
                    item=self.items.get()
                    if item is None:break
                    channel,row=item;streams[channel].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.counts[channel]+=1
        except Exception as ex:self.error=repr(ex)

    def close(self):
        if not self.closed:self.closed=True;self.items.put(None)
        self.thread.join(timeout=15)
        if self.thread.is_alive() or self.error or self.counts!=self.submitted:raise RuntimeError('后台日志未完整写入：'+str(self.error or self.counts))
        return dict(submitted=self.submitted,written=self.counts,flushed=True)


def sample_display(windows,t,end):
    import numpy as np
    times=np.linspace(t,end,180);points=np.empty((len(times),3));starts=np.array([w[0] for w in windows])
    indices=np.searchsorted(starts,times,side='right')-1
    for i in np.unique(indices):
        selected=indices==i;start,curve=windows[i];points[selected]=curve.sample(np.clip(times[selected]-start,0.,curve.duration_s))
    return points.tolist()
