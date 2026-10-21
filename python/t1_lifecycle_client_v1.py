"""Verified long vertical lifecycle tasks, separate from the control child.

msgpackrpc.Future.join waits but does not raise stored errors. get does.
The 60s client timeout exceeds the declared 15/40/45s AirSim vertical task
timeouts; current-image and independent child RPC clients are unchanged.
"""
import time

class VerifiedLifecycleClient:
    def __init__(self,client,log):self.client=client;self.log=log
    def __getattr__(self,name):
        method=getattr(self.client,name)
        if name not in ('takeoffAsync','moveToZAsync','landAsync'):return method
        def request(*args,**kwargs):
            started=time.monotonic();future=method(*args,**kwargs);log=self.log
            class VerifiedTask:
                def join(self):
                    try:
                        result=future.get()
                    except Exception as e:
                        log(dict(kind='LIFECYCLE_RPC_RESULT',method=name,started_monotonic_s=started,
                            completed_monotonic_s=time.monotonic(),status='ERROR',exception_type=type(e).__name__))
                        raise
                    log(dict(kind='LIFECYCLE_RPC_RESULT',method=name,started_monotonic_s=started,
                        completed_monotonic_s=time.monotonic(),status='RETURNED_TRUE' if result is True else 'REJECTED_OR_CANCELLED',
                        declared_rpc_timeout_s=60.,actual_state_verification_still_required=True))
                    if result is not True:raise RuntimeError('VERTICAL_LIFECYCLE_TASK_NOT_ACKNOWLEDGED')
                    return result
            return VerifiedTask()
        return request

def make_client(log):
    import airsim
    return VerifiedLifecycleClient(airsim.MultirotorClient(timeout_value=60),log)
