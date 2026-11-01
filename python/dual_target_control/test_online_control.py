import tempfile,time,unittest
from pathlib import Path
import numpy as np
from .online_curve import target_stream,stream_hash,make_request,fallback,decode,ReferenceTimeline,evaluate


class OnlineReferenceTests(unittest.TestCase):
    def request(self):return make_request(([0,0,5],[0,0,0],[0,0,0]),target_stream([0,0,5],1201)[:2],'hermite_abom')

    def test_seed_stream_is_aircraft_independent_and_reproducible(self):
        self.assertEqual(stream_hash(target_stream([0,0,5],1201)),stream_hash(target_stream([0,0,5],1201)))
        self.assertNotEqual(stream_hash(target_stream([0,0,5],1201)),stream_hash(target_stream([0,0,5],1202)))

    def test_two_representations_same_physical_curve_and_fixed_boundary(self):
        request=self.request();x,checks=fallback(request);self.assertTrue(checks['passed']);a=decode(request,x)
        request['method']='bezier_abom';b=decode(request,x)
        np.testing.assert_allclose(a.sample(np.linspace(0,a.duration_s,120)),b.sample(np.linspace(0,b.duration_s,120)),atol=1e-10)
        np.testing.assert_allclose(a.sample(0.,derivative=1),0,atol=1e-10)
        np.testing.assert_allclose(a.sample(a.duration_s,derivative=1),0,atol=1e-10)
        self.assertTrue(evaluate(request,x)['feasible'])

    def test_nested_pending_handoffs_preserve_prefix_and_c2(self):
        r=self.request();x,_=fallback(r);timeline=ReferenceTimeline(r,x)
        boundary,_=timeline.at(1.);r1=make_request(boundary,r['targets'],'hermite_abom');x1,_=fallback(r1)
        first=timeline.add(1.,r1,x1,'fallback',now=.2)
        old_boundary,_=timeline.at(2.);r2=make_request(old_boundary,r['targets'],'hermite_abom');x2,_=fallback(r2)
        timeline.add(2.,r2,x2,'optimizer',now=.3)
        self.assertFalse(first.get('cancelled',False));self.assertEqual(timeline.at(1.5)[1]['id'],1)
        self.assertEqual(timeline.at(2.)[1]['id'],2)
        for derivative in (0,1,2):np.testing.assert_allclose(timeline.at(2.)[0][derivative],old_boundary[derivative],atol=1e-8)

    def test_reject_jump_and_late_handoff(self):
        r=self.request();x,_=fallback(r);timeline=ReferenceTimeline(r,x)
        with self.assertRaises(ValueError):timeline.add(.2,r,x,'bad',now=.19)
        with self.assertRaises(ValueError):timeline.add(2.,r,x,'bad',now=0.)

    def test_parent_rejects_late_stale_and_unknown_jobs_without_losing_prefix(self):
        from .online_live import OnlineRuntime
        from types import SimpleNamespace
        for condition in ('late','stale','unknown'):
            r=self.request();x,_=fallback(r);timeline=ReferenceTimeline(r,x)
            pending=make_request(timeline.at(2.)[0],r['targets'],'hermite_abom');vector,_=fallback(pending)
            pending.update(job_id=9,epoch=timeline.epoch,handoff_s=2.,target_index=0,deadline_host_s=time.perf_counter()+5,anchor=vector)
            response=dict(type='result',job_id=9,epoch=timeline.epoch,handoff_s=2.,completed_host_s=time.perf_counter(),result=dict(deadline_met=True,parameters=vector,validation=dict(feasible=True)))
            if condition=='late':response['result']['deadline_met']=False
            if condition=='stale':response['epoch']+=1
            if condition=='unknown':response['job_id']=8
            runtime=OnlineRuntime.__new__(OnlineRuntime);runtime.pending=pending;runtime.timeline=timeline;runtime.live_elapsed=.5;runtime.route_index=0
            runtime.next_request_s=100.;runtime.planner=SimpleNamespace(poll=lambda:[response],ready=True);events=[]
            runtime._event=lambda stage,**data:events.append(dict(stage=stage,**data))
            runtime._poll_planner(np.zeros(3),np.zeros(3))
            self.assertEqual(len(timeline.windows),1)
            self.assertEqual(events[-1]['disposition'],dict(late='WALL_DEADLINE',stale='STALE_REFERENCE_OR_TARGET',unknown='UNKNOWN_JOB')[condition])
            if condition=='unknown':self.assertIs(runtime.pending,pending)

    def test_swept_target_gate_preserves_heading_requirement(self):
        from .online_live import OnlineRuntime
        from .core import ControlConfig
        runtime=OnlineRuntime.__new__(OnlineRuntime);runtime.cfg=ControlConfig();runtime.previous_p=np.array([-1.,0.,5.]);runtime.previous_yaw=0.
        runtime._target_queue=lambda:[[0.,0.,5.],[8.,0.,5.]]
        self.assertTrue(runtime._reached(np.array([.8,0.,5.]),0.)[0])
        runtime.previous_yaw=np.pi
        self.assertFalse(runtime._reached(np.array([.8,0.,5.]),np.pi)[0])

    def test_real_worker_is_persistent_and_reports_deadline(self):
        from .online_worker_client import PlannerClient
        with tempfile.TemporaryDirectory() as directory:
            client=PlannerClient(Path(directory))
            try:
                deadline=time.perf_counter()+30;hello=[]
                while not client.ready and time.perf_counter()<deadline:
                    hello+=client.poll();time.sleep(.03)
                self.assertTrue(client.ready);hello+=client.poll();self.assertTrue(any(r['type']=='ready' for r in hello))
                pid=client.process.pid
                for job,method,offset in [(1,'hermite_abom',.8),(2,'bezier_abom_improved_candidate',.8),(3,'hermite_abom',-1.),(4,'hermite_abom',.8)]:
                    request=self.request();request.update(method=method,job_id=job,epoch=1,handoff_s=1.3,deadline_host_s=time.perf_counter()+offset,population=8,max_candidates=32)
                    if job==4:request['search_settings']={'proposal_mode':'gaussian'}
                    before=time.perf_counter();client.submit(request);self.assertLess(time.perf_counter()-before,.02)
                    replies=[];bound=time.perf_counter()+12
                    while not replies and time.perf_counter()<bound:replies=client.poll();time.sleep(.02)
                    self.assertTrue(replies);self.assertEqual(replies[0]['type'],'result',replies)
                    result=replies[0]['result'];self.assertEqual(result['counts']['airsim_search_calls'],0)
                    self.assertEqual(result['counts']['validation_calls'],1)
                    if offset<0:self.assertFalse(result['deadline_met'])
                    if job==4:
                        self.assertEqual(result['timings']['neural_update_s'],0.)
                        self.assertTrue(any(r['source']=='gaussian' for r in result['ledger']))
                        self.assertLessEqual(result['counts']['search_calls'],32)
                    self.assertEqual(client.process.pid,pid)
            finally:client.close()


if __name__=='__main__':unittest.main()
