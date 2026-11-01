import time,unittest
import numpy as np
from .online_curve import make_request,target_stream,fallback,ReferenceTimeline,reference_checks
from .online_study import prepare_request,ARMS,config_for,ResearchRuntime


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.r=make_request(([0,0,5],[0,0,0],[0,0,0]),target_stream([0,0,5],1201)[:2],'hermite_abom_improved_candidate')
        x,_=fallback(self.r);self.timeline=ReferenceTimeline(self.r,x)

    def test_guard_never_postpones_and_locality_keeps_timing_domain(self):
        active=self.timeline.windows[0];handoff=1.3;r=make_request(self.timeline.at(handoff)[0],self.r['targets'],self.r['method'])
        guarded,reason=prepare_request(r,active,handoff,ARMS['PG']);local,reason2=prepare_request(r,active,handoff,ARMS['PGL'])
        self.assertIsNone(reason);self.assertIsNone(reason2)
        self.assertLessEqual(handoff+guarded['upper'][3],active['_curve'].ends[0]+1e-9)
        self.assertTrue(reference_checks(guarded,guarded['anchor'])['passed'])
        np.testing.assert_allclose(np.array(local['upper'])[[0,1,2,4,5,6]],np.array(guarded['upper'])[[0,1,2,4,5,6]]*.25)
        self.assertEqual(local['upper'][3],guarded['upper'][3]);self.assertEqual(local['upper'][7],guarded['upper'][7])

    def test_late_horizon_skips_instead_of_violating_domain(self):
        active=self.timeline.windows[0];handoff=float(active['_curve'].ends[0]-2.9)
        r=make_request(self.timeline.at(handoff)[0],self.r['targets'],self.r['method'])
        guarded,reason=prepare_request(r,active,handoff,ARMS['PG']);self.assertIsNone(guarded);self.assertIn('DOMAIN',reason)

    def test_control_and_scene_frozen_across_arms(self):
        configs=[config_for(arm) for arm in ARMS]
        for cfg in configs[1:]:
            for key in ('control','airsim','preparation'):self.assertEqual(configs[0][key],cfg[key])
            self.assertEqual(configs[0]['online']['targets'],cfg['online']['targets'])
            self.assertEqual(cfg['online']['max_candidates'],100000)

    def test_search_seed_can_change_without_changing_scene(self):
        from types import SimpleNamespace
        runtime=ResearchRuntime.__new__(ResearchRuntime)
        runtime.config=config_for('PGL',1401);runtime.route_index=0
        runtime.cfg=SimpleNamespace(max_speed=6.,max_acceleration=6.)
        targets=runtime.config['online']['targets']
        runtime.config['online']['optimizer_seed']=21
        first=runtime._request(([0,0,5],[0,0,0],[0,0,0]))
        runtime.config['online']['optimizer_seed']=22
        second=runtime._request(([0,0,5],[0,0,0],[0,0,0]))
        self.assertEqual(first['seed'],21);self.assertEqual(second['seed'],22)
        self.assertEqual(first['targets'],second['targets']);self.assertEqual(runtime.config['online']['targets'],targets)


if __name__=='__main__':unittest.main()
