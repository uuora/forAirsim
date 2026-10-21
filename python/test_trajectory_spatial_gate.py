import unittest
from analyze_trajectory_pilot import summarize_spatial_samples
from audit_trajectory_matched_gate import failures


class SpatialGateTests(unittest.TestCase):
    def configuration(self):
        return dict(workspace_min_ned_m=[-5.6,-1.3,-3.6],workspace_max_ned_m=[-2.4,.6,-2.4],planning={'execution_guard_margin_m':.15})

    def test_inner_margin_violation_is_not_hidden_by_velocity_budget(self):
        result=summarize_spatial_samples([{'world_ned_m':[-4,.45175,-3],'velocity_ned_m_s':[.04,0,0]}],self.configuration())
        self.assertEqual(result['status'],'OBSERVED_EXCEEDANCE')
        self.assertEqual(result['entered_inner_margin_samples'],1)
        self.assertEqual(result['outside_outer_workspace_samples'],0)

    def test_missing_positions_cannot_become_positive_spatial_coverage(self):
        result=summarize_spatial_samples([{'world_ned_m':[-4,0,-3]},{}],self.configuration())
        self.assertEqual(result['status'],'PARTIALLY_ASSESSED')

    def test_next_session_gate_rejects_spatial_exceedance(self):
        match=dict(status='ACCEPTED_VERIFIED',acceptance_verified=True,physical_sampled_budget_status='NO_EXCEEDANCE_IN_RECORDED_SAMPLES',
                   command_xy_norm_violations=0,command_xy_norm_invalid_count=0)
        rows=[dict(policy=mode,initial_state_match=match,approach_spatial_budget={'status':'NO_EXCEEDANCE_IN_RECORDED_SAMPLES'}) for mode in ('efficiency','perception')]
        summary=dict(started_flights=2,runs=rows,initial_state_pairs={'paired_session_sample_count':1})
        self.assertEqual(failures(summary),[])
        rows[1]['approach_spatial_budget']['status']='OBSERVED_EXCEEDANCE'
        self.assertIn('perception:APPROACH_SPATIAL_GATE_FAILED',failures(summary))


if __name__=='__main__':unittest.main()
