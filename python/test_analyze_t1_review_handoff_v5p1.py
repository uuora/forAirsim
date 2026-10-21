import copy
import unittest
from analyze_t1_review_handoff_v5p1 import active_stall_evidence, handoff_evidence


class ActiveStallEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.report=dict(configuration=dict(active_lease_stall=dict(enabled=True,duration_s=.65)),
            frames=[dict(frame_id='f1',ticket_id='control-1',lease_stop=dict(ticket_id='control-1',
                declared_lease_end_monotonic_s=10.12,target_setpoint_cleared_monotonic_s=10.14),
                command=dict(command_id='cmd1'))],
            injected_delays=[dict(boundary='AFTER_ACTUAL_NONZERO_DISPATCH_BEFORE_MEDIA_SAVE',
                                 frame_id='f1',ticket_id='control-1',eligibility=dict(command_id='cmd1'))])
        self.run=dict(controller_timing=dict(injected_delay_rows=[dict(
            boundary='AFTER_ACTUAL_NONZERO_DISPATCH_BEFORE_MEDIA_SAVE',
            started_monotonic_s=10.02,completed_monotonic_s=10.68,self_state_receipts_inside=13,
            nonzero_target_active_overlap_rows=[dict(target_frame_id='f1',duration_s=.1)])]),
            target_command_rows=[dict(target_frame_id='f1',purpose='CURRENT_TARGET_TRAJECTORY',
                dispatch_monotonic_s=10.,duration_s=.12,velocity_xy_m_s=[.004,0.],issues=[],command_id='cmd1',
                lease_clear=dict(actual_zero_clear_s=10.14,status='EXPLICIT_ZERO_CLEAR_RECORDED'))])

    def test_actual_active_overlap_and_clear_inside_block(self):
        result=active_stall_evidence(self.report,self.run)
        self.assertEqual(result['status'],'PASS_CONTROLLED_ACTIVE_DISPATCH_STALL')
        self.assertTrue(result['explicit_clear_inside_block'])

    def test_zero_output_expired_lease_or_clear_outside_does_not_pass(self):
        for change in ('zero','late_start','clear_outside','no_query','different_overlap','audit_issue','clear_issue'):
            run=copy.deepcopy(self.run)
            command=run['target_command_rows'][0];block=run['controller_timing']['injected_delay_rows'][0]
            if change=='zero':command['velocity_xy_m_s']=[0.,0.]
            elif change=='late_start':block['started_monotonic_s']=10.2
            elif change=='clear_outside':command['lease_clear']['actual_zero_clear_s']=10.8
            elif change=='no_query':block['self_state_receipts_inside']=0
            elif change=='different_overlap':block['nonzero_target_active_overlap_rows'][0]['target_frame_id']='f2'
            elif change=='clear_issue':command['lease_clear']['issues']=['LEASE_BINDING_MISMATCH']
            else:command['issues']=['WRONG_IDENTITY']
            self.assertEqual(active_stall_evidence(self.report,run)['status'],'FAIL_ACTIVE_STALL_EVIDENCE',change)

    def test_matched_frame_and_stop_ticket_required(self):
        for key in ('ticket_id','clear','declared_end','command_id','short_block'):
            report=copy.deepcopy(self.report)
            frame=report['frames'][0]
            if key=='ticket_id':frame['lease_stop']['ticket_id']='control-2'
            elif key=='clear':frame['lease_stop']['target_setpoint_cleared_monotonic_s']=10.15
            elif key=='declared_end':frame['lease_stop']['declared_lease_end_monotonic_s']=10.16
            elif key=='command_id':report['injected_delays'][0]['eligibility']['command_id']='cmd2'
            run=copy.deepcopy(self.run)
            if key=='short_block':run['controller_timing']['injected_delay_rows'][0]['completed_monotonic_s']=10.60
            self.assertEqual(active_stall_evidence(report,run)['status'],'FAIL_ACTIVE_STALL_EVIDENCE',key)

    def test_not_triggered_and_not_declared_separate(self):
        self.run['controller_timing']['injected_delay_rows']=[]
        self.assertEqual(active_stall_evidence(self.report,self.run)['status'],'FAIL_PREDECLARED_STALL_NOT_OBSERVED')
        self.report['configuration']['active_lease_stall']['enabled']=False
        self.assertEqual(active_stall_evidence(self.report,self.run)['status'],'NOT_DECLARED')


class HandoffReceiptArchitectureTests(unittest.TestCase):
    def test_missing_receipt_fails_both_declared_architectures(self):
        for architecture in ('INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5',
                             'INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5P1'):
            result=handoff_evidence(dict(control_architecture=architecture,
                frames=[dict(frame_id='f1',frame_sequence=1)]))
            self.assertTrue(result['configured'])
            self.assertEqual(result['issues'],[dict(frame_id='f1',issue='HANDOFF_RECEIPT_MISSING')])

    def test_closed_receipt_missing_field_rejected_v5p1(self):
        result=handoff_evidence(dict(control_architecture='INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5P1',
            frames=[dict(frame_id='f1',frame_sequence=1,observer_result=dict(temporal_identity=dict(
                status='UNCERTAIN',review_handoff=dict(event='NO_NEW_REVIEW',status='IDLE',
                    active_request_id=None,pending_request_id=None,confirmation_count=0))))]))
        self.assertEqual(result['issues'][0]['issue'],'HANDOFF_CLOSED_RECEIPT_CONTRACT_INVALID')


if __name__=='__main__':unittest.main()
