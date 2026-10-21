"""Unavailable current zero-hold evidence refuses a proposal without new motion."""
from t1_async_control_v1p4 import AsyncControlV1P4
from t1_zero_hold_proof_v1 import stable_interval_proof

PROOF_REFUSALS={'DECLARED_STABLE_INTERVAL_REQUIRED','ACTUAL_STABLE_INTERVAL_DWELL_REQUIRED',
    'ACTUAL_STATIONARY_ZERO_SAMPLES_REQUIRED','CURRENT_STABLE_ZERO_PROOF_BINDING_REQUIRED'}

class AsyncControlV1P6(AsyncControlV1P4):
    revision='STRICT_STOP_STABLE_INTERVAL_REFUSE_UNAVAILABLE_ZERO_HOLD_V1P6'

    def _consume(self,flight,policy,state,proposal):
        result=super()._consume(flight,policy,state,proposal)
        if result.get('zero_held_stop_proof') is None:return result
        try:result['zero_held_stop_proof']=stable_interval_proof(result['zero_held_stop_proof'])
        except ValueError as error:
            if str(error) not in PROOF_REFUSALS:raise
            result.pop('zero_held_stop_proof')
            result['zero_held_stop_refusal']=dict(status='CURRENT_ZERO_HOLD_PROOF_REFUSED',reason=str(error),
                current_state_timestamp_ns=state['state_timestamp_ns'],current_state_receipt_monotonic_s=state['host_receipt_monotonic_s'],
                actual_zero_dispatch_monotonic_s=flight.last_command,command_dispatched=False,new_target_lease=False,
                clock_rewritten=False,stability_limits_unchanged=True)
            gate=result['motion_gate']
            if gate.get('reason')=='ALREADY_ZERO_WITH_STRICT_SAMPLED_STOP':
                result['motion_gate']=dict(gate,status='WAIT_NO_TARGET',reason='CURRENT_ZERO_HOLD_PROOF_UNAVAILABLE',
                    target_command_authorized=False,command_duration_s=0.,command_velocity_xy_m_s=[0.,0.])
        return result
