"""Validate zero proposals while preserving an already established stationary hold."""
import copy

from t1_async_control import _xy_norm
from t1_async_control_v1p2 import AsyncControlV1P2


class AsyncControlV1P3(AsyncControlV1P2):
    revision = 'STRICT_STOP_VALIDATED_ZERO_HOLD_V1P3'

    def _consume(self, flight, policy, state, proposal):
        now = self.clock()
        if (self._active is None and _xy_norm(proposal['desired_xy_m_s']) > 1e-12
                and (not self._settled or _xy_norm(flight.last_xy) > 1e-12)):
            command = flight.brake_tick(state)
            self._activate_stop(proposal['ticket_id'], command)
            self._idle_window = self._idle_basis = None
            with self._condition:
                self._settled = False
            result = self._rejection(proposal, 'STRICT_STATIONARY_HOLD_REQUIRED_BEFORE_NEW_TARGET', state)
            result['command'] = command
            return result
        held = (self._active is None and self._settled and self._idle_window is not None
            and _xy_norm(proposal['desired_xy_m_s']) <= 1e-12
            and _xy_norm(flight.last_xy) <= 1e-12
            and 0 <= now-flight.last_command < self.idle_zero_refresh_s
            and self._idle_window.observe(state, flight.last_xy, now, flight.last_command))
        if not held:
            return super()._consume(flight, policy, state, proposal)
        # The original policy still checks current frame binding, review age,
        # duplicate use, actual command clock, workspace and own-state freshness.
        gate = policy.decide(frame=proposal['frame'], selection=proposal['selection'], surface=proposal['surface'],
            desired_xy_m_s=proposal['desired_xy_m_s'], actual_velocity_xy_m_s=state['velocity_ned_m_s'][:2],
            self_position_ned_m=state['position_world_ned_m'], previous_command_xy_m_s=flight.last_xy,
            previous_command_monotonic_s=flight.last_command,
            self_state_receipt_monotonic_s=state['host_receipt_monotonic_s'], now=now)
        if gate['status'] == 'ALLOW_BOUNDED_COMMAND':
            gate = dict(gate, status='WAIT_NO_TARGET', reason='ALREADY_ZERO_WITH_STRICT_SAMPLED_STOP',
                validated_gate_status='ALLOW_BOUNDED_COMMAND', target_command_authorized=False,
                command_duration_s=0., command_velocity_xy_m_s=[0., 0.])
        proof = self._idle_window.proof()
        proof['samples'] = copy.deepcopy(proof['samples'][-5:])
        proof.update(actual_zero_dispatch_monotonic_s=flight.last_command,
            current_self_state_receipt_monotonic_s=state['host_receipt_monotonic_s'],
            current_self_state=copy.deepcopy(state), held_zero_without_new_command=True,
            command_timestamp_changed=False, new_target_lease=False,
            proof_scope='last five actual own-state samples from established sampled stop')
        return dict(ticket_id=proposal['ticket_id'], enqueued_monotonic_s=proposal['enqueued_monotonic_s'],
            decision_completed_monotonic_s=self.clock(), motion_gate=gate,
            command_self_state=copy.deepcopy(state), command=None, zero_held_stop_proof=proof)
