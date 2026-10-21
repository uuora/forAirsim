"""Actual FakeFlight receipts and unchanged gates for idle-zero scheduling."""
import threading
import time
import unittest

import test_t1_async_control as fixtures
from t1_async_control_v1p1 import AsyncControlV1P1
from t1_identity_motion_policy import IdentityMotionPolicy
from test_t1_identity_motion_policy import DESCRIPTION, bound_fixture, rebind_surface


class IdleZeroTests(fixtures.AsyncControlTests):
    """Run frozen ownership/lease/fault tests against the revised worker too."""

    def create(self, *, clock=time.monotonic, period_s=.01, real_policy=False):
        holder = {}

        def flight_factory():
            holder["factory_thread"] = threading.get_ident()
            holder["flight"] = fixtures.FakeFlight(self.logs, clock=clock)
            return holder["flight"]

        def policy_factory():
            holder["policy"] = IdentityMotionPolicy("session", 1, DESCRIPTION,
                clock=clock) if real_policy else fixtures.RecordingPolicy()
            return holder["policy"]

        control = AsyncControlV1P1(flight_factory, policy_factory, period_s=period_s,
            clock=clock, idle_zero_refresh_s=.30)
        self.instances[id(control)] = holder
        self.controllers.append(control)
        control.start()
        return control, holder

    def direct(self):
        clock = fixtures.ManualClock()
        flight = fixtures.FakeFlight(self.logs, clock=clock)
        control = AsyncControlV1P1(lambda: flight, lambda: None, clock=clock)
        state = control._read(flight)
        control._idle_tick(flight, state)
        return clock, flight, control

    def tick(self, clock, flight, control, delta):
        clock.advance(delta)
        state = control._read(flight)
        control._idle_tick(flight, state)
        return state

    def established(self):
        clock, flight, control = self.direct()
        self.tick(clock, flight, control, .05)
        self.tick(clock, flight, control, .05)
        self.assertGreater(control.scheduling_receipt()["idle_zero_omission_count"], 0)
        return clock, flight, control

    @staticmethod
    def real_inputs(now, *, frame_id="target-1", sequence=1):
        frame, selection, surface = bound_fixture(frame_id=frame_id, sequence=sequence,
            receipt=now-.10, ready=now-.08)
        selection["semantic_review_deadline_monotonic_s"] = now+8.
        rebind_surface(selection, surface)
        return frame, selection, surface

    def test_startup_has_actual_clear_then_two_distinct_post_clear_samples(self):
        clock, flight, control = self.direct()
        self.assertEqual(len(flight.commands), 1)
        self.assertEqual(flight.commands[0]["velocity_xy_m_s"], [0., 0.])
        self.assertFalse(any(row["kind"] == "IDLE_ZERO_OMITTED" for row in self.logs))
        self.tick(clock, flight, control, .05)
        self.assertEqual(len(flight.commands), 2)
        self.tick(clock, flight, control, .05)
        omitted = [row for row in self.logs if row["kind"] == "IDLE_ZERO_OMITTED"][-1]
        self.assertEqual(omitted["distinct_post_clear_sample_count"], 2)
        self.assertGreaterEqual(omitted["sampled_stop_span_s"], .05-1e-9)
        self.assertEqual(omitted["sampled_stop_clear_monotonic_s"], 10.)

    def test_idle_omission_preserves_last_real_zero_time_and_creates_no_command(self):
        clock, flight, control = self.established()
        before = (flight.last_command, len(flight.commands), len(control.receipts()))
        for _ in range(3):
            self.tick(clock, flight, control, .05)
        self.assertEqual(before, (flight.last_command, len(flight.commands), len(control.receipts())))
        omitted = [row for row in self.logs if row["kind"] == "IDLE_ZERO_OMITTED"][-1]
        self.assertFalse(omitted["command_dispatched"])
        self.assertFalse(omitted["command_receipt_created"])
        self.assertNotIn("command", omitted)
        self.assertEqual(omitted["last_actual_zero_dispatch_monotonic_s"], before[0])
        self.assertEqual(omitted["self_state_receipt_monotonic_s"], clock())

    def test_real_point_three_heartbeat_prevents_unbounded_old_zero_reference(self):
        clock, flight, control = self.established()
        previous = flight.last_command
        self.tick(clock, flight, control, .251)
        self.assertGreater(flight.last_command, previous)
        self.assertEqual(flight.last_command, clock())
        self.assertEqual(flight.commands[-1]["velocity_xy_m_s"], [0., 0.])
        self.assertEqual(control.scheduling_receipt()["idle_zero_refresh_count"], 1)
        refresh = [row for row in self.logs if row["kind"] == "IDLE_ZERO_REFRESH"][-1]
        self.assertEqual(refresh["actual_zero_dispatch_monotonic_s"], flight.last_command)
        self.assertTrue(refresh["real_command_dispatched"])
        for _ in range(20):
            self.tick(clock, flight, control, .05)
            self.assertLess(clock()-flight.last_command, .4)

    def test_fresh_phase_uses_actual_saved_zero_time_and_allows_full_desired(self):
        clock, flight, control = self.established()
        zero_time = flight.last_command
        state = self.tick(clock, flight, control, .10)
        self.assertEqual(flight.last_command, zero_time)
        policy = IdentityMotionPolicy("session", 1, DESCRIPTION, clock=clock)
        frame, selection, surface = self.real_inputs(clock())
        proposal = dict(ticket_id="manual-1", frame=frame, selection=selection,
            surface=surface, desired_xy_m_s=[.009216, 0.], enqueued_monotonic_s=clock())
        decision = control._consume(flight, policy, state, proposal)
        self.assertEqual(decision["motion_gate"]["status"], "ALLOW_BOUNDED_COMMAND")
        self.assertEqual(decision["motion_gate"]["previous_command_monotonic_s"], zero_time)
        self.assertAlmostEqual(decision["motion_gate"]["elapsed_previous_command_s"], clock()-zero_time)
        self.assertAlmostEqual(decision["command"]["velocity_xy_m_s"][0], .009216)
        self.assertEqual(decision["command"]["duration_s"], .12)

    def test_proposal_on_real_zero_clock_tick_still_waits_zero_elapsed(self):
        clock, flight, control = self.established()
        state = self.tick(clock, flight, control, .251)
        policy = IdentityMotionPolicy("session", 1, DESCRIPTION, clock=clock)
        frame, selection, surface = self.real_inputs(clock())
        decision = control._consume(flight, policy, state, dict(ticket_id="manual-1",
            frame=frame, selection=selection, surface=surface, desired_xy_m_s=[.01, 0.],
            enqueued_monotonic_s=clock()))
        self.assertEqual(decision["motion_gate"]["reason"], "INITIAL_ZERO_ELAPSED_PREVENTS_POSITIVE_CHANGE")
        self.assertEqual(decision["command"]["velocity_xy_m_s"], [0., 0.])

    def test_old_point_four_interval_guard_still_rejects_after_unsampled_gap(self):
        clock, flight, control = self.established()
        clock.advance(.5)
        state = control._read(flight)
        policy = IdentityMotionPolicy("session", 1, DESCRIPTION, clock=clock)
        frame, selection, surface = self.real_inputs(clock())
        decision = control._consume(flight, policy, state, dict(ticket_id="manual-1",
            frame=frame, selection=selection, surface=surface, desired_xy_m_s=[.01, 0.],
            enqueued_monotonic_s=clock()))
        self.assertEqual(decision["motion_gate"]["reason"], "PREVIOUS_COMMAND_INTERVAL_EXCEEDED")
        self.assertEqual(decision["command"]["velocity_xy_m_s"], [0., 0.])

    def test_measured_idle_drift_requires_real_brake_and_new_stopping_samples(self):
        clock, flight, control = self.established()
        before = len(flight.commands)
        flight.velocity = [.015, 0., 0.]
        self.tick(clock, flight, control, .05)
        self.assertEqual(len(flight.commands), before+1)
        self.assertEqual(flight.commands[-1]["dispatch_monotonic_s"], clock())
        self.assertIsNone(control._idle_basis["stable"])
        self.tick(clock, flight, control, .05)
        self.tick(clock, flight, control, .05)
        self.assertIsNotNone(control._idle_basis["stable"])

    def test_idle_nonzero_setpoint_is_braked_even_when_measured_velocity_is_zero(self):
        clock, flight, control = self.established()
        flight.last_xy = [.015, 0.]
        flight.last_command = clock()
        before = len(flight.commands)
        self.tick(clock, flight, control, .05)
        self.assertEqual(len(flight.commands), before+1)
        self.assertLess(abs(flight.commands[-1]["velocity_xy_m_s"][0]), .015)
        self.assertFalse(any(row["kind"] == "IDLE_ZERO_OMITTED" and row[
            "decision_monotonic_s"] == clock() for row in self.logs))

    def test_duplicate_sample_cannot_establish_idle_stopping(self):
        clock, flight, control = self.direct()
        state = control._read(flight)
        for _ in range(4):
            control._idle_tick(flight, state)
        self.assertFalse(any(row["kind"] == "IDLE_ZERO_OMITTED" for row in self.logs))
        self.assertEqual(control._idle_basis["post_clear_sample_count"], 1)

    def test_active_expiry_and_independent_stop_receipt_still_clear_explicitly(self):
        control, holder = self.create(period_s=.01)
        ticket = self.submit(control)
        decision = control.await_decision(ticket)
        self.wait_for(lambda: ticket in control.receipts())
        receipt = control.receipts()[ticket]
        clears = [row for row in holder["flight"].commands if row["velocity_xy_m_s"] == [0., 0.]
            and row["dispatch_monotonic_s"] >= decision["command"]["dispatch_monotonic_s"]+.12]
        self.assertTrue(clears)
        self.assertEqual(receipt["target_setpoint_cleared_monotonic_s"], clears[0]["dispatch_monotonic_s"])
        self.assertGreaterEqual(receipt["samples"][-1]["receipt_monotonic_s"]-
            receipt["samples"][0]["receipt_monotonic_s"], .05-1e-9)
        self.wait_for(lambda: any(row["kind"] == "IDLE_ZERO_OMITTED" for row in self.logs))
        self.assertTrue(any(row.get("sampled_stop_basis") == "INDEPENDENT_TARGET_LEASE_STOP_RECEIPT"
            for row in self.logs if row["kind"] == "IDLE_ZERO_OMITTED"))

    def test_real_worker_keeps_sampling_during_zero_omission_and_closes_with_real_zero(self):
        control, holder = self.create(period_s=.01)
        self.wait_for(lambda: control.scheduling_receipt()["idle_zero_omission_count"] >= 3)
        state_count = len([row for row in self.logs if row["kind"] == "SELF_STATE"])
        command_count = len(holder["flight"].commands)
        time.sleep(.03)
        self.assertGreater(len([row for row in self.logs if row["kind"] == "SELF_STATE"]), state_count)
        self.assertEqual(len(holder["flight"].commands), command_count)
        stop = control.close()
        self.assertTrue(stop["explicit_zero"])
        self.assertTrue(any(row["purpose"] == "OWN_STATE_CONTROLLER_CLOSE_ZERO"
            for row in holder["flight"].commands))
        for row in (row for row in self.logs if row["kind"] == "IDLE_ZERO_OMITTED"):
            self.assertLessEqual(row["elapsed_since_last_actual_zero_dispatch_s"], .30)
            self.assertLessEqual(abs(row["actual_velocity_xy_m_s"][0]), .005)

    def test_heartbeat_configuration_is_fixed_and_never_relaxes_original_limits(self):
        for value in (.1, .29, .4, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AsyncControlV1P1(lambda: None, lambda: None, idle_zero_refresh_s=value)
        with self.assertRaises(ValueError):
            AsyncControlV1P1(lambda: None, lambda: None, period_s=.1)


if __name__ == "__main__":
    unittest.main()
