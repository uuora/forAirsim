"""Measured idle zero omission with a real 0.30-second zero heartbeat.

The frozen controller supplies every proposal/lease/stop/fault/close gate.
Only idle scheduling changes. State queries continue; commands and their
timestamps are never synthesized when redundant zero dispatch is omitted.
"""
from __future__ import annotations

import copy
import math

from t1_async_control import AsyncControl, _xy_norm, closed_category


class AsyncControlV1P1(AsyncControl):
    revision = "MEASURED_STOP_IDLE_ZERO_OMISSION_V1P1"

    def __init__(self, flight_factory, policy_factory, period_s=.05, clock=None,
                 idle_zero_refresh_s=.30):
        if (isinstance(idle_zero_refresh_s, bool)
                or not isinstance(idle_zero_refresh_s, (int, float))
                or idle_zero_refresh_s != .30):
            raise ValueError("IDLE_ZERO_REFRESH_MUST_EQUAL_PREDECLARED_030")
        kwargs = dict(period_s=period_s)
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(flight_factory, policy_factory, **kwargs)
        self.idle_zero_refresh_s = float(idle_zero_refresh_s)
        self._idle_basis = None
        self._idle_omissions = 0
        self._idle_refreshes = 0

    def scheduling_receipt(self):
        receipt = super().scheduling_receipt()
        with self._condition:
            receipt.update(controller_revision=self.revision,
                idle_zero_refresh_s=self.idle_zero_refresh_s,
                idle_zero_omission_count=self._idle_omissions,
                idle_zero_refresh_count=self._idle_refreshes,
                omitted_commands_have_no_dispatch_receipt=True)
        return receipt

    def _consume(self, flight, policy, state, proposal):
        # A proposal may create a real command/stop chain. Its independent
        # stop receipt, rather than old idle samples, supplies the next basis.
        self._idle_basis = None
        return super()._consume(flight, policy, state, proposal)

    def _stop_tick(self, flight, state):
        active = self._active
        super()._stop_tick(flight, state)
        if (active is not None and self._active is None and active["clear"] is not None
                and active["stable"] is not None and _xy_norm(flight.last_xy) <= 1e-12):
            samples = [row for row in active["samples"] if row["receipt_monotonic_s"]
                >= max(active["clear"], active["stable"]) and _xy_norm(row["velocity_xy_m_s"]) <= .005]
            self._idle_basis = dict(clear=active["clear"], stable=active["stable"],
                last_sample=state["host_receipt_monotonic_s"],
                post_clear_sample_count=len(samples),
                origin="INDEPENDENT_TARGET_LEASE_STOP_RECEIPT")

    def _idle_brake(self, flight, state, *, refresh=False):
        command = flight.brake_tick(state)
        if _xy_norm(command["velocity_xy_m_s"]) <= 1e-12:
            if self._idle_basis is None:
                self._idle_basis = dict(clear=command["dispatch_monotonic_s"],
                    stable=None, last_sample=None, post_clear_sample_count=0,
                    origin="ACTUAL_IDLE_ZERO_DISPATCH")
        else:
            self._idle_basis = None
        if refresh:
            self._log(flight, dict(kind="IDLE_ZERO_REFRESH", stage=flight.stage,
                actual_zero_dispatch_monotonic_s=command["dispatch_monotonic_s"],
                actual_command_velocity_xy_m_s=command["velocity_xy_m_s"],
                actual_command_id=command.get("command_id"),
                self_state_receipt_monotonic_s=state["host_receipt_monotonic_s"],
                configured_real_zero_refresh_s=self.idle_zero_refresh_s,
                real_command_dispatched=True, target_observation_used=False))
            with self._condition:
                self._idle_refreshes += 1
        return command

    def _idle_tick(self, flight, state):
        now = self.clock()
        receipt = state["host_receipt_monotonic_s"]
        measured_speed = _xy_norm(state["velocity_ned_m_s"])
        actual_zero = (_xy_norm(flight.last_xy) <= 1e-12
            and math.isfinite(flight.last_command) and 0 <= flight.last_command <= receipt <= now)
        fresh = now-receipt <= .12
        if (not actual_zero or not fresh or measured_speed > .005
                or state.get("protection_reasons")):
            self._idle_basis = None
            return self._idle_brake(flight, state)
        basis = self._idle_basis
        if basis is None:
            # Startup must actually clear and subsequently observe stopping.
            return self._idle_brake(flight, state)
        if receipt < basis["clear"] or (basis["last_sample"] is not None
                and receipt < basis["last_sample"]):
            self._idle_basis = None
            return self._idle_brake(flight, state)
        if basis["last_sample"] is None or receipt > basis["last_sample"]:
            basis["post_clear_sample_count"] += 1
            basis["last_sample"] = receipt
            if basis["stable"] is None:
                basis["stable"] = receipt
        established = (basis["post_clear_sample_count"] >= 2 and
            basis["stable"] is not None and receipt-basis["stable"] >= .05-1e-9)
        elapsed = now-flight.last_command
        if not established or elapsed >= self.idle_zero_refresh_s:
            return self._idle_brake(flight, state, refresh=established and
                elapsed >= self.idle_zero_refresh_s)
        row = dict(kind="IDLE_ZERO_OMITTED", stage=flight.stage,
            decision_monotonic_s=now, self_state_receipt_monotonic_s=receipt,
            actual_velocity_xy_m_s=state["velocity_ned_m_s"][:2],
            last_actual_command_velocity_xy_m_s=list(flight.last_xy),
            last_actual_zero_dispatch_monotonic_s=flight.last_command,
            elapsed_since_last_actual_zero_dispatch_s=elapsed,
            configured_real_zero_refresh_s=self.idle_zero_refresh_s,
            sampled_stop_clear_monotonic_s=basis["clear"],
            sampled_stop_started_monotonic_s=basis["stable"],
            sampled_stop_span_s=receipt-basis["stable"],
            distinct_post_clear_sample_count=basis["post_clear_sample_count"],
            sampled_stop_basis=basis["origin"], stationary_xy_threshold_m_s=.005,
            active_target_lease=False, zero_command_omitted=True,
            command_dispatched=False, command_receipt_created=False,
            target_observation_used=False, flight_authorized=False,
            continuous_safety_established=False)
        self._log(flight, row)
        with self._condition:
            self._idle_omissions += 1
        return None

    def _run(self):
        # Frozen worker lifecycle, with only startup/idle scheduling replaced.
        flight = policy = None
        try:
            flight = self.flight_factory()
            policy = self.policy_factory()
            state = self._read(flight)
            self._idle_tick(flight, state)
            with self._condition:
                self._ready = True
                self._condition.notify_all()
            next_tick = self.clock()+self.period_s
            while True:
                with self._condition:
                    if self._closing:
                        break
                    now = self.clock()
                    lease_event = self._active["wait_until"] if self._active and self._active["stop_deadline"] is None else None
                    due = now >= next_tick or (lease_event is not None and now >= lease_event)
                    if self._pending is None and not due:
                        wake = min(next_tick, lease_event) if lease_event is not None else next_tick
                        self._condition.wait(max(0., wake-now))
                        continue
                    proposal = self._pending
                    self._pending = None
                scheduled = next_tick if self.clock() >= next_tick else None
                state = self._read(flight, scheduled=scheduled)
                if scheduled is not None:
                    next_tick += (math.floor(max(0., self.clock()-next_tick)/self.period_s)+1)*self.period_s
                self._stop_tick(flight, state)
                if proposal is not None:
                    decision = self._consume(flight, policy, state, proposal)
                    with self._condition:
                        self._decisions[proposal["ticket_id"]] = copy.deepcopy(decision)
                        self._condition.notify_all()
                    self._log(flight, dict(decision, kind="CONTROL_DECISION", stage=flight.stage))
                elif self._active is None:
                    self._idle_tick(flight, state)
        except BaseException as error:
            with self._condition:
                self._fatal = closed_category(error)
                self._condition.notify_all()
            if flight is not None:
                try:
                    flight.issue([0., 0.], .12, "OWN_STATE_CONTROLLER_FATAL_ZERO")
                except BaseException:
                    pass
        finally:
            if flight is not None:
                try:
                    self._shutdown(flight)
                except BaseException as error:
                    with self._condition:
                        details = closed_category(error)
                        self._fatal = self._fatal+"; CLOSE: "+details if self._fatal else details
            if policy is not None:
                try:
                    policy.close()
                except BaseException as error:
                    with self._condition:
                        self._fatal = self._fatal or "POLICY_CLOSE: "+closed_category(error)
            with self._condition:
                if self._pending is not None:
                    self._decisions[self._pending["ticket_id"]] = self._rejection(
                        self._pending, "CONTROLLER_CLOSED_BEFORE_DECISION")
                    self._pending = None
                self._closed = True
                self._condition.notify_all()
