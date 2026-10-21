"""Offline protocol checks; no AirSim connection or flight commands."""

from copy import deepcopy
import json
import unittest

from refreshed_target_round import RefreshedTargetRound, RoundPhase, RoundThresholds


class RoundProtocolTests(unittest.TestCase):
    def setUp(self):
        self.machine = RefreshedTargetRound()
        self.now = 10.0

    def event(self, event, **changes):
        payload = dict(round_id=self.machine.round_id, target_generation=self.machine.target_generation, now=self.now)
        payload.update(changes)
        return self.machine.submit(event, payload)

    def prepare(self, **changes):
        payload = dict(round_id=self.machine.round_id + 1, target_generation=self.machine.target_generation + 1,
                       actor_name=self.machine.actor_for_round(self.machine.round_id + 1),
                       active_target_actors=[self.machine.actor_for_round(self.machine.round_id + 1)],
                       prepared_at=self.now)
        payload.update(changes)
        return self.event("prepare_round", **payload)

    def ack(self, drone, cancel=False, **changes):
        payload = dict(drone=drone, request_token=self.machine.cancel_token if cancel else self.machine.hold_token,
                       recorded_at=self.now, old_actions_cancelled=True, safe_hold=True, ready=True)
        payload.update(changes)
        return self.event("acknowledge_cancel" if cancel else "acknowledge_hold", **payload)

    def observe(self, drone, eta=3.0, **changes):
        payload = dict(drone=drone, actor_name=self.machine.actor_name,
                       message_id=f"{self.machine.round_id}:{drone}:{self.now}", source="yolo",
                       captured_at=self.now, identity_confirmed=True, ready=True,
                       eta_seconds=eta, estimate_at=self.now if eta is not None else None)
        payload.update(changes)
        return self.event("observe", **payload)

    def ready(self, eta_a=3.0, eta_b=4.0):
        self.assertTrue(self.prepare().accepted)
        for drone in self.machine.DRONES:
            self.assertTrue(self.ack(drone).accepted)
        self.assertTrue(self.observe("DroneA", eta_a).accepted)
        self.assertTrue(self.observe("DroneB", eta_b).accepted)

    def assign(self, policy="fixed_A", **changes):
        payload = dict(policy=policy, assignment_id=f"assignment-{self.machine.round_id}",
                       plan_id=f"plan-{self.machine.round_id}", created_at=self.now)
        payload.update(changes)
        return self.event("assign", **payload)

    def plan(self):
        return {key: self.machine.assignment[key] for key in ("executor", "assignment_id", "plan_id")}

    def dispatch(self, baseline=100, **changes):
        payload = dict(**self.plan(), issued_at=self.now, collision_baseline_timestamp=baseline)
        payload.update(changes)
        return self.event("dispatch", **payload)

    def hit(self, stamp=101, **changes):
        payload = dict(**self.plan(), actor_name=self.machine.actor_name, has_collided=True,
                       collision_timestamp=stamp, reported_at=self.now)
        payload.update(changes)
        return self.event("confirm_hit", **payload)

    def finish_round(self):
        self.assertTrue(self.hit().accepted)
        self.assertTrue(self.ack("DroneA", cancel=True).accepted)
        self.assertTrue(self.ack("DroneB", cancel=True).accepted)

    def assertRejected(self, result, reason):
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, reason)
        self.assertEqual(result.actions, ())

    def test_first_round_requires_both_cancel_hold_acks_then_observations(self):
        result = self.prepare()
        self.assertEqual([a["drone"] for a in result.actions], ["DroneA", "DroneB"])
        self.assertRejected(self.observe("DroneA"), "observation_phase_required")
        self.assertTrue(self.ack("DroneA").accepted)
        self.assertRejected(self.observe("DroneA"), "observation_phase_required")
        self.assertRejected(self.ack("DroneB", safe_hold=False), "safe_cancellation_not_confirmed")
        self.assertTrue(self.ack("DroneB").accepted)
        self.assertTrue(self.observe("DroneA").accepted)
        self.assertRejected(self.assign(), "ready_phase_required")
        self.assertTrue(self.observe("DroneB").accepted)
        self.assertTrue(self.assign().accepted)

    def test_one_target_inventory_and_unique_name_required_every_round(self):
        self.assertRejected(self.prepare(active_target_actors=["BalloonTarget_r0001", "BalloonTarget_r0002"]),
                            "single_active_target_required")
        self.assertRejected(self.prepare(actor_name="BalloonTarget"), "actor_name_mismatch")
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.assertTrue(self.dispatch().accepted)
        self.finish_round()
        self.assertRejected(self.prepare(active_target_actors=["BalloonTarget_r0001", "BalloonTarget_r0002"]),
                            "single_active_target_required")
        self.assertEqual(self.machine.phase, RoundPhase.REFRESH_PENDING)
        self.assertTrue(self.prepare().accepted)
        self.assertEqual(self.machine.actor_name, "BalloonTarget_r0002")

    def test_hit_before_dispatch_wrong_executor_actor_and_old_collision_rejected(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.assertRejected(self.hit(), "contact_or_executing_phase_required")
        self.assertTrue(self.dispatch().accepted)
        self.assertRejected(self.hit(executor="DroneB"), "executor_mismatch")
        self.assertRejected(self.hit(actor_name="BalloonTarget_r0001_String"), "actor_name_mismatch")
        self.assertRejected(self.hit(stamp=100), "old_collision_timestamp")
        self.assertRejected(self.hit(has_collided=False), "contact_not_confirmed")
        self.assertTrue(self.hit().accepted)
        self.assertRejected(self.hit(stamp=102), "duplicate_hit")
        self.assertEqual(self.machine.accepted_hit_rounds, {1})

    def test_cancel_barrier_precedes_refresh_and_new_scene_requires_new_hold(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.assertTrue(self.dispatch().accepted)
        hit = self.hit()
        self.assertEqual(self.machine.phase, RoundPhase.CANCEL_PENDING)
        self.assertEqual({a["operation"] for a in hit.actions}, {"CANCEL_AND_HOLD"})
        self.assertRejected(self.prepare(), "cancellation_barrier_not_complete")
        self.assertTrue(self.ack("DroneA", cancel=True).accepted)
        self.assertRejected(self.prepare(), "cancellation_barrier_not_complete")
        self.assertRejected(self.ack("DroneB", cancel=True, old_actions_cancelled=False), "safe_cancellation_not_confirmed")
        result = self.ack("DroneB", cancel=True)
        self.assertTrue(result.accepted)
        self.assertEqual(result.actions[0]["operation"], "REQUEST_TARGET_REFRESH")
        old_token = self.machine.cancel_token
        self.assertTrue(self.prepare().accepted)
        self.assertRejected(self.ack("DroneA", request_token=old_token), "request_token_mismatch")
        self.assertRejected(self.observe("DroneA"), "observation_phase_required")
        self.assertIsNone(self.machine.assignment)

    def test_late_vlx_old_generation_stale_and_pre_barrier_frames_rejected(self):
        self.ready()
        self.assertRejected(self.observe("DroneA", source="vlx", target_generation=99), "target_generation_mismatch")
        self.now += 1.0
        self.assertRejected(self.observe("DroneA", source="vlx", captured_at=self.now - 0.6), "observation_stale_message")
        self.assertRejected(self.observe("DroneA", source="vlx", captured_at=9.9), "observation_timestamp_before_barrier")
        self.assertRejected(self.observe("DroneA", source="vlx", captured_at=self.now + 0.1), "observation_future_timestamp")
        self.assertTrue(self.observe("DroneA", source="vlx").accepted)
        self.assertRejected(self.observe("DroneA", source="vlx", message_id="different-message"), "old_observation")

    def test_dispatch_rechecks_both_observations_and_assignment_age(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.now += 0.6
        self.assertRejected(self.dispatch(), "observation_stale_message")
        self.now += 0.5
        self.assertRejected(self.dispatch(), "assignment_stale_message")

    def test_alternating_allocates_a_then_b_with_exactly_one_executor(self):
        self.ready()
        result = self.assign("alternating")
        self.assertEqual(result.actions[0]["executor"], "DroneA")
        self.assertEqual(result.actions[0]["observer"], "DroneB")
        self.assertTrue(self.dispatch().accepted)
        self.finish_round()
        self.now += 0.1
        self.ready()
        result = self.assign("alternating")
        self.assertEqual(result.actions[0]["executor"], "DroneB")
        self.assertEqual(result.actions[0]["observer"], "DroneA")
        self.assertTrue(self.dispatch().accepted)

    def test_fixed_a_stays_a_across_refreshed_rounds(self):
        self.ready()
        self.assertTrue(self.assign("fixed_A").accepted)
        self.assertTrue(self.dispatch().accepted)
        self.finish_round()
        self.now += 0.1
        self.ready()
        self.assertTrue(self.assign("fixed_A").accepted)
        self.assertEqual(self.machine.assignment["executor"], "DroneA")

    def test_nearest_eta_uses_only_fresh_estimates_and_tie_breaks_a(self):
        self.ready(eta_a=5.0, eta_b=2.0)
        self.assertTrue(self.assign("nearest_eta").accepted)
        self.assertEqual(self.machine.assignment["executor"], "DroneB")
        tie = RefreshedTargetRound()
        self.machine = tie
        self.ready(eta_a=2.0, eta_b=2.0)
        self.assertTrue(self.assign("nearest_eta").accepted)
        self.assertEqual(self.machine.assignment["executor"], "DroneA")

    def test_eta_nonfinite_negative_stale_missing_and_unconfirmed_rejected(self):
        self.assertTrue(self.prepare().accepted)
        self.assertTrue(self.ack("DroneA").accepted)
        self.assertTrue(self.ack("DroneB").accepted)
        for invalid in (float("nan"), float("inf"), -1.0, True):
            self.assertRejected(self.observe("DroneA", eta=invalid), "invalid_eta")
        self.now += 1
        self.assertRejected(self.observe("DroneA", captured_at=self.now - 0.4, estimate_at=self.now - 0.6),
                            "eta_timestamp_before_barrier")
        self.assertTrue(self.observe("DroneA", eta=None).accepted)
        self.assertTrue(self.observe("DroneB", identity_confirmed=False).accepted)
        self.assertRejected(self.assign("nearest_eta"), "eta_required")
        self.now += 0.1
        self.assertTrue(self.observe("DroneA").accepted)
        self.assertRejected(self.assign("nearest_eta"), "observation_identity_or_readiness_unconfirmed")

    def test_eta_can_expire_independently_of_frame_and_truth_fields_rejected(self):
        self.machine = RefreshedTargetRound(RoundThresholds(max_observation_age_s=1.0, max_eta_age_s=0.1))
        self.ready()
        self.now += 0.2
        self.assertRejected(self.assign("nearest_eta"), "eta_stale_message")
        self.assertRejected(self.event("assign", policy="nearest_eta", assignment_id="x", plan_id="p", created_at=self.now,
                                      target_position=[1, 2, 3]), "event_fields_mismatch")

    def test_old_generation_plan_and_reused_ids_cannot_dispatch(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        old_plan = deepcopy(self.plan())
        self.assertRejected(self.dispatch(plan_id="old-plan"), "plan_id_mismatch")
        self.assertTrue(self.dispatch().accepted)
        self.finish_round()
        self.now += 0.1
        self.ready()
        self.assertRejected(self.assign(assignment_id=old_plan["assignment_id"]), "assignment_or_plan_id_reused")
        self.assertTrue(self.assign().accepted)
        self.assertRejected(self.dispatch(round_id=1, target_generation=1), "round_id_mismatch")
        self.assertRejected(self.dispatch(plan_id=old_plan["plan_id"]), "plan_id_mismatch")
        self.assertTrue(self.dispatch(baseline=101).accepted)
        self.assertRejected(self.hit(actor_name="BalloonTarget_r0001"), "actor_name_mismatch")
        self.assertRejected(self.hit(stamp=101), "old_collision_timestamp")
        self.assertTrue(self.hit(stamp=102).accepted)

    def test_contact_phase_matching_plan_and_fresh_report_required(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.assertTrue(self.dispatch().accepted)
        payload = dict(self.plan(), plan_id="invalid", issued_at=self.now)
        self.assertRejected(self.event("mark_contact", **payload), "plan_id_mismatch")

    def test_contact_phase_is_admitted_and_old_or_nonfinite_report_is_not(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.assertTrue(self.dispatch().accepted)
        self.assertTrue(self.event("mark_contact", **self.plan(), issued_at=self.now).accepted)
        self.assertEqual(self.machine.phase, RoundPhase.CONTACT)
        self.now += 1.1
        self.assertRejected(self.hit(reported_at=10.0), "stale_message")
        self.assertRejected(self.hit(reported_at=float("nan")), "invalid_timestamp")
        self.assertTrue(self.hit().accepted)

    def test_all_events_replay_with_rejections_and_json_safe_nonfinite_inputs(self):
        self.ready()
        self.now += 0.1
        self.assertRejected(self.observe("DroneA", eta=float("nan")), "invalid_eta")
        self.assertTrue(self.assign().accepted)
        self.assertTrue(self.dispatch().accepted)
        self.assertRejected(self.hit(stamp=99), "old_collision_timestamp")
        self.finish_round()
        encoded = json.dumps(self.machine.event_log, allow_nan=False)
        replayed = RefreshedTargetRound.replay(json.loads(encoded))
        self.assertEqual(replayed.snapshot(), self.machine.snapshot())
        tampered = self.machine.event_log
        tampered[-1]["decision"]["reason"] = "pretend_success"
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            RefreshedTargetRound.replay(tampered)

    def test_config_clock_and_unknown_events_reject_visibly(self):
        with self.assertRaises(ValueError):
            RoundThresholds(max_eta_age_s=float("inf"))
        self.assertRejected(self.machine.submit("read_target_truth", {}), "unknown_event")
        self.assertTrue(self.prepare().accepted)
        self.now -= 1
        self.assertRejected(self.ack("DroneA"), "clock_regression")
        self.now = float("nan")
        self.assertRejected(self.ack("DroneA"), "invalid_now")

    def test_malformed_transport_messages_reject_and_replay_without_exception(self):
        self.assertRejected(self.machine.submit("prepare_round", None), "invalid_payload")
        self.assertRejected(self.machine.submit("prepare_round", {}), "event_fields_mismatch")
        self.assertRejected(self.machine.submit(["not-an-event"], {}), "unknown_event")
        encoded = json.dumps(self.machine.event_log, allow_nan=False)
        self.assertEqual(RefreshedTargetRound.replay(json.loads(encoded)).snapshot(), self.machine.snapshot())

    def test_rejected_events_do_not_mutate_round_or_emit_actions(self):
        self.ready()
        self.assertTrue(self.assign().accepted)
        self.assertTrue(self.dispatch().accepted)
        before = self.machine.snapshot()
        for stamp in (True, float("inf"), 0, -1):
            self.assertRejected(self.hit(stamp=stamp), "invalid_collision_timestamp")
            self.assertEqual(self.machine.snapshot(), before)
        self.assertTrue(self.hit().accepted)
        before = self.machine.snapshot()
        self.assertRejected(self.hit(), "duplicate_hit")
        self.assertEqual(self.machine.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
