"""Evaluation-only T1 accounting on an explicitly declared time grid.

All expected ticks and confirmed-started rounds remain in their denominators.
Truth belongs to independent evaluation: this module imports no controller,
detector, simulator, or semantic policy. Intervals are [start, end); the last
interval's end must be declared rather than extrapolated from surviving outputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable


def _time(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class EvaluationTick:
    tick_id: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        _id(self.tick_id, "tick_id")
        if _time(self.end_s, "tick end") <= _time(self.start_s, "tick start"):
            raise ValueError("tick end must exceed its start")


@dataclass(frozen=True)
class EvaluationGrid:
    ticks: tuple[EvaluationTick, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ticks, tuple) or not self.ticks:
            raise ValueError("declare a nonempty immutable evaluation grid")
        seen = set()
        previous_end = None
        for tick in self.ticks:
            if not isinstance(tick, EvaluationTick) or tick.tick_id in seen:
                raise ValueError("invalid or duplicate expected tick")
            seen.add(tick.tick_id)
            if previous_end is not None and tick.start_s != previous_end:
                raise ValueError("evaluation intervals must be ordered and contiguous")
            previous_end = tick.end_s

    @classmethod
    def from_times(cls, times: Iterable[float], *, end_time_s: float,
                   tick_ids: Iterable[str] | None = None) -> EvaluationGrid:
        starts = tuple(_time(value, "grid time") for value in times)
        end = _time(end_time_s, "declared grid end")
        if not starts or any(right <= left for left, right in zip(starts, starts[1:])) or end <= starts[-1]:
            raise ValueError("grid times must be strictly increasing and precede its end")
        ids = tuple(tick_ids) if tick_ids is not None else tuple(f"tick-{index:06d}" for index in range(len(starts)))
        if len(ids) != len(starts):
            raise ValueError("tick ID count differs from expected time count")
        ends = starts[1:] + (end,)
        return cls(tuple(EvaluationTick(identity, start, finish)
                         for identity, start, finish in zip(ids, starts, ends)))

    @property
    def total_duration_s(self) -> float:
        return float(self.ticks[-1].end_s - self.ticks[0].start_s)


@dataclass(frozen=True)
class TruthTick:
    tick_id: str
    target_identity: str | None
    target_visible: bool | None
    coverage_status: str = "VERIFIED"

    def __post_init__(self) -> None:
        _id(self.tick_id, "truth tick ID")
        if self.coverage_status not in {"VERIFIED", "MISSING", "UNCERTAIN"}:
            raise ValueError("invalid truth coverage status")
        if self.coverage_status == "VERIFIED":
            _id(self.target_identity, "verified target identity")
            if not isinstance(self.target_visible, bool):
                raise ValueError("verified visibility must be explicitly true or false")
        else:
            if self.target_identity is not None:
                _id(self.target_identity, "unverified target identity")
            if self.target_visible is not None and not isinstance(self.target_visible, bool):
                raise ValueError("unverified visibility must be bool or None")


@dataclass(frozen=True)
class PredictionTick:
    """OUTPUT is independently validated at this tick; stale labels are INVALID.

    The evaluator must check freshness and association before setting OUTPUT.
    This module does not infer invalidity from target visibility alone.
    """
    tick_id: str
    selected_identity: str | None = None
    status: str = "NO_OUTPUT"

    def __post_init__(self) -> None:
        _id(self.tick_id, "prediction tick ID")
        if self.status not in {"OUTPUT", "NO_OUTPUT", "INVALID"}:
            raise ValueError("invalid prediction status")
        if self.status == "OUTPUT":
            _id(self.selected_identity, "selected identity")
        elif self.selected_identity is not None:
            raise ValueError("a missing/invalid observation cannot carry a selected identity")


def _indexed(rows: Iterable[Any], expected: set[str], row_type: type, name: str) -> dict[str, Any]:
    result = {}
    for row in rows:
        if not isinstance(row, row_type):
            raise ValueError(f"invalid {name} row")
        if row.tick_id not in expected or row.tick_id in result:
            raise ValueError(f"unexpected or duplicate {name} tick")
        result[row.tick_id] = row
    return result


def _spans(rows: list[dict[str, Any]], included: set[str], *,
           state_key: str = "evaluation_state") -> list[dict[str, Any]]:
    spans = []
    for row in rows:
        if row[state_key] not in included:
            continue
        if spans and spans[-1]["end_s"] == row["start_s"]:
            spans[-1]["end_s"] = row["end_s"]
            spans[-1]["tick_ids"].append(row["tick_id"])
        else:
            spans.append({"start_s": row["start_s"], "end_s": row["end_s"],
                          "tick_ids": [row["tick_id"]]})
    for span in spans:
        span["duration_s"] = span["end_s"] - span["start_s"]
    return spans


def _reacquisitions(rows: list[dict[str, Any]], timeout_s: float, *,
                    state_key: str = "evaluation_state") -> list[dict[str, Any]]:
    episodes = []
    previous_correct = False
    active = None
    for row in rows:
        state = row[state_key]
        if state == "UNKNOWN_TRUTH":
            if active is not None:
                active.update(end_s=row["start_s"], duration_s=row["start_s"] - active["start_s"],
                              status="TRUTH_GAP_BEFORE_REACQUISITION", reacquired=False)
                episodes.append(active)
                active = None
            previous_correct = False
            continue
        if state == "LOSS" and previous_correct and active is None:
            active = {"start_s": row["start_s"], "loss_start_tick_id": row["tick_id"],
                      "timeout_s": timeout_s}
        if state == "CORRECT" and active is not None:
            duration = row["start_s"] - active["start_s"]
            active.update(end_s=row["start_s"], duration_s=duration, reacquired=True,
                          recovery_tick_id=row["tick_id"],
                          status="REACQUIRED_WITHIN_TIMEOUT" if duration <= timeout_s else "REACQUIRED_AFTER_TIMEOUT")
            episodes.append(active)
            active = None
        previous_correct = state == "CORRECT"
    if active is not None:
        end = rows[-1]["end_s"]
        duration = end - active["start_s"]
        active.update(end_s=end, duration_s=duration, reacquired=False,
                      status="TIMED_OUT_STILL_LOST" if duration >= timeout_s else "STILL_LOST_AT_END_BEFORE_TIMEOUT")
        episodes.append(active)
    return episodes


def _signal_summary(rows: list[dict[str, Any]], *, signal: str, complete_truth: bool,
                    timeout_s: float) -> dict[str, Any]:
    state_key = f"{signal}_state"
    known_loss = _spans(rows, {"LOSS"}, state_key=state_key)
    conservative_loss = _spans(rows, {"LOSS", "UNKNOWN_TRUTH"}, state_key=state_key)
    correct_spans = _spans(rows, {"CORRECT"}, state_key=state_key)
    lower_loss = max((span["duration_s"] for span in known_loss), default=0.0)
    upper_loss = max((span["duration_s"] for span in conservative_loss), default=0.0)
    episodes = _reacquisitions(rows, timeout_s, state_key=state_key)
    first_correct = next((row["start_s"] for row in rows if row[state_key] == "CORRECT"), None)
    return {"signal": signal, "longest_loss_s": lower_loss if complete_truth else None,
            "longest_loss_lower_bound_s": lower_loss, "longest_loss_upper_bound_s": upper_loss,
            "longest_correct_state_s": max((span["duration_s"] for span in correct_spans), default=0.0),
            "initial_acquisition_duration_s": first_correct - rows[0]["start_s"] if complete_truth and first_correct is not None else None,
            "initial_acquisition_status": "ACQUIRED" if complete_truth and first_correct is not None else
                                          "NOT_ACQUIRED" if complete_truth else "TRUTH_INCOMPLETE",
            "known_loss_spans": known_loss, "correct_state_spans": correct_spans,
            "reacquisition_evaluation_status": "COMPLETE" if complete_truth else "INCOMPLETE_TRUTH_COVERAGE",
            "reacquisition_episodes": episodes,
            "reacquired_within_timeout_count": sum(item["status"] == "REACQUIRED_WITHIN_TIMEOUT" for item in episodes),
            "reacquisition_failure_or_unfinished_count": sum(item["status"] != "REACQUIRED_WITHIN_TIMEOUT" for item in episodes)}


def evaluate_identity_round(grid: EvaluationGrid, truth_rows: Iterable[TruthTick],
                            prediction_rows: Iterable[PredictionTick], *, target_identity: str,
                            reacquisition_timeout_s: float,
                            loss_signal: str = "identity") -> dict[str, Any]:
    """Keep identity correctness separate from effective visible observation.

    Identity credit requires independently verified truth and valid OUTPUT with
    selected_identity == target_identity; visibility is a separate condition.
    Effective visible observation requires identity credit AND target_visible.
    A copied stale label must be passed as INVALID/NO_OUTPUT by the independent
    evaluator. If association cannot be verified, truth coverage is MISSING or
    UNCERTAIN; it is not silently converted to an observed identity error.

    Both loss/reacquisition sequences are returned. Generic loss fields follow
    the explicitly registered ``loss_signal`` (identity by default). The all-
    tick denominator is identical for both rates, including occlusion and gaps.
    """
    if not isinstance(grid, EvaluationGrid):
        raise ValueError("an explicit EvaluationGrid is required")
    _id(target_identity, "registered target identity")
    if loss_signal not in {"identity", "effective_visible_observation"}:
        raise ValueError("loss_signal must name identity or effective_visible_observation")
    timeout = _time(reacquisition_timeout_s, "reacquisition timeout")
    if timeout == 0:
        raise ValueError("reacquisition timeout must be positive")
    ids = {tick.tick_id for tick in grid.ticks}
    truths = _indexed(truth_rows, ids, TruthTick, "truth")
    predictions = _indexed(prediction_rows, ids, PredictionTick, "prediction")
    counts = {name: 0 for name in ("truth_verified_ticks", "truth_missing_ticks", "truth_uncertain_ticks",
        "prediction_missing_ticks", "no_output_ticks", "invalid_output_ticks", "output_ticks",
        "correct_ticks", "effective_visible_observation_ticks", "wrong_identity_ticks", "occluded_ticks", "visible_ticks",
        "nonvisible_target_output_ticks", "direct_identity_switch_count", "target_to_other_switch_count",
        "other_to_target_switch_count", "other_to_other_switch_count", "identity_change_after_gap_count")}
    rows = []
    previous_adjacent_output = None
    previous_known_output = None
    gap_since_output = False
    for tick in grid.ticks:
        truth = truths.get(tick.tick_id)
        prediction = predictions.get(tick.tick_id)
        truth_status = truth.coverage_status if truth is not None else "MISSING"
        counts[f"truth_{truth_status.lower()}_ticks"] += 1
        if prediction is None:
            counts["prediction_missing_ticks"] += 1
        prediction_status = prediction.status if prediction is not None else "NO_OUTPUT"
        selected = prediction.selected_identity if prediction is not None else None
        counts[{"OUTPUT": "output_ticks", "NO_OUTPUT": "no_output_ticks", "INVALID": "invalid_output_ticks"}[
            prediction_status]] += 1
        correct = False
        effective_visible = False
        reason = "UNKNOWN_TRUTH"
        if truth_status == "VERIFIED":
            if truth.target_identity != target_identity:
                raise ValueError("verified truth target differs from the registered designated instance")
            counts["visible_ticks" if truth.target_visible else "occluded_ticks"] += 1
            correct = prediction_status == "OUTPUT" and selected == target_identity
            effective_visible = correct and truth.target_visible
            if correct:
                counts["correct_ticks"] += 1
                if effective_visible:
                    counts["effective_visible_observation_ticks"] += 1
                    reason = "CORRECT_IDENTITY_AND_VISIBLE"
                else:
                    counts["nonvisible_target_output_ticks"] += 1
                    reason = "CORRECT_IDENTITY_TARGET_NONVISIBLE"
            elif prediction_status != "OUTPUT":
                reason = "NO_VALID_OBSERVATION"
            elif selected != target_identity:
                counts["wrong_identity_ticks"] += 1
                reason = "OTHER_INSTANCE"
        # Direct switches require adjacent, independently evaluable outputs.
        # A missing output breaks adjacency and is retained as a loss interval.
        if truth_status != "VERIFIED":
            previous_adjacent_output = previous_known_output = None
            gap_since_output = False
        elif prediction_status == "OUTPUT":
            if previous_adjacent_output is not None and selected != previous_adjacent_output:
                counts["direct_identity_switch_count"] += 1
                direction = ("target_to_other" if previous_adjacent_output == target_identity else
                             "other_to_target" if selected == target_identity else "other_to_other")
                counts[f"{direction}_switch_count"] += 1
            elif gap_since_output and previous_known_output is not None and selected != previous_known_output:
                counts["identity_change_after_gap_count"] += 1
            previous_adjacent_output = previous_known_output = selected
            gap_since_output = False
        else:
            previous_adjacent_output = None
            gap_since_output = previous_known_output is not None
        rows.append({"tick_id": tick.tick_id, "start_s": tick.start_s, "end_s": tick.end_s,
                     "duration_s": tick.end_s - tick.start_s, "truth_coverage": truth_status,
                     "target_visible": truth.target_visible if truth_status == "VERIFIED" else None,
                     "prediction_present": prediction is not None,
                     "prediction_status": prediction_status, "selected_identity": selected,
                     "identity_correct": correct if truth_status == "VERIFIED" else None,
                     "effective_visible_observation_correct": effective_visible if truth_status == "VERIFIED" else None,
                     "identity_state": "UNKNOWN_TRUTH" if truth_status != "VERIFIED" else
                                       "CORRECT" if correct else "LOSS",
                     "effective_visible_observation_state": "UNKNOWN_TRUTH" if truth_status != "VERIFIED" else
                                         "CORRECT" if effective_visible else "LOSS",
                     "evaluation_state": "UNKNOWN_TRUTH" if truth_status != "VERIFIED" else
                                         "CORRECT" if (correct if loss_signal == "identity" else effective_visible) else "LOSS",
                     "reason": reason})
    complete_truth = counts["truth_verified_ticks"] == len(grid.ticks)
    correct_duration = sum(row["duration_s"] for row in rows if row["identity_state"] == "CORRECT")
    effective_visible_duration = sum(row["duration_s"] for row in rows if row["effective_visible_observation_state"] == "CORRECT")
    visible_duration = sum(row["duration_s"] for row in rows if row["target_visible"] is True)
    verified_duration = sum(row["duration_s"] for row in rows if row["truth_coverage"] == "VERIFIED")
    unknown_duration = grid.total_duration_s - verified_duration
    signal_metrics = {signal: _signal_summary(rows, signal=signal, complete_truth=complete_truth, timeout_s=timeout)
                      for signal in ("identity", "effective_visible_observation")}
    primary = dict(signal_metrics[loss_signal])
    primary.pop("signal")
    return {"status": "FULLY_EVALUABLE" if complete_truth else "INCOMPLETE_TRUTH_COVERAGE",
            "target_identity": target_identity,
            "denominator_policy": "all predeclared evaluation intervals; no-output and occluded ticks retained",
            "formula_identity_correct_allticks": "sum(dt * valid_OUTPUT * identity_matches_designated) / sum(dt)",
            "formula_effective_visible_observation": "sum(dt * valid_OUTPUT * identity_matches_designated * target_visible) / sum(dt)",
            "valid_OUTPUT_contract": "freshness and independent instance association validated by evaluator; stale labels must be INVALID",
            "expected_ticks": len(grid.ticks), "evaluation_duration_s": grid.total_duration_s,
            **counts, "truth_verified_duration_s": verified_duration,
            "truth_unknown_duration_s": unknown_duration,
            "identity_correct_duration_s": correct_duration,
            "identity_correct_allticks_time_fraction": correct_duration / grid.total_duration_s if complete_truth else None,
            "identity_correct_tick_fraction": counts["correct_ticks"] / len(grid.ticks) if complete_truth else None,
            "identity_correct_time_fraction": correct_duration / grid.total_duration_s if complete_truth else None,
            "identity_correct_time_fraction_lower_bound": correct_duration / grid.total_duration_s,
            "identity_correct_time_fraction_upper_bound": (correct_duration + unknown_duration) / grid.total_duration_s,
            "effective_visible_observation_duration_s": effective_visible_duration,
            "effective_visible_observation_tick_fraction": counts["effective_visible_observation_ticks"] / len(grid.ticks) if complete_truth else None,
            "effective_visible_observation_time_fraction": effective_visible_duration / grid.total_duration_s if complete_truth else None,
            "effective_visible_observation_time_fraction_lower_bound": effective_visible_duration / grid.total_duration_s,
            "effective_visible_observation_time_fraction_upper_bound": (effective_visible_duration + unknown_duration) / grid.total_duration_s,
            "target_visible_time_fraction": visible_duration / grid.total_duration_s if complete_truth else None,
            "loss_signal": loss_signal, "loss_and_reacquisition_signals": signal_metrics,
            "longest_identity_loss_s": signal_metrics["identity"]["longest_loss_s"],
            "longest_effective_visible_observation_loss_s": signal_metrics["effective_visible_observation"]["longest_loss_s"],
            "identity_reacquisition_episodes": signal_metrics["identity"]["reacquisition_episodes"],
            "effective_visible_observation_reacquisition_episodes": signal_metrics["effective_visible_observation"]["reacquisition_episodes"],
            **primary,
            "ticks": rows, "truth_used_for_control": False,
            "confidence_calibration_estimated": False}


@dataclass(frozen=True)
class RoundRecord:
    round_id: str
    start_state: str
    outcome: str
    safety_passed: bool | None = None
    identity_and_observation_passed: bool | None = None

    def __post_init__(self) -> None:
        _id(self.round_id, "round ID")
        if self.start_state not in {"CONFIRMED_STARTED", "CONFIRMED_NOT_STARTED", "UNKNOWN"}:
            raise ValueError("invalid round start state")
        if self.outcome not in {"SUCCESS", "FAILED", "INTERRUPTED", "NO_FEASIBLE_TRAJECTORY",
                                "NO_REACQUISITION", "NOT_STARTED", "UNKNOWN"}:
            raise ValueError("invalid round outcome")
        for value in (self.safety_passed, self.identity_and_observation_passed):
            if value is not None and not isinstance(value, bool):
                raise ValueError("round gates must be bool or None")
        if self.start_state == "CONFIRMED_STARTED" and self.outcome == "NOT_STARTED":
            raise ValueError("started round has contradictory not-started outcome")
        if self.start_state == "CONFIRMED_NOT_STARTED" and self.outcome == "SUCCESS":
            raise ValueError("a not-started round cannot succeed")


def round_success_summary(records: Iterable[RoundRecord], *,
                          expected_round_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Use documented start evidence; never infer starts or discard failures."""
    records = tuple(records)
    seen = set()
    for record in records:
        if not isinstance(record, RoundRecord) or record.round_id in seen:
            raise ValueError("invalid or duplicate round record")
        seen.add(record.round_id)
    expected = tuple(expected_round_ids) if expected_round_ids is not None else tuple(
        record.round_id for record in records)
    expected_set = set()
    for identity in expected:
        _id(identity, "expected round ID")
        if identity in expected_set:
            raise ValueError("duplicate expected round ID")
        expected_set.add(identity)
    if not seen <= expected_set:
        raise ValueError("round record outside the expected round registry")
    missing_ids = [identity for identity in expected if identity not in seen]
    by_id = {record.round_id: record for record in records}
    records = tuple(by_id.get(identity, RoundRecord(identity, "UNKNOWN", "UNKNOWN")) for identity in expected)
    started = [record for record in records if record.start_state == "CONFIRMED_STARTED"]
    successful = [record for record in started if record.outcome == "SUCCESS"
                  and record.safety_passed is True and record.identity_and_observation_passed is True]
    verified_success_ids = {record.round_id for record in successful}
    unknown_start = [record.round_id for record in records if record.start_state == "UNKNOWN"]
    return {"registered_rounds": len(records), "missing_round_records": len(missing_ids),
            "missing_round_record_ids": missing_ids,
            "expected_registry_supplied": expected_round_ids is not None,
            "confirmed_started_rounds": len(started),
            "verified_successful_rounds": len(successful),
            "task_success_all_confirmed_started": len(successful) / len(started) if started else None,
            "confirmed_started_non_successful_rounds": len(started) - len(successful),
            "confirmed_not_started_rounds": sum(record.start_state == "CONFIRMED_NOT_STARTED" for record in records),
            "unknown_start_rounds": len(unknown_start), "unknown_start_round_ids": unknown_start,
            "start_reconciliation_status": "COMPLETE" if not unknown_start else "UNKNOWN_STARTS_REMAIN",
            "successful_round_ids": [record.round_id for record in successful],
            "non_successful_started_round_ids": [record.round_id for record in started if record.round_id not in verified_success_ids],
            "declared_success_missing_or_failed_gate_round_ids": [record.round_id for record in started
                if record.outcome == "SUCCESS" and record.round_id not in verified_success_ids],
            "started_outcome_counts": {outcome: sum(record.outcome == outcome for record in started)
                                      for outcome in sorted({record.outcome for record in started})},
            "denominator_policy": "all confirmed-started rounds, including interrupted/unfeasible/unknown outcomes"}
