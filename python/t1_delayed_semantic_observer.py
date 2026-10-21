"""RGB-only chain-reviewed observer with sanitized transport clocks.

The original source frame remains the semantic time origin. The HTTP proxy
records clocks and closed status categories only; URLs, headers, payloads,
response text and exception messages never enter the evidence record.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import threading
import time

from t1_continuous_semantic_observer import ContinuousSemanticObserver, OBSERVABILITY, create_observer
from t1_delayed_semantic_bridge import (
    DelayedBridgeConfig, DelayedSemanticPolicy, DelayedChainIdentityTracker,
)


def transport_error_category(error):
    import requests
    if isinstance(error, requests.exceptions.JSONDecodeError):
        return "JSON_DECODE_FAILURE"
    if isinstance(error, requests.exceptions.ReadTimeout):
        return "READ_TIMEOUT"
    if isinstance(error, requests.exceptions.ConnectTimeout):
        return "CONNECT_TIMEOUT"
    if isinstance(error, requests.exceptions.HTTPError):
        return "HTTP_STATUS_FAILURE"
    if isinstance(error, requests.exceptions.RequestException):
        return "REQUEST_TRANSPORT_ERROR"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return "VALUE_OR_RESPONSE_SCHEMA_ERROR"
    return "BACKEND_ERROR"


class _ResponseProxy:
    def __init__(self, response, owner, record):
        self.response, self.owner, self.record = response, owner, record

    def raise_for_status(self):
        try:
            self.response.raise_for_status()
        except Exception:
            self.owner.mark(self.record, http_status_check="FAILED")
            raise
        self.owner.mark(self.record, http_status_check="PASSED")

    def json(self):
        started = self.owner.clock()
        try:
            value = self.response.json()
        except Exception:
            self.owner.mark(self.record, json_parse_started_monotonic_s=started,
                json_parse_completed_monotonic_s=self.owner.clock(), json_parse_status="FAILED")
            raise
        self.owner.mark(self.record, json_parse_started_monotonic_s=started,
            json_parse_completed_monotonic_s=self.owner.clock(), json_parse_status="PARSED")
        if not isinstance(value, dict):
            payload_status = "NONOBJECT_PAYLOAD"
        elif "error" in value or ("code" in value and value["code"] != 200):
            payload_status = "SERVICE_ERROR_DECLARED"
        elif not isinstance(value.get("choices"), list) or not value["choices"]:
            payload_status = "NO_CHOICES"
        else:
            payload_status = "CHOICES_PRESENT_CONVERSION_PENDING"
        self.owner.mark(self.record, service_payload_status=payload_status)
        return value

    def __getattr__(self, name):
        return getattr(self.response, name)


class _SessionProxy:
    def __init__(self, session, owner):
        self.session, self.owner = session, owner

    def post(self, *args, **kwargs):
        record = self.owner.active_record
        self.owner.mark(record, http_started_monotonic_s=self.owner.clock())
        try:
            response = self.session.post(*args, **kwargs)
        except Exception as error:
            self.owner.mark(record, http_completed_monotonic_s=self.owner.clock(),
                http_status_class="NO_RESPONSE", transport_error=transport_error_category(error))
            raise
        status = response.status_code
        status_class = f"HTTP_{status // 100}XX" if isinstance(status, int) and 100 <= status <= 599 else "UNKNOWN"
        self.owner.mark(record, http_completed_monotonic_s=self.owner.clock(), http_status_class=status_class)
        return _ResponseProxy(response, self.owner, record)

    def __getattr__(self, name):
        return getattr(self.session, name)


class TracedGroundingBackend:
    """One semantic worker owns transport; main thread only takes copies."""
    def __init__(self, backend, *, clock=time.monotonic):
        self.backend, self.clock = backend, clock
        self.vlx_client = backend.vlx_client
        self.lock = threading.Lock()
        self.records = []
        self.active_record = None
        self.vlx_client.session = _SessionProxy(self.vlx_client.session, self)

    def mark(self, record, **values):
        if record is not None:
            with self.lock:
                record.update(values)

    def snapshot(self):
        with self.lock:
            return deepcopy(self.records)

    def warmup(self, path):
        return self.backend.warmup(path)

    def detect_frame(self, *args, **kwargs):
        return self.backend.detect_frame(*args, **kwargs)

    def __call__(self, request):
        record = dict(request_binding=request.binding(), backend_started_monotonic_s=self.clock(),
            json_parse_status="NOT_ATTEMPTED", grounding_conversion_status="NOT_COMPLETED",
            request_mode="image_text", proposals_transmitted=False)
        with self.lock:
            self.records.append(record)
            self.active_record = record
        try:
            reply = self.backend(request)
        except Exception as error:
            self.mark(record, backend_completed_monotonic_s=self.clock(), backend_status="FAILED",
                error_category=transport_error_category(error), grounding_conversion_status="FAILED")
            raise
        self.mark(record, backend_completed_monotonic_s=self.clock(), backend_status="REPLY_RETURNED",
            reply_completed_monotonic_s=reply.completed_monotonic_s,
            grounding_conversion_status="CONVERTED",
            returned_box_count=len(reply.result.get("result_bbox_list", [])))
        return reply


class DelayedSemanticObserver(ContinuousSemanticObserver):
    def __init__(self, backend, *, clock=time.monotonic, policy_factory=DelayedSemanticPolicy,
                 tracker_factory=DelayedChainIdentityTracker, model_receipt=None):
        super().__init__(backend, clock=clock, policy_factory=policy_factory,
            tracker_factory=tracker_factory, review_period_s=3., model_receipt=model_receipt)

    def begin(self, session_id, description, warmup_rgb_path, *, recovery_observability):
        if self.policy is not None or self._closed:
            raise ValueError("Observer requires one new instance per session")
        if recovery_observability not in OBSERVABILITY:
            raise ValueError("Explicit recovery observability declaration required")
        self.session_id, self.description = session_id, description
        self.recovery_observability = recovery_observability
        self.warmup_receipt = self.backend.warmup(warmup_rgb_path)
        self.policy = self.policy_factory(self.backend, session_id=session_id, target_generation=1,
            description=description, config=DelayedBridgeConfig(), clock=self.clock)
        self.tracker = self.tracker_factory(session_id, 1, description)
        return dict(warmup=self.warmup_receipt, model=self.model_receipt,
            policy_config=asdict(self.policy.config), tracker_config=asdict(self.tracker.config),
            review_period_s=self.review_period_s, recovery_observability=recovery_observability,
            input_scope="RGB + self camera + constant task description only",
            time_origin="original semantic source readback; never rebased")

    def process(self, observation):
        result = super().process(observation)
        if hasattr(self.backend, "snapshot"):
            result["sanitized_transport_trace"] = self.backend.snapshot()
        result.update(semantic_policy="EVERY_FRAME_UNIQUE_CHAIN_REVIEW_V3",
            source_age_budget_s=8., transport_timeout_setting_s=10.,
            transport_timeout_scope="requests connect/read setting; not a total wall-clock guarantee")
        return result

    def close(self):
        if self._closed:
            return self.close_receipt
        started = self.clock()
        # Flight runner invokes this after independent stop/landing. Joining
        # cannot extend a target command or lend authority to a late result.
        policy_receipt = self.policy.close(wait=True) if self.policy is not None else None
        if self.tracker is not None:
            self.tracker.close()
        self._closed = True
        self.close_receipt = dict(status="CLOSED", processed_frames=self._process_count,
            request_submissions=self._submitted_count,
            valid_semantic_current_frame_certificates=self._valid_count,
            unobservable_refusal_frames=self._observability_refusal_count,
            final_tracker=self._last_tracker, vlx_usage_http_response_counters=self._usage(),
            model_binding=self.model_receipt,
            policy_config=asdict(self.policy.config) if self.policy else None,
            tracker_config=asdict(self.tracker.config) if self.tracker else None,
            request_terminal_trace=policy_receipt,
            sanitized_transport_trace=self.backend.snapshot() if hasattr(self.backend, "snapshot") else [],
            close_wall_s=self.clock()-started, ground_only=True, flight_authorized=False)
        return self.close_receipt


class DelayedFlightObserver:
    def __init__(self, detector_config_path):
        original = create_observer(detector_config_path)
        # Existing factory performs its private credential loading internally.
        # No credentials or endpoint values are read into evidence here.
        original.backend.vlx_client.timeout = 10.
        backend = TracedGroundingBackend(original.backend)
        self.core = DelayedSemanticObserver(backend, model_receipt=original.model_receipt)
        self.close_receipt = None

    def begin(self, *args, **kwargs):
        return self.core.begin(*args, **kwargs)

    def process(self, observation):
        result = self.core.process(observation)
        result.update(ground_only=False, platform_stage="AIRBORNE_RGB_CHAIN_REVIEW",
            motion_authority="separate_current_frame_and_self_state_guard", flight_authorized=False)
        return result

    def close(self):
        self.close_receipt = dict(self.core.close())
        self.close_receipt.update(ground_only=False, platform_stage="AIRBORNE_RGB_CHAIN_REVIEW")
