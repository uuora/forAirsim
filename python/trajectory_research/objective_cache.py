"""Exact reuse of complete measurements from a declared deterministic oracle.

Memoization is a standard baseline, not a new optimization principle. This
wrapper is appropriate only when the objective, reset state, environment,
controller, numerical settings and all random seeds remain fixed. The caller
must explicitly assert that contract and identify its scope with context_id.
It is not suitable for an evolving environment, stochastic flights, or an
objective whose side effects affect subsequent search decisions.

Inputs are copied into an exact float64 domain. No decimal rounding or
near-neighbour matching occurs. Distinct float64 bit patterns, including
signed zero, remain distinct keys. A returned payload is an independent deep
copy; its timing fields still describe the ORIGINAL measurement. Consumers
must persist the separate ledger to distinguish CACHE_HIT from a fresh call.
The searcher's own 'fe' counter is a proposal counter when this wrapper is
used; stats['oracle_calls'] counts actual underlying calls instead.

The cache is local to one instance and supports sequential callers only.
No empirical determinism or physical safety is established by this wrapper.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from numbers import Complex, Real
import time

import numpy as np


def _parameters(value):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("parameters must be a nonempty finite real vector") from error
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf":
        raise ValueError("parameters must be a nonempty finite real vector")
    if not np.isfinite(raw).all():
        raise ValueError("parameters must be a nonempty finite real vector")
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.array(raw, dtype="<f8", order="C", copy=True)
    if not np.isfinite(vector).all():
        raise ValueError("parameters exceed the finite float64 domain")
    # Do not quietly collapse different integers or extended-precision floats
    # onto a common float64 key. Ordinary float32 inputs convert exactly.
    if raw.dtype.kind in "iu":
        if any(int(a) != float(b) for a, b in zip(raw, vector)):
            raise ValueError("integer parameters must be exactly representable in float64")
    elif raw.dtype.itemsize > np.dtype("float64").itemsize:
        if not np.array_equal(raw, vector.astype(raw.dtype)):
            raise ValueError("parameters must be exactly representable in float64")
    return vector


def _score(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("objective score must be a finite real scalar")
    try:
        score = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("objective score must be a finite real scalar") from error
    if not math.isfinite(score):
        raise ValueError("objective score must be a finite real scalar")
    return score


def _finite_payload(value):
    """Reject nonfinite numeric payloads without changing their representation."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_payload(key)
            _finite_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_payload(item)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "iufb":
            if not np.isfinite(value).all():
                raise ValueError("objective payload contains a nonfinite number")
        elif value.dtype.kind == "c":
            raise ValueError("objective payload contains non-real numbers")
        elif value.dtype.kind == "O":
            for item in value.flat:
                _finite_payload(item)
    elif isinstance(value, (bool, np.bool_)):
        return
    elif isinstance(value, Real):
        try:
            finite = math.isfinite(float(value))
        except (ValueError, OverflowError) as error:
            raise ValueError("objective payload contains a nonfinite number") from error
        if not finite:
            raise ValueError("objective payload contains a nonfinite number")
    elif isinstance(value, Complex):
        raise ValueError("objective payload contains non-real numbers")


@dataclass(frozen=True)
class _Entry:
    result: dict
    source_evaluation_id: int
    score: float
    feasible: bool
    key_digest: str
    oracle_wall_s: float


class DeterministicObjectiveCache:
    """Callable exact cache with a separate measurement-provenance ledger.

    ``deterministic=True`` is required, rather than silently assuming that a
    simulator or experiment is repeatable. Create a NEW instance when context
    changes. max_entries=None is unbounded within this instance; otherwise
    positive integer capacity uses LRU replacement, including cache-hit refresh.

    Results require finite real score, boolean feasible, and a complete status
    (absent/None or EVALUATED). Infeasible complete results are cacheable.
    Exceptions, invalid results and partial PRUNED results are never cached.
    Error attempts are recorded and re-raised. Invalid input requests fail
    before becoming proposals. NumPy-valued payloads remain NumPy-valued; only
    the ledger/stats promise standard JSON-compatible finite values.
    """

    def __init__(self, objective, context_id, *, deterministic=False, max_entries=None):
        if not callable(objective):
            raise ValueError("objective must be callable")
        if not isinstance(context_id, str) or not context_id.strip():
            raise ValueError("a nonempty immutable context_id is required")
        if deterministic is not True:
            raise ValueError("caller must explicitly assert deterministic=True")
        if max_entries is not None and (isinstance(max_entries, (bool, np.bool_))
                or not isinstance(max_entries, (int, np.integer)) or max_entries < 1):
            raise ValueError("max_entries must be a positive integer or None")
        self._objective = objective
        self._context_id = context_id
        context = context_id.encode("utf-8")
        self._digest_prefix = len(context).to_bytes(8, "big") + context
        self._max_entries = None if max_entries is None else int(max_entries)
        self._cache = OrderedDict()
        self._ledger = []
        self._proposals = self._oracle_calls = self._hits = self._successes = self._failures = 0
        self._evictions = self._peak_entries = 0
        self._lookup_wall_s = self._oracle_wall_s = self._miss_wall_s = self._hit_wall_s = 0.

    @property
    def context_id(self):
        return self._context_id

    @property
    def max_entries(self):
        return self._max_entries

    @property
    def ledger(self):
        return deepcopy(self._ledger)

    @property
    def last_record(self):
        return None if not self._ledger else deepcopy(self._ledger[-1])

    @property
    def stats(self):
        return dict(context_id=self._context_id, deterministic_contract=True,
            proposals=self._proposals, oracle_calls=self._oracle_calls,
            successful_evaluations=self._successes, failed_evaluations=self._failures,
            hits=self._hits, cache_entries=len(self._cache), peak_cache_entries=self._peak_entries,
            max_entries=self._max_entries, evictions=self._evictions,
            lookup_wall_s=self._lookup_wall_s, oracle_wall_s=self._oracle_wall_s,
            miss_wall_s=self._miss_wall_s, hit_wall_s=self._hit_wall_s,
            key_policy="exact little-endian float64 bits plus vector size; no rounding",
            payload_timing_scope="cached payload timings remain the original measurement",
            accounting="oracle_calls includes failed attempts; proposal count = oracle_calls + hits")

    def __call__(self, parameters):
        started = time.perf_counter()
        vector = _parameters(parameters)
        key = vector.size.to_bytes(8, "big") + vector.tobytes(order="C")
        digest = hashlib.sha256(self._digest_prefix + key).hexdigest()
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
        lookup_s = time.perf_counter() - started
        self._lookup_wall_s += lookup_s
        self._proposals += 1
        common = dict(proposal_index=self._proposals, context_id=self._context_id,
            key_digest=digest, parameters=vector.tolist(), lookup_wall_s=lookup_s)
        if entry is not None:
            result = deepcopy(entry.result)
            duration = time.perf_counter() - started
            self._hits += 1
            self._hit_wall_s += duration
            self._ledger.append(dict(**common, status="CACHE_HIT", result_reused=True,
                source_evaluation_id=entry.source_evaluation_id, score=entry.score,
                feasible=entry.feasible, oracle_wall_s=0.,
                source_oracle_wall_s=entry.oracle_wall_s, proposal_wall_s=duration,
                payload_timing_scope="original measurement; not a fresh evaluation"))
            return result

        self._oracle_calls += 1
        evaluation_id = self._oracle_calls
        oracle_started = time.perf_counter()
        oracle_s = None
        try:
            original = self._objective(vector.copy())
            oracle_s = time.perf_counter() - oracle_started
            if not isinstance(original, Mapping) or "score" not in original or "feasible" not in original:
                raise ValueError("objective must return score and boolean feasible")
            if original.get("status") not in (None, "EVALUATED"):
                raise ValueError("only complete objective results may be cached")
            score = _score(original["score"])
            if not isinstance(original["feasible"], (bool, np.bool_)):
                raise ValueError("objective feasibility must be boolean")
            feasible = bool(original["feasible"])
            _finite_payload(original)
            stored = deepcopy(dict(original))
            result = deepcopy(stored)
        except BaseException as error:
            if oracle_s is None:
                oracle_s = time.perf_counter() - oracle_started
            duration = time.perf_counter() - started
            self._oracle_wall_s += oracle_s
            self._miss_wall_s += duration
            self._failures += 1
            self._ledger.append(dict(**common, status="EVALUATION_ERROR", result_reused=False,
                source_evaluation_id=evaluation_id, score=None, feasible=None,
                oracle_wall_s=oracle_s, source_oracle_wall_s=oracle_s,
                proposal_wall_s=duration, error_type=type(error).__name__))
            raise
        self._cache[key] = _Entry(stored, evaluation_id, score, feasible, digest, oracle_s)
        evicted = None
        if self._max_entries is not None and len(self._cache) > self._max_entries:
            _, evicted = self._cache.popitem(last=False)
            self._evictions += 1
        self._peak_entries = max(self._peak_entries, len(self._cache))
        self._successes += 1
        duration = time.perf_counter() - started
        self._oracle_wall_s += oracle_s
        self._miss_wall_s += duration
        self._ledger.append(dict(**common, status="FULL_EVALUATION", result_reused=False,
            source_evaluation_id=evaluation_id, score=score, feasible=feasible,
            oracle_wall_s=oracle_s, source_oracle_wall_s=oracle_s, proposal_wall_s=duration,
            evicted_source_evaluation_id=None if evicted is None else evicted.source_evaluation_id,
            evicted_key_digest=None if evicted is None else evicted.key_digest,
            payload_timing_scope="original measurement"))
        return result


__all__ = ["DeterministicObjectiveCache"]
