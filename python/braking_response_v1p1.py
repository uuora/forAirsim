"""Bounded measured-velocity acquisition for the next braking protocol version."""
import math
from braking_response import entry_check, validate_measurement


def validate_acquisition(config):
    m = validate_measurement(config)
    for key in ('entry_acquisition_timeout_s', 'entry_stable_s'):
        value = m[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid entry acquisition parameter: '+key)
    if m['entry_acquisition_timeout_s'] > 3. or not .25 <= m['entry_stable_s'] <= .5:
        raise ValueError('Entry acquisition outside declared development range')
    return m


def acquired_velocity_interval(samples, pulse, measurement):
    """Require a fresh continuous acceptable suffix, not a single chosen sample."""
    suffix = []
    for sample in reversed(samples):
        if sample['protection_reasons'] or not entry_check(sample, pulse, measurement)['accepted']:
            break
        if suffix and (not 0 < suffix[-1]['host_monotonic_s']-sample['host_monotonic_s'] <= measurement['max_query_gap_s']
            or suffix[-1]['query_state_timestamp_ns'] <= sample['query_state_timestamp_ns']):
            break
        suffix.append(sample)
    elapsed = suffix[0]['host_monotonic_s']-suffix[-1]['host_monotonic_s'] if suffix else 0.
    return dict(accepted=bool(suffix and elapsed >= measurement['entry_stable_s']),
        acceptable_suffix_sample_count=len(suffix), acceptable_suffix_host_s=elapsed,
        accepted_state=suffix[0] if suffix and elapsed >= measurement['entry_stable_s'] else None)
