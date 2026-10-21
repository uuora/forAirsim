"""Select post-request queries by the unique origin record, not coarse host time."""
from braking_response import summarize_pulse as legacy_summary


def summarize_pulse(pulse, measurement, model):
    brake=pulse.get('brake_first_request')
    if brake is None:return legacy_summary(pulse,measurement,model)
    rows=pulse.get('samples',[]);origin=brake['origin_sample']
    indices=[i for i,row in enumerate(rows) if row==origin]
    if len(indices)!=1:
        return dict(status='ORIGIN_QUERY_NOT_UNIQUE',counts_as_measured_stopping=False,
            origin_selection='explicit_unique_saved_query_index')
    following=rows[indices[0]+1:]
    if any(row['host_monotonic_s']<brake['dispatch_host_s'] for row in following):
        return dict(status='POST_ORIGIN_QUERY_BEFORE_REQUEST',counts_as_measured_stopping=False,
            origin_selection='explicit_unique_saved_query_index')
    result=legacy_summary(dict(pulse,samples=following),measurement,model)
    result['origin_selection']='explicit_unique_saved_query_index'
    return result
