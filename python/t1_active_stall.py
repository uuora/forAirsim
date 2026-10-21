"""Bound a predeclared main-thread stall to an actual nonzero control receipt.

Eligibility uses only dispatch receipts. Recorded event replay must separately
prove actual active overlap and explicit clearing during the blocked interval.
"""
import math


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def active_stall_eligibility(row, now, *, already_triggered=False, cleared_tickets=()):
    result = dict(eligible=False, reason='NO_CURRENT_NONZERO_TARGET_DISPATCH',
                  minimum_remaining_lease_s=.04, input_scope='control dispatch receipts only')
    if already_triggered:
        return dict(result, reason='PREDECLARED_SINGLE_STALL_ALREADY_TRIGGERED')
    if not _finite(now):
        return dict(result, reason='INVALID_STALL_CLOCK')
    command = row.get('command') or {}
    gate = row.get('motion_gate') or {}
    ticket = row.get('ticket_id')
    velocity = command.get('velocity_xy_m_s')
    if (not isinstance(ticket, str) or not ticket or ticket in cleared_tickets
            or command.get('purpose') != 'CURRENT_TARGET_TRAJECTORY'
            or gate.get('target_command_authorized') is not True
            or gate.get('status') != 'ALLOW_BOUNDED_COMMAND'
            or command.get('target_frame_id') != row.get('frame_id')
            or not isinstance(velocity, (list, tuple)) or len(velocity) != 2
            or not all(_finite(v) for v in velocity)
            or not 1e-6 < math.hypot(*velocity) <= .03 + 1e-12):
        return result
    dispatch, duration = command.get('dispatch_monotonic_s'), command.get('duration_s')
    if not _finite(dispatch) or not _finite(duration) or not 0 < duration <= .12:
        return dict(result, reason='INVALID_DECLARED_SHORT_LEASE')
    end = dispatch + duration
    if now < dispatch or end - now < .04:
        return dict(result, reason='INSUFFICIENT_REMAINING_ACTUAL_DISPATCH_LEASE',
                    declared_lease_end_monotonic_s=end)
    return dict(result, eligible=True, reason='ACTUAL_NONZERO_DISPATCH_WITH_AT_LEAST_40MS_LEASE',
                ticket_id=ticket, command_id=command.get('command_id'),
                frame_id=row.get('frame_id'), dispatch_monotonic_s=dispatch,
                declared_lease_end_monotonic_s=end, eligibility_checked_monotonic_s=now,
                remaining_declared_lease_s=end-now)
