"""Declared .4 s schedule with shorter intervals only at phase boundaries."""
import math

PHASES=("initial_pair","target_hidden_with_distractor_at_old_target","return_target_right")


def boundary_clipped_slots(configuration):
    if (configuration.get("timing_grid") != "BOUNDARY_CLIPPED_0P4"
            or configuration.get("minimum_period_s") != .4):
        raise ValueError("Explicit .4-second boundary-clipped protocol required")
    phases=configuration["phases"]
    if [row["phase_id"] for row in phases]!=list(PHASES):
        raise ValueError("Three declared phases required")
    rows,start=[],0.
    for phase,duration in zip(phases,(5.,2.,5.)):
        if phase["host_duration_s"]!=duration:
            raise ValueError("Phase durations changed")
        count=math.ceil(duration/.4-1e-9)
        for index in range(count):
            a=round(start+index*.4,10)
            b=round(min(start+duration,start+(index+1)*.4),10)
            if b-a < .2-1e-9:
                raise ValueError("Unexpected short boundary interval")
            rows.append({"frame_sequence":len(rows)+1,"phase_id":phase["phase_id"],
                "planned_start_relative_s":a,"planned_end_relative_s":b,
                "capture_status":"NOT_CAPTURED","not_captured_reason":"SESSION_NOT_STARTED"})
        start+=duration
    return rows
