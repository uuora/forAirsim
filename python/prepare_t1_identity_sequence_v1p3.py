"""Freeze .4 s CUDA pilot from measured loop cost; historical grids unchanged."""
import argparse
import json
from pathlib import Path
import prepare_t1_identity_sequence as base
from t1_identity_schedule import boundary_clipped_slots


def prepare(output,campaign):
    inventory,builder=base.SOURCE_FILES,base.build_protocol
    def revised(*args,**kwargs):
        protocol=builder(*args,**kwargs)
        protocol.update(collector_version="v1p3_CUDA_bound_RGB_nominal_0p4",
            protocol_id="t1_continuous_identity_v1p3_"+protocol["episode_id"]+"_20261004",
            timing_grid="BOUNDARY_CLIPPED_0P4",minimum_period_s=.4)
        protocol["timing_contract"]="31 predeclared intervals, nominal .4s and phase-end .2s; 12 host seconds; every missed/late interval retained"
        protocol["timing_repair"]={"reason":"Three .2s development protocols failed stable deadline coverage",
            "unchanged":"scene/descriptions/weights/thresholds/policy/tracker; ground only",
            "device":"CUDA 0", "processing_order":"bound RGB decision before depth/seg/mask evaluation",
            "metadata_scope":"phase/end ground and actor receipts; one intrinsic read before timing",
            "sampling":"nominal .4s; clips intervals at 5/7/12s without revising previous grids"}
        if len(boundary_clipped_slots(protocol))!=31:
            raise ValueError("Expected declared 31-slot protocol")
        return protocol
    try:
        base.SOURCE_FILES=(*inventory,"python/t1_identity_sequence_probe_v1p3.py",
            "python/prepare_t1_identity_sequence_v1p3.py","scripts/run-t1-identity-sequence-v1p3.ps1",
            "python/t1_identity_schedule.py","python/test_t1_identity_sequence_v1p3.py")
        base.build_protocol=revised
        plan=base.prepare(output,campaign,weights_config=base.ROOT/"configs/t1_balloon_detector_runtime_cuda_v1.json")
        plan["planned_slots_per_session"]=31
        plan["timing_grid"]="BOUNDARY_CLIPPED_0P4"
        base.save(Path(output)/"plan.json",plan)
        return plan
    finally:
        base.SOURCE_FILES,base.build_protocol=inventory,builder


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--campaign",type=Path,required=True)
    args=parser.parse_args()
    plan=prepare(args.output,args.campaign)
    print(json.dumps({"status":plan["status"],"planned_slots":124,
        "plan":str(args.output.resolve()/"plan.json")},ensure_ascii=False))
