"""Freeze GPU/RGB-first timing pilot; original campaigns remain immutable."""
import argparse
import json
from pathlib import Path
import prepare_t1_identity_sequence as base


def prepare(output,campaign):
    inventory,builder=base.SOURCE_FILES,base.build_protocol
    def revised(*args,**kwargs):
        protocol=builder(*args,**kwargs)
        protocol["collector_version"]="v1p2_bound_RGB_before_evaluation_CUDA"
        protocol["timing_repair"]={
            "device":"CUDA 0, same trained weights/confidence/NMS/image size",
            "processing_order":"RGB persisted/bound then observer completes; depth/seg/mask evaluation afterward",
            "metadata":"Scene intrinsics before schedule; ground/actor readbacks at phase boundaries/end",
            "encoding":"lossless PNG compression 0; file hashes may change but pixel content is not altered",
            "unchanged":"all 60 slots, 5/2/5 host seconds, target/distractor poses, descriptions, masks and policy/tracker",
            "scope":"ground observer only; scene acceptance after observer is independent evaluation and never authorizes flight"}
        return protocol
    try:
        base.SOURCE_FILES=(*inventory,"python/t1_identity_sequence_probe_v1p2.py",
            "python/prepare_t1_identity_sequence_v1p2.py","scripts/run-t1-identity-sequence-v1p2.ps1",
            "python/test_t1_identity_sequence_v1p2.py")
        base.build_protocol=revised
        return base.prepare(output,campaign,weights_config=base.ROOT/"configs/t1_balloon_detector_runtime_cuda_v1.json")
    finally:
        base.SOURCE_FILES,base.build_protocol=inventory,builder


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--campaign",type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.output,args.campaign)
    print(json.dumps({"status":result["status"],"plan":str(args.output.resolve()/"plan.json")},ensure_ascii=False))
