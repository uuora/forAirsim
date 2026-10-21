"""Preregister the timing repair without altering V1 protocol/source receipts."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import prepare_t1_identity_sequence as base


def prepare(output, campaign):
    inventory = base.SOURCE_FILES
    old_builder = base.build_protocol
    def builder(*args, **kwargs):
        protocol = old_builder(*args, **kwargs)
        protocol["collector_version"] = "v1p1_phase_scoped_metadata"
        protocol["timing_repair"] = {
            "unchanged": "same 5/2/5 seconds, .2s slots, boxes, masks, descriptions, weights and observer",
            "change": "camera intrinsics once before timing; ground/actor pose RPC at phase boundaries; one report save after each observer",
            "ground_scope": "phase boundary and end readback; not per-frame fresh ground state",
            "evaluation": "every planned slot and actual receipt retained; no retrospective grid change"}
        return protocol
    try:
        base.SOURCE_FILES = (*inventory,
            "python/t1_identity_sequence_probe_v1p1.py",
            "python/prepare_t1_identity_sequence_v1p1.py",
            "scripts/run-t1-identity-sequence-v1p1.ps1",
            "python/test_t1_identity_sequence_v1p1.py")
        base.build_protocol = builder
        return base.prepare(output,campaign)
    finally:
        base.SOURCE_FILES, base.build_protocol = inventory, old_builder


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--campaign",required=True,type=Path)
    arguments=parser.parse_args()
    plan=prepare(arguments.output,arguments.campaign)
    print(json.dumps({"status":plan["status"],"planned_sessions":4,"planned_slots":240,
        "plan":str(arguments.output.resolve()/"plan.json")},ensure_ascii=False))
