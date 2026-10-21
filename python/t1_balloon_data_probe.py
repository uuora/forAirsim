"""Versioned camera data entry: direct references to precreated scene actors.

The legacy probe remains byte-for-byte intact. A full scene-name listing is
not used to decide whether an already-created actor should be spawned again.
"""
import argparse
import json
from pathlib import Path

import airsim
import numpy as np
import t1_ground_perception_probe as probe
from setup_scene import vector


class ControlledMeshRegistry:
    def __init__(self):
        self.created = set()
        self.audit = []

    def ensure(self, client, name, asset, position, scale):
        pose = airsim.Pose(airsim.Vector3r(*position), airsim.Quaternionr())
        before = np.asarray(vector(client.simGetObjectPose(name).position), float)
        if name not in self.created and not np.isfinite(before).all():
            actual = client.simSpawnObject(name, asset, pose, airsim.Vector3r(*scale), False)
            if actual != name:
                raise ValueError("Controlled spawn returned an unexpected name")
            self.created.add(name)
            operation = "FIRST_SPAWN"
        else:
            if not np.isfinite(before).all():
                raise ValueError("Known controlled actor became unavailable; refuse respawn")
            self.created.add(name)
            if not client.simSetObjectPose(name, pose, teleport=True):
                raise ValueError("Controlled actor position update failed")
            if not client.simSetObjectScale(name, airsim.Vector3r(*scale)):
                raise ValueError("Controlled actor scale update failed")
            operation = "UPDATE_DIRECT_ACTOR_REFERENCE"
        observed = np.asarray(vector(client.simGetObjectPose(name).position), float)
        if not np.isfinite(observed).all() or not np.allclose(observed, position, atol=.02):
            raise ValueError("Controlled actor observed pose disagrees")
        self.audit.append({"name":name,"operation":operation,"requested_world_ned_m":list(position),
            "observed_world_ned_m":observed.tolist()})
        return observed.tolist()


def run(output, protocol):
    registry=ControlledMeshRegistry()
    previous=probe.ensure_mesh
    probe.ensure_mesh=registry.ensure
    try:
        return probe.run(output,protocol)
    finally:
        probe.ensure_mesh=previous
        report_path=output.resolve()/"report.json"
        if report_path.is_file():
            report=json.loads(report_path.read_text(encoding="utf-8"))
            report["source_sha256"][Path(__file__).name]=probe.digest(__file__)
            report["actor_registry_audit"]={"lookup":"direct simGetObjectPose; known actors never respawned",
                "created_names":sorted(registry.created),"operations":registry.audit}
            probe.save(report_path,report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True)
    a=p.parse_args();run(a.output,a.protocol)
