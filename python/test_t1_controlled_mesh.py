import unittest
from types import SimpleNamespace
import airsim
from t1_balloon_data_probe import ControlledMeshRegistry


class FakeScene:
    def __init__(self):self.poses={};self.spawns=[]
    def simGetObjectPose(self,name):
        return SimpleNamespace(position=airsim.Vector3r(*self.poses.get(name,[float('nan')]*3)))
    def simSpawnObject(self,name,asset,pose,scale,physics):
        self.spawns.append(name);self.poses[name]=[pose.position.x_val,pose.position.y_val,pose.position.z_val];return name
    def simSetObjectPose(self,name,pose,teleport):
        self.poses[name]=[pose.position.x_val,pose.position.y_val,pose.position.z_val];return True
    def simSetObjectScale(self,name,scale):return True
    def simListSceneObjects(self):raise AssertionError("Actor registry must not use a full list")


class ControlledMeshTests(unittest.TestCase):
    def test_known_actor_updates_without_list_or_respawn(self):
        client=FakeScene();registry=ControlledMeshRegistry()
        registry.ensure(client,"balloon","sphere",[0,0,-3],[1,1,1])
        registry.ensure(client,"balloon","sphere",[0,1,-3],[1,1,1])
        self.assertEqual(client.spawns,["balloon"])
        self.assertEqual(registry.audit[-1]["operation"],"UPDATE_DIRECT_ACTOR_REFERENCE")
    def test_known_missing_actor_rejects_respawn(self):
        client=FakeScene();registry=ControlledMeshRegistry()
        registry.ensure(client,"balloon","sphere",[0,0,-3],[1,1,1]);client.poses.clear()
        with self.assertRaises(ValueError):registry.ensure(client,"balloon","sphere",[0,1,-3],[1,1,1])
        self.assertEqual(len(client.spawns),1)
    def test_preexisting_direct_actor_reused(self):
        client=FakeScene();client.poses["balloon"]=[0,0,-3];registry=ControlledMeshRegistry()
        registry.ensure(client,"balloon","sphere",[0,1,-3],[1,1,1]);self.assertFalse(client.spawns)


if __name__=="__main__":unittest.main()
