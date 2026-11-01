"""Analytic scene geometry regressions, without a simulator."""
from copy import deepcopy
import unittest
from curve_airsim_sphere_scene import scene_hash,validate_scene,swept_clearance


def scene():
    return dict(scene_id='test',coordinate_frame='local_zup_m',seed=1101,
        obstacles=[dict(name='CurveSphere_Test',centre=[0,0,0],radius=.4)],
        vehicle_envelope_radius_m=.3,safety_margin_m=.1)


class SphereSceneTests(unittest.TestCase):
    def test_chord_detects_obstacle_between_free_endpoints(self):
        value=swept_clearance([[-2,0,0],[2,0,0]],scene())
        self.assertAlmostEqual(value['measured_chord_min_envelope_clearance_m'],-.7)
        self.assertFalse(value['clearance_margin_passed'])

    def test_tangent_and_stationary_path(self):
        for points in ([[-2,.8,0],[2,.8,0]],[[0,.8,0]],[[0,.8,0],[0,.8,0]]):
            self.assertAlmostEqual(swept_clearance(points,scene())['measured_chord_min_envelope_clearance_m'],.1)

    def test_scene_hash_ignores_dictionary_order_but_tracks_geometry(self):
        a=scene();b=dict(reversed(list(a.items())))
        self.assertEqual(scene_hash(a),scene_hash(b))
        b=deepcopy(b);b['obstacles'][0]['radius']=.5
        self.assertNotEqual(scene_hash(a),scene_hash(b))

    def test_rejects_non_owned_mesh_and_duplicate_names(self):
        a=scene();a['obstacles']*=2
        with self.assertRaises(ValueError):validate_scene(a)
        a=scene();a['obstacles'][0]['name']='UserActor'
        with self.assertRaises(ValueError):validate_scene(a)


if __name__=='__main__':unittest.main()
