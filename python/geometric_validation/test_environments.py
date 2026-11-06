"""Pure task selection, independent reference checks and comparison isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .audit import frozen_plan_issues
from .environments import ROOT, CATALOG, load_catalog, config_for, get_environment, task_summary, matching_comparison, load_environment_config
from .planners import construct, plan, METHODS
from .runtime import load_config


class EnvironmentTests(unittest.TestCase):
    def test_four_new_presets_plus_unchanged_baseline(self):
        entries=load_catalog()['environments']
        self.assertEqual([e['id'] for e in entries],['baseline','straight','planar_s','climb_descent','large_turn'])
        original=load_config(config_for('baseline'))
        for entry in entries:
            cfg=load_config(config_for(entry['id']))
            for key in ('controller','limits','execution','airsim','visualization'):
                self.assertEqual(cfg[key],original[key],(entry['id'],key))
            self.assertEqual([c['method'] for c in cfg['cases'].values()],list(METHODS))
            for case in cfg['cases']:
                self.assertEqual(frozen_plan_issues(cfg,case),[],(entry['id'],case))

    def test_presets_have_distinct_geometry_and_task_dimensions(self):
        straight=load_config(config_for('straight'));planar=load_config(config_for('planar_s'))
        climb=load_config(config_for('climb_descent'));turn=load_config(config_for('large_turn'))
        self.assertAlmostEqual(task_summary(straight)['polyline_length_m'],12.)
        self.assertEqual(task_summary(straight)['max_direction_change_deg'],0.)
        self.assertEqual(task_summary(planar)['altitude_min_m'],task_summary(planar)['altitude_max_m'])
        self.assertEqual(task_summary(planar)['intermediate_points'],3)
        self.assertEqual(task_summary(climb)['altitude_min_m'],4.)
        self.assertEqual(task_summary(climb)['altitude_max_m'],9.)
        self.assertGreaterEqual(task_summary(turn)['max_direction_change_deg'],134.)
        self.assertLess(task_summary(turn)['max_direction_change_deg'],180.)

    def test_degenerate_constant_axes_preserve_hermite_coefficient_shape(self):
        task=get_environment('straight')['task'];times=task['initial_segment_durations']
        a=construct('hermite',task['points_ned'],times)
        b=construct('bezier',task['points_ned'],times)
        self.assertEqual(a.shape,(3,8,3))
        np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(a[:,:,1],0,atol=1e-12)

    def test_time_search_handles_equal_active_peaks_on_several_segments(self):
        base=json.loads((ROOT/'configs/geometric_planner_task.json').read_text(encoding='utf-8'))
        for method in ('bezier','hermite'):
            result=plan(method,get_environment('straight')['task'],base['time_search'])
            self.assertTrue(result['planning']['success'])
            self.assertLess(sum(result['segment_durations']),18.)
            self.assertLessEqual(result['planning']['peak_speed_mps'],1.6+1e-6)

    def test_loading_and_selecting_presets_cannot_open_rpc_or_launch(self):
        with patch('socket.create_connection',side_effect=AssertionError('Unexpected RPC')), \
             patch('subprocess.Popen',side_effect=AssertionError('Unexpected launch')):
            for entry in load_catalog()['environments']:
                cfg=json.loads(config_for(entry['id']).read_text(encoding='utf-8'))
                self.assertGreater(task_summary(cfg)['polyline_length_m'],0)

    def test_stale_frozen_environment_is_rejected_without_regenerating(self):
        path=config_for('straight');cfg=load_environment_config('straight')
        cfg['comparison']['task']['points_ned'][1][0]+=.1
        read_text=Path.read_text
        def altered(p,*args,**kwargs):
            return json.dumps(cfg) if p.resolve()==path else read_text(p,*args,**kwargs)
        with patch.object(Path,'read_text',altered), \
             patch('geometric_validation.environments.prepare_environment',side_effect=AssertionError('Unexpected planning')):
            with self.assertRaisesRegex(ValueError,'stale'):
                load_environment_config('straight')

    def test_comparison_never_falls_back_to_another_environment_or_gate(self):
        straight=load_config(config_for('straight'));planar=load_config(config_for('planar_s'))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'geometric_three_planners_1';folder.mkdir()
            (folder/'index.html').write_text('synthetic',encoding='utf-8')
            (folder/'config_frozen.json').write_text(json.dumps(straight),encoding='utf-8')
            self.assertEqual(matching_comparison(straight,temp),folder/'index.html')
            self.assertIsNone(matching_comparison(planar,temp))
            changed=copy.deepcopy(straight);changed['limits']['position_error_m']+=.1
            self.assertIsNone(matching_comparison(changed,temp))

    def test_invalid_catalog_rejects_duplicate_ids_and_exact_reversal(self):
        catalog=load_catalog()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'catalog.json'
            invalid=copy.deepcopy(catalog);invalid['environments'][1]['id']='baseline'
            path.write_text(json.dumps(invalid),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_catalog(path)
            invalid=copy.deepcopy(catalog);invalid['environments'][1]['task']['points_ned'][2]=[0,0,-5]
            path.write_text(json.dumps(invalid),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'reversal'):
                load_catalog(path)
