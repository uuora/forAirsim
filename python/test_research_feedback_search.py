"""Diagnostic information, capacity and evaluation accounting are separable."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from trajectory_research.feedback_search import run_feedback_search


UPSTREAM = Path(__file__).resolve().parents[1]/"tmp/abom_deploy_probe/upstream"
HAS_NEURAL = importlib.util.find_spec("torch") is not None and (UPSTREAM/"abom_net.py").is_file()


def objective(x, *, zeros=False):
    features = np.zeros(5) if zeros else np.array([x[0]+1., x[1]+1., x[-1]+1., float(x@x), abs(x[0])])
    feasible = bool(x[0] >= 0)
    return dict(score=float(x@x)+(0. if feasible else 5.), feasible=feasible,
                score_details=dict(control_features=features.tolist()))


class FeedbackValidationTests(unittest.TestCase):
    def test_malformed_features_abort_on_first_evaluation_in_every_mode(self):
        bad_values = ([1.,2.], [0.,0.,0.,0.,float('nan')],
                      [0.,0.,0.,0.,float('inf')], [True]*5, ['0']*5, [[0.]*5])
        for mode in ('scalar','diagnostic','zero','shuffled'):
            for features in bad_values:
                calls=[]
                def invalid(x):
                    calls.append(x)
                    return dict(score=0.,feasible=True,score_details=dict(control_features=features))
                with self.subTest(mode=mode,features=features),self.assertRaises(ValueError):
                    run_feedback_search(invalid,[-1.]*3,[1.]*3,mode=mode,
                                        population=4,budget=8,upstream='unused')
                self.assertEqual(len(calls),1)

    def test_missing_features_and_invalid_requests_fail_visibly(self):
        with self.assertRaisesRegex(ValueError,'control_features'):
            run_feedback_search(lambda x:dict(score=0.,feasible=True),[-1.],[1.],
                                population=4,budget=4,upstream='unused')
        for options in (dict(mode='estimated'),dict(population=3),dict(budget=7),
                        dict(seed=True),dict(dropout=1.),dict(dropout=float('nan')),
                        dict(initialization='random'),dict(anchor=[2.,2.,2.])):
            kwargs=dict(population=4,budget=8,upstream='unused')
            kwargs.update(options)
            with self.subTest(options=options),self.assertRaises(ValueError):
                run_feedback_search(objective,[-1.]*3,[1.]*3,**kwargs)

    def test_objective_exceptions_are_not_converted_to_scores(self):
        def broken(x): raise RuntimeError('rollout error')
        with self.assertRaisesRegex(RuntimeError,'rollout error'):
            run_feedback_search(broken,[-1.],[1.],population=4,budget=4,upstream='unused')


@unittest.skipUnless(HAS_NEURAL,'optional torch and pinned upstream source required')
class FeedbackNeuralTests(unittest.TestCase):
    def run_mode(self,mode,fn=objective,**options):
        kwargs=dict(mode=mode,seed=23,population=4,budget=20,upstream=UPSTREAM,
                    anchor=[.1,.1,.1],dropout=.1)
        kwargs.update(options)
        return run_feedback_search(fn,[-1.]*3,[1.]*3,**kwargs)

    def test_common_initialization_capacity_and_exact_call_budget(self):
        outputs={}
        for mode in ('scalar','diagnostic','zero','shuffled'):
            calls=[]
            def counted(x):
                calls.append(x.copy())
                return objective(x)
            result=self.run_mode(mode,counted)
            self.assertEqual(len(calls),20)
            self.assertEqual(result['full_evaluations'],20)
            self.assertEqual(result['diagnostic_only_evaluations'],0)
            self.assertFalse(result['screening_enabled'])
            self.assertEqual([r['fe'] for r in result['evaluations']],list(range(1,21)))
            self.assertEqual(result['best_score'],min(r['score'] for r in result['evaluations']))
            self.assertTrue(any(not r['feasible'] for r in result['evaluations']))
            self.assertEqual(result['network_input_columns'],1 if mode=='scalar' else 6)
            self.assertTrue(all(result[k]>=0 for k in ('elapsed_s','evaluation_s',
                'neural_generation_s','neural_update_s','feature_preparation_s')))
            json.dumps(result,allow_nan=False)
            outputs[mode]=result
        for mode in ('diagnostic','zero','shuffled'):
            self.assertEqual(outputs[mode]['initial_population_normalized'],outputs['scalar']['initial_population_normalized'])
            self.assertEqual(outputs[mode]['parameter_count'],outputs['zero']['parameter_count'])
            self.assertEqual(outputs[mode]['initial_model_sha256'],outputs['zero']['initial_model_sha256'])
        self.assertLess(outputs['scalar']['parameter_count'],outputs['zero']['parameter_count'])

    def test_zero_actual_diagnostics_make_diagnostic_and_zero_identical(self):
        fn=lambda x:objective(x,zeros=True)
        diagnostic=self.run_mode('diagnostic',fn)
        zero=self.run_mode('zero',fn)
        self.assertEqual(diagnostic['initial_model_sha256'],zero['initial_model_sha256'])
        self.assertEqual([r['parameters'] for r in diagnostic['evaluations']],
                         [r['parameters'] for r in zero['evaluations']])
        self.assertEqual(diagnostic['final_population_normalized'],zero['final_population_normalized'])
        self.assertEqual(diagnostic['final_scores'],zero['final_scores'])

    def test_feature_ledger_replays_parent_selection_and_consumption(self):
        for mode in ('scalar','diagnostic','zero','shuffled'):
            with self.subTest(mode=mode):
                result=self.run_mode(mode,budget=28)
                records={r['fe']:r for r in result['evaluations']}
                self.assertEqual(len(result['feature_ledger']),6)
                for event,history in zip(result['feature_ledger'],result['history']):
                    self.assertEqual(event['parent_evaluation_ids'],history['parent_evaluation_ids'])
                    parent=[records[i] for i in event['parent_evaluation_ids']]
                    self.assertEqual(event['parent_scores'],[r['score'] for r in parent])
                    self.assertEqual(event['parent_control_features'],[r['control_features'] for r in parent])
                    self.assertEqual(event['parent_normalized_parameters'],[r['normalized_parameters'] for r in parent])
                    consumed=np.asarray(event['consumed_features'])
                    np.testing.assert_array_equal(consumed[:,0],np.arange(4)/3.)
                    if mode=='scalar': self.assertEqual(consumed.shape,(4,1))
                    elif mode=='zero': np.testing.assert_array_equal(consumed[:,1:],np.zeros((4,5)))
                    else:
                        raw=np.asarray(event['parent_control_features'])
                        permutation=event['diagnostic_row_permutation']
                        self.assertEqual(sorted(permutation),list(range(4)))
                        np.testing.assert_array_equal(consumed[:,1:],raw[permutation])
                for index,fe in enumerate(result['final_parent_evaluation_ids']):
                    self.assertEqual(result['final_control_features'][index],records[fe]['control_features'])

    def test_shuffle_determinism_and_global_rng_isolation(self):
        import torch
        np.random.seed(106)
        np_before=np.random.get_state()
        torch.manual_seed(209)
        torch_before=torch.random.get_rng_state().clone()
        first=self.run_mode('shuffled',initialization='lhs')
        second=self.run_mode('shuffled',initialization='lhs')
        self.assertEqual([r['parameters'] for r in first['evaluations']],
                         [r['parameters'] for r in second['evaluations']])
        self.assertEqual(first['feature_ledger'],second['feature_ledger'])
        np_after=np.random.get_state()
        self.assertEqual(np_before[0],np_after[0])
        np.testing.assert_array_equal(np_before[1],np_after[1])
        self.assertEqual(np_before[2:],np_after[2:])
        self.assertTrue(torch.equal(torch_before,torch.random.get_rng_state()))

    def test_recorded_diagnostics_are_not_mutated_by_reused_objective_dict(self):
        shared=dict(score=0.,feasible=True,score_details=dict(control_features=[0.]*5))
        calls=[]
        def reused(x):
            calls.append(1)
            shared['score']=float(x@x)
            shared['score_details']['control_features'][0]=float(len(calls))
            return shared
        result=self.run_mode('diagnostic',reused)
        self.assertEqual([r['details']['control_features'][0] for r in result['evaluations']],
                         list(range(1,21)))


if __name__=='__main__': unittest.main()
