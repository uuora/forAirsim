"""Exact measurement reuse must not change search outcomes or hide real calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from trajectory_research.objective_cache import DeterministicObjectiveCache
from trajectory_research.search import run_search


UPSTREAM = Path(__file__).resolve().parents[1]/"tmp/abom_deploy_probe/upstream"
HAS_NEURAL = importlib.util.find_spec("torch") is not None and (UPSTREAM/"abom_net.py").is_file()


def objective(x):
    return dict(score=float(x@x), feasible=bool(x[0] >= 0),
                score_details=dict(control_features=x.tolist(), measured_wall_s=.125))


class ObjectiveCacheTests(unittest.TestCase):
    def test_reuses_complete_infeasible_measurement_with_explicit_provenance(self):
        calls=[]
        def oracle(x):
            calls.append(x.copy())
            return objective(x)
        cache=DeterministicObjectiveCache(oracle,"frozen-scene-a",deterministic=True)
        first=cache([-1.,2.])
        second=cache(np.array([-1.,2.]))
        self.assertEqual(first,second)
        self.assertFalse(second['feasible'])
        self.assertEqual(len(calls),1)
        self.assertEqual(cache.stats['proposals'],2)
        self.assertEqual(cache.stats['oracle_calls'],1)
        self.assertEqual(cache.stats['hits'],1)
        self.assertEqual([r['status'] for r in cache.ledger],['FULL_EVALUATION','CACHE_HIT'])
        self.assertEqual([r['source_evaluation_id'] for r in cache.ledger],[1,1])
        self.assertEqual(cache.last_record['oracle_wall_s'],0.)
        self.assertIn('not a fresh evaluation',cache.last_record['payload_timing_scope'])
        json.dumps(dict(stats=cache.stats,ledger=cache.ledger),allow_nan=False)

    def test_payload_and_ledger_are_defensive_copies(self):
        original=objective(np.array([.5,1.]))
        cache=DeterministicObjectiveCache(lambda x:original,"copies",deterministic=True)
        first=cache([.5,1.])
        original['score_details']['control_features'][0]=999
        first['score_details']['control_features'][1]=888
        first['score']=-7
        second=cache([.5,1.])
        self.assertEqual(second['score'],1.25)
        self.assertEqual(second['score_details']['control_features'],[.5,1.])
        ledger=cache.ledger
        ledger[0]['parameters'][0]=555
        self.assertEqual(cache.ledger[0]['parameters'],[.5,1.])
        last=cache.last_record
        last['score']=-99
        self.assertEqual(cache.last_record['score'],1.25)

    def test_objective_cannot_mutate_caller_parameters_or_cache_key(self):
        def oracle(x):
            x[:]=3.
            return objective(x)
        cache=DeterministicObjectiveCache(oracle,"input-copy",deterministic=True)
        point=np.array([1.,2.])
        cache(point)
        np.testing.assert_array_equal(point,[1.,2.])
        self.assertEqual(cache([1.,2.])['score'],18.)
        self.assertEqual(cache.stats['oracle_calls'],1)

    def test_exact_bits_keep_nearby_points_and_signed_zero_distinct(self):
        cache=DeterministicObjectiveCache(objective,"exact",deterministic=True)
        points=[[1.],[float(np.nextafter(1.,2.))],[0.],[-0.]]
        for point in points: cache(point)
        self.assertEqual(cache.stats['oracle_calls'],4)
        self.assertEqual(len({r['key_digest'] for r in cache.ledger}),4)
        cache([1.])
        self.assertEqual(cache.stats['hits'],1)
        # Different numeric input representations may share an exactly equal
        # float64 value; no loss of information occurs for binary .25.
        cache([.25]);cache(np.array([.25],dtype=np.float32))
        self.assertEqual(cache.stats['hits'],2)

    def test_context_instances_do_not_share_measurements(self):
        a=DeterministicObjectiveCache(lambda x:dict(score=1.,feasible=True),"A",deterministic=True)
        b=DeterministicObjectiveCache(lambda x:dict(score=2.,feasible=True),"B",deterministic=True)
        self.assertEqual(a([0.])['score'],1.)
        self.assertEqual(b([0.])['score'],2.)
        self.assertNotEqual(a.last_record['key_digest'],b.last_record['key_digest'])
        for cache in (a,b):
            self.assertEqual(cache.stats['oracle_calls'],1)
            with self.assertRaises(AttributeError): cache.context_id="changed"

    def test_invalid_contract_and_inputs_fail_before_proposal(self):
        for kwargs in ({},{'deterministic':False},{'deterministic':np.bool_(True)}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                DeterministicObjectiveCache(objective,"scope",**kwargs)
        for context in (None,'','   ',17):
            with self.subTest(context=context),self.assertRaises(ValueError):
                DeterministicObjectiveCache(objective,context,deterministic=True)
        for capacity in (True,False,0,-1,1.5,np.inf):
            with self.subTest(capacity=capacity),self.assertRaises(ValueError):
                DeterministicObjectiveCache(objective,"scope",deterministic=True,max_entries=capacity)
        cache=DeterministicObjectiveCache(objective,"valid",deterministic=True)
        for point in ([],[[1.]],['x'],[True],[1+0j],[np.nan],[np.inf],np.array([2**53+1],dtype=np.int64)):
            with self.subTest(point=point),self.assertRaises(ValueError): cache(point)
        self.assertEqual(cache.stats['proposals'],0)
        self.assertEqual(cache.stats['oracle_calls'],0)

    def test_errors_invalid_and_partial_results_are_never_cached(self):
        bad_results=[{},dict(score=np.nan,feasible=True),dict(score=True,feasible=True),
            dict(score=0.,feasible=1),dict(score=0.,feasible=True,status='PRUNED'),
            dict(score=0.,feasible=True,score_details={'bad':np.inf})]
        for bad in bad_results:
            calls=[]
            def oracle(x): calls.append(1);return deepcopy(bad)
            cache=DeterministicObjectiveCache(oracle,"bad",deterministic=True)
            for _ in range(2):
                with self.assertRaises(ValueError): cache([0.])
            self.assertEqual(len(calls),2)
            self.assertEqual(cache.stats['hits'],0)
            self.assertEqual(cache.stats['cache_entries'],0)
            self.assertEqual(cache.stats['failed_evaluations'],2)
            json.dumps(cache.ledger,allow_nan=False)

    def test_raised_error_propagates_and_attempt_ids_are_not_reused(self):
        calls=[]
        def oracle(x):
            calls.append(1)
            if len(calls)==1: raise RuntimeError('transient')
            return objective(x)
        cache=DeterministicObjectiveCache(oracle,"error",deterministic=True)
        with self.assertRaisesRegex(RuntimeError,'transient'): cache([1.])
        cache([1.]);cache([1.])
        self.assertEqual([r['source_evaluation_id'] for r in cache.ledger],[1,2,2])
        self.assertEqual([r['status'] for r in cache.ledger],['EVALUATION_ERROR','FULL_EVALUATION','CACHE_HIT'])
        self.assertEqual(cache.stats['proposals'],3)
        self.assertEqual(cache.stats['oracle_calls'],2)
        self.assertEqual(cache.stats['successful_evaluations'],1)
        self.assertEqual(cache.stats['failed_evaluations'],1)
        self.assertEqual(cache.stats['hits'],1)

    def test_lru_capacity_records_evictions_and_new_sources(self):
        cache=DeterministicObjectiveCache(objective,"lru",deterministic=True,max_entries=2)
        for x in (1.,2.,1.,3.,2.): cache([x])
        self.assertEqual([r['source_evaluation_id'] for r in cache.ledger],[1,2,1,3,4])
        self.assertEqual(cache.stats['oracle_calls'],4)
        self.assertEqual(cache.stats['hits'],1)
        self.assertEqual(cache.stats['evictions'],2)
        self.assertEqual(cache.stats['peak_cache_entries'],2)
        self.assertEqual(cache.stats['cache_entries'],2)
        self.assertEqual(cache.ledger[3]['evicted_source_evaluation_id'],2)
        self.assertEqual(cache.ledger[4]['evicted_source_evaluation_id'],1)

    def assert_same_search(self,a,b):
        for key in ('initial_population_normalized','best_parameters','best_score','history'):
            self.assertEqual(a[key],b[key])
        self.assertEqual([(r['parameters'],r['score'],r['feasible'],r['details']) for r in a['evaluations']],
                         [(r['parameters'],r['score'],r['feasible'],r['details']) for r in b['evaluations']])

    def test_same_generic_search_stream_and_actual_call_accounting(self):
        for method in ('random','gaussian'):
            kwargs=dict(method=method,lower=[-1.]*3,upper=[1.]*3,seed=14,population=8,budget=40,anchor=[0.]*3)
            cache=DeterministicObjectiveCache(objective,"search-"+method,deterministic=True)
            if method=='gaussian':
                def proposals(parents,count,*unused,**kwargs): return np.repeat(parents[[0]],count,axis=0)
                with patch('trajectory_research.search.paired_perturbation',side_effect=proposals):
                    original=run_search(objective,**kwargs)
                    cached=run_search(cache,**kwargs)
                self.assertEqual(cache.stats['oracle_calls'],8)
                self.assertEqual(cache.stats['hits'],32)
            else:
                original=run_search(objective,**kwargs)
                cached=run_search(cache,**kwargs)
            self.assert_same_search(original,cached)
            self.assertEqual(cache.stats['proposals'],40)
            self.assertEqual(cache.stats['oracle_calls']+cache.stats['hits'],40)
            self.assertEqual(len(cache.ledger),40)

    @unittest.skipUnless(HAS_NEURAL,"pinned optional ABOM source and torch required")
    def test_same_pinned_abom_stream_with_exact_reuse(self):
        kwargs=dict(lower=[-1.]*4,upper=[1.]*4,method='abom',seed=29,
                    population=8,budget=64,anchor=[0.]*4,upstream=UPSTREAM)
        cache=DeterministicObjectiveCache(objective,"abom",deterministic=True)
        original=run_search(objective,**kwargs)
        cached=run_search(cache,**kwargs)
        self.assert_same_search(original,cached)
        self.assertGreater(cache.stats['hits'],0)
        self.assertEqual(cache.stats['oracle_calls']+cache.stats['hits'],64)


if __name__=='__main__': unittest.main()
