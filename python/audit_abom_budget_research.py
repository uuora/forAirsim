"""Replay the selection/accounting claims from frozen paired-search records."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(folder):
    folder=Path(folder)
    manifest=load(folder/'manifest.json')
    for source,digest in manifest['sources'].items():
        assert hashlib.sha256((folder/'source_snapshot'/Path(source).name).read_bytes()).hexdigest()==digest
    rows=load(folder/'rows.json')
    assert rows, 'no completed runs'
    if manifest['protocol']=='pinned-abom-dropout-calibration-v1':
        return audit_dropout(folder,manifest,rows)
    total_proposals=total_pruned=total_cases=0
    for row in rows:
        geometry='family' in row
        name=(f"{row['family']}_{row['layout']}_{row['method']}_{row['seed']}" if geometry
              else f"{row['method']}_{row['seed']}")
        raw=load(folder/f'{name}.json')
        assert raw['summary']==row
        baseline=raw['plain']
        budget=row.get('proposals',row.get('budget'))
        assert len(baseline['evaluations'])==budget
        for mode in (['screened'] if geometry else ['fixed','adaptive']):
            other=raw[mode]
            assert len(other['evaluations'])==budget
            assert other['best_score']==baseline['best_score']
            assert other['best_parameters']==baseline['best_parameters']
            assert other['initial_population_normalized']==baseline['initial_population_normalized']
            assert [r['best'] for r in baseline['history']]==[r['best'] for r in other['history']]
            assert [r['scale'] for r in baseline['history']]==[r['scale'] for r in other['history']]
            pruned=0
            for a,b in zip(baseline['evaluations'],other['evaluations']):
                assert a['parameters']==b['parameters']
                if b['status']=='PRUNED':
                    pruned+=1
                    assert b['score'] is None and b['feasible'] is None and b['fe'] is None
                    assert b['lower_bound']<=a['score']+1e-14
                    assert b['lower_bound']>=b['selection_threshold']
                    assert a['score']>=b['selection_threshold']
                else:
                    assert a['score']==b['score'] and a['feasible']==b['feasible']
            assert pruned==other['pruned_count']
            assert other['full_evaluations']+pruned==budget
            total_proposals+=budget
            total_pruned+=pruned
            if not geometry:
                count=np.zeros(3)
                failures=np.zeros(3)
                seconds=np.zeros(3)
                assert len(other['case_order_ledger'])==budget
                cases=0
                for event in other['case_order_ledger']:
                    rate=(failures+1)/(count+2)
                    cost=np.divide(seconds,count,out=np.ones(3),where=count>0)
                    priority=rate/np.maximum(cost,1e-9)
                    expected=np.argsort(-priority,kind='stable').tolist() if mode=='adaptive' else [0,1,2]
                    assert event['order']==expected
                    assert np.array_equal(event['priority_before'],priority)
                    assert len(event['case_observations'])==event['completed_cases']
                    assert [o['case_index'] for o in event['case_observations']]==expected[:event['completed_cases']]
                    for obs in event['case_observations']:
                        i=obs['case_index']
                        assert obs['wall_s']>=0
                        count[i]+=1
                        failures[i]+=not obs['passed']
                        seconds[i]+=obs['wall_s']
                    cases+=event['completed_cases']
                assert cases==other['completed_case_rollouts']==row['modes'][mode]['case_rollouts']
                total_cases+=cases
        assert row['equivalent']
    expected=(len(manifest['args']['families'])*len(manifest['args']['layouts'])*len(manifest['args']['methods'])*len(manifest['args']['seeds'])
              if 'families' in manifest['args'] else len(manifest['args']['methods'])*len(manifest['args']['seeds']))
    assert len(rows)==expected, 'campaign incomplete'
    # Check published arithmetic independently from aggregate summary.
    for group in load(folder/'summary.json'):
        chosen=[r for r in rows if r['method']==group['method'] and
                ('family' not in r or (r['family']==group['family'] and r['layout']==group['layout']))]
        assert group['n']==len(chosen)
        if 'family' in group:
            ratios=[r['screened_wall_s']/r['plain_wall_s'] for r in chosen]
            saved=sum(r['pruned'] for r in chosen)/sum(r['proposals'] for r in chosen)
            assert saved==group['calls_saved_fraction']
        else:
            mode=group['mode']
            ratios=[r['modes'][mode]['wall_s']/r['modes']['plain']['wall_s'] for r in chosen]
            saved=1-sum(r['modes'][mode]['case_rollouts'] for r in chosen)/sum(r['modes']['plain']['case_rollouts'] for r in chosen)
            assert saved==group['case_rollout_reduction']
        assert float(np.median(ratios))==group['median_wall_ratio']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json') if p.name!='audit.json'}
    result=dict(status='PASS',completed_pairs_or_triples=len(rows),screened_proposals_audited=total_proposals,
                pruned_records_audited=total_pruned,observed_screened_case_rollouts=total_cases,
                source_snapshot_hashes_verified=True,raw_record_sha256=hashes,
                scope='deterministic record consistency, bounds vs measured paired baseline, summary arithmetic, and adaptive-order replay; no empirical flight or formal interval proof')
    (folder/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {k:v for k,v in result.items() if k!='raw_record_sha256'}


def audit_dropout(folder,manifest,rows):
    from run_abom_dropout_research import dropout_label
    assert len(rows)==manifest['expected_runs']
    initial={}
    indexed={}
    for row in rows:
        key=(row['family'],row['layout'],row['seed'])
        raw=load(folder/f"{row['family']}_{row['layout']}_{row['seed']}_dropout_{dropout_label(row['dropout'])}.json")
        assert raw['summary']==row
        run=raw['search']
        assert len(run['evaluations'])==row['budget']==run['full_evaluations']
        assert run['pruned_count']==0 and not run['screening_enabled']
        assert run['best_score']==min(e['score'] for e in run['evaluations'])==row['best_score']
        assert row['failed_candidate_count']==sum(not e['feasible'] for e in run['evaluations'])
        assert raw['selected_result_recheck']['score']==run['best_score']
        assert raw['selected_result_recheck']['feasible']==row['feasible']
        if key in initial: assert initial[key]==run['initial_population_normalized']
        initial[key]=run['initial_population_normalized']
        indexed[key+(row['dropout'],)]=row
    observed=[]
    for block in manifest['execution_plan']:
        for i,dropout in enumerate(block['dropout_order']):
            row=indexed[(block['family'],block['layout'],block['seed'],dropout)]
            assert row['execution_order_index']==i
            observed.append(row)
    assert observed==rows
    for group in load(folder/'summary.json'):
        chosen=[r for r in rows if (r['family'],r['layout'],r['dropout'])==(group['family'],group['layout'],group['dropout'])]
        assert group['n']==len(chosen)
        assert group['mean_score']==float(np.mean([r['best_score'] for r in chosen]))
        assert group['median_wall_s']==float(np.median([r['wall_s'] for r in chosen]))
    for group in load(folder/'paired_comparisons.json'):
        differences=np.array([indexed[(group['family'],group['layout'],seed,group['dropout'])]['best_score']-
                              indexed[(group['family'],group['layout'],seed,.95)]['best_score'] for seed in manifest['args']['seeds']])
        assert group['mean_paired_score_difference']==float(np.mean(differences))
        assert group['better_runs']==sum(differences<0)
        rng=np.random.default_rng(20261029)
        boot=np.mean(differences[rng.integers(0,len(differences),(5000,len(differences)))],axis=1)
        assert group['exploratory_paired_bootstrap95']==np.quantile(boot,[.025,.975]).tolist()
    result=dict(status='PASS',runs=len(rows),proposals=sum(r['budget'] for r in rows),
                shared_initialization_and_execution_order_verified=True,summary_and_paired_arithmetic_verified=True,
                source_snapshot_hashes_verified=True,
                raw_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json') if p.name!='audit.json'},
                scope='record consistency and arithmetic; no heldout-scene generalization or flight claim')
    (folder/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return {k:v for k,v in result.items() if k!='raw_record_sha256'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folders',nargs='+')
    args=parser.parse_args()
    for folder in args.folders:
        print(json.dumps(dict(folder=folder,**audit(folder)),ensure_ascii=False))
