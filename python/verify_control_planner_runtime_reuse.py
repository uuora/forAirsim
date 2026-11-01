"""Cold/warm same-case reproducibility check; zero AirSim calls."""
import argparse,json,hashlib,time
from pathlib import Path
from control_target_planner_client import PlannerClient


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    client=PlannerClient();receipts=[]
    try:
        for name in ('cold','warm'):
            receipts.append(client.generate(output/name,1101,[8.,0.,5.]))
    finally:client.close()
    groups=json.loads((output/'cold/airsim_replay_plan.json').read_text())['plans']
    proof=[]
    for plan in groups:
        name=plan['group_id'];a=json.loads((output/f'cold/groups/{name}.json').read_text());b=json.loads((output/f'warm/groups/{name}.json').read_text())
        assert a['parameters']==b['parameters'] and a['selected_validation']==b['selected_validation']
        assert a['counts']==b['counts']
        if a['search']:
            for x,y in zip(a['search']['evaluations'],b['search']['evaluations']):
                assert x['parameters']==y['parameters'] and x['score']==y['score'] and x['feasible']==y['feasible']
        proof.append(dict(group_id=name,same_selected_reference=True,same_proposals_and_scores=True,
            counts=a['counts'],cold_search_wall_s=a['timings']['search_wall_s'],warm_search_wall_s=b['timings']['search_wall_s']))
    result=dict(status='PASS',requests=receipts,groups=proof,airsim_calls=0,
        cold_total_wall_s=receipts[0]['parent_total_planning_wall_s'],warm_total_wall_s=receipts[1]['parent_total_planning_wall_s'],
        scope='One paired runtime-reuse development check. Same seeded network initializations and complete calls; import reuse is engineering, not a new ABOM principle or efficacy proof.')
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('requests','groups')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.output)
