"""Small live local-model integration check, with no dispatch or flight."""
import json,sys
from fixed_map_mission import OUT,read,save,utc
from fixed_map_language import interpret

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    task=read(OUT/'fixed_version/task.json');folder=OUT/'interface_revision_20261009'
    cases=[('暂停任务并悬停','pause',True),('返回起飞平台并落地','return',True),
           ('让DroneB起飞接近气球','none',False),('把高度改成5米后开始绕障接近目标','none',False),
           ('看看那边','none',False)]
    rows=[]
    for text,action,accepted in cases:
        try:record=interpret(text,task)
        except Exception as error:record=dict(instruction=text,validated=False,validation_error=str(error),flight_executed=False)
        p=record.get('proposal',{})
        record.update(expected_action=action,expected_accepted=accepted,
            expectation_pass='proposal' in record and record['validated']==accepted and (not accepted or p.get('action')==action))
        rows.append(record);save(folder/'llm_small_live_checks.json',dict(utc=utc(),scope='five live text cases; no statistical reliability inference; no command dispatch',rows=rows))
        print(json.dumps(record,ensure_ascii=False),flush=True)
    print('completed',sum(r['expectation_pass'] for r in rows),'/',len(rows))

if __name__=='__main__':main()
