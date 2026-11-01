"""Pure queue state machine for explicitly selected methods and policies."""
from copy import deepcopy
import json,time,uuid
from pathlib import Path


class CurveCampaign:
    def __init__(self,config,packages,selections,repeats,folder):
        if not selections or not 1<=repeats<=10:raise ValueError('请选择方法，重复次数1至10')
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.base=deepcopy(config);self.packages={k:Path(v).resolve() for k,v in packages.items()}
        self.slots=[dict(policy=p,method=m,repeat=r+1,execution_id=uuid.uuid4().hex,status='NOT_RUN') for r in range(repeats) for p,m in selections]
        self.index=0;self.phase='READY';self.stop_requested=False;self.deadline=0.;self.old_path=None
        self.save()

    @property
    def active(self):return self.phase not in ('COMPLETED','STOPPED','CLOSED','NEEDS_LANDING')

    def save(self):
        (self.folder/'campaign.json').write_text(json.dumps(dict(phase=self.phase,index=self.index,stop_requested=self.stop_requested,
            config=self.base,packages={k:str(v) for k,v in self.packages.items()},slots=self.slots,
            scope=('Each explicitly selected slot performs fresh model planning after measured arrival; repeated seed is not an independent seed' if 'preparation' in self.base else 'Explicit execution repeats, same frozen plans; repeats are not independent optimizer trainings')),ensure_ascii=False,indent=2),encoding='utf-8')

    def stop_after_current(self):
        self.stop_requested=True
        if self.phase=='READY':self.phase='STOPPED'
        self.save()

    def close(self):self.stop_requested=True;self.phase='CLOSED';self.save()

    def tick(self,runtime):
        if not self.active:return
        snap=runtime.snapshot()
        if self.phase=='READY':
            if self.stop_requested:self.phase='STOPPED';self.save();return
            if self.index==len(self.slots):self.phase='COMPLETED';self.save();return
            slot=self.slots[self.index];path=self.packages[slot['policy']]
            document=json.loads(path.read_text(encoding='utf-8'));plan=next(p for p in document['plans'] if p['group_id']==slot['method'])
            cfg=deepcopy(self.base);cfg['curve_experiment'].update(plan=plan,plans_file=str(path),
                plans_sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(),time_scale=plan['route']['time_scale'],execution_id=slot['execution_id'],batch_id=self.folder.name)
            cfg['airsim']['initial_height']=plan['boundary']['p0'][2]
            if 'preparation' in cfg:
                from prepared_route_panel import task_from_spec
                cfg['preparation'].update(task=task_from_spec(plan['route']),method=slot['method'])
            self.old_path=snap.get('run_dir');self.phase='EXECUTING';self.deadline=time.monotonic()+240
            slot['status']='STARTING';self.save();runtime.configure(cfg);runtime.start();return
        slot=self.slots[self.index]
        if snap.get('execution_id')!=slot['execution_id']:
            if time.monotonic()>self.deadline:
                slot.update(status='DISPATCH_TIMEOUT',error='运行请求未被确认');self.phase='STOPPED';self.save()
            return
        if snap.get('running') and snap.get('status')!='ERROR':return
        if snap.get('status') not in ('ERROR','STOPPED'):return
        changed=snap.get('run_dir') and snap.get('run_dir')!=self.old_path
        if snap['status']=='STOPPED' and not changed:return
        slot.update(status='FAILED' if snap['status']=='ERROR' else 'COMPLETED',run_dir=snap.get('run_dir') if changed else None,
            message=snap.get('message'),control_metrics=snap.get('control_metrics') if changed else None)
        if changed:
            summary=Path(snap['run_dir'])/'summary.json'
            if summary.is_file():
                saved=json.loads(summary.read_text(encoding='utf-8'));slot['outcome']=saved['outcome']
                if saved['outcome']!='COMPLETED':slot['status']=saved['outcome']
        self.index+=1
        if snap.get('running'):self.phase='NEEDS_LANDING'
        elif self.stop_requested:self.phase='STOPPED'
        else:self.phase='READY'
        self.save()
