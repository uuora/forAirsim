"""Persistent experiment catalogue and reversible cleanup of derived views only.

Raw runs, failed runs, plans and unknown directories never qualify for cleanup.
No directory is relocated: archived views retain their original restore location.
"""
import csv,hashlib,html,json,os,sqlite3,stat,time,uuid,zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
KINDS={'flight':'曲线飞行','control':'原03控制日志','legacy':'其他原始日志','campaign':'批量实验','plan':'冻结规划方案','view':'派生图表','research':'研究/验证资料'}
STATUS={'PASS':'控制通过','CONTROL_FAIL':'控制未通过','FAILED':'运行失败','STOPPED':'中止/关闭','INVALID':'证据无效/待核验','INCOMPLETE':'不完整/待检查','COMPLETE':'已完成','UNKNOWN':'未判定'}
VIEW_FILES={'index.html','trajectory.png','trajectory.svg','trajectory.pdf','projection.png','projection.svg','projection.pdf','error.png','error.svg','error.pdf','speed_heading.png','speed_heading.svg','speed_heading.pdf','comparison.png','comparison.svg','comparison.pdf','metrics.png','metrics.svg','metrics.pdf','comparison.csv','指标中文.csv','audit.json','provenance.json','plot_data.npz','plot_style.json','figure_index.json','definitions.json','数据字段说明.csv'}

def read_json(path,default=None):
    try:
        if path.stat().st_size>12_000_000:raise ValueError('metadata exceeds 12MB')
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        return default if isinstance(default,dict) and not isinstance(value,dict) else value
    except (OSError,ValueError):return default

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def checked(root,path):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    resolved=path.resolve()
    if root not in resolved.parents:raise ValueError('路径不在工作区内：'+str(path))
    # Reject all symlinks/junctions, including ancestors within the workspace.
    for candidate in [path,*path.parents]:
        if candidate==root:break
        if candidate.exists():
            info=candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('不处理链接或目录联接：'+str(candidate))
    return resolved

def tree_files(path):
    files=[]
    for directory,children,names in os.walk(path,followlinks=False):
        for name in children:
            child=Path(directory)/name
            if child.is_symlink() or getattr(child.lstat(),'st_file_attributes',0)&0x400:raise ValueError('目录包含链接或联接')
        for name in names:
            p=Path(directory)/name
            if p.is_symlink() or getattr(p.lstat(),'st_file_attributes',0)&0x400:raise ValueError('目录包含链接')
            files.append(p)
    return sorted(files)

class ExperimentStore:
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.home=self.root/'experiment_data';self.home.mkdir(parents=True,exist_ok=True)
        self.database=self.home/'catalog.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS annotations (id TEXT PRIMARY KEY, tags TEXT, note TEXT, pinned INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS workspace_visibility (id TEXT PRIMARY KEY, visibility TEXT NOT NULL)')
    def connect(self):return sqlite3.connect(self.database,timeout=20)
    def annotate(self,record_id,*,tags='',note='',pinned=False):
        if len(tags)>500 or len(note)>4000:raise ValueError('标签或备注过长')
        with self.connect() as db:db.execute('INSERT OR REPLACE INTO annotations VALUES (?,?,?,?)',(record_id,tags,note,int(pinned)))
    def records(self):
        with self.connect() as db:
            notes={r[0]:dict(tags=r[1],note=r[2],pinned=bool(r[3])) for r in db.execute('SELECT * FROM annotations')}
            rows=[json.loads(r[0]) for r in db.execute('SELECT metadata FROM records')]
            visibility=dict(db.execute('SELECT id,visibility FROM workspace_visibility'))
        for row in rows:row.update(notes.get(row['id'],notes.get(row.get('original_id'),dict(tags='',note='',pinned=False))))
        self.curate(rows,visibility)
        return sorted(rows,key=lambda r:r['modified'],reverse=True)
    def curate(self,rows,visibility):
        """Current control workspace first; old evidence remains addressable."""
        preset=read_json(self.root/'configs/curve_live_protocol14.json',{})
        plans={str((self.root/p).parent.resolve()) for p in preset.get('packages',{}).values()}
        main='reports/route_protocol14_all12_v1';primary='reports/route_dashboard_v2'
        for row in rows:
            key=row['id'];kind=row['kind'];name=row['name'];current=False
            if kind=='flight' and str(row['protocol']) in ('14','15','16'):
                suite=row.get('suite')
                if suite is None:
                    summary=read_json(Path(row['path'])/'summary.json',{})
                    suite=((summary.get('config') or {}).get('curve_experiment') or {}).get('batch_id','')
                current=not suite.startswith('route_protocol14_') or suite=='route_protocol14_all12_v1'
                row['display_name']='控制飞行 · '+{'stop':'逐点停稳','pass':'连续过点'}.get(row['policy'],'单目标')
                if str(row['protocol'])=='16':row['display_name']='连续在线飞行 · '+row['name']
            elif kind=='plan':
                current=row['path'] in plans or name.startswith('route_plans_')
                if current:row['display_name']={'stop':'逐点停稳','pass':'连续过点'}.get(row['policy'],'')+' · 参考方案'
            elif kind=='campaign':
                current=key==main or name.startswith('route_campaign_')
                if key==main:row['display_name']='已验证参考批次 · 两组×六方法'
            elif kind=='view':
                current=key==primary or key.startswith(primary+'/') or key=='reports/control_view_protocol15_final' or key.startswith('reports/control_view_protocol15_final/') or (str(row['protocol']) in ('14','15','16') and key.startswith('reports/control_view_') and bool(row.get('input_key')))
                if key.startswith(primary+'/'):row['display_name']={'stop':'逐点停稳','pass':'连续过点'}.get(row['policy'],'')+' · 六方法对比图'
            row['collection']='current' if current else 'history'
        # Keep one current view for the same raw inputs; retain older versions.
        by_input={}
        for row in rows:
            if row['collection']=='current' and row.get('input_key'):by_input.setdefault(row['input_key'],[]).append(row)
        for group in by_input.values():
            winner=max(group,key=lambda r:(r['id'].startswith(primary+'/'),r['modified']))
            for row in group:
                if row is not winner:row['collection']='history'
        for row in rows:
            if row.get('view_role')=='collection' or (row['kind']=='view' and not row.get('input_key') and not row.get('archive_id')):
                row['collection']='current' if any(child['collection']=='current' and child['id'].startswith(row['id']+'/') for child in rows) else 'history'
                if row['id']==primary:row['display_name']='当前六方法对比图 · 停稳/连续分组'
            override=visibility.get(row['id'])
            if override in ('current','history'):row['collection']=override
            elif row.get('pinned') and not row.get('archive_id'):row['collection']='current'
    def set_visibility(self,record_ids,visibility):
        if visibility not in ('current','history','auto'):raise ValueError('无效的工作区分类')
        records={r['id']:r for r in self.records()};ids=list(dict.fromkeys(record_ids))
        if not ids or any(key not in records for key in ids):raise ValueError('请选择已有记录')
        if visibility=='history' and any(records[key].get('pinned') for key in ids):raise ValueError('请先取消固定保留，再收起到历史')
        with self.connect() as db:db.executemany('INSERT OR REPLACE INTO workspace_visibility VALUES (?,?)',[(key,visibility) for key in ids])
        with (self.home/'history_actions.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(time=time.time(),records=ids,visibility=visibility),ensure_ascii=False)+'\n')
        self.export();return self.records()
    def classify(self,path,source):
        row=dict(id=path.relative_to(self.root).as_posix(),path=str(path),name=path.name,kind='legacy' if source=='logs' else 'research',status='UNKNOWN',
            protocol='',method='',policy='',seed='',condition='',mode='',error='',modified=path.stat().st_mtime,size_bytes=0,file_count=0)
        manifest=read_json(path/'live_manifest.json',{});summary=read_json(path/'summary.json',{});online=read_json(path/'online_manifest.json',{})
        if online:
            settings=(online.get('conditions') or {}).get('online') or {};metrics=summary.get('online_metrics') or {}
            row.update(kind='flight',protocol=16,method=online.get('method',''),policy='online',seed=settings.get('seed',''),condition=online.get('condition_fingerprint',''),
                mode=online.get('mode',''),outcome=summary.get('outcome','INCOMPLETE'),error=summary.get('error') or '')
            row['status']='INCOMPLETE' if not summary else 'FAILED' if summary.get('outcome')=='FAILED' else 'STOPPED' if summary.get('outcome') not in ('DURATION_COMPLETE','STREAM_COMPLETE') else 'PASS' if metrics.get('tracking_checks_passed') else 'CONTROL_FAIL'
            if summary.get('sources_unchanged') is False:row.update(status='INVALID',error='在线飞行期间源码变化')
            row['metrics']={k:metrics.get(k) for k in ('position_rmse_m','executed_duration_s','targets_reached')}
        elif manifest:
            plan=manifest.get('plan') or {};route=plan.get('route') or {};metrics=summary.get('metrics') or {}
            row.update(kind='flight',protocol=manifest.get('protocol_revision',''),method=manifest.get('method',''),policy=route.get('policy','single'),seed=plan.get('seed',''),
                condition=manifest.get('condition_fingerprint',''),mode=manifest.get('mode',''),outcome=summary.get('outcome','INCOMPLETE'),error=summary.get('error') or summary.get('metric_error') or '')
            row['suite']=((summary.get('config') or {}).get('curve_experiment') or {}).get('batch_id','')
            row['status']='INCOMPLETE' if not summary else 'FAILED' if summary.get('outcome')=='FAILED' else 'STOPPED' if summary.get('outcome')!='COMPLETED' else 'PASS' if metrics.get('control_error_gates_passed') is True else 'CONTROL_FAIL' if metrics else 'INCOMPLETE'
            if summary.get('sources_unchanged') is False:row.update(status='INVALID',error='执行期间源码变化；原始记录保留，不进入公平排名。')
            row['metrics']={k:metrics.get(k) for k in ('position_rmse_m','reference_duration_s','quality_term','terminal_position_error_m')}
        elif source=='logs' and path.name.startswith('dual_target_'):
            row.update(kind='control',mode=summary.get('mode',''),outcome=summary.get('outcome','INCOMPLETE'),status='FAILED' if summary.get('outcome')=='FAILED' else 'COMPLETE' if summary.get('outcome')=='COMPLETED' else 'STOPPED' if summary else 'INCOMPLETE')
        elif (path/'campaign.json').is_file():
            campaign=read_json(path/'campaign.json',{});row.update(kind='campaign',status='COMPLETE' if campaign.get('phase')=='COMPLETED' else 'INCOMPLETE',outcome=campaign.get('phase','UNKNOWN'),slots=len(campaign.get('slots',[])))
        elif (path/'airsim_replay_plan.json').is_file():
            config=read_json(path/'config.json',{});route=config.get('route') or {};row.update(kind='plan',status='COMPLETE' if (read_json(path/'completion.json',{}) or {}).get('status')=='COMPLETE' else 'UNKNOWN',policy=route.get('policy',''),seed=route.get('seed',''))
        elif self.is_view(path):
            provenance=read_json(path/'provenance.json',{});row.update(kind='view',status='COMPLETE',condition=provenance.get('condition_fingerprint',''))
            row['input_key']=hashlib.sha256(json.dumps(provenance.get('input_sha256'),sort_keys=True).encode()).hexdigest()
            methods=[]
            for source_path in provenance.get('input_sha256',{}):
                if Path(source_path).name not in ('live_manifest.json','online_manifest.json'):continue
                try:source_manifest=read_json(checked(self.root,source_path),{})
                except ValueError:continue
                if not source_manifest:continue
                plan=source_manifest.get('plan') or {};route=plan.get('route') or {}
                online_settings=(source_manifest.get('conditions') or {}).get('online') or {}
                row.update(protocol=source_manifest.get('protocol_revision',''),policy='online' if online_settings else route.get('policy','single'),seed=online_settings.get('seed',plan.get('seed','')),mode=source_manifest.get('mode',''))
                method=source_manifest.get('method')
                if isinstance(method,str) and method not in methods:methods.append(method)
            if methods:row.update(method=methods[0] if len(methods)==1 else f'{len(methods)}种方法对比',methods=methods)
        for key in ('method','policy','mode','condition'):
            if not isinstance(row[key],str):raise TypeError('元数据字段须为字符串：'+key)
        try:
            files=tree_files(path);row.update(size_bytes=sum(p.stat().st_size for p in files),file_count=len(files))
        except (OSError,ValueError) as exc:row['error']=str(exc)
        return row
    @staticmethod
    def is_view(path):
        data=read_json(path/'provenance.json',{})
        return bool((path/'index.html').is_file() and isinstance(data.get('input_sha256'),dict) and data['input_sha256'] and data.get('condition_fingerprint'))
    def safe_classify(self,path,source):
        try:return self.classify(path,source)
        except (OSError,TypeError,KeyError,AttributeError) as exc:
            # Invalid nested metadata must remain visible, rather than making
            # the entire catalogue fail or silently dropping a failed trial.
            return dict(id=path.relative_to(self.root).as_posix(),path=str(path),name=path.name,kind='legacy' if source=='logs' else 'research',
                status='INCOMPLETE',protocol='',method='',policy='',seed='',condition='',mode='',error='元数据结构异常：'+str(exc),modified=path.stat().st_mtime if path.exists() else 0.,size_bytes=0,file_count=0)
    def scan(self):
        rows=[]
        for source in ('logs','reports'):
            parent=self.root/source
            if not parent.exists():continue
            for path in sorted(parent.iterdir()):
                if not path.is_dir() or path.is_symlink():continue
                try:checked(self.root,path);rows.append(self.safe_classify(path,source))
                except (OSError,ValueError):continue
                if source=='reports':
                    views=[*path.iterdir(),*(path/'data_view').glob('*')]
                    found=False
                    for view in views:
                        if view.is_dir() and self.is_view(view):rows.append(self.safe_classify(view,source))
                        if view.is_dir() and self.is_view(view):found=True
                    if found and rows[-1]['id']!=path.relative_to(self.root).as_posix():
                        # A router is a classified result collection, but only
                        # independently regenerable leaf views can be cleaned.
                        parent=next(r for r in rows if r['id']==path.relative_to(self.root).as_posix())
                        if parent['kind']=='research':parent.update(kind='view',status='COMPLETE',view_role='collection',policy='multi')
        # Explicit archived state remains findable without restoring the files.
        for entry in self.archives():
            if entry['status'] in ('ARCHIVED','PARTIAL_ARCHIVE'):
                archived=dict(id='archive:'+entry['id'],path=str(self.home/'archive'/(entry['id']+'.json')),name=entry['original'],kind='view',status='ARCHIVED',protocol='',method='',policy='',seed='',condition='',mode='',error='',modified=entry['created'],size_bytes=entry.get('archive_bytes',0),file_count=len(entry['files']),archive_id=entry['id'],original_id=entry['original'])
                archived.update({k:v for k,v in entry.get('classification',{}).items() if k in ('protocol','method','policy','seed','condition','mode')});rows.append(archived)
        counts=Counter(r.get('input_key') for r in rows if r.get('input_key'))
        for row in rows:row['same_input_views']=counts.get(row.get('input_key'),0)
        with self.connect() as db:
            db.execute('DELETE FROM records');db.executemany('INSERT INTO records VALUES (?,?)',[(r['id'],json.dumps(r,ensure_ascii=False)) for r in rows])
        self.export();return self.records()
    def index_paths(self,paths):
        rows=[]
        for value in dict.fromkeys(str(p) for p in paths):
            path=checked(self.root,value)
            source=path.relative_to(self.root).parts[0]
            if source not in ('logs','reports') or not path.is_dir():continue
            row=self.safe_classify(path,source);views=[p for p in [*path.iterdir(),*(path/'data_view').glob('*')] if p.is_dir() and self.is_view(p)]
            if views and row['kind']=='research':row.update(kind='view',status='COMPLETE',view_role='collection',policy='multi')
            rows.append(row);rows.extend(self.safe_classify(p,source) for p in views)
        with self.connect() as db:db.executemany('INSERT OR REPLACE INTO records VALUES (?,?)',[(r['id'],json.dumps(r,ensure_ascii=False)) for r in rows])
        self.export();return self.records()
    def export(self):
        rows=self.records();atomic_json(self.home/'catalog.json',dict(updated=time.time(),records=rows))
        fields=['id','collection','kind','status','protocol','policy','method','seed','size_bytes','file_count','same_input_views','tags','note','pinned','path']
        with (self.home/'catalog.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        counts=Counter(KINDS.get(r['kind'],r['kind']) for r in rows)
        body='<html lang="zh-CN"><meta charset="utf-8"><title>实验数据目录</title><style>body{font:14px system-ui;margin:30px;color:#173346}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #dde5eb;text-align:left}a{color:#137a83}</style><h1>实验数据目录</h1><p>索引不搬移原始日志；失败和不完整记录保留。'+html.escape(str(dict(counts)))+'</p><table><tr><th>目录</th><th>类别</th><th>结果</th><th>协议/组</th><th>方法</th><th>标签</th></tr>'
        for r in rows:body+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['id'],KINDS.get(r['kind'],r['kind']),STATUS.get(r['status'],r['status']),str(r['protocol'])+'/'+str(r['policy']),r['method'],r.get('tags','')])+'</tr>'
        (self.home/'index.html').write_text(body+'</table><p>请在Qt面板“数据管理”中筛选、备注、查看图表或恢复归档。</p></html>',encoding='utf-8')
    def protected_paths(self,active=()):
        paths=[self.root/'reports/route_protocol14_all12_v1/data_view',self.root/'reports/live_curve_protocol13_comparison_v3',self.root/'reports/route_dashboard_v2']
        paths.extend(Path(p) for p in active if p)
        with self.connect() as db:
            for key, in db.execute('SELECT id FROM annotations WHERE pinned=1'):
                if not key.startswith('archive:'):paths.append(self.root/key)
        return [p.resolve() for p in paths]
    def preview_archive(self,record_ids,active=()):
        result=[];protected=self.protected_paths(active);chosen=[]
        for record_id in dict.fromkeys(record_ids):
            path=checked(self.root,record_id)
            if (self.root/'reports').resolve() not in path.parents:raise ValueError('只允许归档reports中的派生图表')
            if any(path==p or path in p.parents or p in path.parents for p in protected):raise ValueError('目录受保护、被固定或正在使用：'+record_id)
            if any(path in p.parents or p in path.parents for p in chosen):raise ValueError('不能同时选父目录和子目录')
            if not self.is_view(path):raise ValueError('不是可重新生成的图表目录；原始数据、方案和未知目录受保护：'+record_id)
            chosen.append(path);files=tree_files(path)
            for file in files:
                rel=file.relative_to(path)
                paged=file.stem.startswith('speed_heading_') and file.stem.removeprefix('speed_heading_').isdigit() and file.suffix in ('.png','.svg','.pdf')
                plot_source=len(rel.parts)==2 and rel.parts[0]=='plot_source' and file.name in ('curve_results.py','curve_plots.py','curve_audit.py','metric_explanations.py')
                if not plot_source and ((file.name not in VIEW_FILES and not paged) or (len(rel.parts)>1 and rel.parts[0]!='字段说明')):raise ValueError('包含非派生或未知文件，拒绝清理：'+str(rel))
            result.append(dict(original=path.relative_to(self.root).as_posix(),files=[dict(path=p.relative_to(path).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in files]))
        return dict(created=time.time(),entries=result,bytes=sum(f['size'] for e in result for f in e['files']),scope='Verified zip archive of regenerable views; raw evidence stays in place')
    def archive(self,preview,active=()):
        if not preview.get('entries'):raise ValueError('没有选择可归档图表')
        current=self.preview_archive([e['original'] for e in preview['entries']],active)
        if current['entries']!=preview['entries']:raise ValueError('预览后文件已变化，请重新预览')
        receipts=[];home=self.home/'archive';home.mkdir(exist_ok=True)
        for entry in current['entries']:
            path=checked(self.root,entry['original']);key=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:10]
            archive=home/(key+'.zip');manifest=home/(key+'.json')
            classification=self.safe_classify(path,'reports')
            state=dict(entry,id=key,created=time.time(),status='PREPARING',classification={k:classification[k] for k in ('protocol','method','policy','seed','condition','mode')})
            atomic_json(manifest,state)
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
                for file in entry['files']:output.write(checked(path,file['path']),file['path'])
            with zipfile.ZipFile(archive) as saved:
                for file in entry['files']:
                    if hashlib.sha256(saved.read(file['path'])).hexdigest()!=file['sha256']:raise ValueError('归档校验失败，原文件未清理')
            # Recheck after copying. Only verified immutable content is unlinked.
            if self.preview_archive([entry['original']],active)['entries'][0]!=entry:raise ValueError('归档期间目录变化，原文件未清理')
            state.update(status='PARTIAL_ARCHIVE',archive_bytes=archive.stat().st_size);atomic_json(manifest,state)
            try:
                for file in entry['files']:
                    target=checked(path,file['path'])
                    if sha(target)!=file['sha256']:raise ValueError('文件在清理前发生变化')
                    target.unlink()
                for directory in sorted((p for p in path.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):checked(self.root,directory).rmdir()
                checked(self.root,path).rmdir();state['status']='ARCHIVED'
            finally:atomic_json(manifest,state)
            receipts.append(state)
        self.scan();return receipts
    def archives(self):
        return [v for p in sorted((self.home/'archive').glob('*.json'),reverse=True) if (v:=read_json(p,{})) and v.get('id')]
    def restore(self,archive_id):
        if not archive_id or any(c not in '0123456789_abcdef' for c in archive_id):raise ValueError('无效归档编号')
        manifest=self.home/'archive'/(archive_id+'.json');entry=read_json(manifest,{})
        if entry.get('status') not in ('ARCHIVED','PARTIAL_ARCHIVE'):raise ValueError('归档状态不能恢复')
        path=checked(self.root,entry['original'])
        if (self.root/'reports').resolve() not in path.parents:raise ValueError('恢复位置不是reports派生目录')
        with zipfile.ZipFile(self.home/'archive'/(archive_id+'.zip')) as saved:
            for item in entry['files']:
                target=checked(path,item['path']);content=saved.read(item['path'])
                if hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('归档内容损坏')
                for parent in target.parents:
                    if parent.exists() and not parent.is_dir():raise ValueError('恢复目录位置已有普通文件：'+str(parent))
                    if parent==self.root:break
                if target.exists() and sha(target)!=item['sha256']:raise ValueError('原位置已有不同文件，拒绝覆盖：'+str(target))
            for item in entry['files']:
                target=checked(path,item['path'])
                if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(saved.read(item['path']))
        entry['status']='RESTORED';entry['restored']=time.time();atomic_json(manifest,entry);self.scan();return path

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--scan',action='store_true');args=parser.parse_args()
    store=ExperimentStore(args.root);rows=store.scan() if args.scan else store.records();print(json.dumps(dict(records=len(rows),categories=dict(Counter(r['kind'] for r in rows)),index=str(store.home/'index.html')),ensure_ascii=False))
