"""Create native editable Origin graphs from saved T1 evidence, with no new tests."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/origin_authoring_runtime'))
import originpro as op

SOURCES={
    'detector':ROOT/'logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json',
    'age':ROOT/'logs/t1_process_source_age_figures_zh_v11_v1p3_20261008/data.json',
    'campaign':ROOT/'logs/t1_process_campaign_with_supplement_v11_20261008/report.json',
}
BLACK='#171717'
BLUE='#285F9E'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worksheet(name,columns):
    w=op.new_sheet(lname=name)
    w.cols=len(columns)
    for i,(label,vals) in enumerate(columns):w.from_list(i,vals,lname=label)
    return w


def page(name,xlabel,ylabel,xlim,ylim):
    g=op.new_graph(lname=name,template='scatter')
    gl=g[0]
    g.activate()
    op.lt_exec('page.width=page.resx*3.74;page.height=page.resy*2.95;page.autosize=0;')
    gl.lt_exec('layer.unit=1;layer.left=22;layer.top=8;layer.width=72;layer.height=70;'
               'layer.tickstyle=2;layer.tickL=4;layer.tickW=0.7;layer.title=0;'
               'layer.x.thickness=0.8;layer.y.thickness=0.8;'
               'layer.x.opposite=1;layer.y.opposite=1;'
               'layer.x.showgrids=0;layer.y.showgrids=0;'
               'layer.x.minorticks=1;layer.y.minorticks=1;'
               'layer.x.label.font=font(Times New Roman);layer.y.label.font=font(Times New Roman);'
               'layer.x.label.pt=10;layer.y.label.pt=10;'
               'xb.font=font(宋体);yl.font=font(宋体);xb.fsize=11;yl.fsize=11;'
               'xb.bold=0;yl.bold=0;')
    gl.axis('x').title=xlabel
    gl.axis('y').title=ylabel
    gl.set_xlim(*xlim)
    gl.set_ylim(*ylim)
    g.set_int('aa',1)
    return g,gl


def add(gl,w,y,x,index,kind='s',small=False):
    p=gl.add_plot(w,coly=y,colx=x,type=kind)
    p.color=[BLACK,BLUE][index]
    p.symbol_kind=[2,1][index]
    p.symbol_interior=[2,1][index]
    p.symbol_size=3.2 if small else 5
    gl.lt_exec(f'set {p.lt_range()} -w 1;')
    return p


def legend(gl,text,x,y):
    lg=gl.label('legend')
    lg.text=text
    lg.set_int('showframe',0)
    lg.set_int('fsize',9)
    lg.set_int('font',int(op.lt_int('font(宋体)')))
    lg.set_float('x',x)
    lg.set_float('y',y)


def export(g,output,name,receipt):
    g.activate()
    for suffix in ['png','pdf','svg']:
        target=output/f'{name}.{suffix}'
        returned=g.save_fig(str(target),width=2400 if suffix=='png' else 0)
        if not target.exists():raise RuntimeError(f'ORIGIN_EXPORT_FAILED {target} {returned}')
        receipt.append({'path':str(target),'sha256':sha(target),'native_page':g.name})
    print('EXPORTED',name,flush=True)


def ecdf(values):
    vals=sorted(values)
    x,y=[vals[0]],[0.]
    for i,v in enumerate(vals):
        x.extend([v,v]);y.extend([i/len(vals),(i+1)/len(vals)])
    return x,y


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    output=ap.parse_args().output.resolve()
    if output.exists():raise FileExistsError('NEW_OUTPUT_REQUIRED')
    output.mkdir(parents=True)
    d={k:json.loads(p.read_text(encoding='utf-8')) for k,p in SOURCES.items()}
    receipt=[]
    # originpro uses OriginExt.Application, an independent Origin instance.
    version=op.lt_float('@V')
    op.set_show(False)
    try:
        models=['original_COCO_sports_ball','specialist_balloon']
        episodes=sorted(d['detector']['summary_by_episode_and_model'][models[0]])
        for metric,label,name in [('recall','实例召回率 (%)','01_recall_paired'),
                                  ('precision','检测精确率 (%)','02_precision_paired')]:
            cols=[('Method',[1,2])]
            for ep in episodes:
                vals=[d['detector']['summary_by_episode_and_model'][m][ep][metric]*100 for m in models]
                cols.append((ep,vals))
            w=worksheet(name,cols)
            g,gl=page(name,'检测模型',label,(.65,2.35,1),(0,108,20))
            for i in range(2):add(gl,w,i+1,0,i,'linesymbol')
            gl.lt_exec('layer.x.minorticks=0;layer.x.label.type=10;'
                       'layer.x.label.string$="原COCO映射 专用气球模型";'
                       'layer.x.label.font=font(宋体);')
            legend(gl,r'\l(1) 测试会话1'+'\n'+r'\l(2) 测试会话2',1.75,24)
            export(g,output,name,receipt)
        cols=[]
        for m in models:
            frames=sorted([r for r in d['detector']['frames'] if r['model']==m],key=lambda r:(r['episode_id'],r['frame_id']))
            cols.extend([(m+'_index',list(range(1,len(frames)+1))),
                         (m+'_ms',[r['elapsed_ns']/1e6 for r in frames])])
        w=worksheet('FrameLatency',cols)
        g,gl=page('Latency','测试帧序号','检测耗时 (ms)',(0,50,10),(40,65,5))
        for i in range(2):add(gl,w,i*2+1,i*2,i,small=True)
        legend(gl,r'\l(1) 原COCO映射'+'\n'+r'\l(2) 专用气球模型',33,62)
        export(g,output,'03_latency_scatter',receipt)
        cols=[]
        for s in d['age']['sessions']:
            pts=s['actual_target_points']
            cols.extend([(s['session_id']+'_slot',[p['slot'] for p in pts]),
                         (s['session_id']+'_margin_ms',[p['first_clear_margin_s']*1000 for p in pts])])
        w=worksheet('SourceDeadlineMargin',cols)
        g,gl=page('SourceMargin','主测量槽编号','首次清零截止余量 (ms)',(0,400,100),(0,260,50))
        for i in range(2):add(gl,w,i*2+1,i*2,i,small=True)
        legend(gl,r'\l(1) 50 mm先导'+'\n'+r'\l(2) 独立重复',115,45)
        export(g,output,'04_clear_margin_scatter',receipt)
        ids=['t1_process_50_v11_20261007','t1_process_50_v11_rep1_20261007']
        runs=[next(r for r in d['campaign']['runs'] if r['session_id']==sid) for sid in ids]
        for stage,label,name in [('frame_processing','帧处理完成跨度 (ms)','05_processing_ecdf'),
                                  ('historical_full','含停止的完成跨度 (ms)','06_completion_ecdf')]:
            cols=[]
            for r in runs:
                vals=[f['recorded_timing'][stage]['after_due_s']*1000 for f in r['frame_timing_rows']
                      if f['recorded_timing'][stage]['after_due_s'] is not None]
                x,y=ecdf(vals)
                cols.extend([(r['session_id']+'_ms',x),(r['session_id']+'_cdf',y)])
            w=worksheet(name,cols)
            w.cols=6;w.from_list(4,[400,400],lname='Budget_ms');w.from_list(5,[0,1],lname='Budget_line')
            g,gl=page(name,label,'累积概率',(0,1200 if stage=='historical_full' else 800,200),(0,1.05,.2))
            for i in range(2):add(gl,w,i*2+1,i*2,i,'line')
            p=gl.add_plot(w,coly=5,colx=4,type='line');p.color='#777777'
            gl.lt_exec(f'set {p.lt_range()} -d 2;set {p.lt_range()} -w 0.7;')
            legend(gl,r'\l(1) 50 mm先导'+'\n'+r'\l(2) 独立重复',850 if stage=='historical_full' else 590,.22)
            export(g,output,name,receipt)
        project=output/'T1_论文科研图_Origin.opju'
        op.save(str(project))
        if not project.exists():raise RuntimeError('NATIVE_PROJECT_NOT_SAVED')
        for k,p in SOURCES.items():shutil.copyfile(p,output/f'input_{k}.json')
        shutil.copyfile(__file__,output/Path(__file__).name)
        json_out=dict(status='ORIGIN_GENERATED_PENDING_VISUAL_REVIEW',origin_version=version,
            project={'path':str(project),'sha256':sha(project)},files=receipt,
            inputs=[{'role':k,'path':str(p),'sha256':sha(p)} for k,p in SOURCES.items()],
            new_flights=0,cloud_requests=0,formal_efficacy=False,
            source={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
            authoring='Native Origin 2024 graphs and worksheets; independent COM instance',
            visual_review=False,caption_location='external manuscript; no canvas title or footnote')
        (output/'origin_artifact_receipt.json').write_text(json.dumps(json_out,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PROJECT',project,flush=True)
    finally:op.exit()


if __name__=='__main__':main()
