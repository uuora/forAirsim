"""Typeset native-Origin companion figures from identical source data."""
import argparse,csv,json,hashlib,shutil,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
from matplotlib import font_manager as fm

ROOT=Path(__file__).resolve().parents[1]
BLACK,BLUE='#161616','#245D91'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new(xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(3.74,2.95))
    fig.subplots_adjust(left=.19,bottom=.20,right=.96,top=.95)
    ax.set_xlabel(xlabel,labelpad=3);ax.set_ylabel(ylabel,labelpad=4)
    ax.tick_params(direction='in',which='both',top=True,right=True,width=.65,labelsize=8.5)
    ax.tick_params(which='major',length=3.3);ax.tick_params(which='minor',length=1.7)
    for sp in ax.spines.values():sp.set_linewidth(.7);sp.set_color('black')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return fig,ax


def save(fig,ax,out,name,records):
    for t in ax.get_xticklabels()+ax.get_yticklabels():
        if not any('\u4e00' <= c <= '\u9fff' for c in t.get_text()):t.set_fontname('Times New Roman')
    for ext in ['png','pdf','svg']:
        p=out/f'{name}.{ext}';fig.savefig(p,dpi=600,facecolor='white')
        records.append(dict(path=str(p),sha256=sha(p)))
    plt.close(fig)


def key(ax,loc):ax.legend(frameon=False,fontsize=7.5,loc=loc,handlelength=1.5,borderpad=.15,labelspacing=.35)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    font=Path('C:/Windows/Fonts/simsun.ttc');fm.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':fm.FontProperties(fname=str(font)).get_name(),'font.size':9,
                         'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    inputs={'detector':ROOT/'logs/t1_balloon_yolo_evaluation_v1p2_20261004/report.json',
            'age':ROOT/'logs/t1_process_source_age_figures_zh_v11_v1p3_20261008/data.json',
            'campaign':ROOT/'logs/t1_process_campaign_with_supplement_v11_20261008/report.json'}
    d={k:json.loads(v.read_text(encoding='utf-8')) for k,v in inputs.items()};records=[];points=[]
    models=['original_COCO_sports_ball','specialist_balloon']
    episodes=sorted(d['detector']['summary_by_episode_and_model'][models[0]])
    for metric,title,name in [('recall','实例召回率 (%)','01_recall_paired'),('precision','检测精确率 (%)','02_precision_paired')]:
        fig,ax=new('检测模型',title)
        for i,ep in enumerate(episodes):
            vals=[d['detector']['summary_by_episode_and_model'][m][ep][metric]*100 for m in models]
            for m,v in zip(models,vals):points.append(dict(figure=name,source_id=ep,series=m,x=1 if m==models[0] else 2,y=v))
            ax.plot([1,2],vals,lw=.85,marker=['o','s'][i],ms=4,mfc='white' if i==0 else BLUE,
                    mec=[BLACK,BLUE][i],mew=.75,color=[BLACK,BLUE][i],label=f'测试会话{i+1}')
        ax.set(xlim=(.65,2.35),ylim=(0,108));ax.set_xticks([1,2],['原COCO映射','专用气球模型'])
        ax.xaxis.set_minor_locator(MultipleLocator(1));ax.yaxis.set_major_locator(MultipleLocator(20))
        key(ax,'lower right');save(fig,ax,out,name,records)
    fig,ax=new('测试帧序号','检测耗时 (ms)')
    for i,m in enumerate(models):
        rows=sorted([r for r in d['detector']['frames'] if r['model']==m],key=lambda r:(r['episode_id'],r['frame_id']))
        vals=[r['elapsed_ns']/1e6 for r in rows];xs=list(range(1,len(rows)+1))
        ax.plot(xs,vals,ls='none',marker=['o','s'][i],ms=2.8,mfc='white' if i==0 else BLUE,
                mec=[BLACK,BLUE][i],mew=.55,label=['原COCO映射','专用气球模型'][i])
        for x,r,y in zip(xs,rows,vals):points.append(dict(figure='03_latency_scatter',source_id=r['episode_id']+'::'+r['frame_id'],series=m,x=x,y=y))
    ax.set(xlim=(0,50),ylim=(40,67));ax.xaxis.set_major_locator(MultipleLocator(10));ax.yaxis.set_major_locator(MultipleLocator(5))
    key(ax,'upper right');save(fig,ax,out,'03_latency_scatter',records)
    fig,ax=new('主测量槽编号','首次清零截止余量 (ms)')
    for i,s in enumerate(d['age']['sessions']):
        pts=s['actual_target_points'];xs=[r['slot'] for r in pts];ys=[r['first_clear_margin_s']*1000 for r in pts]
        ax.plot(xs,ys,ls='none',marker=['o','s'][i],ms=2.5,mfc='white' if i==0 else BLUE,
                mec=[BLACK,BLUE][i],mew=.45,label=['50 mm先导','独立重复'][i])
        for r,x,y in zip(pts,xs,ys):points.append(dict(figure='04_clear_margin_scatter',source_id=r['frame_id'],series=s['session_id'],x=x,y=y))
    ax.set(xlim=(0,400),ylim=(0,260));ax.xaxis.set_major_locator(MultipleLocator(100));ax.yaxis.set_major_locator(MultipleLocator(50))
    key(ax,'lower left');save(fig,ax,out,'04_clear_margin_scatter',records)
    ids=['t1_process_50_v11_20261007','t1_process_50_v11_rep1_20261007']
    runs=[next(r for r in d['campaign']['runs'] if r['session_id']==sid) for sid in ids]
    for stage,label,name in [('frame_processing','帧处理完成跨度 (ms)','05_processing_ecdf'),('historical_full','含停止的完成跨度 (ms)','06_completion_ecdf')]:
        fig,ax=new(label,'累积概率')
        for i,r in enumerate(runs):
            vals=sorted(f['recorded_timing'][stage]['after_due_s']*1000 for f in r['frame_timing_rows'] if f['recorded_timing'][stage]['after_due_s'] is not None)
            ys=np.arange(1,len(vals)+1)/len(vals)
            ax.step([vals[0]]+vals,[0]+list(ys),where='post',lw=1,color=[BLACK,BLUE][i],ls=['-','--'][i],label=['50 mm先导','独立重复'][i])
            for f in r['frame_timing_rows']:
                t=f['recorded_timing'][stage]
                if t['after_due_s'] is not None:points.append(dict(figure=name,source_id=f['frame_id'],series=r['session_id'],x=t['after_due_s']*1000,y=''))
        ax.axvline(400,color='#777777',ls=':',lw=.7)
        ax.set(xlim=(0,1200 if stage=='historical_full' else 800),ylim=(0,1.05));ax.xaxis.set_major_locator(MultipleLocator(200));ax.yaxis.set_major_locator(MultipleLocator(.2))
        key(ax,'lower right');save(fig,ax,out,name,records)
    with (out/'figure_point_mapping.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(points[0]));wr.writeheader();wr.writerows(points)
    shutil.copyfile(__file__,out/Path(__file__).name)
    receipt=dict(status='GENERATED_PENDING_VISUAL_REVIEW',authoring='Matplotlib publication typesetting companion to native Origin project',
                 origin_export_issue='Native export has unexplained text overbars; native project remains editable. This companion is separately rendered from identical saved data.',
                 files=records,inputs=[dict(role=k,path=str(v),sha256=sha(v)) for k,v in inputs.items()],
                 font=dict(path=str(font),sha256=sha(font)),runtime=dict(python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__),
                 new_flights=0,cloud_requests=0,inferential_statistics=False,
                 plot_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
                 point_mapping=dict(path=str(out/'figure_point_mapping.csv'),sha256=sha(out/'figure_point_mapping.csv')))
    (out/'publication_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Figures',out)


if __name__=='__main__':main()
