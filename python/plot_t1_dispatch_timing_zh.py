"""Chinese scientific chart from exact saved dispatch evidence."""
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from plot_t1_identity_zh import choose_font,layout_audit,digest,save_json


def plot(analysis,output):
    analysis,output=Path(analysis).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_DISPATCH_FIGURE_REQUIRED')
    data=json.loads(analysis.read_text(encoding='utf-8'));runs=data['runs']
    if not runs:raise ValueError('SAVED_DISPATCH_RUNS_REQUIRED')
    output.mkdir(parents=True)
    font,_=choose_font({ord(c) for c in '反馈派发时效主帧恢复核查'},None)
    font_manager.fontManager.addfont(str(font));prop=font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({'font.family':prop.get_name(),'axes.unicode_minus':True,'pdf.fonttype':42,'font.size':10})
    labels=['零门控' if '_zero_' in r['session_id'] else '100毫米' if '_100_' in r['session_id'] else
        '50毫米重复' if '_rep' in r['session_id'] else '50毫米先导' for r in runs]
    ticks=[label+'\n总核查'+('通过' if r['independent_status'].startswith('PASS') else '未通过') for label,r in zip(labels,runs)]
    colors=['#356b74','#597ca0','#82a394','#a37950']
    fig,axes=plt.subplots(1,2,figsize=(14,6.8))
    for i,(r,label) in enumerate(zip(runs,labels)):
        f=r['feedback_rows']; values=[p['state_age_at_dispatch_s']*1000 for p in f]
        axes[0].scatter([i]*len(values),values,s=8,alpha=.35,color=colors[i%4])
        if values:axes[0].plot([i-.13,i+.13],[max(values)]*2,color=colors[i%4],lw=2)
        rejected=[p['state_age_at_rejection_s']*1000 for p in r['refusal_rows']]
        if rejected:axes[0].scatter([i]*len(rejected),rejected,marker='x',s=55,color='#b65046',zorder=5)
    axes[0].axhline(120,color='#b65046',ls='--',label='原自状态时效120毫秒')
    axes[0].set_xticks(range(len(labels)),ticks);axes[0].set_ylabel('实际本地检查时的自状态年龄（毫秒）')
    axes[0].set_title('反馈派发年龄：点为通过，叉为过期拒绝');axes[0].legend(loc='upper left',fontsize=9);axes[0].grid(axis='y',alpha=.2)
    x=list(range(len(runs)))
    axes[1].bar([i-.17 for i in x],[r['planned'] for r in runs],width=.34,color='#d9e2e5',label='计划主帧')
    axes[1].bar([i+.17 for i in x],[r['main_completed'] for r in runs],width=.34,
        color=['#356b74' if r['independent_status'].startswith('PASS') else '#b65046' for r in runs],label='完整主帧')
    axes[1].set_xticks(x,ticks);axes[1].set_ylabel('主帧数');axes[1].set_title('原测量分母与全部尝试');axes[1].grid(axis='y',alpha=.2)
    for i,r in enumerate(runs):
        axes[1].text(i,r['planned']+12,f"{r['main_completed']}/{r['main_started']}/{r['planned']}",ha='center',fontsize=9)
    axes[1].set_ylim(0,max(r['planned'] for r in runs)*1.18+10);axes[1].legend(loc='upper left',fontsize=9)
    fig.suptitle('任务一：反馈派发时效与完整测量（V10P2开发验证）',fontsize=16,y=.97)
    notices=[]
    for r,label in zip(runs,labels):
        group=r['groups']['ALL_PERCEPTION_XY'];net='未测量' if r['net_mm'] is None else f"{r['net_mm']:.3f}毫米"
        notices.append(f"{label}：拒绝{group['refused_attempts']}次、恢复{group['recovered_ticks']}次；净位移{net}")
    for index in range((len(notices)+1)//2):fig.text(.06,.135-index*.035,'；'.join(notices[index*2:index*2+2]),fontsize=10)
    fig.text(.06,.045,'年龄统计含初始化和收尾，细分阶段见分析。完整/开始/计划主帧分别记账；本地检查不是服务端应用时刻。',fontsize=9,color='#586571')
    fig.text(.06,.019,'相同时钟读数不代表零耗时；未触发恢复不能证明真实阻塞下的收益。固定高度XY结果不代表三维优化或正式效果。',fontsize=9,color='#586571')
    fig.subplots_adjust(left=.08,right=.96,bottom=.24,top=.86,wspace=.28)
    png=output/'feedback_dispatch_zh.png';pdf=output/'feedback_dispatch_zh.pdf'
    fig.savefig(png,dpi=170);fig.savefig(pdf);a=layout_audit(fig);plt.close(fig);shutil.copyfile(__file__,output/Path(__file__).name)
    save_json(output/'artifact_receipt.json',dict(status='GENERATED_PENDING_VISUAL_REVIEW',analysis=dict(path=str(analysis),sha256=digest(analysis)),
        layout_audit=a,figures=[dict(path=str(p),sha256=digest(p)) for p in (png,pdf)],font_path=str(font),
        visual_review=False,pdf_rendered=False,new_flights=0,cloud_requests=False))
    print(png);return png


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.analysis,a.output)
