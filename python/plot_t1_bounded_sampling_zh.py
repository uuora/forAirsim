"""Chinese development-clock figure, with nested components kept separate."""
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from plot_t1_identity_zh import choose_font,layout_audit,digest,save_json

def plot(analysis,output):
    analysis=Path(analysis).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_CHINESE_FIGURES_REQUIRED')
    data=json.loads(analysis.read_text());runs=[r for r in data['runs'] if r['main_frames']]
    if not runs:raise ValueError('ACTUAL_MAIN_FRAMES_REQUIRED')
    output.mkdir(parents=True);font,_=choose_font({ord(c) for c in '有界后台记录采样中文'},None)
    font_manager.fontManager.addfont(str(font));prop=font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({'font.family':prop.get_name(),'axes.unicode_minus':True,'pdf.fonttype':42,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(14.6,6.8))
    labels=[]
    for r in runs:
        label='零门控' if 'zero' in r['session_id'] else ('100毫米' if '_100_' in r['session_id'] else '50毫米')
        if '_rep' in r['session_id']:label+='重复'
        labels.append(label)
    colors=['#356b74','#597ca0','#82a394','#a37950']
    for index,(r,label) in enumerate(zip(runs,labels)):
        spans=r['components'];axes[0].plot([0,1,2],[spans['PRODUCER_SPAN']['p95_ms'],spans['ACTUAL_MEDIA_COMPLETION_SPAN']['p95_ms'],r['start_offset']['p95_ms']],
            marker='o',label=label,color=colors[index%len(colors)])
    axes[0].set_xticks([0,1,2],['生产线程跨度','实际媒体完成跨度','采集开始偏移'])
    axes[0].set_ylabel('95分位数（毫秒，最近秩）');axes[0].set_title('实际时钟与采样偏移');axes[0].grid(axis='y',alpha=.2);axes[0].legend(loc='lower left')
    r=next((r for r in reversed(runs) if '_50_' in r['session_id']),runs[-1])
    components=[('CAPTURE','图像读取'),('RGB_SAVE_HASH','RGB保存与校验'),('SAVED_RGB_READ_DETECTION_AND_CANDIDATES','读图、检测及候选'),
        ('YOLO_DETECT_CALL','其中：YOLO调用'),('SURFACE','表面估计'),('PLANNING','轨迹规划'),('CONTROL_SUBMIT_AND_AWAIT','控制提交及等待'),
        ('DEPTH_SEG_SAVE_HASH','后台深度与分割保存'),('TRUTH_EVALUATION','后台独立评分')]
    pairs=[(key,label) for key,label in components if key in r['components']]
    y=list(range(len(pairs)));values=[r['components'][key]['p95_ms'] for key,_ in pairs]
    axes[1].barh(y,values,color=['#8aa0a4' if key!='YOLO_DETECT_CALL' else '#b99a72' for key,_ in pairs])
    axes[1].set_yticks(y,[label for _,label in pairs]);axes[1].invert_yaxis();axes[1].set_xlabel('95分位数（毫秒，最近秩）')
    axes[1].set_title('最新50毫米试次的组件区间');axes[1].grid(axis='x',alpha=.2)
    for yi,value in zip(y,values):axes[1].text(value+1,yi,f'{value:.0f}',va='center',fontsize=9)
    axes[1].set_xlim(0,max(values)*1.15+5)
    fig.suptitle('任务一：有界后台记录与组件计时（受控开发）',fontsize=16,y=.97)
    notices=[f'{label} {r["completed_main_frames"]}/{r["main_frames"]}/{r["planned_main_slots"]}完成/开始/计划，完整晚到{r["timing_summary"]["historical_full"]["counts"].get("LATE",0)}帧'
        for r,label in zip(runs,labels)]
    for line_index in range((len(notices)+1)//2):
        fig.text(.06,.112-line_index*.035,'；'.join(notices[line_index*2:line_index*2+2]),fontsize=10)
    fig.text(.06,.028,'YOLO区间嵌套于检测与候选区间；生产与后台保存可能重叠，以上分位数不能相加。开发核查不代表正式方法效果。',fontsize=9,color='#586571')
    fig.subplots_adjust(left=.08,right=.96,bottom=.22,top=.86,wspace=.62)
    png=output/'bounded_sampling_zh.png';pdf=output/'bounded_sampling_zh.pdf';fig.savefig(png,dpi=170);fig.savefig(pdf)
    audit=layout_audit(fig);plt.close(fig);shutil.copyfile(__file__,output/Path(__file__).name)
    save_json(output/'artifact_receipt.json',dict(status='GENERATED_PENDING_VISUAL_REVIEW',analysis=dict(path=str(analysis),sha256=digest(analysis)),
        layout_audit=audit,figures=[dict(path=str(p),sha256=digest(p)) for p in (png,pdf)],font_path=str(font),
        visual_review=False,pdf_rendered=False,new_flights=0,cloud_requests=False))
    print(png);return png
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.analysis,a.output)
