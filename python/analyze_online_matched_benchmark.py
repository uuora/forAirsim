"""Matched-request plots: never average raw scores of different problems."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from dual_target_control.curve_results import STYLE,table
from dual_target_control.online_results import read_lines


def analyze(folder):
    folder=Path(folder).resolve();status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status']=='RUNNING':raise ValueError('基准尚未结束')
    design=json.loads((folder/'design_frozen.json').read_text(encoding='utf-8'));rows=read_lines(folder/'results.jsonl')
    provenance=json.loads((folder/'provenance.json').read_text(encoding='utf-8'))
    if hashlib.sha256((folder/'design_frozen.json').read_bytes()).hexdigest()!=provenance['design_sha256']:raise ValueError('冻结基准变化')
    if not status['sources_unchanged']:raise ValueError('基准运行期源码变化，不能汇总')
    paired=[];lookup={(r['request_index'],r['seed'],r['mode'],r['budget'],r['arm']):r for r in rows}
    for r in rows:
        if r['arm']!='PGL':continue
        for control in ('AGL','RGL'):
            key=(r['request_index'],r['seed'],r['mode'],r['budget'],control);other=lookup.get(key)
            both=bool(other and r['usable_result'] and other['usable_result'])
            if r['mode']=='candidate':both=bool(both and r['equal_candidate_count_met'] and other['equal_candidate_count_met'])
            paired.append(dict(request_index=r['request_index'],seed=r['seed'],mode=r['mode'],budget=r['budget'],control=control,both_usable=both,
                delta_score=r['usable_selected_score']-other['usable_selected_score'] if both else None))
    with (folder/'paired_differences.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    colors={'AGL':'#82623e','PGL':'#ce5175','RGL':'#345b64'};n=len(design['requests'])
    fig,axes=plt.subplots(1,n,figsize=(5*n,4.8),squeeze=False)
    for i,ax in enumerate(axes[0]):
        for arm,color in colors.items():
            points=[]
            for b in (.2,.4,.8):
                selected=[r for r in rows if r['mode']=='wall' and r['request_index']==i and r['arm']==arm and r['budget']==b]
                values=[r['usable_selected_score'] for r in selected if r['usable_result']]
                points.append(float(np.median(values)) if values else np.nan)
                ax.scatter([b]*len(values),values,color=color,alpha=.4,s=24)
                ax.annotate(f'{len(values)}/{len(selected)}',(b,points[-1]),xytext=(3,4),textcoords='offset points',fontsize=8,color=color)
            ax.plot([.2,.4,.8],points,'.-',color=color,label=arm)
        ax.set(title=f'冻结请求{i+1}（不同图分数不平均）',xlabel='完整收件预算 / s',ylabel='按时验证可行的选中分数（小为好）');ax.legend()
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(folder/('budget_quality.'+ext),dpi=170)
    plt.close(fig)
    summary=[]
    for mode in ('wall','candidate'):
        for arm in colors:
            selected=[r for r in rows if r['mode']==mode and r['arm']==arm];expected=sum(j['mode']==mode and j['arm']==arm for j in design['jobs'])
            summary.append(dict(mode=mode,arm=arm,expected=expected,recorded=len(selected),usable=sum(r['usable_result'] for r in selected),
                equal32=sum(r.get('equal_candidate_count_met') is True for r in selected) if mode=='candidate' else None))
    wins=[]
    for mode in ('wall','candidate'):
        for control in ('AGL','RGL'):
            selected=[r for r in paired if r['mode']==mode and r['control']==control];valid=[r['delta_score'] for r in selected if r['both_usable']]
            wins.append(dict(mode=mode,candidate='PGL',control=control,all_pairs=len(selected),usable_pairs=len(valid),
                better=sum(v < -1e-10 for v in valid),tied=sum(abs(v)<=1e-10 for v in valid),worse=sum(v > 1e-10 for v in valid)))
    result=dict(status=status,summary=summary,paired_win_counts=wins,
        scope='Old1201 frozen development requests; one common persistent worker startup; per-job initialization/validation and IPC included. No AirSim flight or significance evidence.')
    (folder/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8">'+STYLE+'<main><h1>相同请求：搜索预算与质量</h1><p>每请求4个优化种子、三种预算、AGL/PGL/RGL同域同锚点；另补等32候选档。主质量要求工作进程完成与结果收件均按时，且选中验证可行。逾期质量仅在原始记录保留。</p>'
    page+='<section>'+table(['模式','算法','计划任务','已记录','可用结果','完成32候选'],[[r[k] for k in ('mode','arm','expected','recorded','usable','equal32')] for r in summary])+'</section>'
    page+='<section><h2>PGL逐问题配对胜负</h2><p>仅当双方结果可用时判断分数胜负，完整配对分母另列；分数接近1e−10内算持平。不能跨请求平均原始分数。</p>'+table(['模式','对照','全部配对','双方可用','更好','持平','更差'],[[r[k] for k in ('mode','control','all_pairs','usable_pairs','better','tied','worse')] for r in wins])+'</section>'
    page+='<section><img src="budget_quality.png"><p>散点为原始重复，线为可用结果中位数，数字为可用／已记录；失效不填虚构分数。</p></section><section><a href="metrics.csv">完整指标</a> · <a href="paired_differences.csv">配对差</a> · <a href="results.jsonl">逐任务账本</a> · <a href="analysis.json">状态与解释</a><p>网络启动／预热见startup.json；本试验为简化模型诊断，AirSim调用0。AI协助分析，结果需后续独立场景验证。</p></section></main>'
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    (folder/'analysis_provenance.json').write_text(json.dumps(dict(analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('design_frozen.json','status.json','results.jsonl','metrics.csv','provenance.json')}),indent=2),encoding='utf-8')
    return folder/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(analyze(a.folder))
