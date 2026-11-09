"""Read-only comparison of recorded flights, with raw samples and no smoothing."""
import html
import hashlib
import json
from pathlib import Path

import numpy as np


STYLES = {
    'bezier':dict(color='#007c91',linestyle='-',linewidth=2.6,marker='o'),
    'hermite':dict(color='#b6417a',linestyle='--',linewidth=1.5,marker='v'),
    'bspline':dict(color='#a66a00',linestyle='-.',linewidth=1.7,marker='s'),
}


def _frame_world(world, displayed, points, title, locator, display=False):
    if not display:world.scatter(*points.T,color='#445764',marker='x',s=28)
    world.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Height (m)',title=title)
    cloud=np.concatenate(displayed+[points]);low=cloud.min(axis=0);high=cloud.max(axis=0)
    span=high-low
    minimum=np.array([1.,1.,1.]) if display else max(1.,span.max()*.25)
    span=np.maximum(span,minimum)*1.12
    center=(low+high)/2
    for setter,c,extent in zip((world.set_xlim,world.set_ylim,world.set_zlim),center,span):
        setter(c-extent/2,c+extent/2)
    aspect=np.array([1.55,1.1,.8]) if display else span
    world.set_box_aspect(aspect.copy(),zoom=1.12 if display else 1.)
    if display:
        elevation=43 if np.ptp(points[:,2])<.1 else 28
        world.set_proj_type('ortho');world.view_init(elev=elevation,azim=-58)
        for axis in (world.xaxis,world.yaxis,world.zaxis):
            axis.set_pane_color((.975,.984,.99,.5))
            axis._axinfo['grid'].update(color='#dce5eb',linewidth=.6)
        world.tick_params(labelsize=9,pad=2)
        for index,point in enumerate(points):
            label='S' if index==0 else 'E' if index==len(points)-1 else 'W'+str(index)
            color='#32816d' if index==0 else '#cf4854' if index==len(points)-1 else '#fff'
            world.scatter(*point,color=color,edgecolor='#445764',marker='D' if label=='E' else 'o',s=38,depthshade=False)
            world.text(point[0],point[1],point[2]+span[2]*.035,label,fontsize=9,color='#34434f')
    for axis in (world.xaxis,world.yaxis,world.zaxis):axis.set_major_locator(locator(nbins=3))
    return dict(axis_limits=np.array([center-span/2,center+span/2]).T.tolist(),
                display_box_ratio=aspect.tolist(),metric_box_ratio=span.tolist(),
                camera=dict(projection='orthographic' if display else 'perspective',
                            elevation_deg=world.elev,azimuth_deg=world.azim))


def render(folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    folder=Path(folder).resolve()
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    cfg=json.loads((folder/'config_frozen.json').read_text(encoding='utf-8'))
    fig=plt.figure(figsize=(12,8),layout='constrained')
    world=fig.add_subplot(2,2,1,projection='3d')
    position=fig.add_subplot(2,2,2);attitude=fig.add_subplot(2,2,3);speed=fig.add_subplot(2,2,4)
    displayed=[];recordings=[]
    for item in summary['results']:
        method=item['method'];path=item.get('run_dir')
        if not path:
            continue
        path=Path(path)
        rows=[json.loads(line) for line in (path/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        rows=[r for r in rows if r.get('phase')=='TRACKING']
        if not rows:
            continue
        t=np.asarray([r['t'] for r in rows]);xyz=np.asarray([r['position'] for r in rows]);xyz[:,2]*=-1
        reference=np.asarray([r['reference']['p'] for r in rows]);reference[:,2]*=-1
        displayed.extend((xyz,reference))
        errors=np.linalg.norm(xyz-reference,axis=1)
        angles=np.degrees([r['attitude_reference_error_rad'] for r in rows])
        speeds=np.linalg.norm([r['velocity'] for r in rows],axis=1)
        recordings.append((method,t,xyz,reference,errors,angles,speeds))
        style=dict(STYLES[method],markersize=3.5,markerfacecolor='white',
                   markevery=(len(recordings)-1,max(3,len(t)//12)))
        world.plot(*xyz.T,label=method,**style)
        world.plot(*reference.T,color=style['color'],linestyle=':',linewidth=1.,alpha=.55)
        position.plot(t,errors,label=method,**style)
        attitude.plot(t,angles,label=method,**style)
        speed.plot(t,speeds,label=method,**style)
    points=np.array(cfg['comparison']['task']['points_ned']);points[:,2]*=-1
    _frame_world(world,displayed,points,'Measured: legend styles / reference: dotted',MaxNLocator)
    position.axhline(cfg['limits']['position_error_m'],color='#333333',linestyle=':')
    attitude.axhline(np.degrees(cfg['limits']['attitude_error_rad']),color='#333333',linestyle=':')
    speed.axhline(cfg['comparison']['passage']['min_actual_speed_mps'],color='#777777',linestyle=':',label='via-point speed gate')
    for ax,label in ((position,'Position error (m)'),(attitude,'SO(3) reference error (deg)'),(speed,'Actual speed (m/s)')):
        ax.set(xlabel='Simulation time from release (s)',ylabel=label);ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    if world.get_legend_handles_labels()[0]:world.legend(fontsize=8)
    fig.savefig(folder/'comparison.png',dpi=160)
    fig.savefig(folder/'comparison.svg')
    plt.close(fig)
    # Independent rows retain visibility even when equivalent methods overlap exactly.
    separate=plt.figure(figsize=(16,3.3*max(1,len(recordings))),layout='constrained')
    separate.get_layout_engine().set(wspace=.15,hspace=.08)
    if not recordings:separate.text(.5,.5,'No tracking samples',ha='center',va='center')
    labels=('Position error (m)','SO(3) reference error (deg)','Actual speed (m/s)')
    for i,(method,t,xyz,reference,errors,angles,speeds) in enumerate(recordings):
        view=separate.add_subplot(len(recordings),4,i*4+1,projection='3d')
        style=dict(STYLES[method],markersize=3.,markerfacecolor='white',markevery=max(1,len(t)//12))
        view.plot(*xyz.T,label='Measured',**style)
        view.plot(*reference.T,color='#60666d',linestyle=':',linewidth=1.,label='Reference')
        _frame_world(view,displayed,points,method,MaxNLocator);view.legend(fontsize=7)
        for column,(values,label,cap) in enumerate(zip((errors,angles,speeds),labels,
                (cfg['limits']['position_error_m'],np.degrees(cfg['limits']['attitude_error_rad']),
                 cfg['comparison']['passage']['min_actual_speed_mps'])),2):
            ax=separate.add_subplot(len(recordings),4,i*4+column)
            ax.plot(t,values,label=method,**style);ax.axhline(cap,color='#777777',linestyle=':')
            ax.set(xlabel='Simulation time (s)',ylabel=label,title=method);ax.grid(alpha=.2)
            combined=(position,attitude,speed)[column-2]
            ax.set_xlim(combined.get_xlim());ax.set_ylim(combined.get_ylim())
    separate.savefig(folder/'by_method.png',dpi=160)
    separate.savefig(folder/'by_method.svg')
    plt.close(separate)
    spatial=plt.figure(figsize=(10,6),layout='constrained')
    spatial.get_layout_engine().set(rect=(0,.06,1,.88))
    spatial_view=spatial.add_subplot(111,projection='3d')
    for index,(method,t,xyz,reference,*_) in enumerate(recordings):
        style={k:v for k,v in STYLES[method].items() if k!='marker'}
        style['linewidth']=3.2 if method=='bezier' else 1.8 if method=='hermite' else 2.6
        spatial_view.plot(*xyz.T,label=method,**style)
        spatial_view.plot(*reference.T,color='#8d9ba6',linestyle=':',linewidth=.8,alpha=.6)
    spatial_view.plot([],[],[],color='#60666d',linestyle=':',label='Reference')
    framing=_frame_world(spatial_view,displayed,points,'',MaxNLocator,display=True)
    spatial_view.legend(fontsize=10,loc='upper center',bbox_to_anchor=(.5,1.07),ncol=4,frameon=False)
    caption=spatial.text(.5,.015,'Display axis ratio X:Y:H = 1.55:1.10:0.80 (independent scales)',
                        fontsize=9,color='#59636e',ha='center')
    spatial.savefig(folder/'trajectory_3d.png',dpi=160)
    spatial.savefig(folder/'trajectory_3d.svg')
    spatial_view.set_box_aspect(framing['metric_box_ratio'])
    caption.set_text('Metric view: equal axis length per metre')
    spatial.savefig(folder/'trajectory_3d_metric.png',dpi=160)
    spatial.savefig(folder/'trajectory_3d_metric.svg')
    plt.close(spatial)
    (folder/'trajectory_3d_view.json').write_text(json.dumps(dict(framing,
        default_mode='display',coordinate_units='metres; X/Y world axes, H=-NED_Z',
        samples_by_method={method:len(t) for method,t,*_ in recordings},
        scope='Axis display ratio only; unmodified recorded samples and shared limits across methods and modes'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    from .trajectory_view import campaign_view, write_view
    interactive=write_view(folder,[campaign_view(folder)])
    spatial_content=('<iframe class="trajectory-frame" src="trajectory_view.html" title="可交互三维轨迹对比"></iframe>'
                     if interactive else '<img src="trajectory_3d.png" alt="三种方案实测与参考三维轨迹">')
    def value(x):
        if x is True:return '通过'
        if x is False:return '未通过'
        if x is None:return '未运行 / 无记录'
        if isinstance(x,float):return f'{x:.4f}'
        return html.escape(str(x))
    headings=['方案','正常完成','位置/姿态','连续过点','审计','安全收尾','实际时间(s)','参考时间(s)','位置峰值(m)','姿态峰值(°)']
    keys=['method','normal_completion','tracking_passed','passage_passed','audit_passed','safe_cleanup','actual_time_s',
          'reference_time_s','position_peak_m','attitude_peak_deg']
    body=''.join('<tr>'+''.join('<td>'+value(r.get(k))+'</td>' for k in keys)+'</tr>' for r in summary['results'])
    links=''.join('<li>'+html.escape(r['method'])+': '+
        ('<a href="'+html.escape(str(Path(r['run_dir']).relative_to(folder)).replace('\\','/')+'/report.json')+'">report.json</a>'
         if r.get('run_dir') else value(r.get('error',r['status'])))+'</li>' for r in summary['results'])
    environment=summary.get('environment',cfg.get('comparison',{}).get('environment',{}))
    environment_label=html.escape(environment.get('label','E0 · 基准三维过点'))
    title=html.escape(summary.get('title','三种轨迹 · 实测对比'))
    coverage=('本页只包含所选已有记录，未选择的方案不计为失败。' if summary.get('selection') else '每方案一次开发覆盖。')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+title+'''</title>
<style>body{font:15px sans-serif;margin:28px;color:#20242a;line-height:1.6}table{border-collapse:collapse;width:100%}
td,th{padding:7px 9px;border-bottom:1px solid #d7dce1;text-align:left}img{width:100%;max-width:1200px}a{color:#007486}
.trajectory-frame{border:0;width:100%;height:720px}.table-scroll{overflow-x:auto}table{min-width:1050px}
h1{font-size:24px}h2{font-size:18px}@media(max-width:620px){body{margin:12px;font-size:14px}}</style>
<h1>'''+title+'''</h1><p>'''+environment_label+''' · 本批条件一致性：'''+value(summary['comparable_conditions'])+'''。</p>
<h2>三维轨迹对比</h2>'''+spatial_content+'''
<p>展示轴比例 X:Y:H = 1.55:1.10:0.80（非等米制）</p>
<p><a href="trajectory_3d.svg">三维图SVG</a> · <a href="trajectory_3d.png">三维图PNG</a> ·
<a href="trajectory_3d_metric.png">等米制图PNG</a> · <a href="trajectory_3d_metric.svg">等米制图SVG</a></p>
<p>'''+coverage+'''Bézier 与 Hermite 是等价换基，不作独立算法优劣结论。时间搜索只约束参考运动学，尚未自动搜索真实闭环的最快可行时间。</p>
<div class="table-scroll"><table><thead><tr>'''+''.join('<th>'+h+'</th>' for h in headings)+'''</tr></thead><tbody>'''+body+'''</tbody></table></div>
<p>实际时间含末端稳定保持，起飞、准备与降落另记。规划耗时见原始报告，含不同的首次导入开销，不直接排名。</p>
<img src="comparison.png" alt="三方案实测轨迹与误差叠加图">
<h2>各方案实测记录</h2><img src="by_method.png" alt="Bézier、Hermite、B样条分别显示的实测与参考">
<p><a href="comparison.csv">完整 CSV</a> · <a href="summary.json">汇总 JSON</a> ·
<a href="comparison.svg">叠加图SVG</a> · <a href="by_method.svg">分方案SVG</a> · <a href="design_frozen.json">冻结条件</a></p>
<ul>'''+links+'''</ul><script>window.addEventListener('message',event=>{const f=document.querySelector('.trajectory-frame');if(f&&event.source===f.contentWindow&&event.data&&event.data.type==='trajectory-height')f.style.height=Math.max(500,Math.min(1800,event.data.height))+'px';});</script></html>'''
    (folder/'index.html').write_text(page,encoding='utf-8')
    plot_source=folder/'plot_source';plot_source.mkdir(exist_ok=True)
    (plot_source/'comparison.py').write_bytes(Path(__file__).read_bytes())
    inputs=[folder/'config_frozen.json',folder/'summary.json',folder/'comparison.csv']
    inputs += [Path(r['run_dir'])/'samples.jsonl' for r in summary['results'] if r.get('run_dir')]
    outputs=['index.html','comparison.png','comparison.svg','by_method.png','by_method.svg',
             'trajectory_3d.png','trajectory_3d.svg','trajectory_3d_metric.png','trajectory_3d_metric.svg','trajectory_3d_view.json']
    if interactive:outputs+=['trajectory_view.html','trajectory_view_provenance.json']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    provenance=dict(inputs={str(p):sha(p) for p in inputs},plot_source_sha256=sha(plot_source/'comparison.py'),
                    outputs={p:sha(folder/p) for p in outputs},
                    scope='Read-only rendering of recorded samples; no new flight, smoothing, sensor or camera evidence')
    (folder/'plot_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return folder/'index.html'
