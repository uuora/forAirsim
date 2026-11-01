"""Scientific plots from recorded states: stable method colours, vector export.

No interpolation, smoothing or synthetic trajectories are used for display.
"""
import json,math,threading
from pathlib import Path
import numpy as np

PLOT_LOCK=threading.RLock()
COLORS={'bezier_ordinary':'#64748b','hermite_ordinary':'#0f766e','bezier_abom':'#2563eb','hermite_abom':'#7c3aed','bezier_abom_improved_candidate':'#d97706','hermite_abom_improved_candidate':'#be123c'}
SHORT={'bezier_ordinary':'Bézier 基线','hermite_ordinary':'Hermite 基线','bezier_abom':'Bézier / ABOM','hermite_abom':'Hermite / ABOM','bezier_abom_improved_candidate':'Bézier / 扰动候选','hermite_abom_improved_candidate':'Hermite / 扰动候选'}


def render_plots(records,output):
    with PLOT_LOCK:return _render(records,Path(output))


def _render(records,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output.mkdir(parents=True,exist_ok=True)
    figures=[];series=[];arrays={}
    for index,(folder,summary,trace,manifest) in enumerate(records):
        states=trace['states'];commands=trace['commands'];method=summary['method']
        if not states:continue
        t=np.array([s['reference_elapsed_s'] for s in states]);p=np.array([s['position_zup_m'] for s in states]);q=np.array([s['reference_position_zup_m'] for s in states]);v=np.array([s['velocity_zup_mps'] for s in states])
        qv=np.array([s.get('reference_velocity_zup_mps',[np.nan]*3) for s in states]);cmd=np.array([c['limited'] for c in commands]).reshape(-1,3)
        yaw=np.degrees(np.unwrap([s['yaw_rad'] for s in states]));cyaw=np.degrees(np.unwrap(np.radians([c['yaw_command_deg'] for c in commands])))
        item=dict(t=t,p=p,q=q,v=v,qv=qv,cmd=cmd,yaw=yaw,cyaw=cyaw,color=COLORS.get(method,'#334155'),method=method,label=SHORT.get(method,method),id=index,summary=summary,manifest=manifest)
        series.append(item)
        for key in ('t','p','q','v','qv','cmd','yaw','cyaw'):arrays[f'run_{index}_{key}']=item[key]
    np.savez_compressed(output/'plot_data.npz',**arrays)
    config={'font.family':'Microsoft YaHei','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.edgecolor':'#c4d0d8','axes.labelcolor':'#334155','text.color':'#18364a','xtick.color':'#526777','ytick.color':'#526777','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#d9e2e8','grid.alpha':.55,'grid.linewidth':.6,'axes.axisbelow':True,'figure.facecolor':'white','axes.facecolor':'white','axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42}
    def save(fig,name,title):
        for suffix in ('png','svg','pdf'):fig.savefig(output/(name+'.'+suffix),dpi=190,bbox_inches='tight',facecolor='white')
        plt.close(fig);figures.append((name,title))
    def legend(ax,**kwargs):
        handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles))
        if unique:ax.legend(unique.values(),unique.keys(),frameon=False,fontsize=8,**kwargs)
    def unique_reference(item,seen):
        # A numeric equality check avoids duplicating only genuinely equal
        # saved reference samples. Different time sampling is not fabricated.
        for old in seen:
            if item['q'].shape==old['q'].shape and np.allclose(item['q'],old['q'],atol=1e-9,rtol=0) and np.allclose(item['t'],old['t'],atol=1e-9,rtol=0):return False
        seen.append(item);return True
    with plt.rc_context(config):
        fig=plt.figure(figsize=(11.8,6.8));ax=fig.add_subplot(111,projection='3d');seen=[];method_seen=set()
        for item in series:
            label=item['label'] if item['method'] not in method_seen else '_nolegend_';method_seen.add(item['method'])
            ax.plot(*item['p'].T,color=item['color'],lw=1.7,alpha=.85,label=label)
            if unique_reference(item,seen):ax.plot(*item['q'].T,color='#1e293b',lw=1.1,ls='--',alpha=.6,label='_nolegend_')
        route=records[0][3]['plan'].get('route',{});points=route.get('waypoints') or []
        for i,point in enumerate(points):ax.scatter(*point,c='#e05252',s=32,depthshade=False);ax.text(*point,'  '+str(i+1),fontsize=9)
        if series:
            start=series[0]['q'][0];ax.scatter(*start,c='#0f766e',marker='s',s=40);ax.text(*start,' 起点',fontsize=9)
            all_points=np.vstack([s[k] for s in series for k in ('p','q')])
            low=all_points.min(axis=0);high=all_points.max(axis=0);pad=max(float(np.ptp(all_points,axis=0).max()),1)*.05
            low-=pad;high+=pad
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_zlim(low[2],high[2])
            ax.set_box_aspect(high-low);ax.set_proj_type('ortho')
        ax.set(xlabel='X /米',ylabel='Y /米',zlabel='高度 /米',title='参考与实际轨迹  |  虚线：参考；彩色实线：实际')
        ax.view_init(elev=25,azim=-58);ax.legend(frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(1.01,.95))
        fig.subplots_adjust(left=.01,right=.79,bottom=.03,top=.95);save(fig,'trajectory','三维参考与实际轨迹')
        fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
        for ax,dims,labels in zip(axes,[(0,1),(0,2)],[('X /米','Y /米'),('X /米','高度 /米')]):
            for item in series:
                ax.plot(item['q'][:,dims[0]],item['q'][:,dims[1]],ls='--',lw=.9,color='#64748b',alpha=.45)
                ax.plot(item['p'][:,dims[0]],item['p'][:,dims[1]],lw=1.6,color=item['color'],alpha=.8,label=item['label'])
            for i,point in enumerate(points):ax.scatter(point[dims[0]],point[dims[1]],c='#e05252',s=25,zorder=5);ax.annotate(str(i+1),(point[dims[0]],point[dims[1]]),xytext=(5,5),textcoords='offset points')
            ax.set(xlabel=labels[0],ylabel=labels[1]);ax.set_aspect('equal',adjustable='datalim');ax.margins(.13)
        axes[0].set_title('水平投影 · XY');axes[1].set_title('高度投影 · XZ');legend(axes[0],loc='best');save(fig,'projection','水平与高度投影（等物理比例）')
        fig,ax=plt.subplots(figsize=(12,4.7),layout='constrained');maximum=.5
        for item in series:
            error=np.linalg.norm(item['p']-item['q'],axis=1);maximum=max(maximum,float(error.max())*1.15)
            ax.plot(item['t'],error,color=item['color'],lw=1.6,alpha=.82,label=item['label'])
        ax.axhline(.5,color='#b84a4a',ls='--',lw=1.15,label='采样峰值门槛 0.50米');ax.axhspan(.5,maximum*1.05,color='#f9e6e3',alpha=.7,zorder=-1)
        ax.set(xlabel='参考经过时间 /秒（未归一化）',ylabel='同步位置误差 /米',title='同步跟踪误差 · 使用原始采样，不平滑',ylim=(0,maximum*1.05));legend(ax,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3);save(fig,'error','同步跟踪误差与门槛')
        # Small multiples keep reference, actual and command interpretable;
        # at most six runs per image, but every selected run is exported.
        for page in range(math.ceil(len(series)/6)):
            subset=series[page*6:(page+1)*6];fig,axes=plt.subplots(len(subset),2,figsize=(12,2.7*len(subset)+.6),squeeze=False,layout='constrained')
            for row,item in enumerate(subset):
                left,right=axes[row];t=item['t'];n=len(item['cmd']);color=item['color']
                left.plot(t,np.linalg.norm(item['v'],axis=1),color=color,lw=1.7,label='实际速度')
                left.plot(t,np.linalg.norm(item['qv'],axis=1),color='#334155',ls='--',lw=1,label='参考速度')
                left.plot(t[:n],np.linalg.norm(item['cmd'],axis=1),color='#b07832',ls=':',lw=1.2,label='速度指令')
                right.plot(t,item['yaw'],color=color,lw=1.6,label='实际朝向');right.plot(t[:len(item['cyaw'])],item['cyaw'],ls='--',color='#64748b',lw=1,label='偏航指令')
                left.set(title=f"{item['label']} · 记录{item['id']+1}",ylabel='速度 /米每秒',xlabel='参考时间 /秒');right.set(title='机头朝向（连续展开）',ylabel='角度 /度',xlabel='参考时间 /秒')
                low,high=right.get_ylim()
                if high-low<10:mid=(high+low)/2;right.set_ylim(mid-5,mid+5)
                legend(left,loc='upper right',ncol=3);legend(right,loc='upper right',ncol=2)
            name='speed_heading' if page==0 else 'speed_heading_'+str(page+1)
            save(fig,name,'逐条记录：参考速度、实际速度、指令与朝向'+(f'（第{page+1}页）' if len(series)>6 else ''))
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
        methods=list(dict.fromkeys(s[1]['method'] for s in records))
        fresh=records[0][3]['protocol_revision']==15
        specs=[('reference_duration_s','参考执行时长 /秒',False,None),('position_rmse_m','同步位置RMSE /米',False,None),('max_position_error_m','采样峰值误差 /米',False,.5),('planning_wall_s','本次完整模型规划墙钟 /秒' if fresh else '历史模型规划耗时 /秒',True,None)]
        for ax,(key,title,planning,gate) in zip(axes.flat,specs):
            maximum=gate or 0.
            for y,method in enumerate(methods):
                selected=[s for s in records if s[1]['method']==method];values=[]
                for j,(_,summary,_,_) in enumerate(selected):
                    evidence=summary.get('planning_evidence') or {}
                    value=(evidence.get('parent_total_planning_wall_s') if fresh else (evidence.get('timings') or {}).get('search_plus_selected_validation_wall_s')) if planning else (summary.get('metrics') or {}).get(key)
                    if value is None:continue
                    maximum=max(maximum,float(value))
                    values.append(value);offset=0 if len(selected)==1 else (j/(len(selected)-1)-.5)*.28
                    passed=(summary.get('metrics') or {}).get('control_error_gates_passed') is True and summary['outcome']=='COMPLETED'
                    ax.scatter(value,y+offset,color=COLORS.get(method,'#334155'),marker='o' if passed else 'x',s=36,zorder=3)
                    if len(selected)==1:ax.annotate(f'{value:.3f}',(value,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
                if len(values)>=2:ax.plot([np.median(values)]*2,[y-.2,y+.2],color=COLORS.get(method,'#334155'),lw=2)
                if not values:ax.text(.02,y,'无有效值',transform=ax.get_yaxis_transform(),va='center',color='#8b98a4')
            ax.set(yticks=range(len(methods)),yticklabels=[SHORT.get(m,m)+f"  n={sum(s[1]['method']==m for s in records)}" for m in methods],xlabel=title,title=title)
            ax.invert_yaxis();ax.set_xlim(0,max(maximum,1e-3)*1.27);ax.margins(y=.15);ax.grid(axis='y',visible=False)
            if gate:ax.axvline(gate,color='#b84a4a',ls='--',lw=1)
        fig.suptitle('同条件所选记录 · 圆点为成功，叉号为未通过；多次记录显示原始点与中位数',fontsize=12)
        save(fig,'comparison','执行质量与规划开销（不合并不同任务）')
    style=dict(version=2,colors=COLORS,scope='Recorded samples only; no smoothing; physical coordinates; elapsed seconds; no statistical superiority claim',records=[dict(index=i,path=str(r[0]),method=r[1]['method']) for i,r in enumerate(records)])
    (output/'plot_style.json').write_text(json.dumps(style,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'figure_index.json').write_text(json.dumps([dict(name=n,title=t,png=n+'.png',svg=n+'.svg',pdf=n+'.pdf') for n,t in figures],ensure_ascii=False,indent=2),encoding='utf-8')
    return figures
