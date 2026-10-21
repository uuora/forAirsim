"""Publication-style dimensions, exact planner primitives and ground preview package."""
import argparse,base64,csv,hashlib,html,json,math,os,shutil
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from enclosed_lab_scene import ROOT,CONFIG,bounds

OUT=ROOT/'reports/enclosed_uav_lab_v1_20261008'
COLORS=dict(low_block='#d8a925',tall_block='#80539b',medium_block='#3f9699',container='#3565a0',crate_stack='#b08350',
    utility_cabinet='#528763',bench='#9c754a',barrier_0='#d28430',barrier_1='#d28430',overhead_frame='#c67834')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def export_geometry(scene,build):
    primitives=[]
    actor_names={p['id']:p['unreal_actor_name'] for p in build['parts']}
    for p in scene['parts']:
        if not p['collides']:continue
        q=dict(id=p['id'],unreal_actor_name=actor_names[p['id']],group=p['group'],role=p['role'],source_mesh=p['mesh'],collision_shape=p['collision_shape'],
            planner_shape=p['planner_shape'],center_xyz_m=p['center_xyz_m'],bounds_xyz_m=p['engineering_bounds_xyz_m'])
        if p['planner_shape']=='sphere':q['radius_m']=p['size_xyz_m'][0]/2
        else:q['size_xyz_m']=p['size_xyz_m']
        primitives.append(q)
    save(OUT/'planning_geometry.json',dict(schema_version=1,coordinate_system=scene['coordinate_system'],
        scene_config_sha256=sha(CONFIG),representation='union of individually identified AABBs and spheres',
        clearance_inflation_applied=False,physical_body_clearance_to_be_set_before_flight=True,primitives=primitives,
        excluded_decoration_ids=[p['id'] for p in scene['parts'] if not p['collides']],controller_connection='NOT_CONNECTED',flight_validation='NOT_RUN'))
    with (OUT/'geometry_correspondence.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['actor_id','unreal_actor_name','group','role','visible_mesh','collision_shape','planner_shape',
            'center_x_m','center_y_m','center_z_m','size_x_m','size_y_m','size_z_m','collides'])
        for p in scene['parts']:writer.writerow([p['id'],actor_names[p['id']],p['group'],p['role'],p['mesh'],p['collision_shape'],p['planner_shape'],*p['center_xyz_m'],*p['size_xyz_m'],p['collides']])
    return primitives

def dimensions(scene):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.8})
    fig,ax=plt.subplots(figsize=(12,8))
    ax.add_patch(Rectangle((-6,-18),60,36,facecolor='#f4f5f5',edgecolor='#555',lw=1.5))
    groups={}
    for p in scene['parts']:
        if p['group'] not in COLORS or not p['collides']:continue
        lo,hi=bounds(p)
        ax.add_patch(Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],facecolor=COLORS[p['group']],edgecolor='#333',lw=.6))
        groups.setdefault(p['group'],[]).append(p)
    labels={'low_block':('Low block\n1.2 m',(-1.5,-3)), 'tall_block':('Tall block\n5.5 m',(-1,-4)),
        'medium_block':('Medium block\n3.5 m',(-.5,3)), 'container':('Container\n6 × 3 × 3 m',(0,-3)),
        'crate_stack':('Crate stack\n2.4 m',(-.8,3)), 'utility_cabinet':('Cabinet\n3.8 m',(2,3)),
        'bench':('Bench\n1.36 m',(-1,3)), 'barrier_0':('Barrier\n1.30 m',(-6,-1)),
        'barrier_1':('Barrier\n1.30 m',(1,-3)), 'overhead_frame':('Open frame\n5.025 m',(1,4))}
    for group,parts in groups.items():
        lo=[min(bounds(p)[0][i] for p in parts) for i in range(3)];hi=[max(bounds(p)[1][i] for p in parts) for i in range(3)]
        x,y=(lo[0]+hi[0])/2,(lo[1]+hi[1])/2;label,offset=labels[group]
        ax.annotate(label,(x,y),(x+offset[0],y+offset[1]),ha='center',va='center',fontsize=9,
            arrowprops=dict(arrowstyle='-',lw=.5,color='#888'),bbox=dict(fc='white',ec='none',pad=1,alpha=.85))
    for x,name in [(0,'Start platform'),(48,'Target platform')]:
        ax.add_patch(Rectangle((x-3,-4),6,8,fc='#dde5eb',ec='#3a5365',lw=1.3))
        ax.text(x,6,name+'\n6 × 8 × 0.6 m',ha='center',va='center',fontsize=10)
    for y,name in [(-1.8,'UAV A'),(1.8,'UAV B')]:
        ax.plot([-.3,.3],[y-.3,y+.3],c='#194d7a',lw=2);ax.plot([-.3,.3],[y+.3,y-.3],c='#194d7a',lw=2)
        ax.text(-3.8,y,name,ha='right',va='center',fontsize=9,color='#194d7a')
    for p in scene['parts']:
        if p['role']!='target':continue
        x,y,z=p['center_xyz_m'];r=p['size_xyz_m'][0]/2
        ax.add_patch(Circle((x,y),r,fc='#c6473d' if p['material']!='balloon_yellow' else '#e8b528',ec='#555',lw=.7))
        if p['material']=='balloon_white_stripe':ax.plot([x-r*.85,x+r*.85],[y,y],c='white',lw=2)
    ax.annotate('',(48,-13),(0,-13),arrowprops=dict(arrowstyle='<->',color='#333',lw=.8))
    ax.text(24,-12.3,'48 m between platform centres',ha='center',fontsize=11)
    for x in [0,48]:ax.plot([x,x],[-13.8,-11.8],c='#777',lw=.6)
    ax.annotate('',(54,-21),(-6,-21),arrowprops=dict(arrowstyle='<->',color='#333',lw=.8))
    ax.text(24,-20.2,'60 m',ha='center')
    ax.annotate('',(58,18),(58,-18),arrowprops=dict(arrowstyle='<->',color='#333',lw=.8))
    ax.text(59,0,'36 m',rotation=90,va='center')
    ax.text(24,20,'Enclosed UAV laboratory — configured geometry',ha='center',fontsize=15)
    ax.text(24,-25,'Walls: 3.2 m high; top open. Ground preview only; no measured flight trajectory.',ha='center',fontsize=10)
    ax.set_aspect('equal');ax.set_xlim(-13,62);ax.set_ylim(-27,23)
    ax.set_xlabel('X / m (forward)');ax.set_ylabel('Y / m (right)');ax.spines[['top','right']].set_visible(False)
    ax.set_xticks(range(0,49,12));ax.set_yticks(range(-18,19,6));fig.tight_layout()
    for ext in ['svg','pdf','png']:fig.savefig(OUT/('dimensions.'+ext),dpi=600 if ext=='png' else None,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def independent_checks(scene,build,preview,primitives):
    rows=[]
    def check(name,ok,evidence):
        rows.append(dict(requirement=name,status='PASS' if ok else 'FAIL',evidence=evidence))
    check('asset inventory',json.loads((ROOT/'logs/enclosed_lab_asset_inventory_v2_20261008.json').read_text())['status']=='PASS','UE Asset Registry and loaded primitive bounds')
    check('matching configuration hash',build['config_sha256']==sha(CONFIG),'editor construction bound to configuration bytes')
    expected=set(p['id'] for p in scene['parts'])
    check('visible/collision/planner actor population',expected==set(p['id'] for p in build['parts'])==set(p['id'] for p in preview['geometry_readbacks']),str(len(scene['parts']))+' individual parts')
    check('native actor name mapping',{p['id']:p['unreal_actor_name'] for p in build['parts']}=={p['id']:p['unreal_actor_name'] for p in preview['geometry_readbacks']},'recorded editor/runtime alias mapping matches exactly')
    check('exact planner population',set(p['id'] for p in scene['parts'] if p['collides'])==set(p['id'] for p in primitives),'every colliding part has one primitive; decoration excluded')
    check('editor visible/collision geometry',build['status']=='PASS_EDITOR_GEOMETRY_COLLISION','isolated simple collision crossing and outside ray per part; visible bounds readback')
    check('runtime actor geometry',preview['status']=='PASS_GROUND_PREVIEW_GEOMETRY','all actor position/scale/orientation readbacks after map load')
    check('grounded two drones',all(v['ready'] for v in preview['ground_after_capture'].values()),preview['ground_after_capture'])
    check('requested actual Unreal views',set(['overhead','drone_front','target_close']).issubset(preview['views']),'unmodified Scene PNG with camera information and SHA256')
    check('no flight or target motion',preview['new_flights']==0 and preview['flight_motion_requests'] is False and preview['target_motion_requests'] is False,'preview script has no arm/control-acquisition/vehicle-movement calls')
    check('48m platform spacing',math.dist([0,0],[48,0])==scene['platform_center_distance_m'],'configured start and target platform centres; runtime positions checked')
    # This prevents inadvertently placing the new everyday objects onto a bay.
    obstacle=[p for p in scene['parts'] if p['role']=='obstacle' and p['collides']]
    separated=True
    for x in [0,48]:
        for p in obstacle:
            lo,hi=bounds(p)
            if lo[0]<x+3 and hi[0]>x-3 and lo[1]<4 and hi[1]>-4:separated=False
    check('platforms separated from task obstacles',separated,'no obstacle XY footprint overlaps either platform')
    status='PASS_FIRST_ROUND_GROUND_PREVIEW' if all(r['status']=='PASS' for r in rows) else 'FAIL'
    save(OUT/'acceptance_checks.json',dict(status=status,checks=rows,flight_validation='NOT_RUN',
        limitations=['Collision rays sample six axial faces and three outside lines per component; do not establish all-contact physical safety.',
            'Sphere physics is an analytic sphere; visible sphere is tessellated. Boxes match their six planar faces.',
            'No overhead roof; perimeter walls enclose the sides.',
            'Static planner primitives are exported; closed-loop integration and clearance tuning are not tested.']))
    return status

def package():
    if (OUT/'provenance.json').exists():raise FileExistsError('SEALED_DELIVERY_REQUIRES_NEW_OUTPUT')
    OUT.mkdir(parents=True,exist_ok=True)
    scene=json.loads(CONFIG.read_text(encoding='utf-8'))
    latest=json.loads((ROOT/'logs/enclosed_lab_preview_latest.json').read_text(encoding='utf-8'))
    preview=json.loads(Path(latest['report']).read_text(encoding='utf-8'))
    session=json.loads(Path(latest['session']).read_text(encoding='utf-8'))
    build=json.loads(Path(session['build_report']).read_text(encoding='utf-8'))
    primitives=export_geometry(scene,build);dimensions(scene);status=independent_checks(scene,build,preview,primitives)
    images={}
    for name,view in preview['views'].items():
        target=OUT/(name+'.png');shutil.copyfile(view['path'],target)
        if sha(target)!=view['sha256']:raise ValueError('IMAGE_COPY_CHANGED')
        images[name]='data:image/png;base64,'+base64.b64encode(target.read_bytes()).decode()
    images['dimensions']='data:image/svg+xml;base64,'+base64.b64encode((OUT/'dimensions.svg').read_bytes()).decode()
    captions=dict(overhead='实际 Unreal 俯视图',overview='实际 Unreal 场地总览',drone_front='实际 DroneA 起点前视图',
        target_close='实际 Unreal 目标近景',start_close='实际 Unreal 无人机平台近景',dimensions='尺寸示意 · 配置几何，不是实测航迹')
    tabs=''.join('<button data-view="'+k+'">'+v+'</button>' for k,v in captions.items())
    panes=''.join('<figure id="'+k+'" '+('hidden' if k!='overview' else '')+'><img src="'+images[k]+'" alt="'+v+'"><figcaption>'+v+'</figcaption></figure>' for k,v in captions.items())
    rows=''.join('<tr><td>'+html.escape(p['id'])+'</td><td>'+html.escape(p['group'])+'</td><td>'+html.escape(str(p['size_xyz_m']))+'</td><td>'+p['collision_shape']+'</td><td>'+p['planner_shape']+'</td></tr>' for p in scene['parts'])
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>无人机围合实验场 · 第一轮地面预览</title>
<style>body{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:#f2f4f5;color:#172a3a;margin:0}main{max-width:1200px;margin:auto;padding:30px}h1{font-size:27px;margin:0}p{max-width:1000px}nav{display:flex;gap:7px;flex-wrap:wrap;margin:20px 0}button{padding:9px 15px;background:white;border:1px solid #b8c4ce;border-radius:4px;cursor:pointer}button.active{background:#224f71;color:white}figure{margin:0;background:white;padding:12px;border:1px solid #d5dce1}figure img{display:block;width:100%;max-height:800px;object-fit:contain}figcaption{padding:8px;color:#425668}.metrics{display:flex;gap:35px;flex-wrap:wrap;padding:15px 0}.metrics b{font-size:23px}summary{cursor:pointer;margin:20px 0}table{width:100%;font-size:12px;border-collapse:collapse;background:white}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}input{padding:8px;width:350px;max-width:90%}.status{color:#1b6947}.note{background:white;padding:16px;border-left:3px solid #345d7c}</style>
<main><h1>无人机围合实验场</h1><p>第一轮交付：地面场景预览与几何检查。所有仿真照片来自实际 Unreal Scene 相机，尺寸图由同一场景配置生成。</p>
<div class="metrics"><span><b>60 × 36 m</b><br>场地尺寸</span><span><b>48 m</b><br>平台中心间距</span><span><b>2 架</b><br>地面无人机</span><span><b>3 只</b><br>静止气球</span><span><b>0 次</b><br>本轮飞行</span></div>
<nav>'''+tabs+'''</nav>'''+panes+'''
<p class="status">已满足：四周围墙、两个平台、远距离布局、高矮不同的彩色障碍、日常物体组合、实际多视角画面、部件与碰撞/规划几何对应。</p>
<p class="note">尚未满足：没有接入新地图的闭环控制、没有进行绕障飞行或双机接替、没有验证远距离识别效果。场地顶部开放。日常物体是基础网格组合的简化模型；涂装、文字、标线与细小饰面明确不参与碰撞。碰撞线段核查不等于全接触或飞行安全验证。</p>
<details><summary>查看逐部件的可见模型 / 碰撞体 / 规划几何对应</summary><input id="filter" placeholder="筛选部件名称或类别"><table><thead><tr><th>部件 ID</th><th>类别</th><th>尺寸 XYZ / m</th><th>碰撞形状</th><th>规划形状</th></tr></thead><tbody>'''+rows+'''</tbody></table></details>
<p>源配置：configs/enclosed_uav_lab_v1.json。规划数据：planning_geometry.json。证据与限制：acceptance_checks.json、provenance.json。</p></main>
<script>const buttons=[...document.querySelectorAll('nav button')];function show(id){document.querySelectorAll('figure').forEach(x=>x.hidden=x.id!==id);buttons.forEach(x=>x.classList.toggle('active',x.dataset.view===id));}buttons.forEach(x=>x.onclick=()=>show(x.dataset.view));show('overview');document.querySelector('#filter').oninput=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));</script></html>'''
    (OUT/'preview.html').write_text(document,encoding='utf-8')
    sources=[CONFIG,ROOT/'configs/enclosed_uav_lab_preview_settings_v1.json',ROOT/'python/enclosed_lab_scene.py',ROOT/'python/enclosed_lab_preview.py',
        ROOT/'python/export_enclosed_lab_preview.py',ROOT/'scripts/unreal/build_enclosed_uav_lab.py',ROOT/'docs/围合无人机实验场_第一轮地面预览.md']
    for source in sources:
        target=OUT/'source_snapshot'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for source,name in [(Path(session['build_report']),'editor_build_report.json'),(Path(latest['report']),'runtime_preview_report.json'),
                        (ROOT/'logs/enclosed_lab_asset_inventory_v2_20261008.json','asset_inventory.json')]:
        shutil.copyfile(source,OUT/name)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='provenance.json'}
    asset_folder=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabV1'
    assets={str(p.relative_to(ROOT)):sha(p) for p in asset_folder.rglob('*') if p.suffix in ('.uasset','.umap')}
    save(OUT/'provenance.json',dict(status=status,scene_config=str(CONFIG),scene_config_sha256=sha(CONFIG),
        editor_build_report=session['build_report'],editor_build_report_sha256=sha(session['build_report']),
        actual_preview_report=latest['report'],actual_preview_report_sha256=sha(latest['report']),source_session=latest['session'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},unreal_asset_sha256=assets,
        exported_files_sha256=files,AI_assistance='AI assisted procedural scene authoring, software implementation and image review; human acceptance pending.',
        claim_scope='ground preview and bounded geometry checks; no flight, algorithm efficacy or detection result'))
    print(json.dumps(dict(status=status,output=str(OUT)),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--open',action='store_true');args=parser.parse_args()
    if args.output:OUT=args.output.resolve()
    elif (OUT/'provenance.json').exists():OUT=ROOT/'reports'/('enclosed_uav_lab_preview_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    package()
    save(ROOT/'reports/enclosed_lab_preview_latest.json',dict(output=str(OUT),preview=str(OUT/'preview.html')))
    if args.open:os.startfile(OUT/'preview.html')
