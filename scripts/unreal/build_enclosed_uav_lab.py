"""UE4.27 editor-only construction and geometry/collision checks; no flight."""
import json,hashlib,math,traceback,os
from datetime import datetime
from pathlib import Path
import unreal as ue

ROOT=Path('D:/forAirsim')
CONFIG=Path(os.environ.get('ENCLOSED_LAB_CONFIG',str(ROOT/'configs/enclosed_uav_lab_v1.json')))
BASE=os.environ.get('ENCLOSED_LAB_BASE','/Game/EnclosedUAVLabV1')
OUT=ROOT/'logs'/('enclosed_lab_build_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
LATEST=ROOT/'logs'/('enclosed_lab_manual_build_latest.json' if BASE=='/Game/EnclosedUAVLabManual' else 'enclosed_lab_build_latest.json')
report=dict(status='RUNNING',flight_commands=False,map_scope=BASE,config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest())
tools=ue.AssetToolsHelpers.get_asset_tools()
mel=ue.MaterialEditingLibrary

def vec(v):return [float(v.x),float(v.y),float(v.z)]
def expression(mat,kind,**properties):
    n=mel.create_material_expression(mat,kind)
    for k,v in properties.items():n.set_editor_property(k,v)
    return n

def material(name,rgb,rough=.7):
    path=BASE+'/Materials'
    m=ue.load_asset(path+'/M_'+name) or tools.create_asset('M_'+name,path,ue.Material,ue.MaterialFactoryNew())
    mel.delete_all_material_expressions(m)
    col=expression(m,ue.MaterialExpressionConstant3Vector,constant=ue.LinearColor(*rgb,1))
    if name=='balloon_white_stripe':
        inputs=[]
        for key in ('P','O'):
            item=ue.CustomInput();item.set_editor_property('input_name',key);inputs.append(item)
        n=expression(m,ue.MaterialExpressionCustom,code='return abs(P.z-O.z)<18 ? float3(.92,.92,.90) : float3(.68,.015,.012);',
                     output_type=ue.CustomMaterialOutputType.CMOT_FLOAT3,
                     inputs=inputs)
        p=expression(m,ue.MaterialExpressionWorldPosition)
        o=expression(m,ue.MaterialExpressionObjectPositionWS)
        mel.connect_material_expressions(p,'',n,'P');mel.connect_material_expressions(o,'',n,'O');col=n
    mel.connect_material_property(col,'',ue.MaterialProperty.MP_BASE_COLOR)
    r=expression(m,ue.MaterialExpressionConstant,r=rough)
    mel.connect_material_property(r,'',ue.MaterialProperty.MP_ROUGHNESS)
    mel.recompile_material(m);ue.EditorAssetLibrary.save_loaded_asset(m)
    return m

def collision_mesh(name,source,shape):
    path=BASE+'/Meshes/SM_'+name
    mesh=ue.load_asset(path) or ue.EditorAssetLibrary.duplicate_asset(source,path)
    if not mesh:raise RuntimeError('MESH_DUPLICATION_FAILED')
    ue.EditorStaticMeshLibrary.remove_collisions(mesh)
    if ue.EditorStaticMeshLibrary.add_simple_collisions(mesh,shape)<0:raise RuntimeError('COLLISION_GENERATION_FAILED')
    if ue.EditorStaticMeshLibrary.get_simple_collision_count(mesh)!=1 or ue.EditorStaticMeshLibrary.get_convex_collision_count(mesh)!=0:
        raise RuntimeError('EXPECTED_ONE_EXPLICIT_PRIMITIVE')
    ue.EditorAssetLibrary.save_loaded_asset(mesh)
    return mesh

def spawn(part,meshes,mats):
    actor=ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor,ue.Vector(*[100*x for x in part['center_xyz_m']]))
    actor.set_actor_label(part['id']);actor.set_editor_property('tags',[ue.Name(part['id']),ue.Name(part['group'])])
    if BASE=='/Game/EnclosedUAVLabManual':actor.set_folder_path(ue.Name('Lab/'+part['group']))
    c=actor.static_mesh_component;c.set_static_mesh(meshes[part['mesh']]);c.set_material(0,mats[part['material']])
    c.set_mobility(ue.ComponentMobility.STATIC)
    c.set_collision_profile_name('BlockAll' if part['collides'] else 'NoCollision')
    c.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS if part['collides'] else ue.CollisionEnabled.NO_COLLISION)
    actor.set_actor_scale3d(ue.Vector(*part['size_xyz_m']))
    c.set_editor_property('cast_shadow',part['collides'])
    return actor

def text(name,value,pos,size,rotation=(0,180,0),color=(225,230,235)):
    a=ue.EditorLevelLibrary.spawn_actor_from_class(ue.TextRenderActor,ue.Vector(*[100*x for x in pos]),
        ue.Rotator(pitch=rotation[0],yaw=rotation[1],roll=rotation[2]))
    a.set_actor_label('EL1_Label_'+name)
    if BASE=='/Game/EnclosedUAVLabManual':a.set_folder_path(ue.Name('Lab/labels'))
    c=a.get_component_by_class(ue.TextRenderComponent)
    c.set_text(value);c.set_world_size(size*100);c.set_text_render_color(ue.Color(*color,255))
    c.set_horizontal_alignment(ue.HorizTextAligment.EHTA_CENTER)
    c.set_collision_enabled(ue.CollisionEnabled.NO_COLLISION)
    return a

def ray(world,start,end,ignored):
    result=ue.SystemLibrary.line_trace_single(world,ue.Vector(*start),ue.Vector(*end),ue.TraceTypeQuery.TRACE_TYPE_QUERY1,
        False,ignored,ue.DrawDebugTrace.NONE,True)
    if result is None:return dict(blocking=False,impact_cm=None,actor_label=None)
    # UE4 Python returns the OutHit struct; BreakHitResult exposes its bool and actor.
    hit=result if isinstance(result,ue.HitResult) else next(x for x in result if isinstance(x,ue.HitResult))
    # UE4.27 HitResult.to_tuple invokes its native BreakHitResult function.
    # Field order is pinned to Engine/Classes/Kismet/GameplayStatics.h.
    fields=hit.to_tuple()
    if len(fields)!=17:raise RuntimeError('UNEXPECTED_NATIVE_HIT_RESULT_LAYOUT')
    return dict(blocking=bool(fields[0]),impact_cm=vec(fields[5]),actor_label=fields[9].get_actor_label() if fields[9] else None)

try:
    scene=json.loads(CONFIG.read_text(encoding='utf-8'))
    if BASE not in ('/Game/EnclosedUAVLabV1','/Game/EnclosedUAVLabManual') or not scene['unreal_map'].startswith(BASE+'/Maps/'):
        raise RuntimeError('MAP_SCOPE_MISMATCH')
    if scene['flight_authorized'] is not False or scene['delivery_stage']!='GROUND_PREVIEW_ONLY':raise RuntimeError('PREVIEW_SCOPE_REQUIRED')
    ids=set()
    for p in scene['parts']:
        if p['id'] in ids or not p['id'].startswith('EL1_'):raise RuntimeError('DUPLICATE_OR_UNSCOPED_PART_ID')
        ids.add(p['id'])
        if p['mesh'] not in ('Box','Sphere') or len(p['center_xyz_m'])!=3 or len(p['size_xyz_m'])!=3:
            raise RuntimeError('UNSUPPORTED_PART_GEOMETRY')
        if not all(math.isfinite(v) for v in p['center_xyz_m']+p['size_xyz_m']) or min(p['size_xyz_m'])<=0:
            raise RuntimeError('INVALID_PART_DIMENSIONS_'+p['id'])
        if p['mesh']=='Sphere' and max(p['size_xyz_m'])-min(p['size_xyz_m'])>.00001:
            raise RuntimeError('SPHERE_REQUIRES_UNIFORM_SCALE_'+p['id'])
        # Derive geometry from edited position/size; never trust cached bounds from old JSON.
        p['engineering_bounds_xyz_m']=[[p['center_xyz_m'][i]+s*p['size_xyz_m'][i]/2 for i in range(3)] for s in [-1,1]]
        p['collision_shape']=('box' if p['mesh']=='Box' else 'sphere') if p['collides'] else 'none'
        p['planner_shape']=('aabb' if p['mesh']=='Box' else 'sphere') if p['collides'] else 'none'
    palette=dict(floor=(.23,.25,.27),wall=(.36,.40,.44),dark=(.045,.06,.075),pad=(.13,.16,.19),yellow=(.95,.58,.045),
        orange=(.87,.23,.055),purple=(.30,.13,.42),teal=(.035,.33,.34),blue=(.045,.17,.38),blue_light=(.09,.28,.53),
        wood=(.37,.21,.105),green=(.095,.29,.17),green_light=(.17,.41,.25),white=(.9,.91,.92),line=(.65,.69,.72),
        red=(.68,.015,.012),balloon_yellow=(.95,.60,.025),balloon_white_stripe=(.68,.015,.012))
    mats={k:material(k,v,.38 if k.startswith('balloon') or k=='red' else .72) for k,v in palette.items()}
    meshes=dict(Box=collision_mesh('Box','/Engine/BasicShapes/Cube',ue.ScriptingCollisionShapeType.BOX),
                Sphere=collision_mesh('Sphere','/Engine/BasicShapes/Sphere',ue.ScriptingCollisionShapeType.SPHERE))
    report['mesh_collision_readbacks']={k:dict(path=m.get_path_name(),simple_count=ue.EditorStaticMeshLibrary.get_simple_collision_count(m),
        convex_count=ue.EditorStaticMeshLibrary.get_convex_collision_count(m),complexity=str(ue.EditorStaticMeshLibrary.get_collision_complexity(m))) for k,m in meshes.items()}
    if ue.EditorAssetLibrary.does_asset_exist(scene['unreal_map']):
        if not ue.EditorLevelLibrary.load_level(scene['unreal_map']):raise RuntimeError('LOAD_FAILED')
        for a in ue.EditorLevelLibrary.get_all_level_actors():
            if a.get_actor_label().startswith('EL1_'):ue.EditorLevelLibrary.destroy_actor(a)
    elif not ue.EditorLevelLibrary.new_level(scene['unreal_map']):raise RuntimeError('NEW_MAP_FAILED')
    world=ue.EditorLevelLibrary.get_editor_world()
    world.get_world_settings().set_editor_property('default_game_mode',ue.load_class(None,'/Script/AirSim.AirSimGameMode'))
    actors=[spawn(p,meshes,mats) for p in scene['parts']]
    start=ue.EditorLevelLibrary.spawn_actor_from_class(ue.PlayerStart,ue.Vector(0,0,100))
    start.set_actor_label('EL1_PlayerStart')
    sun=ue.EditorLevelLibrary.spawn_actor_from_class(ue.DirectionalLight,ue.Vector(0,0,1800),ue.Rotator(pitch=-48,yaw=-38,roll=0))
    sun.set_actor_label('EL1_Sun');sun.light_component.set_mobility(ue.ComponentMobility.MOVABLE)
    sun.light_component.set_editor_property('intensity',4.0);sun.light_component.set_atmosphere_sun_light(True)
    sky=ue.EditorLevelLibrary.spawn_actor_from_class(ue.SkyAtmosphere,ue.Vector(0,0,0));sky.set_actor_label('EL1_Atmosphere')
    light=ue.EditorLevelLibrary.spawn_actor_from_class(ue.SkyLight,ue.Vector(0,0,1000));light.set_actor_label('EL1_SkyLight')
    light.light_component.set_mobility(ue.ComponentMobility.MOVABLE);light.light_component.set_editor_property('intensity',1.0)
    light.light_component.set_editor_property('real_time_capture',True)
    text('Start','START / UAV A + B',[0,-5.4,.018],.75,(90,0,0))
    text('Target','TARGET / BALLOONS',[48,-5.4,.018],.65,(90,0,0))
    for x in [0,12,24,36,48]:text('Distance_'+str(x),str(x)+' m',[x,-12,.018],.85,(90,0,0))
    for name,label,pos,width in [('low','1.2 m',[7.98,-4,.8],.34),('tall','5.5 m',[15.48,-1.5,3.4],.36),
        ('medium','3.5 m',[23.23,5,2.4],.38),('container','CONTAINER / 3.0 m',[27.98,-4.5,2.1],.35),
        ('cabinet','CABINET / 3.8 m',[35.98,4,2.7],.30)]:text(name,label,pos,width)
    report['parts']=[]
    for p,a in zip(scene['parts'],actors):
        c=a.static_mesh_component;origin,extent=a.get_actor_bounds(False)
        actual_center=[x/100 for x in vec(origin)];actual_size=[2*x/100 for x in vec(extent)]
        err=max(abs(actual_center[i]-p['center_xyz_m'][i]) for i in range(3))
        err=max(err,max(abs(actual_size[i]-p['size_xyz_m'][i]) for i in range(3)))
        if err>.005:raise RuntimeError('VISIBLE_BOUNDS_MISMATCH_'+p['id'])
        ignore=[x for x in ue.EditorLevelLibrary.get_all_level_actors() if x!=a]
        center=[100*x for x in p['center_xyz_m']];faces=[];outside_rays=[];errors=[]
        for axis in range(3):
            half=50*p['size_xyz_m'][axis]
            for direction in [-1,1]:
                begin=list(center);end=list(center)
                begin[axis]+=direction*(half+25);end[axis]-=direction*(half+25)
                hit=ray(world,begin,end,ignore)
                if hit['blocking']!=p['collides'] or (p['collides'] and hit['actor_label']!=p['id']):
                    raise RuntimeError('COLLISION_RAY_MISMATCH_'+p['id']+'_'+str(hit))
                if p['collides']:
                    error=abs(hit['impact_cm'][axis]/100-(p['center_xyz_m'][axis]+direction*p['size_xyz_m'][axis]/2))
                    if error>.015:raise RuntimeError('COLLISION_FACE_OFFSET_'+p['id'])
                    errors.append(error)
                faces.append(dict(axis=axis,entry_side=direction,result=hit))
            begin=list(center);end=list(center);perp=(axis+1)%3
            begin[axis]-=half+25;end[axis]+=half+25
            begin[perp]+=50*p['size_xyz_m'][perp]+5;end[perp]=begin[perp]
            hit=ray(world,begin,end,ignore)
            if hit['blocking']:raise RuntimeError('COLLISION_EXCEEDS_BOUNDS_'+p['id'])
            outside_rays.append(dict(axis=axis,result=hit))
        collision_surface_error=max(errors) if errors else None
        report['parts'].append(dict(id=p['id'],unreal_actor_name=a.get_name(),group=p['group'],visible_bounds_error_m=err,collision_enabled=str(c.get_collision_enabled()),
            collision_profile=str(c.get_collision_profile_name()),planner_shape=p['planner_shape'],collision_shape=p['collision_shape'],
            configured_bounds_xyz_m=p['engineering_bounds_xyz_m'],face_rays=faces,outside_rays=outside_rays,collision_face_error_m=collision_surface_error))
    ue.EditorLevelLibrary.set_level_viewport_camera_info(ue.Vector(-1600,-3300,3200),ue.Rotator(pitch=-28,yaw=39,roll=0))
    if not ue.EditorLevelLibrary.save_current_level():raise RuntimeError('SAVE_MAP_FAILED')
    ue.EditorAssetLibrary.save_directory(BASE,only_if_is_dirty=False,recursive=True)
    report.update(status='PASS_EDITOR_GEOMETRY_COLLISION',map=scene['unreal_map'],parts_checked=len(actors),
        editor_rays='six axial face rays and three outside rays per part; other actors ignored; simple collision; 15mm face tolerance; not an airborne contact test')
except Exception:
    report.update(status='FAIL',error=traceback.format_exc())
finally:
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    LATEST.write_text(json.dumps(dict(report=str(OUT),status=report['status']),indent=2),encoding='utf-8')
    ue.log('ENCLOSED_LAB_BUILD_'+report['status'])
