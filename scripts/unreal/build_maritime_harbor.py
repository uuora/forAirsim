"""Run inside UE4.27 Python commandlet. Builds original meshes and a saved map.

Visual ocean only: shader displacement is not a buoyancy/hydrodynamics model.
All generated assets live under /Game/MaritimeHarborV2; existing maps are untouched.
"""
import json
import math
from pathlib import Path
import traceback
import unreal as ue

ROOT = Path('D:/forAirsim')
SOURCE = ROOT/'assets'/'maritime_harbor_v2'
OUT = ROOT/'logs'/'maritime_harbor_build.json'
BASE = '/Game/MaritimeHarborV2'
MAP = BASE+'/Maps/MaritimeHarbor'
SOURCE.mkdir(parents=True, exist_ok=True)


class Mesh:
    def __init__(self):
        self.v, self.f = [], []
    def poly(self, points):
        start = len(self.v)
        self.v.extend(points)
        for i in range(1, len(points)-1):
            self.f.append((start, start+i, start+i+1))
    def box(self, center, size):
        x,y,z = center
        a,b,c = [n/2 for n in size]
        pts = [(x-a,y-b,z-c),(x+a,y-b,z-c),(x+a,y+b,z-c),(x-a,y+b,z-c),
               (x-a,y-b,z+c),(x+a,y-b,z+c),(x+a,y+b,z+c),(x-a,y+b,z+c)]
        for face in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
            self.poly([pts[i] for i in face])
    def tube(self, a, b, radius, sides=10):
        delta = [b[i]-a[i] for i in range(3)]
        length = math.sqrt(sum(t*t for t in delta))
        axis = [t/length for t in delta]
        helper = [0,0,1] if abs(axis[2]) < .9 else [0,1,0]
        u = [axis[1]*helper[2]-axis[2]*helper[1],axis[2]*helper[0]-axis[0]*helper[2],axis[0]*helper[1]-axis[1]*helper[0]]
        norm = math.sqrt(sum(t*t for t in u))
        u = [t/norm for t in u]
        v = [axis[1]*u[2]-axis[2]*u[1],axis[2]*u[0]-axis[0]*u[2],axis[0]*u[1]-axis[1]*u[0]]
        rings = [[tuple(p[j]+radius*(math.cos(2*math.pi*i/sides)*u[j]+math.sin(2*math.pi*i/sides)*v[j]) for j in range(3)) for i in range(sides)] for p in (a,b)]
        for i in range(sides):
            k=(i+1)%sides
            self.poly([rings[0][i],rings[0][k],rings[1][k],rings[1][i]])
        self.poly(list(reversed(rings[0])))
        self.poly(rings[1])
    def torus(self, center, major, minor, axis='x'):
        rings=[]
        for i in range(32):
            t=2*math.pi*i/32
            row=[]
            for j in range(8):
                q=2*math.pi*j/8
                p=((major+minor*math.cos(q))*math.cos(t),(major+minor*math.cos(q))*math.sin(t),minor*math.sin(q))
                p=(p[2],p[0],p[1]) if axis=='x' else (p[0],p[2],p[1]) if axis=='y' else p
                row.append(tuple(p[k]+center[k] for k in range(3)))
            rings.append(row)
        for i in range(32):
            for j in range(8):
                self.poly([rings[i][j],rings[(i+1)%32][j],rings[(i+1)%32][(j+1)%8],rings[i][(j+1)%8]])
    def save(self, name):
        path=SOURCE/(name+'.obj')
        with path.open('w') as f:
            f.write('# Original procedural research vessel / ocean mesh; units centimetres\n')
            f.write('o '+name+'\ns 1\n')
            for x,y,z in self.v:
                f.write('v %.6f %.6f %.6f\n'%(x*100,y*100,z*100))
            for face in self.f:
                f.write('f %d %d %d\n'%tuple(i+1 for i in face))
        return path


def boat_geometry():
    meshes={k:Mesh() for k in ['Hull','White','Deck','Windows','Orange','Metal','Rubber']}
    hull,white,deck,glass,orange,metal,rubber=[meshes[k] for k in meshes]
    stations=[(-6,1.55),(-5.6,1.9),(-4,2.0),(-2,2.0),(0,1.95),(2,1.75),(3.8,1.35),(5,.87),(6,.38),(6.6,.035)]
    rings=[]
    for x,w in stations:
        rings.append([(x,-w,.7),(x,-w*.93,-.1),(x,-w*.50,-.85),(x,0,-1.03),(x,w*.50,-.85),(x,w*.93,-.1),(x,w,.7)])
    for i in range(len(rings)-1):
        for j in range(6):
            hull.poly([rings[i][j],rings[i+1][j],rings[i+1][j+1],rings[i][j+1]])
        deck.poly([rings[i][0],rings[i][6],rings[i+1][6],rings[i+1][0]])
    hull.poly(rings[0])
    hull.poly(list(reversed(rings[-1])))
    # White gunwale stripe and dark rubber rubbing strip along both sides.
    for side in [-1,1]:
        for i in range(len(stations)-1):
            x,w=stations[i]; nx,nw=stations[i+1]
            white.poly([(x,side*(w+.008),.40),(nx,side*(nw+.008),.40),(nx,side*(nw+.008),.63),(x,side*(w+.008),.63)])
            rubber.tube((x,side*w,.73),(nx,side*nw,.73),.055)
        # Side rail, stanchions and mid-height safety rail.
        line=[(x,side*(w-.12),1.52) for x,w in stations[:-1]]
        line.append((6.25,side*.13,1.52))
        for a,b in zip(line,line[1:]):
            metal.tube(a,b,.027)
            metal.tube((a[0],a[1],1.14),(b[0],b[1],1.14),.02)
        for x,y,z in line:
            metal.tube((x,y,.75),(x,y,z),.026)
    metal.tube((-6,-1.43,1.52),(-6,1.43,1.52),.027)
    # Cabin with sloping forward windshield and overhanging roof.
    p=[(-2.4,-1.28,.78),(1.65,-1.28,.78),(1.65,1.28,.78),(-2.4,1.28,.78),
       (-2.4,-1.13,2.82),(1.22,-1.13,2.82),(1.22,1.13,2.82),(-2.4,1.13,2.82)]
    for face in [(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)]:
        white.poly([p[i] for i in face])
    for side in [-1,1]:
        for lo,hi in [(-2.1,-.95),(-.78,.35),(.5,1.03)]:
            glass.poly([(lo,side*1.235,1.48),(hi,side*1.235,1.48),(hi,side*1.155,2.53),(lo,side*1.155,2.53)])
        # Rear cabin door, side handrails and life ring.
        metal.tube((-2.46,side*.65,1.15),(-2.46,side*.65,2.30),.022)
        orange.torus((-2.45,side*.78,1.9),.32,.075,'x')
    for lo,hi in [(-1.04,-.05),(.05,1.04)]:
        glass.poly([(1.50,lo,1.46),(1.50,hi,1.46),(1.282,hi,2.53),(1.282,lo,2.53)])
    orange.box((-.57,0,2.94),(4.2,2.85,.18))
    white.box((-.9,0,3.13),(1.6,1.15,.15))
    metal.tube((-.9,0,3.19),(-.9,0,4.9),.058)
    metal.tube((-.9,-.65,4.45),(-.9,.65,4.45),.035)
    white.box((-.9,0,4.75),(.30,1.55,.17))
    rubber.tube((-.9,.5,4.5),(-.9,.5,5.5),.014)
    rubber.tube((-1.15,-.45,3.3),(-1.15,-.45,4.5),.012)
    # Foredeck hatch, winch, aft working deck benches and vents.
    white.box((3.35,0,.83),(1.35,1.15,.20))
    metal.box((3.35,0,.96),(1.12,.9,.06))
    orange.box((-4.3,0,.9),(1.6,1.4,.23))
    rubber.tube((-4.3,-.38,1.18),(-4.3,.38,1.18),.26,20)
    for side in [-1,1]:
        white.box((-4.65,side*1.30,1.0),(1.7,.32,.42))
        for x in [-5.1,-3.5,2.8]:
            rubber.torus((x,side*1.95,.23),.25,.09,'y')
        metal.box((4.6,side*.63,.87),(.25,.12,.26))
    return meshes


def generate():
    paths={name:mesh.save('MRV_'+name) for name,mesh in boat_geometry().items()}
    ocean=Mesh(); count=200; extent=160
    for y in range(count+1):
        for x in range(count+1):
            ocean.v.append((-extent+2*extent*x/count,-extent+2*extent*y/count,0))
    for y in range(count):
        for x in range(count):
            a=y*(count+1)+x; b=a+1; c=a+count+1; d=c+1
            ocean.f.extend([(a,b,d),(a,d,c)])
    paths['Ocean']=ocean.save('OceanGrid')
    return paths


tools=ue.AssetToolsHelpers.get_asset_tools()
mel=ue.MaterialEditingLibrary


def node(mat, cls, **props):
    result=mel.create_material_expression(mat,cls)
    for k,v in props.items(): result.set_editor_property(k,v)
    return result


def scalar(mat,val): return node(mat,ue.MaterialExpressionConstant,r=val)


def custom(mat,code,output):
    inputs=[]
    for name in ['P','T']:
        arg=ue.CustomInput(); arg.set_editor_property('input_name',name); inputs.append(arg)
    result=node(mat,ue.MaterialExpressionCustom,code=code,output_type=output,inputs=inputs)
    world=node(mat,ue.MaterialExpressionWorldPosition)
    time=node(mat,ue.MaterialExpressionTime)
    if not mel.connect_material_expressions(world,'',result,'P') or not mel.connect_material_expressions(time,'',result,'T'):
        raise RuntimeError('Custom material input connection failed')
    return result


def material(name,rgb,rough=.4,metallic=0):
    path=BASE+'/Materials'
    mat=ue.load_asset(path+'/'+name) or tools.create_asset(name,path,ue.Material,ue.MaterialFactoryNew())
    mel.delete_all_material_expressions(mat)
    mat.set_editor_property('two_sided',True)
    col=node(mat,ue.MaterialExpressionConstant3Vector,constant=ue.LinearColor(*rgb,1))
    mel.connect_material_property(col,'',ue.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(scalar(mat,rough),'',ue.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(scalar(mat,metallic),'',ue.MaterialProperty.MP_METALLIC)
    return mat


def finish_material(mat):
    mel.recompile_material(mat)
    ue.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def make_water(name='M_AnimatedOcean', strength=1.0):
    mat=material(name,(.014,.055,.072),.30,0)
    mat.set_editor_property('tangent_space_normal',False)
    # Three metre-scale travelling waves; world position and displacement in cm.
    # Incommensurate wavelengths and directions reduce the previous striped pattern.
    # Absolute displacement bound = 43 cm; deck underside is 260 cm above datum.
    waves=[(.81,.59,.31,.73,16,0),(.96,-.28,.53,1.02,10,1.7),(.24,.97,.89,1.31,7,3.1),(-.61,.79,1.43,1.69,4,.8),(.72,-.69,2.17,2.03,3,2.5),(.98,.19,3.31,2.42,2,4.2),(-.42,.91,4.87,2.81,1,5.3)]
    wave='float2 q=P.xy*.01; float h=0; float2 g=0; '
    for i,(dx,dy,k,w,a,phase) in enumerate(waves):
        wave+='float p%d=dot(q,float2(%s,%s))*%s+T*%s+%s; h+=%s*sin(p%d); g+=float2(%s,%s)*%s*cos(p%d); '%(i,dx,dy,k,w,phase,a,i,dx,dy,a*.01*k,i)
    wave+='h*=%s; g*=%s; '%(strength,strength)
    offset=custom(mat,wave+'return float3(0,0,h);',ue.CustomMaterialOutputType.CMOT_FLOAT3)
    # UE4.27 hides WPO from its Python enum: use our narrow editor-only bridge.
    if not ue.MaritimeMaterialLibrary.connect_world_position_offset(mat,offset):
        raise RuntimeError('World-position-offset connection failed')
    normal=custom(mat,wave+'float footprint=max(length(ddx(q)),length(ddy(q))); float fade=1-smoothstep(.025,.25,footprint); g+=fade*float2(.022*cos(dot(q,float2(17.3,11.7))+T*2.7),.018*sin(dot(q,float2(-13.1,21.9))+T*3.1)); g*=1-smoothstep(.4,3,footprint); return normalize(float3(-g,1));',ue.CustomMaterialOutputType.CMOT_FLOAT3)
    mel.connect_material_property(normal,'',ue.MaterialProperty.MP_NORMAL)
    color=custom(mat,wave+'return lerp(float3(.008,.035,.047),float3(.014,.061,.073),saturate(.5+h*.006));',ue.CustomMaterialOutputType.CMOT_FLOAT3)
    mel.connect_material_property(color,'',ue.MaterialProperty.MP_BASE_COLOR)
    return finish_material(mat)


def import_mesh(path):
    dest=BASE+'/Meshes'
    existing=ue.load_asset(dest+'/'+path.stem)
    if existing: return existing
    task=ue.AssetImportTask()
    for k,v in dict(filename=str(path),destination_path=dest,destination_name=path.stem,automated=True,replace_existing=False,save=True).items():
        task.set_editor_property(k,v)
    opts=ue.FbxImportUI()
    opts.set_editor_property('import_mesh',True)
    opts.set_editor_property('import_materials',False)
    opts.set_editor_property('import_textures',False)
    opts.set_editor_property('import_as_skeletal',False)
    data=opts.get_editor_property('static_mesh_import_data')
    data.set_editor_property('combine_meshes',True)
    data.set_editor_property('generate_lightmap_u_vs',False)
    data.set_editor_property('auto_generate_collision',False)
    data.set_editor_property('convert_scene',False)
    data.set_editor_property('normal_import_method',ue.FBXNormalImportMethod.FBXNIM_COMPUTE_NORMALS)
    task.set_editor_property('options',opts)
    tools.import_asset_tasks([task])
    imported=task.get_editor_property('imported_object_paths')
    if not imported: raise RuntimeError('No imported mesh: '+str(path))
    return ue.load_asset(imported[0])


def actor(mesh,mat,name,pos=(0,0,0),scale=(1,1,1)):
    a=ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor,ue.Vector(*pos))
    a.set_actor_label(name)
    a.set_editor_property('tags',[ue.Name(name)])
    c=a.static_mesh_component
    c.set_static_mesh(mesh)
    c.set_material(0,mat)
    c.set_mobility(ue.ComponentMobility.MOVABLE)
    c.set_collision_enabled(ue.CollisionEnabled.NO_COLLISION)
    c.set_editor_property('cast_shadow',True)
    a.set_actor_scale3d(ue.Vector(*scale))
    return a


def build():
    OUT.write_text(json.dumps({'status':'RUNNING'}))
    paths=generate()
    palette={'Hull':((.018,.055,.092),.31,.2),'White':((.82,.86,.88),.32,.12),'Deck':((.28,.34,.35),.62,0),'Windows':((.012,.09,.135),.10,.52),'Orange':((.95,.18,.028),.31,.05),'Metal':((.58,.65,.68),.24,.78),'Rubber':((.012,.015,.018),.75,0)}
    mats={k:finish_material(material('M_'+k,*v)) for k,v in palette.items()}
    water=make_water()
    make_water('M_OceanCalm',.5)
    make_water('M_OceanRaised',1.5)
    meshes={k:import_mesh(v) for k,v in paths.items()}
    for k,m in meshes.items():
        m.set_material(0,water if k=='Ocean' else mats[k])
        ue.EditorAssetLibrary.save_loaded_asset(m)
    if ue.EditorAssetLibrary.does_asset_exist(MAP):
        if not ue.EditorLevelLibrary.load_level(MAP): raise RuntimeError('Cannot load generated map')
        for a in ue.EditorLevelLibrary.get_all_level_actors():
            if a.get_actor_label().startswith('MH2_'):
                ue.EditorLevelLibrary.destroy_actor(a)
    elif not ue.EditorLevelLibrary.new_level(MAP):
        raise RuntimeError('Could not create maritime map')
    world=ue.EditorLevelLibrary.get_editor_world()
    world.get_world_settings().set_editor_property('default_game_mode',ue.load_class(None,'/Script/AirSim.AirSimGameMode'))
    # Saved environment: original research launch, animated ocean, three balloons.
    for k in palette: actor(meshes[k],mats[k],'MH2_Vessel_'+k,(2200,0,0))
    ocean=actor(meshes['Ocean'],water,'MH2_OceanGrid',(2200,0,0))
    ocean.static_mesh_component.set_editor_property('bounds_scale',2.0)
    plane=ue.load_asset('/Engine/BasicShapes/Plane')
    actor(plane,water,'MH2_DistantOcean',(0,0,-40),(15000,15000,1))
    sphere=ue.load_asset('/Engine/BasicShapes/Sphere')
    cylinder=ue.load_asset('/Engine/BasicShapes/Cylinder')
    for i,(pos,rgb) in enumerate([((2600,950,150),(.95,.025,.02)),((3300,-1000,170),(1,.62,.025)),((1100,650,145),(.95,.04,.06))]):
        balloon=finish_material(material('M_Balloon'+str(i),rgb,.25,.05))
        actor(sphere,balloon,'MH2_Balloon_'+str(i),pos,(1.1,1.1,1.3))
        actor(cylinder,mats['Rubber'],'MH2_String_'+str(i),(pos[0],pos[1],pos[2]/2),(.012,.012,pos[2]/100))
    # Fixed observation pad for the two landed AirSim vehicles.
    cube=ue.load_asset('/Engine/BasicShapes/Cube')
    pad=actor(cube,mats['Deck'],'MH2_ObservationPad',(-400,0,280),(12,12,.4))
    pad.static_mesh_component.set_collision_enabled(ue.CollisionEnabled.QUERY_AND_PHYSICS)
    for i,(x,y) in enumerate([(-900,-500),(-900,500),(100,-500),(100,500)]):
        actor(cylinder,mats['Metal'],'MH2_PlatformPile_'+str(i),(x,y,60),(.6,.6,4.4))
    for i,y in enumerate([-550,550]):
        actor(cube,mats['Orange'],'MH2_PlatformFascia_'+str(i),(-400,y,285),(12,.18,.5))
        for j,x in enumerate(range(-950,151,220)):
            actor(cylinder,mats['Metal'],'MH2_PlatformPost_%d_%d'%(i,j),(x,y,350),(.055,.055,1))
        actor(cube,mats['White'],'MH2_PlatformRail_'+str(i),(-400,y,400),(11.2,.055,.055))
    # Two visible landing bays, each with a white H and perimeter stripes.
    for i,y in enumerate([-200,200]):
        actor(cube,mats['Hull'],'MH2_LandingBay_'+str(i),(-400,y,301),(3.2,3.2,.02))
        for j,x in enumerate([-450,-350]):
            actor(cube,mats['White'],'MH2_HSide_%d_%d'%(i,j),(x,y,303),(.12,1.5,.02))
        actor(cube,mats['White'],'MH2_HBar_'+str(i),(-400,y,303),(1.1,.12,.02))
        for j,dy in enumerate([-160,160]):
            actor(cube,mats['Orange'],'MH2_BayEdge_%d_%d'%(i,j),(-400,y+dy,303),(3.2,.08,.02))
    start=ue.EditorLevelLibrary.spawn_actor_from_class(ue.PlayerStart,ue.Vector(-400,0,420))
    start.set_actor_label('MH2_PlayerStart')
    sun=ue.EditorLevelLibrary.spawn_actor_from_class(ue.DirectionalLight,ue.Vector(0,0,1000),ue.Rotator(-28,-35,0))
    sun.set_actor_label('MH2_Sun')
    sun.light_component.set_mobility(ue.ComponentMobility.MOVABLE)
    sun.light_component.set_editor_property('intensity',5.0)
    sun.light_component.set_atmosphere_sun_light(True)
    sky=ue.EditorLevelLibrary.spawn_actor_from_class(ue.SkyAtmosphere,ue.Vector(0,0,0)); sky.set_actor_label('MH2_Atmosphere')
    skylight=ue.EditorLevelLibrary.spawn_actor_from_class(ue.SkyLight,ue.Vector(0,0,300)); skylight.set_actor_label('MH2_Skylight')
    skylight.light_component.set_mobility(ue.ComponentMobility.MOVABLE)
    skylight.light_component.set_editor_property('intensity',1.0)
    skylight.light_component.set_editor_property('real_time_capture',True)
    fog=ue.EditorLevelLibrary.spawn_actor_from_class(ue.ExponentialHeightFog,ue.Vector(0,0,-100)); fog.set_actor_label('MH2_Haze')
    fog.component.set_editor_property('fog_density',.004)
    ue.EditorLevelLibrary.set_level_viewport_camera_info(ue.Vector(650,-2100,1150),ue.Rotator(-18,53,0))
    if not ue.EditorLevelLibrary.save_current_level(): raise RuntimeError('Map save failed')
    ue.EditorAssetLibrary.save_directory(BASE,only_if_is_dirty=False,recursive=True)
    OUT.write_text(json.dumps({'status':'PASS','map':MAP,'source_meshes':{k:str(v) for k,v in paths.items()},'ocean_model':'time_driven_visual_displacement_not_hydrodynamics','vessel':'original_procedural_12m_research_launch','objects':'vessel_and_three_balloon_placeholders'},indent=2))
    ue.log('MARITIME_HARBOR_BUILD_PASS')


try:
    build()
except Exception:
    OUT.write_text(json.dumps({'status':'FAIL','error':traceback.format_exc()},indent=2))
    raise
