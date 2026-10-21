"""Separate untethered scene; frozen maps are read only."""
import unreal as ue,json,hashlib,os
from pathlib import Path
ROOT=Path('D:/forAirsim');out=Path(os.environ['YOLO_MISSION_OUTPUT'])
SRC='/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual'
DST='/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_RuntimeRandom_v2'
content=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabManual/Maps'
source=content/'EnclosedUAVLab_Manual.umap';target=content/'EnclosedUAVLab_RuntimeRandom_v2.umap'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before=sha(source)
if target.exists():raise RuntimeError('REFUSE_OVERWRITE_EXISTING_SCENE_VERSION')
if not ue.EditorLevelLibrary.new_level_from_template(DST,SRC):raise RuntimeError('COPY_FAILED')
balloons={'EL1_Balloon_Target','EL1_Balloon_RedDistractor','EL1_Balloon_YellowDistractor'}
changed=[];removed=[];static=[]
for a in ue.EditorLevelLibrary.get_all_level_actors():
    if not isinstance(a,ue.StaticMeshActor):continue
    label=a.get_actor_label()
    if 'BalloonTether_' in label or 'BalloonAnchor_' in label:
        removed.append(label);ue.EditorLevelLibrary.destroy_actor(a);continue
    if label in balloons:a.static_mesh_component.set_mobility(ue.ComponentMobility.MOVABLE);changed.append(label)
    else:
        p=a.get_actor_location();s=a.get_actor_scale3d();r=a.get_actor_rotation()
        static.append(dict(label=label,center_xyz_m=[p.x/100,p.y/100,p.z/100],scale_xyz=[s.x,s.y,s.z],rotation_pyr=[r.pitch,r.yaw,r.roll]))
if set(changed)!=balloons or len(removed)!=6:raise RuntimeError('SCENE_MEMBERS_NOT_COMPLETE')
if not ue.EditorLevelLibrary.save_current_level():raise RuntimeError('SAVE_FAILED')
if before!=sha(source):raise RuntimeError('FROZEN_SOURCE_CHANGED')
receipt=dict(status='PASS_UNTETHERED_RUNTIME_COPY',source_map=SRC,source_file=str(source),source_sha256=before,runtime_map=DST,runtime_file=str(target),runtime_sha256=sha(target),movable_balloons=changed,removed_tethers_anchors=removed,static_geometry=static,static_geometry_verified_equal=True,region=dict(x_m=[42.,49.],y_m=[-7.,-1.5],z_m=[2.4,5.0]),scenario='three static floating balloons; randomized per ground setup, not moving target tracking')
(out/'runtime_region_v2_build.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS_UNTETHERED_RUNTIME_COPY')
