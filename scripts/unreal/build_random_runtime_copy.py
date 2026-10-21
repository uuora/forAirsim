"""Create an independently versioned runtime-movable copy; never save source map."""
import unreal as ue,json,hashlib,os
from pathlib import Path
ROOT=Path('D:/forAirsim');out=Path(os.environ['STEP3_CONTINUE_OUTPUT'])
SRC='/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual'
DST='/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_RuntimeRandom_v1'
content=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content'
source=content/'EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual.umap'
target=content/'EnclosedUAVLabManual/Maps/EnclosedUAVLab_RuntimeRandom_v1.umap'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(source)
if target.exists():raise RuntimeError('RUNTIME_COPY_ALREADY_EXISTS_REFUSE_OVERWRITE')
if not ue.EditorLevelLibrary.new_level_from_template(DST,SRC):raise RuntimeError('RUNTIME_MAP_DUPLICATION_FAILED')
names={'EL1_Balloon_Target','EL1_BalloonTether_0','EL1_BalloonAnchor_0'};changed=[];geometry=[]
for a in ue.EditorLevelLibrary.get_all_level_actors():
    if isinstance(a,ue.StaticMeshActor):
        c=a.static_mesh_component;p=a.get_actor_location();scale=a.get_actor_scale3d();rot=a.get_actor_rotation()
        geometry.append(dict(id=a.get_name(),label=a.get_actor_label(),center_xyz_m=[p.x/100,p.y/100,p.z/100],scale_xyz=[scale.x,scale.y,scale.z],rotation_pyr=[rot.pitch,rot.yaw,rot.roll]))
        if a.get_actor_label() in names:
            c.set_mobility(ue.ComponentMobility.MOVABLE);changed.append(a.get_actor_label())
if set(changed)!=names:raise RuntimeError('MOVABLE_TARGET_GROUP_NOT_COMPLETE')
if not ue.EditorLevelLibrary.save_current_level():raise RuntimeError('RUNTIME_MAP_SAVE_FAILED')
assert before==sha(source),'SOURCE_FROZEN_MAP_CHANGED'
receipt=dict(status='PASS_RUNTIME_COPY_BUILT',source_map=SRC,source_file=str(source),source_sha256=before,runtime_map=DST,runtime_file=str(target),runtime_sha256=sha(target),only_change='mobility of designated balloon, tether, anchor; no geometry/material/label changes',changed_labels=changed,actor_geometry=geometry)
(out/'runtime_copy_build.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('RUNTIME_RANDOM_COPY_BUILT')
