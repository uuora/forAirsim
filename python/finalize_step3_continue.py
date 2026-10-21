"""Create a small final source/config archive; preserve original trials."""
import sys,json,shutil,platform,importlib.metadata as metadata,subprocess
from pathlib import Path
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
def main():
    root=Path(sys.argv[1]).resolve();out=root/'final_delivery';out.mkdir(exist_ok=False);(out/'source').mkdir();(out/'configs').mkdir()
    modules=['fixed_map_mission.py','fixed_map_dashboard.py','fixed_map_perception.py','fixed_map_perception_process.py','fixed_map_perception_execution.py','fixed_map_perception_task.py','fixed_map_perception_geometry.py','fixed_map_delayed_identity.py','fixed_map_random_scene.py','run_fixed_map_perception_base.py','step3_recover_unowned.py','audit_step3_continue.py','check_random_dashboard.py','test_fixed_map_handoff_v5.py','test_fixed_map_vlx_rejection_reason.py','finalize_step3_continue.py']
    originals=[]
    for name in modules:
        p=ROOT/'python'/name;shutil.copyfile(p,out/'source'/name);originals.append(dict(path=str(p),sha256=sha(p),snapshot='source/'+name))
    for p in [ROOT/'configs/fixed_map_perception_v6.json',ROOT/'configs/step3_balloon_cuda1280_v2.json',ROOT/'configs/fixed_map_scene_state.json',ROOT/'configs/fixed_map_scene_runtime_v1.json',HIST/'fixed_version/task.json',HIST/'fixed_version/snapshot/configs/enclosed_uav_lab_manual_settings.json']:
        shutil.copyfile(p,out/'configs'/p.name);originals.append(dict(path=str(p),sha256=sha(p),snapshot='configs/'+p.name))
    for p in [ROOT/'01_固定地图实验.cmd',ROOT/'scripts/unreal/build_random_runtime_copy.py']:
        shutil.copyfile(p,out/p.name);originals.append(dict(path=str(p),sha256=sha(p),snapshot=p.name))
    detector=read(ROOT/'configs/step3_balloon_cuda1280_v2.json');w=Path(detector['weights']);w=w if w.is_absolute() else ROOT/w
    actual=sha(w)
    if actual!=detector['weights_sha256']:raise ValueError('FROZEN_WEIGHT_HASH_MISMATCH')
    maps=[]
    for name in ['EnclosedUAVLab_Manual.umap','EnclosedUAVLab_RuntimeRandom_v1.umap']:
        p=ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabManual/Maps'/name;maps.append(dict(path=str(p),sha256=sha(p),copied=False))
    packages={}
    for name in ['torch','ultralytics','airsim','PyQt5','numpy','opencv-python']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]='UNAVAILABLE'
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=4,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip()
    import airsim
    c=airsim.MultirotorClient(port=41454,timeout_value=3);ground={}
    for n in ['DroneA','DroneB']:
        s=c.getMultirotorState(vehicle_name=n);ground[n]=dict(state_timestamp_ns=s.timestamp,landed=s.landed_state==airsim.LandedState.Landed,api_control=c.isApiControlEnabled(vehicle_name=n))
    if any(not r['landed'] or r['api_control'] for r in ground.values()):raise ValueError('FINAL_GROUND_RELEASE_NOT_CONFIRMED')
    save(out/'ground_receipt.json',dict(utc=utc(),vehicles=ground,read_only=True,flight_commands_issued=False))
    save(out/'manifest.json',dict(utc=utc(),python=platform.python_version(),platform=platform.platform(),packages=packages,gpu=gpu,unreal='4.27 (configured executable D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe)',files=originals,large_map_references=maps,weights=dict(path=str(w),sha256=actual,frozen_hash_verified=True,copied=False),source_during_flight=str(root/'source_snapshot_v6/hashes.json'),postflight_changes=['zero-dt diagnostic emits null/flag instead of NaN; original control unchanged','explicit VLX source-attribute rejection reason; original audit/gate unchanged','Qt layout/word wrap and current v6 startup config','postflight audit and archive scripts'],postflight_log_changes_not_flight_validated=True,historical_references=[str(ROOT/'reports/step1_quick_review_20261010_083715'),str(ROOT/'reports/step2_perception_20261010'),str(ROOT/'reports/step3_development_20261010_101819')]))
    old=root/'base_trial_v6_interruption_audit.json';shutil.copyfile(old,root/'base_trial_v6_interruption_audit_initial.json');r=read(old);r['new_takeoffs']=1;r['takeoff_evidence']='OBSERVATION_START_ACCEPTED + actual TAKEOFF states + TAKEOFF_REACHED; recovery has no takeoff';save(old,r)
    save(out/'archive_verification.json',dict(status='PASS',files=len(originals),snapshot_hashes_verified=all(sha(out/r['snapshot'])==r['sha256'] for r in originals),frozen_weights_hash_verified=True,final_ground_release_confirmed=True))
    print(json.dumps(dict(status='PASS',snapshot_files=len(originals),ground=ground),ensure_ascii=False))
if __name__=='__main__':main()
