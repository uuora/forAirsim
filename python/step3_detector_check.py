"""One declared frozen-weight inference-resolution repair, on recorded RGB.

This is a replay diagnostic, never a fresh online localization certificate.
"""
import json,time,sys
from pathlib import Path
import cv2
from fixed_map_perception import read,save,sha,utc,ROOT

def main():
    out=Path(sys.argv[1]).resolve()
    src=out/'online_observation_v1/frames/frame-000246/rgb.png'
    cfg=read(ROOT/'configs/t1_balloon_detector_runtime_cuda_v1.json')
    cfg.update(imgsz=1280,status='STAGE3_DEVELOPMENT_REPAIR_PENDING_ONLINE_CHECK')
    dest=ROOT/'configs/step3_balloon_cuda1280_v2.json'
    declaration=dict(created_utc=utc(),version='step3_detector_runtime_v2',scope='one targeted runtime/resolution repair; validation follows declaration',basis='actual 1600x900 RGB shows visible spheres about 140px wide; 640 resize reduces them to about 56px and replay yielded zero candidates; 1280 preserves about112px',changes=dict(device='0',imgsz=1280),unchanged=dict(weights_sha256=cfg['weights_sha256'],confidence=cfg['confidence'],nms_iou=cfg['nms_iou'],frame_max_age_s=.5),risks=['higher GPU cost may still violate .5s freshness','detector generalization may still fail; zero output remains failure'],model_training=False,source_rgb=str(src),source_sha256=sha(src),acceptance='fresh online candidate and valid depth required separately; stop same-class repairs after this check')
    save(out/'detector_repair_v2_declaration.json',declaration);save(dest,cfg)
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    import torch
    torch.set_num_threads(1)
    from specialist_balloon_detector import SpecialistBalloonDetector
    detector=SpecialistBalloonDetector(cfg['weights'],confidence=cfg['confidence'],imgsz=cfg['imgsz'],device=cfg['device'],nms_iou=cfg['nms_iou'])
    image=cv2.imread(str(src));timing=[]
    for _ in range(3):
        start=time.monotonic();result=detector.detect(image);timing.append(time.monotonic()-start)
    for c in result['candidates']:
        x,y,w,h=map(int,c['bbox_xyxy_pixels']);cv2.rectangle(image,(x,y),(w,h),(0,255,255),2)
    cv2.imwrite(str(out/'detector_repair_v2_replay_overlay.png'),image)
    save(out/'detector_repair_v2_replay_result.json',dict(utc=utc(),mode='recorded_current_map_RGB_replay_NOT_online',declaration_sha256=sha(out/'detector_repair_v2_declaration.json'),config=cfg,config_sha256=sha(dest),source_sha256=sha(src),wall_s=timing,result=result))
    print(json.dumps(dict(candidate_count=result['candidate_count'],wall_s=timing),ensure_ascii=False))

if __name__=='__main__':main()
