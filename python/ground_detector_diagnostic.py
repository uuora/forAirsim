"""Bounded frozen-weight replay diagnosis. No control or target truth inputs."""
import sys,time,json
from pathlib import Path
import cv2,numpy as np
from fixed_map_perception import ROOT,read,save,sha,utc
from fixed_map_resources import memory_headroom
from t1_source_attribute_guard import band_evidence

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False)
    original=ROOT/'reports/step3_continue_20261010_152319/online_base_v6'
    cfg=read(ROOT/'configs/step3_balloon_cuda1280_v2.json')
    chosen=[original/'frames/frame-000266/rgb.png']
    matches=[]
    for p in sorted((original/'frames').glob('*/perception.json')):
        r=read(p)
        if r.get('current_attribute_audit',{}).get('matching_candidate_ids'):matches.append(p.parent/'rgb.png')
    chosen+=matches[:1]+matches[-1:]
    chosen=list(dict.fromkeys(chosen))
    save(out/'declaration.json',dict(utc=utc(),scope='GROUND_DEVELOPMENT_REPLAY_NOT_FRESH_ONLINE_NOT_FLIGHT',source_files=[dict(path=str(p),sha256=sha(p)) for p in chosen],weights_sha256=cfg['weights_sha256'],confidence=.25,nms_iou=.5,original_imgsz=1280,diagnostics=['original full frame1280','full frame640 diagnostic','uniform overlapping full-height square tiles1280'],training=False,maximum_images=3,maximum_repair_candidates=1,criterion='same frozen weights/thresholds; descriptor guard unchanged; choose at most one preregistered inference-interface change if supported; current online ground no target is retained',truth_input=False))
    budget=memory_headroom();save(out/'memory_before.json',budget)
    if min(budget['physical_available_bytes'],budget['commit_available_bytes'])<3*2**30:raise ValueError('DIAGNOSTIC_HOST_MEMORY_BELOW_3GIB')
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.5,0)
    from specialist_balloon_detector import SpecialistBalloonDetector
    if sha(cfg['weights'])!=cfg['weights_sha256']:raise ValueError('FROZEN_WEIGHT_CHANGED')
    d=SpecialistBalloonDetector(cfg['weights'],confidence=.25,imgsz=1280,device='0',nms_iou=.5)
    d.detect(cv2.imread(str(chosen[0])))
    rows=[]
    for p in chosen:
        image=cv2.imread(str(p));h,w=image.shape[:2]
        modes=[('full1280',image,(0,0),1280),('full640',image,(0,0),640)]
        side=min(h,w)
        if w>side:
            for x in [0,w-side]:modes.append((f'tile{x}',image[:,x:x+side],(x,0),1280))
        for name,roi,offset,size in modes:
            d.imgsz=size;begin=time.monotonic();result=d.detect(roi);elapsed=time.monotonic()-begin
            mapped=[]
            for c in result['candidates']:
                x0,y0,x1,y1=c['bbox_xyxy_pixels'];box=[x0+offset[0],y0+offset[1],x1+offset[0],y1+offset[1]]
                mapped.append(dict(box=box,confidence=c['confidence'],tile_edge_truncated=c['truncated'],band=band_evidence(image,box)))
            record=dict(mode='REPLAY_NOT_ONLINE',source=str(p),source_sha256=sha(p),inference_mode=name,imgsz=size,offset_xy=list(offset),roi_shape=list(roi.shape),wall_s=elapsed,result=result,mapped=mapped)
            rows.append(record);save(out/(p.parent.name+'_'+name+'.json'),record)
            canvas=image.copy()
            for c in mapped:
                x0,y0,x1,y1=map(round,c['box']);cv2.rectangle(canvas,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.imwrite(str(out/(p.parent.name+'_'+name+'.png')),canvas)
    save(out/'diagnostic_results.json',dict(utc=utc(),mode='REPLAY_DIAGNOSTIC_NOT_ONLINE',rows=rows))
    print(json.dumps([dict(frame=Path(r['source']).parent.name,mode=r['inference_mode'],wall_s=r['wall_s'],candidates=len(r['mapped']),white_band_candidates=sum(c['band']['status']=='ACCEPTED' for c in r['mapped']),boxes=[c['box'] for c in r['mapped']]) for r in rows],ensure_ascii=False))
if __name__=='__main__':main()
