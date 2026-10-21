"""Offline candidate/attribute diagnostics. Saved frames never become live evidence."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from fixed_map_perception import ROOT,read,save,sha,utc,restore_inference_threads


def run(output):
    output=Path(output).resolve();output.mkdir(exist_ok=False,parents=True)
    batch=ROOT/'reports/v17_repeatability_20261011_163839'
    selection=[('development',3,300),('development',3,400),('development',3,500)]
    selection += [('reserved_diagnostic',1,n) for n in [200,250,300,350,400]]
    selection += [('reserved_diagnostic',2,n) for n in [210,230,260,280]]
    selection += [('reserved_diagnostic',3,n) for n in [220,250,350,450]]
    cases=[]
    for split,trial,frame in selection:
        folder=batch/f'trial_{trial:02d}/online_trial/online/frames'/f'frame-{frame:06d}'
        original=read(folder/'perception.json')
        cases.append(dict(split=split,trial=trial,frame_id=folder.name,rgb=str(folder/'rgb.png'),rgb_sha256=sha(folder/'rgb.png'),original_valid=original['valid'],original_candidates=len(original['semantic_frame']['candidates']),original_refusals=original['rejection_reasons']))
    save(output/'SELECTION.json',dict(utc=utc(),cases=cases,options=[dict(imgsz=640,confidence=.25),dict(imgsz=960,confidence=.25)],weights_unchanged=True,identity_thresholds_unchanged=True,scope='source-bound offline diagnostics; no human truth labels, no recall/precision claim; reserved rows are correlated same-layout frames, not independent generalization'))
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    import torch
    from specialist_balloon_detector import SpecialistBalloonDetector
    from t1_source_attribute_guard import band_evidence
    dc=read(ROOT/'configs/step3_balloon_cpu_dual_v9.json');assert sha(dc['weights'])==dc['weights_sha256']
    detector=SpecialistBalloonDetector(dc['weights'],confidence=.25,imgsz=640,device='cpu',nms_iou=.5)
    first=cv2.imread(cases[0]['rgb']);detector.detect(first);restore_inference_threads(dict(torch_threads=6),torch)
    rows=[]
    for case in cases:
        assert sha(case['rgb'])==case['rgb_sha256']
        image=cv2.imread(case['rgb'])
        for size in [640,960]:
            detector.imgsz=size
            started=time.monotonic();detected=detector.detect(image);inference=time.monotonic()-started
            audits=[]
            for candidate in detected['candidates']:
                audit=band_evidence(image,candidate['bbox_xyxy_pixels'])
                audits.append(dict(box=candidate['bbox_xyxy_pixels'],score=candidate['confidence'],band_status=audit['status'],band_reason=audit['reason']))
            rows.append(dict(**case,imgsz=size,inference_s=inference,actual_threads=torch.get_num_threads(),candidate_count=len(audits),described_attribute_matches=sum(a['band_status']=='ACCEPTED' for a in audits),candidates=audits,mode='OFFLINE_NOT_CURRENT_CONTROL_INPUT'))
        save(output/'ROWS.json',rows)
    stats={}
    for split in ['development','reserved_diagnostic']:
        for size in [640,960]:
            selected=[r for r in rows if r['split']==split and r['imgsz']==size]
            stats[split+'_'+str(size)]=dict(frames=len(selected),frames_with_one_attribute_match=sum(r['described_attribute_matches']==1 for r in selected),ambiguous_frames=sum(r['described_attribute_matches']>1 for r in selected),median_inference_s=float(np.median([r['inference_s'] for r in selected])),max_inference_s=max(r['inference_s'] for r in selected))
    save(output/'DIAGNOSTIC_SUMMARY.json',dict(utc=utc(),stats=stats,weights_sha256=dc['weights_sha256'],ai_disclosure='Codex selected source-bound real frames and ran offline diagnostics; no flight samples or expert labels generated',interpretation='attribute-matching candidates, not measured target recall; online timing still requires prospective validation'))
    print(stats)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');run(parser.parse_args().output)
