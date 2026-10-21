"""Source-bound development timing, counterbalanced thread order, no flight."""
import argparse
import time
from pathlib import Path
import numpy as np
from fixed_map_perception import ROOT,read,save,sha,utc,restore_inference_threads
from fixed_map_resources import memory_headroom


def run(output):
    output=Path(output).resolve();output.mkdir(exist_ok=False,parents=True)
    selection=[('v19_repeatability_20261011_192354',1,n) for n in [200,243,284,350,400,450,506]]
    selection += [('v17_repeatability_20261011_163839',3,n) for n in [300,400,500]]
    selection += [('v17_repeatability_20261011_163839',2,n) for n in [260,280]]
    cases=[]
    for batch,trial,frame in selection:
        rgb=ROOT/'reports'/batch/f'trial_{trial:02d}/online_trial/online/frames'/f'frame-{frame:06d}'/'rgb.png'
        cases.append(dict(path=str(rgb),sha256=sha(rgb),case_id=f'{batch}_trial{trial}_frame{frame}'))
    memory=memory_headroom()
    if min(memory['physical_available_bytes'],memory['commit_available_bytes'])<3*2**30:raise ValueError('MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB')
    dc=read(ROOT/'configs/step3_balloon_cpu_dual_v9.json');assert sha(dc['weights'])==dc['weights_sha256']
    save(output/'PROTOCOL.json',dict(utc=utc(),cases=cases,orders=[[2,4,6],[6,4,2]],weights_sha256=dc['weights_sha256'],memory=memory,scope='Already-consumed correlated development images; timing only, no accuracy/generalization/flight acceptance claim',same_image_order_each_condition=True,detector_policy='adaptive_full640_960_v18',no_training=True,no_aircraft_commands=True))
    from train_t1_balloon_yolo import configure_offline_runtime
    configure_offline_runtime()
    import torch,cv2
    from specialist_balloon_detector import SpecialistBalloonDetector
    from fixed_map_conditional_scale import ConditionalScaleBalloonDetector
    from t1_source_attribute_guard import band_evidence
    detector=SpecialistBalloonDetector(dc['weights'],confidence=dc['confidence'],imgsz=640,device='cpu',nms_iou=dc['nms_iou'])
    images=[]
    for case in cases:
        assert sha(case['path'])==case['sha256'];images.append(cv2.imread(case['path']))
    for size in [640,960]:detector.imgsz=size;detector.detect(images[0])
    detector.imgsz=640;rows=[]
    for repeat,order in enumerate([[2,4,6],[6,4,2]],1):
        for threads in order:
            restore_inference_threads(dict(torch_threads=threads),torch)
            adaptive=ConditionalScaleBalloonDetector(detector)
            adaptive.detect(images[0]);adaptive.primary_imgsz=640
            for case,image in zip(cases,images):
                start=time.monotonic();detected=adaptive.detect(image);elapsed=time.monotonic()-start
                matches=[c for c in detected['candidates'] if not c['truncated'] and band_evidence(image,c['bbox_xyxy_pixels'])['status']=='ACCEPTED']
                rows.append(dict(case_id=case['case_id'],repeat=repeat,threads=threads,actual_threads=torch.get_num_threads(),elapsed_s=elapsed,candidate_count=detected['candidate_count'],attribute_matches=len(matches),scale=detected['conditional_scale_receipt'],candidate_boxes=[c['bbox_xyxy_pixels'] for c in detected['candidates']]))
            save(output/'ROWS.json',rows)
            print(dict(repeat=repeat,threads=threads,p90_s=float(np.percentile([r['elapsed_s'] for r in rows if r['repeat']==repeat and r['threads']==threads],90))),flush=True)
    summary={}
    for threads in [2,4,6]:
        chosen=[r for r in rows if r['threads']==threads];durations=[r['elapsed_s'] for r in chosen]
        summary[str(threads)]=dict(n=len(chosen),median_s=float(np.median(durations)),p90_s=float(np.percentile(durations,90)),max_s=max(durations),unique_attribute_frames=sum(r['attribute_matches']==1 for r in chosen),ambiguous_frames=sum(r['attribute_matches']>1 for r in chosen))
    stable=all(len({r['attribute_matches'] for r in rows if r['case_id']==case['case_id']})==1 for case in cases)
    save(output/'BENCHMARK_RESULT.json',dict(utc=utc(),summary=summary,attribute_match_counts_same_across_conditions=stable,memory_after=memory_headroom(),scope='Descriptive development timing. Counterbalancing does not control unrelated system load or establish flight reliability.'))
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');run(parser.parse_args().output)
