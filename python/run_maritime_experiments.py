"""Run four profiles sequentially and verify frame/annotation linkage."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=int,default=10); args=ap.parse_args()
if args.pairs<1: ap.error('pairs must be positive')
summary={'status':'RUNNING','runs':[]}
output=ROOT/'logs'/('maritime_experiments_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
try:
    for name in ['baseline','low_sun','raised_waves','occlusion']:
        subprocess.run([sys.executable,str(ROOT/'python/capture_dual_cameras.py'),'--pairs',str(args.pairs),'--ground-truth','--scenario',name],check=True)
        folder=Path(json.loads((ROOT/'logs/dual_camera_latest.json').read_text())['output'])
        manifest=json.loads((folder/'manifest.json').read_text())
        frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        annotations=[json.loads(s) for s in (folder/'evaluation_ground_truth/annotations.jsonl').read_text().splitlines()]
        assert manifest['status']=='PASS' and manifest['scene_restored_to_baseline']
        assert len(frames)==len(annotations)==2*args.pairs
        assert len({r['image'] for r in frames})==len(frames)
        for f,a in zip(frames,annotations):
            assert f['image']==a['image'] and f['timestamp_ns']==a['image_timestamp_ns']
            assert len(a['targets'])==4 and len({t['target_id'] for t in a['targets']})==4
            image=folder/f['image']
            assert hashlib.sha256(image.read_bytes()).hexdigest()==f['sha256']
            assert cv2.imread(str(image)).shape==(720,1280,3)
            for t in a['targets']:
                if 'bbox_xyxy_pixels' in t:
                    x1,y1,x2,y2=t['bbox_xyxy_pixels']; assert 0<=x1<x2<=1280 and 0<=y1<y2<=720
        summary['runs'].append({'scenario':name,'directory':str(folder),'images_verified':len(frames),'annotations_verified':len(annotations),'review_required':True,'restored':True})
        output.write_text(json.dumps(summary,indent=2))
    summary['status']='PASS'
except Exception as exc:
    summary.update(status='FAIL',error=str(exc)); raise
finally:
    output.write_text(json.dumps(summary,indent=2)); print(output)
