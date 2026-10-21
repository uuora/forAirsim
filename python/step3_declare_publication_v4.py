import sys
from pathlib import Path
from fixed_map_perception import ROOT,read,save,utc,sha

def main():
    out=Path(sys.argv[1]).resolve();old=read(out/'base_trial_v3_result.json')
    if not old['final_state']['landed'] or not old['api_control_released']:raise ValueError('NOT_RELEASED_GROUND')
    cfg=read(ROOT/'configs/fixed_map_perception_v3.json');cfg.update(schema='step3_direct_publication_v4',publication_pipeline_version='direct_frame_publish_v4')
    save(ROOT/'configs/fixed_map_perception_v4.json',cfg)
    save(out/'publication_v4_declaration.json',dict(utc=utc(),version='direct_publication_v4',basis='v3 had genuine designated online positive but two submissions failed CURRENT_FRAME_EXPIRED; submitted source age about .515s',repair='publish directly upon completed frame or invalidation; parent reads changed atomic file at .01s rather than producer periodic .1s plus consumer .05s; actual source timestamps unchanged',unchanged=['weights/confidence/NMS/imgsz/GPU repair','2nd target version white-band descriptor','current age .5s, source8s, response6s, chain gap.5s and3 fresh confirmations','flight speed/accel/height/envelope/clearance/arrival/stop criteria'],old_failures=str(out/'base_trial_v3_result.json'),maximum_flight_rechecks=1,maximum_preview_attempts=1,on_failure='retain blockage, return land release; no further same-class repair',risk='publishing still costs time; post-planning and post-execution checks must reject expired source',config_sha256=sha(ROOT/'configs/fixed_map_perception_v4.json')))

if __name__=='__main__':main()
