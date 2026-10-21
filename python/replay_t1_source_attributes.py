"""Evaluate fixed descriptor guard on sealed semantic sources, no API/flight."""
import argparse
from pathlib import Path
import json
import shutil
from analyze_t1_identity_flight import read_json,digest,verify_file,mask_truth,identity_score,source_checks
from analyze_t1_delayed_identity import frame_from_dict,save
from t1_semantic_perception import SemanticRequest,SemanticReply
from t1_source_attribute_guard import verify_source_attribute


def replay(roots,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    results=[];checks=[]
    for root in roots:
        root=Path(root).resolve();manifest=read_json(root/'manifest.json');report=read_json(manifest['report']);plan=read_json(report['plan'])
        checks+=[verify_file(manifest['report'],manifest['report_sha256'],'sealed_report')]+source_checks(plan,root,report)
        raw={row['frame_id']:row for row in report['acquisition_frames']+report['frames']}
        seen=set()
        for row in raw.values():
            proof=row.get('observer_result',{}).get('temporal_identity',{}).get('delayed_chain_proof')
            if not proof:continue
            binding=proof['request_binding'];request_id=binding['request_id']
            if request_id in seen:continue
            seen.add(request_id);source=raw[binding['frame_id']];frame=frame_from_dict(source['observer_result']['semantic_frame'])
            rgb=source['media']['rgb'];checks.append(verify_file(rgb['path'],rgb['sha256'],'source_rgb'))
            object.__setattr__(frame,'encoded_image',Path(rgb['path']).read_bytes())
            request=SemanticRequest(request_id,binding['session_id'],binding['target_generation'],plan['description'],binding['description_sha256'],binding['submitted_monotonic_s'],frame)
            box=proof['semantic_box_xyxy_pixels'];reply=SemanticReply(request.binding(),proof['reply_completed_monotonic_s'],{'result_bbox_list':[dict(zip(('xmin','ymin','xmax','ymax'),box))]})
            audit=verify_source_attribute(request,reply)
            # Independent evaluation only. It never enters attribute guard input.
            import cv2
            seg=source['media']['segmentation'];checks.append(verify_file(seg['path'],seg['sha256'],'source_segmentation'))
            truth=mask_truth(cv2.imread(seg['path']),report['segmentation_palette_rgb'])
            identity=identity_score(dict(status='TEMPORAL_LOCAL_ASSOCIATION_VALID',selected_candidate=dict(bbox_xyxy_pixels=box)),truth)
            results.append(dict(run=str(root),source_frame_id=frame.frame_id,request_id=request_id,source_identity=identity,attribute_audit=audit))
    status='PASS' if all(c['status']=='PASS' for c in checks) else 'SEALED_INPUT_FAILURE'
    report=dict(kind='T1_SOURCE_ATTRIBUTE_DEVELOPMENT_REPLAY',status=status,checks=checks,results=results,
        prospective_flight_validated=False,formal_efficacy_results=False,
        scope='Retrospective development sources used after failure diagnosis; not independent generalization or counterfactual closed-loop performance',
        ai_assistance='AI implemented guard and replay; author reviews conclusions')
    save(output/'report.json',report);sources=[]
    for name in ('replay_t1_source_attributes.py','t1_source_attribute_guard.py'):
        file=Path(__file__).with_name(name);snap=output/name;shutil.copyfile(file,snap);sources.append(dict(path=str(file.resolve()),snapshot_path=str(snap),sha256=digest(file)))
    save(output/'artifact_receipt.json',dict(sources=sources,report_sha256=digest(output/'report.json')))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=replay(args.runs,args.output)
    print(json.dumps(dict(status=result['status'],semantic_sources=len(result['results']),accepted=sum(r['attribute_audit']['status']=='ACCEPTED' for r in result['results']),rejected=sum(r['attribute_audit']['status']=='REJECTED' for r in result['results']))))
