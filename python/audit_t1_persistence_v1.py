"""Read-only byte/clock audit; queue admission is never a completion proof."""
from pathlib import Path
import hashlib,json,math


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(report):
    issues=[];frames=report.get('acquisition_frames',[])+report.get('frames',[])
    media_ids=set();event_count=0
    for row in frames:
        frame_id=row.get('frame_id');proof=row.get('evidence_media_record',{})
        try:
            if row.get('status')!='FRAME_COMPLETED':raise ValueError('FRAME_NOT_ACTUALLY_PERSISTED')
            path=Path(proof['path'])
            with path.open('rb') as stream:stream.seek(proof['byte_offset']);line=stream.read(proof['byte_count'])
            if hashlib.sha256(line).hexdigest()!=proof['line_sha256'] or not line.endswith(b'\n'):
                raise ValueError('IMMUTABLE_MEDIA_RECORD_BYTES_CHANGED')
            original=json.loads(line)
            for key in ('frame_id','slot_index','phase_id','session_id','capture_started_host_s','capture_received_host_s',
                'observer_result','surface','command','motion_gate','media','evaluation_truth','producer_pipeline_completed_monotonic_s'):
                if original.get(key)!=row.get(key):raise ValueError('MEDIA_RECORD_FRAME_BINDING_CHANGED')
            if frame_id in media_ids:raise ValueError('DUPLICATE_MEDIA_FRAME_RECORD')
            media_ids.add(frame_id)
            for item in row['media'].values():
                if digest(item['path'])!=item['sha256']:raise ValueError('SAVED_MEDIA_DIGEST_CHANGED')
            done=row['frame_processing_completed_monotonic_s'];flushed=proof['flush_completed_monotonic_s']
            producer=row['producer_pipeline_completed_monotonic_s']
            if (not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (done,flushed,producer))
                or done!=max(flushed,producer) or flushed<row['depth_segmentation_persistence_completed_monotonic_s']):
                raise ValueError('ACTUAL_MEDIA_COMPLETION_CLOCK_REQUIRED')
        except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
            issues.append(dict(frame_id=frame_id,issue=str(error) if isinstance(error,ValueError) else type(error).__name__))
    writer=report.get('evidence_writer',{})
    if (writer.get('status')!='CLOSED' or writer.get('thread_alive') is not False or writer.get('max_inflight_frames')!=3
        or writer.get('maximum_observed_inflight',4)>3 or writer.get('accepted')!=len(frames)
        or writer.get('completed')!=len(frames) or writer.get('collected')!=len(frames)):
        issues.append(dict(issue='FRAME_WRITER_CLOSURE_OR_DENOMINATOR_MISMATCH'))
    journal=report.get('event_journal_close',{})
    try:
        raw=Path(journal['event_path']).read_bytes();receipts=Path(journal['flush_receipt_path']).read_text(encoding='utf-8').splitlines()
        offset=0;sequence=0
        for line in receipts:
            receipt=json.loads(line);chunk=raw[receipt['byte_offset']:receipt['byte_offset']+receipt['byte_count']]
            if (receipt['byte_offset']!=offset or hashlib.sha256(chunk).hexdigest()!=receipt['payload_sha256']
                or receipt['flush_completed_monotonic_s']<receipt['serialization_write_started_monotonic_s']):
                raise ValueError('EVENT_BATCH_BYTES_OR_CLOCK_CHANGED')
            events=[json.loads(x) for x in chunk.splitlines()]
            if not events or receipt['first_sequence']!=sequence or receipt['last_sequence']!=sequence+len(events)-1:
                raise ValueError('EVENT_BATCH_SEQUENCE_MISMATCH')
            for event in events:
                if event['journal_sequence']!=sequence or event['journal_enqueued_monotonic_s']>receipt['flush_completed_monotonic_s']:
                    raise ValueError('EVENT_SEQUENCE_OR_FLUSH_CLOCK_INVALID')
                sequence+=1
            offset+=receipt['byte_count']
        if (offset!=len(raw) or journal.get('status')!='CLOSED' or journal.get('thread_alive') is not False
            or journal.get('capacity')!=4096 or journal.get('overflow') is not False
            or sequence!=journal.get('submitted') or sequence!=journal.get('written')):
            raise ValueError('EVENT_JOURNAL_CLOSURE_OR_COVERAGE_MISMATCH')
        event_count=sequence
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
        issues.append(dict(issue=str(error) if isinstance(error,ValueError) else type(error).__name__))
    return dict(status='PASS_ACTUAL_BOUNDED_PERSISTENCE' if not issues and frames else 'FAIL_OR_MISSING_PERSISTENCE_EVIDENCE',
        audited_frames=len(frames),audited_events=event_count,issues=issues,queue_admission_used_as_completion=False,
        file_flush_scope='Python flush only; fsync/power-loss durability not established',new_flights=0)
