"""Disk backpressure, immutable byte bindings and true completion clocks."""
import copy,json,tempfile,time,threading,unittest
from pathlib import Path
import cv2
import numpy as np

from t1_bounded_evidence_v1 import BoundedFrameEvidence,BoundedEventJournal,digest
from t1_timed_review_observer_v1 import TimedMethods,TimedAttributeBackend,TimedReviewFlightObserver
from audit_t1_persistence_v1 import audit
from t1_control_settlement import completion_timing

ROOT=Path(__file__).resolve().parents[1]


class BoundedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='bounded_evidence_test_',dir=ROOT/'logs')
        self.folder=Path(self.tmp.name).resolve();(self.folder/'frames').mkdir()
        self.workers=[];self.releases=[]
        self.depth=np.ones((12,16),np.float64);self.seg=np.zeros((12,16,3),np.uint8)
        self.palette={42:[255,0,0],43:[0,255,0],44:[0,0,255]}

    def tearDown(self):
        for release in self.releases:release.set()
        for worker in self.workers:worker.close(timeout=3.)
        assert self.folder.is_relative_to((ROOT/'logs').resolve());self.tmp.cleanup()

    def frame(self,number=0):
        name=f'test_f{number:03d}';p=self.folder/'frames'/(name+'_rgb.png')
        self.assertTrue(cv2.imwrite(str(p),np.full((12,16,3),25,np.uint8)))
        now=time.monotonic()
        return dict(frame_id=name,slot_index=number,phase_id='visible',session_id='test',status='CAPTURED_EVIDENCE_PENDING',
            experiment_scope='MEASURED_RESEARCH',media={'rgb':{'path':str(p),'sha256':digest(p)}},
            capture_started_host_s=now,capture_received_host_s=now,observer_result={'anchor':{'candidate':1}},
            surface={'status':'SURFACE_REJECTED'},motion_gate={'status':'ZERO'},command=None)

    def writer(self,hook=None):
        w=BoundedFrameEvidence(self.folder,media_hook=hook);self.workers.append(w);return w

    def submit(self,w,row):
        ticket=w.submit(row,self.depth,self.seg,self.palette)
        ticket.complete({'producer_pipeline_completed_monotonic_s':time.monotonic()});return ticket

    def wait(self,predicate,timeout=3.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.005)
        self.fail('Bounded worker did not reach expected state')

    def completed_report(self):
        w=self.writer();self.submit(w,self.frame());w.close();frames=w.drain()
        j=BoundedEventJournal(self.folder/'execution.jsonl');self.workers.append(j)
        j.submit({'kind':'SELF_STATE','stage':'TEST','state_timestamp_ns':1});journal=j.close()
        return dict(frames=frames,acquisition_frames=[],evidence_writer=w.receipt(),event_journal_close=journal)

    def test_queue_admission_does_not_hide_slow_media_or_late_completion(self):
        entered=threading.Event();release=threading.Event();self.releases.append(release)
        def hold(row):entered.set();release.wait(2.)
        w=self.writer(hold);row=self.frame();due=time.monotonic();self.submit(w,row)
        self.assertTrue(entered.wait(1.));self.assertEqual(w.drain(),[])
        time.sleep(.42);release.set();w.close();saved=w.drain()[0]
        self.assertGreaterEqual(saved['frame_processing_completed_monotonic_s'],due+.4)
        saved.update(due_monotonic_s=due,deadline_monotonic_s=due+.4)
        self.assertEqual(completion_timing(saved)['frame_processing_timing'],'LATE')
        self.assertEqual(completion_timing(saved)['historical_full_completion_timing'],'LATE')

    def test_three_job_bound_includes_completed_but_undrained_receipts(self):
        w=self.writer()
        for i in range(3):self.submit(w,self.frame(i))
        self.wait(lambda:w.completed==3)
        with self.assertRaisesRegex(RuntimeError,'EVIDENCE_INFLIGHT_BUFFER_FULL'):w.submit(self.frame(3),self.depth,self.seg,self.palette)
        self.assertEqual(w.maximum_inflight,3)
        self.assertEqual(len(w.drain()),3);self.submit(w,self.frame(4));w.close()
        self.assertEqual(len(w.drain()),1)

    def test_snapshot_isolated_from_later_input_mutation(self):
        entered=threading.Event();release=threading.Event();self.releases.append(release)
        def hold(row):entered.set();release.wait(2.)
        w=self.writer(hold);row=self.frame();self.submit(w,row);self.assertTrue(entered.wait(1.))
        self.depth[:]=-99;self.seg[:]=[0,0,255];row['observer_result']['anchor']['candidate']=2
        release.set();w.close();saved=w.drain()[0]
        self.assertEqual(saved['observer_result']['anchor']['candidate'],1)
        self.assertTrue(np.all(np.load(saved['media']['depth']['path'])==1))
        self.assertEqual(saved['evaluation_truth']['target']['visible_pixel_count'],0)

    def test_media_ready_still_waits_for_actual_producer_completion(self):
        w=self.writer();row=self.frame();ticket=w.submit(row,self.depth,self.seg,self.palette)
        time.sleep(.04);self.assertEqual(w.drain(),[])
        producer=time.monotonic();ticket.complete({'producer_pipeline_completed_monotonic_s':producer})
        w.close();saved=w.drain()[0]
        self.assertGreaterEqual(saved['frame_processing_completed_monotonic_s'],producer)

    def test_write_failure_exposes_category_and_no_completion(self):
        def fail(row):raise OSError('secret-like internal detail must not enter receipt')
        w=self.writer(fail);self.submit(w,self.frame());w.close()
        self.assertEqual(w.drain(),[])
        self.assertEqual(w.receipt()['error_type'],'OSError')
        self.assertNotIn('secret-like',json.dumps(w.receipt()))
        with self.assertRaisesRegex(RuntimeError,'EVIDENCE_WORKER_FAILED'):w.check()

    def test_queued_changed_rgb_fails_before_evaluation(self):
        entered=threading.Event();release=threading.Event();self.releases.append(release)
        def hold(row):
            if row['slot_index']==0:entered.set();release.wait(2.)
        w=self.writer(hold);self.submit(w,self.frame());self.assertTrue(entered.wait(1.))
        second=self.frame(1);self.submit(w,second);Path(second['media']['rgb']['path']).write_bytes(b'changed')
        release.set();w.close();saved=w.drain()
        self.assertEqual([r['slot_index'] for r in saved],[0]);self.assertEqual(w.receipt()['status'],'FAILED')

    def test_auditor_accepts_exact_bytes_and_rejects_queue_clock_substitution(self):
        report=self.completed_report();self.assertEqual(audit(report)['status'],'PASS_ACTUAL_BOUNDED_PERSISTENCE')
        changed=copy.deepcopy(report);changed['frames'][0]['frame_processing_completed_monotonic_s']-=.01
        self.assertEqual(audit(changed)['status'],'FAIL_OR_MISSING_PERSISTENCE_EVIDENCE')

    def test_media_record_byte_tampering_is_rejected(self):
        report=self.completed_report();p=Path(report['frames'][0]['evidence_media_record']['path'])
        data=bytearray(p.read_bytes());data[0]=ord('[');p.write_bytes(data)
        self.assertEqual(audit(report)['status'],'FAIL_OR_MISSING_PERSISTENCE_EVIDENCE')

    def test_event_batch_byte_tampering_is_rejected(self):
        report=self.completed_report();Path(report['event_journal_close']['event_path']).write_bytes(b'{}\n')
        self.assertEqual(audit(report)['status'],'FAIL_OR_MISSING_PERSISTENCE_EVIDENCE')

    def test_event_journal_conserves_fifo_order_and_exact_byte_coverage(self):
        j=BoundedEventJournal(self.folder/'events.jsonl');self.workers.append(j)
        for i in range(257):j.submit({'kind':'TEST','input_sequence':i})
        receipt=j.close();rows=[json.loads(x) for x in j.path.read_text().splitlines()]
        self.assertEqual([r['input_sequence'] for r in rows],list(range(257)))
        self.assertEqual([r['journal_sequence'] for r in rows],list(range(257)))
        self.assertEqual(receipt['written'],257)
        raw=j.path.read_bytes();batches=[json.loads(x) for x in j.flush_path.read_text().splitlines()]
        self.assertEqual(sum(r['byte_count'] for r in batches),len(raw))
        import hashlib
        for r in batches:self.assertEqual(hashlib.sha256(raw[r['byte_offset']:r['byte_offset']+r['byte_count']]).hexdigest(),r['payload_sha256'])

    def test_event_backpressure_is_bounded_and_failure_remains_visible(self):
        entered=threading.Event();release=threading.Event();self.releases.append(release)
        def hold():entered.set();release.wait(2.)
        j=BoundedEventJournal(self.folder/'events.jsonl',write_hook=hold);self.workers.append(j)
        j.submit({'kind':'TEST'});self.assertTrue(entered.wait(1.))
        for i in range(4096):j.submit({'kind':'TEST','index':i})
        with self.assertRaisesRegex(RuntimeError,'CONTROL_EVENT_BUFFER_OVERFLOW'):j.submit({'kind':'REJECTED'})
        release.set();receipt=j.close()
        self.assertEqual(receipt['status'],'FAILED');self.assertTrue(receipt['overflow'])
        self.assertEqual(receipt['submitted'],receipt['written'])

    def test_nested_timers_preserve_original_objects_and_exception(self):
        class Owner:
            def inner(self,value):return value
            def outer(self,value):return self.inner(value)
        owner=Owner();timer=TimedMethods();timer.wrap(owner,'inner','INNER');timer.wrap(owner,'outer','OUTER')
        timer.active=[];obj={'readback_receipt_monotonic_s':1.};self.assertIs(owner.outer(obj),obj)
        self.assertEqual(obj,{'readback_receipt_monotonic_s':1.})
        spans=timer.active;self.assertEqual([r['component'] for r in spans],['INNER','OUTER'])
        self.assertGreaterEqual(spans[-1]['completed_monotonic_s'],spans[0]['completed_monotonic_s'])

    def test_timed_attribute_guard_preserves_accepted_reply_and_source_binding(self):
        from test_t1_source_attribute_guard import request_reply
        image=np.full((100,100,3),(30,30,220),np.uint8);image[40:60,10:90]=230
        request,reply=request_reply(image)
        class Backend:
            vlx_client=None
            active_record={}
            def __call__(self,request):return reply
            def mark(self,record,**values):record.update(values)
        backend=Backend();self.assertIs(TimedAttributeBackend(backend)(request),reply)
        trace=backend.active_record
        self.assertEqual(trace['source_attribute_audit']['request_binding'],request.binding())
        self.assertEqual(trace['source_attribute_guard_status'],'ACCEPTED')
        self.assertLessEqual(trace['source_attribute_guard_started_monotonic_s'],trace['source_attribute_guard_completed_monotonic_s'])

    def test_timed_attribute_guard_rejects_plain_red_without_reply_rewrite(self):
        from test_t1_source_attribute_guard import request_reply
        from t1_source_attribute_guard import SourceAttributeRejection
        request,reply=request_reply(np.full((100,100,3),(30,30,220),np.uint8))
        original=copy.deepcopy(reply.result)
        class Backend:
            vlx_client=None
            active_record={}
            def __call__(self,request):return reply
            def mark(self,record,**values):record.update(values)
        backend=Backend()
        with self.assertRaises(SourceAttributeRejection):TimedAttributeBackend(backend)(request)
        self.assertEqual(backend.active_record['source_attribute_guard_status'],'REJECTED')
        self.assertEqual(reply.result,original)

    def test_timed_observer_factory_composition_does_not_call_network(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Detector:
            def detect(self,value):return value
        detector=Detector()
        class Backend:
            vlx_client=SimpleNamespace(session=object())
            def __init__(self):self.detector=detector
            def detect_frame(self,value):return self.detector.detect(value)
        original=SimpleNamespace(backend=Backend(),model_receipt={'synthetic':True})
        with patch('t1_delayed_semantic_observer.create_observer',return_value=original):
            observer=TimedReviewFlightObserver('synthetic')
        observer.timing.active=[];obj={'source_time':1.}
        self.assertIs(observer.inner.core.backend.detect_frame(obj),obj)
        self.assertEqual([r['component'] for r in observer.timing.active],['YOLO_DETECT_CALL','SAVED_RGB_READ_DETECTION_AND_CANDIDATES'])
        self.assertEqual(obj,{'source_time':1.})


if __name__=='__main__':unittest.main()
