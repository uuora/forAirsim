"""Bounded disk workers; no simulator, detector, network or motion authority."""
from __future__ import annotations

import copy
from dataclasses import dataclass,field
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import threading
import time

import cv2
import numpy as np


def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('UNSUPPORTED_EVIDENCE_JSON_VALUE')


def encoded(row):
    return (json.dumps(row,ensure_ascii=False,allow_nan=False,default=scalar)+'\n').encode('utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


@dataclass
class ProducerTicket:
    frame_id:str
    _ready:threading.Event=field(default_factory=threading.Event,repr=False)
    _metadata:dict|None=field(default=None,repr=False)

    def complete(self,metadata):
        if self._ready.is_set():raise RuntimeError('PRODUCER_COMPLETION_ALREADY_BOUND')
        done=metadata.get('producer_pipeline_completed_monotonic_s')
        if not isinstance(done,(int,float)) or isinstance(done,bool) or not math.isfinite(done):
            raise ValueError('ACTUAL_PRODUCER_COMPLETION_CLOCK_REQUIRED')
        self._metadata=copy.deepcopy(metadata);self._ready.set()


class BoundedFrameEvidence:
    """At most three queued, active or undrained jobs in total; receipt is actual."""
    def __init__(self,output,*,max_inflight=3,clock=time.monotonic,media_hook=None):
        if max_inflight!=3:raise ValueError('FROZEN_THREE_FRAME_INFLIGHT_BOUND_REQUIRED')
        self.output=Path(output).resolve();self.clock=clock;self.media_hook=media_hook
        if not (self.output/'frames').is_dir():raise ValueError('EXISTING_FRAME_OUTPUT_DIRECTORY_REQUIRED')
        self.records_path=self.output/'evidence_media_records.jsonl'
        if self.records_path.exists():raise FileExistsError('NEW_MEDIA_RECORDS_REQUIRED')
        self._slots=threading.BoundedSemaphore(max_inflight)
        self._jobs=queue.Queue(maxsize=max_inflight);self._done=queue.Queue(maxsize=max_inflight)
        self._stop=threading.Event();self._failed=threading.Event();self._error_type=None
        self._ids=set();self.accepted=0;self.completed=0;self.collected=0;self.maximum_inflight=0
        self._thread=threading.Thread(target=self._work,name='T1EvidenceDisk',daemon=True);self._thread.start()

    def check(self):
        if self._failed.is_set():raise RuntimeError('EVIDENCE_WORKER_FAILED')
        if self._stop.is_set():raise RuntimeError('EVIDENCE_WORKER_CLOSED')

    def submit(self,row,depth,segmentation,palette):
        self.check()
        frame_id=row.get('frame_id')
        if not isinstance(frame_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',frame_id) or frame_id in self._ids:
            raise ValueError('UNIQUE_SAFE_EVIDENCE_FRAME_ID_REQUIRED')
        rgb=row.get('media',{}).get('rgb',{})
        if Path(rgb.get('path','')).resolve()!=self.output/'frames'/(frame_id+'_rgb.png'):
            raise ValueError('BOUND_RGB_FRAME_PATH_REQUIRED')
        if not isinstance(depth,np.ndarray) or not isinstance(segmentation,np.ndarray) or depth.shape!=segmentation.shape[:2]:
            raise ValueError('BOUND_DEPTH_SEGMENTATION_GEOMETRY_REQUIRED')
        if not self._slots.acquire(blocking=False):raise RuntimeError('EVIDENCE_INFLIGHT_BUFFER_FULL')
        try:
            snapshot=copy.deepcopy(row);ticket=ProducerTicket(frame_id)
            job=(snapshot,np.array(depth,copy=True),np.array(segmentation,copy=True),copy.deepcopy(palette),ticket)
            self._jobs.put_nowait(job);self._ids.add(frame_id);self.accepted+=1
            self.maximum_inflight=max(self.maximum_inflight,self.accepted-self.collected)
            return ticket
        except Exception:
            self._slots.release();raise

    def drain(self):
        rows=[]
        while True:
            try:row=self._done.get_nowait()
            except queue.Empty:break
            rows.append(row);self.collected+=1;self._slots.release()
        return rows

    def _persist(self,row,depth,seg,palette,ticket,records):
        from t1_ground_perception_probe import visible_mask
        row['evidence_worker_started_monotonic_s']=self.clock()
        rgb=row['media']['rgb']
        if digest(rgb['path'])!=rgb['sha256']:raise ValueError('RGB_CHANGED_BEFORE_MEDIA_PERSISTENCE')
        if self.media_hook:self.media_hook(row)
        row['truth_evaluation_started_monotonic_s']=self.clock()
        row['evaluation_truth']={key:visible_mask(seg,palette[n]) for key,n in [('target',42),('distractor',43),('occluder',44)]}
        row['truth_evaluation_completed_monotonic_s']=self.clock()
        row['depth_segmentation_persistence_started_monotonic_s']=self.clock()
        for key,suffix,values in [('depth','_depth.npy',depth),('segmentation','_seg.png',seg)]:
            path=self.output/'frames'/(row['frame_id']+suffix)
            if path.exists():raise FileExistsError('MEDIA_FRAME_ALREADY_EXISTS')
            if key=='depth':
                with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
            elif not cv2.imwrite(str(path),values,[cv2.IMWRITE_PNG_COMPRESSION,0]):
                raise RuntimeError('SEGMENTATION_SAVE_FAILED')
            row['media'][key]=dict(path=str(path),sha256=digest(path))
        row['depth_segmentation_persistence_completed_monotonic_s']=self.clock()
        if not ticket._ready.wait(1.):raise RuntimeError('PRODUCER_COMPLETION_BINDING_MISSING')
        row.update(ticket._metadata)
        if ticket._metadata['producer_pipeline_completed_monotonic_s']>self.clock():raise ValueError('FUTURE_PRODUCER_CLOCK')
        row['evidence_record_serialization_started_monotonic_s']=self.clock()
        line=encoded(dict(row,status='MEDIA_READY_BEFORE_RECORD_FLUSH'))
        row['evidence_record_serialization_completed_monotonic_s']=self.clock()
        offset=records.tell();records.write(line);records.flush()
        flushed=self.clock()
        row['evidence_media_record']=dict(path=str(self.records_path),byte_offset=offset,byte_count=len(line),line_sha256=hashlib.sha256(line).hexdigest(),
            flush_completed_monotonic_s=flushed,scope='Python file write and flush; not fsync or power-loss durability')
        row['frame_processing_completed_monotonic_s']=max(flushed,row['producer_pipeline_completed_monotonic_s'])
        row['status']='FRAME_COMPLETED'
        return row

    def _work(self):
        try:
            with self.records_path.open('xb') as records:
                while not self._stop.is_set() or not self._jobs.empty():
                    try:row,depth,seg,palette,ticket=self._jobs.get(timeout=.01)
                    except queue.Empty:continue
                    result=self._persist(row,depth,seg,palette,ticket,records)
                    # This later progress copy does not redefine media completion.
                    name='acquisition_frame_results.jsonl' if row.get('experiment_scope')=='INITIAL_IDENTITY_ACQUISITION' else 'frame_results.jsonl'
                    with (self.output/name).open('ab') as progress:progress.write(encoded(result));progress.flush()
                    self.completed+=1;self._done.put_nowait(result)
        except Exception as error:
            self._error_type=type(error).__name__;self._failed.set()

    def close(self,timeout=5.):
        self._stop.set();self._thread.join(timeout)
        if self._thread.is_alive():raise RuntimeError('EVIDENCE_WORKER_JOIN_TIMEOUT')
        return self.receipt()

    def receipt(self):
        return dict(status='FAILED' if self._failed.is_set() else 'CLOSED' if not self._thread.is_alive() else 'RUNNING',
            max_inflight_frames=3,maximum_observed_inflight=self.maximum_inflight,accepted=self.accepted,completed=self.completed,collected=self.collected,
            error_type=self._error_type,records_path=str(self.records_path),thread_alive=self._thread.is_alive(),motion_authority=False)


class BoundedEventJournal:
    """Single owner serializes event batches; flush receipts bind exact bytes."""
    def __init__(self,path,*,capacity=4096,clock=time.monotonic,write_hook=None):
        if capacity!=4096:raise ValueError('FROZEN_EVENT_CAPACITY_REQUIRED')
        self.path=Path(path);self.flush_path=self.path.with_name('event_journal_flush_receipts.jsonl')
        if self.path.exists() or self.flush_path.exists():raise FileExistsError('NEW_EVENT_JOURNAL_REQUIRED')
        self.clock=clock;self.write_hook=write_hook;self.queue=queue.Queue(maxsize=capacity)
        self._stop=threading.Event();self._failed=threading.Event();self.overflow=threading.Event();self._lock=threading.Lock()
        self.submitted=0;self.written=0;self.flush_count=0;self.error_type=None
        self._thread=threading.Thread(target=self._work,name='T1EventDisk',daemon=True);self._thread.start()

    def check(self):
        if self._failed.is_set():raise RuntimeError('EVENT_JOURNAL_WRITE_FAILED')
        if self.overflow.is_set():raise RuntimeError('CONTROL_EVENT_BUFFER_OVERFLOW')

    def submit(self,row):
        self.check()
        if self._stop.is_set():raise RuntimeError('EVENT_JOURNAL_CLOSED')
        with self._lock:
            item=copy.deepcopy(row);item['journal_sequence']=self.submitted
            item['journal_enqueued_monotonic_s']=self.clock()
            try:self.queue.put_nowait(item)
            except queue.Full:
                self.overflow.set();raise RuntimeError('CONTROL_EVENT_BUFFER_OVERFLOW')
            self.submitted+=1

    def _work(self):
        try:
            with self.path.open('xb') as events,self.flush_path.open('xb') as receipts:
                while not self._stop.is_set() or not self.queue.empty():
                    try:first=self.queue.get(timeout=.01)
                    except queue.Empty:continue
                    batch=[first]
                    while len(batch)<128:
                        try:batch.append(self.queue.get_nowait())
                        except queue.Empty:break
                    started=self.clock()
                    if self.write_hook:self.write_hook()
                    payload=b''.join(encoded(row) for row in batch);offset=events.tell()
                    events.write(payload);events.flush();completed=self.clock()
                    receipts.write(encoded(dict(first_sequence=batch[0]['journal_sequence'],last_sequence=batch[-1]['journal_sequence'],
                        byte_offset=offset,byte_count=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),
                        serialization_write_started_monotonic_s=started,flush_completed_monotonic_s=completed,
                        scope='serialization plus Python write and flush; not fsync')));receipts.flush()
                    self.written+=len(batch);self.flush_count+=1
        except Exception as error:
            self.error_type=type(error).__name__;self._failed.set()

    def close(self,timeout=5.):
        self._stop.set();self._thread.join(timeout)
        if self._thread.is_alive():raise RuntimeError('EVENT_JOURNAL_JOIN_TIMEOUT')
        return self.receipt()

    def receipt(self):
        return dict(status='FAILED' if self._failed.is_set() or self.overflow.is_set() else 'CLOSED' if not self._thread.is_alive() else 'RUNNING',
            capacity=4096,submitted=self.submitted,written=self.written,flush_batches=self.flush_count,
            error_type=self.error_type,overflow=self.overflow.is_set(),thread_alive=self._thread.is_alive(),
            event_path=str(self.path),flush_receipt_path=str(self.flush_path),motion_authority=False)
