"""Replay closed child events into parent evidence after RPC ownership transfer.

Original child files and their clocks remain authoritative. The later parent
journal admission clock denotes replay, not execution or measurement time.
"""
import json
from pathlib import Path
import time


def replay_closed_trace(controller, journal):
    if not controller.can_transfer_rpc():
        raise RuntimeError('PROCESS_TRACE_REPLAY_WITHOUT_VERIFIED_TRANSFER')
    folder = controller.output
    count = 0
    with (folder/'execution.jsonl').open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            row['child_journal_sequence'] = row.pop('journal_sequence')
            row['child_journal_enqueued_monotonic_s'] = row.pop('journal_enqueued_monotonic_s')
            row['process_trace_replay'] = True
            row['process_trace_path'] = str(folder/'execution.jsonl')
            journal.submit(row)
            count += 1
            # Control has stopped before any wait. Bound replay bursts instead
            # of overflowing the parent's unchanged 4096 event capacity.
            if count % 512 == 0:
                wait_for_written(journal)
    wait_for_written(journal)
    return dict(status='REPLAYED_VERIFIED_CLOSED_PROCESS_TRACE', events=count,
        execution_clock_rebased=False, post_control_replay=True,
        original_child_event_path=str(folder/'execution.jsonl'))


def wait_for_written(journal):
    target = journal.submitted
    deadline = time.monotonic()+5.
    while journal.written < target and time.monotonic() < deadline:
        journal.check()
        time.sleep(.005)
    if journal.written < target:
        raise TimeoutError('PROCESS_TRACE_REPLAY_WRITE_TIMEOUT')
