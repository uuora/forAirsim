"""Local, append-only candidate/review storage. Never trains or runs a model."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from language_plan import SYSTEM_PROMPT, validate_plan


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + ' must be a nonempty string')
    return value.strip()


def content_hash(row):
    """Bind reviews to all stored content and provenance, not just the label."""
    payload = {key: value for key, value in row.items() if key != 'content_hash'}
    return hashlib.sha256(_json(payload).encode('utf-8')).hexdigest()


def _read_jsonl(path):
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'{path.name}:{number}: malformed JSON') from exc
        if not isinstance(value, dict):
            raise ValueError(f'{path.name}:{number}: expected an object')
        result.append(value)
    return result


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Read existing content before appending, so a partial prior write stays visible.
    _read_jsonl(path)
    prefix = b''
    if path.exists() and path.stat().st_size:
        with path.open('rb') as existing:
            existing.seek(-1, os.SEEK_END)
            if existing.read(1) != b'\n':
                prefix = b'\n'
    payload = prefix + (_json(value) + '\n').encode('utf-8')
    with path.open('ab') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_row(row):
    for name in ('id', 'scenario_group', 'split', 'source'):
        _nonempty(row.get(name), name)
    messages = row.get('messages')
    if not isinstance(messages, list) or len(messages) not in (2, 3):
        raise ValueError('A row requires system/user messages and an optional proposed assistant')
    expected_roles = ['system', 'user', 'assistant'][:len(messages)]
    if any(not isinstance(message, dict) or message.get('role') != role
           for message, role in zip(messages, expected_roles)):
        raise ValueError('Message roles must be system, user, then optional assistant')
    for message in messages:
        _nonempty(message.get('content'), 'message content')
    if len(messages) == 3:
        validate_plan(json.loads(messages[2]['content']))


class DatasetStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.seed_path = self.folder / 'seed.jsonl'
        self.candidate_path = self.folder / 'candidates.jsonl'
        self.review_path = self.folder / 'reviews.jsonl'

    def rows(self):
        result, identifiers = [], set()
        for row in _read_jsonl(self.seed_path) + _read_jsonl(self.candidate_path):
            _validate_row(row)
            if row['id'] in identifiers:
                raise ValueError('Duplicate row id: ' + row['id'])
            identifiers.add(row['id'])
            row['content_hash'] = content_hash(row)
            result.append(row)
        return result

    def _current_row(self, row):
        _validate_row(row)
        supplied_hash = content_hash(row)
        if row.get('content_hash', supplied_hash) != supplied_hash:
            raise ValueError('Row content no longer matches its content_hash')
        for current in self.rows():
            if current['id'] == row['id']:
                if current['content_hash'] != supplied_hash:
                    raise ValueError('Row has changed; reload before reviewing')
                return current
        raise ValueError('Row is not stored in this dataset')

    def _events(self):
        events = _read_jsonl(self.review_path)
        for event in events:
            for field in ('event_id', 'row_id', 'row_hash', 'reviewer', 'reviewed_at',
                          'source', 'split', 'scenario_group'):
                _nonempty(event.get(field), field)
            if event.get('decision') not in ('approved', 'rejected'):
                raise ValueError('Invalid review decision')
            plan = event.get('expected_plan')
            if plan is not None:
                validate_plan(plan)
            elif event['decision'] == 'approved':
                raise ValueError('Approved review requires a valid expected plan')
            if not isinstance(event.get('note'), str):
                raise ValueError('Review note must be a string')
        return events

    @staticmethod
    def _matching_review(row, events):
        matches = [event for event in events
                   if event['row_id'] == row['id']
                   and event['row_hash'] == row['content_hash']
                   and all(event[key] == row[key]
                           for key in ('source', 'split', 'scenario_group'))]
        # File order is the event order; timestamps never reorder human decisions.
        return matches[-1] if matches else None

    def review_for(self, row):
        current = self._current_row(row)
        return self._matching_review(current, self._events())

    def review(self, row, decision, reviewer, expected_plan, note=''):
        current = self._current_row(row)
        if decision not in ('approved', 'rejected'):
            raise ValueError('decision must be approved or rejected')
        reviewer = _nonempty(reviewer, 'reviewer')
        if expected_plan is not None:
            validate_plan(expected_plan)
        elif decision == 'approved':
            raise ValueError('Approval requires a valid expected plan')
        if not isinstance(note, str):
            raise ValueError('note must be a string')
        self._events()
        event = dict(event_id=uuid4().hex, row_id=current['id'],
                     row_hash=current['content_hash'], decision=decision,
                     reviewer=reviewer, expected_plan=expected_plan,
                     note=note, reviewed_at=_now())
        event.update({key: current[key] for key in ('source', 'split', 'scenario_group')})
        _append(self.review_path, event)
        return event

    def collect_candidate(self, text, proposed_plan=None, source='gui_capture',
                          scenario_group=None, *, split='development', source_metadata=None):
        text = _nonempty(text, 'text')
        source = _nonempty(source, 'source')
        if split not in ('development', 'smoke_eval', 'validation', 'test'):
            raise ValueError('Unknown dataset split')
        if proposed_plan is not None:
            validate_plan(proposed_plan)
        if source_metadata is not None and not isinstance(source_metadata, dict):
            raise ValueError('source_metadata must be an object')
        row_id = 'capture_' + uuid4().hex
        group = row_id if scenario_group is None else _nonempty(scenario_group, 'scenario_group')
        if any(row['scenario_group'] == group and row['split'] != split for row in self.rows()):
            raise ValueError('A scenario_group cannot cross dataset splits')
        row = dict(id=row_id, scenario_group=group, split=split, source=source,
                   review_status='pending_human_review', collected_at=_now(),
                   messages=[dict(role='system', content=SYSTEM_PROMPT),
                             dict(role='user', content=text)])
        if source_metadata is not None:
            row['source_metadata'] = source_metadata
        if proposed_plan is not None:
            row['messages'].append(dict(role='assistant', content=_json(proposed_plan)))
        _append(self.candidate_path, row)
        row['content_hash'] = content_hash(row)
        return row

    def export_training(self, output_dir):
        rows, events = self.rows(), self._events()
        reserved_groups = {row['scenario_group'] for row in rows if row['split'] != 'development'}
        # Catch exact instruction leakage even when the collector chooses a new group.
        reserved_prompts = {row['messages'][1]['content'].strip() for row in rows
                            if row['split'] != 'development'}
        records, included, excluded = [], [], []
        for row in rows:
            review = self._matching_review(row, events)
            reason = None
            if row['split'] != 'development':
                reason = 'reserved_split'
            elif row['scenario_group'] in reserved_groups:
                reason = 'scenario_group_leakage'
            elif row['messages'][1]['content'].strip() in reserved_prompts:
                reason = 'reserved_instruction_overlap'
            elif review is None:
                reason = ('stale_review' if any(event['row_id'] == row['id'] for event in events)
                          else 'pending_human_review')
            elif review['decision'] != 'approved':
                reason = 'rejected'
            provenance = {key: row[key] for key in ('id', 'scenario_group', 'split', 'source')}
            provenance['content_hash'] = row['content_hash']
            if reason:
                excluded.append(dict(provenance, reason=reason))
                continue
            messages = row['messages'][:2] + [dict(role='assistant', content=_json(review['expected_plan']))]
            records.append(dict(provenance, messages=messages))
            included.append(dict(provenance, review_event_id=review['event_id'],
                                 reviewer=review['reviewer'], reviewed_at=review['reviewed_at']))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Each export is a new snapshot; an existing training file is never overwritten.
        token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
        output = output_dir / ('sft_' + token + '.jsonl')
        payload = ''.join(_json(record) + '\n' for record in records).encode('utf-8')
        with output.open('xb') as stream:
            stream.write(payload)
        manifest_path = output.with_suffix('.manifest.json')
        counts = {}
        for exclusion in excluded:
            counts[exclusion['reason']] = counts.get(exclusion['reason'], 0) + 1
        manifest = dict(schema_version=1, exported_at=_now(), dataset_folder=str(self.folder.resolve()),
                        output_path=str(output.resolve()), manifest_path=str(manifest_path.resolve()),
                        sha256=hashlib.sha256(payload).hexdigest(), total_rows=len(rows),
                        exported_count=len(records), excluded_count=len(excluded),
                        exclusion_counts=counts, included=included, excluded=excluded,
                        status='ready_for_training_review' if records else 'empty_no_approved_development_rows',
                        review_identity_note='Reviewer names are operator declarations, not authenticated identities.')
        with manifest_path.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return manifest
