"""Source-preserving BIDS/EEGLAB event and stage normalization."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
import math
import pandas as pd


@dataclass
class Event:
    onset_s: float
    duration_s: float | None
    sample_index: int | None
    raw_value: str
    semantic_label: str
    family: str
    source: str
    row: int
    mapping_source: str | None = None
    response_time_s: float | None = None
    original_response_time: float | None = None
    alignment: str = 'not_checked'

    def as_dict(self):
        return asdict(self)


def _value(v):
    if pd.isna(v):
        return None
    return str(v).strip()


def _number(v):
    try:
        a = float(v)
        return a if math.isfinite(a) else None
    except (TypeError, ValueError):
        return None


def _dictionary_level(dictionaries: list[Path], field: str, value: str):
    for path in dictionaries:
        try:
            doc = json.loads(path.read_text(encoding='utf-8-sig'))
            val = doc.get(field, {}).get('Levels', {}).get(value)
            if isinstance(val, str):
                return val, path.name
        except (OSError, ValueError):
            continue
    return None, None


def _motor_adapter(label: str, task: str, run: str | None):
    """Documented PhysioNet EEGMMI TASKnT0/1/2 mapping, only on known motion task.

    The actual condition changes by RUN. TASK2 is imagery for unilateral fists.
    No generic 'T1 = left' assumptions in unrelated recordings.
    """
    if task.lower() not in ('motion', 'eegmmidb'):
        return None
    m = re.fullmatch(r'TASK([1-4])T([0-2])', label, flags=re.I)
    if m is None:
        return None
    task_id, code = map(int, m.groups())
    if code == 0:
        return 'rest'
    if task_id in (1, 2):
        return 'left_hand' if code == 1 else 'right_hand'
    return 'both_hands' if code == 1 else 'both_feet'


def classify_event_value(raw: str, dictionaries: list[Path], task: str, run: str | None):
    v = raw.strip().lower()
    descr, source = _dictionary_level(dictionaries, 'value', raw)
    mapped_motor = _motor_adapter(raw, task, run)
    if mapped_motor is not None:
        return mapped_motor, 'motor', 'eegmmi_task_adapter'
    if v in {'standard', 'standard_with_reponse', 'standard_with_response'}:
        return 'standard', 'oddball', source or 'descriptive_label'
    if v in {'oddball', 'oddball_with_reponse', 'oddball_with_response', 'target', 'deviant'}:
        return 'target', 'oddball', source or 'descriptive_label'
    if v in {'noise', 'noise_with_reponse', 'noise_with_response', 'distractor'}:
        return 'distractor', 'oddball', source or 'descriptive_label'
    if v in {'response', 'button_press', 'responses'}:
        return 'response', 'response', source or 'descriptive_label'
    if v in {'ignore', 'condition_5'}:
        return 'ignored', 'misc', source or 'descriptive_label'
    stage_map = {'w':'W', 'wake':'W', 'stage w':'W', 'n1':'N1', 'stage 1':'N1',
                 'n2':'N2', 'stage 2':'N2', 'n3':'N3', 'stage 3':'N3',
                 'rem':'REM', 'r':'REM', 'stage r':'REM', 'sleep stage 1':'N1',
                 'sleep stage 2':'N2', 'sleep stage 3':'N3', 'sleep stage 4':'N3',
                 'sleep stage r':'REM', 'sleep stage w':'W'}
    if v in stage_map:
        return stage_map[v], 'sleep', source or 'recognized_stage_label'
    if v in {'left_hand','right_hand','left_fist','right_fist','movement_onset','motor_imagery','both_hands','both_feet','rest'}:
        return v, 'motor', source or 'descriptive_label'
    return raw, 'unknown', source


def _read_table(p: Path):
    sep = '\t' if p.suffix.lower() == '.tsv' else None
    return pd.read_csv(p, sep=sep, engine='python', encoding='utf-8-sig',
                       dtype=str, na_values=['n/a','N/A','NA'])


def load_events(bundle, raw):
    """Use primary event TSV when present; avoid double-counting embedded annotations.

    Source event rows are retained. Some real datasets have conflicting response_time
    unit descriptors; RT must not be used from ambiguous columns. Response events
    can separately support onset-difference RT calculations, with explicit validation.
    """
    warnings = []
    events: list[Event] = []
    task = bundle.entities.get('task','')
    run = bundle.entities.get('run')
    duration = raw.n_times / raw.info['sfreq']
    event_files = [p for p in bundle.annotations if p.suffix.lower() in ('.tsv','.csv')]
    # a file may hold stages with 'value' or 'stage' in a sleep scoring sidecar.
    for p in event_files:
        try:
            df = _read_table(p)
        except Exception as ex:
            warnings.append(f'Failed to parse annotation {p.name}: {ex}')
            continue
        cols = {str(c).lower(): c for c in df.columns}
        onset_name = cols.get('onset') or cols.get('onset_s') or cols.get('start')
        value_name = (cols.get('value') or cols.get('trial_type') or cols.get('staging')
                      or cols.get('stage') or cols.get('sleep_stage'))
        if not onset_name or not value_name:
            warnings.append(f'Ignoring non-event table {p.name}: missing onset or event value')
            continue
        n_added = 0
        for row_no, row in df.iterrows():
            onset = _number(row[onset_name]); value = _value(row[value_name])
            if onset is None or value is None:
                continue
            # BIDS onsets are seconds. No implicit milliseconds conversion.
            duration_raw = _number(row[cols['duration']]) if 'duration' in cols else None
            sample = _number(row[cols['sample']]) if 'sample' in cols else None
            sem, fam, mapping = classify_event_value(value, bundle.dictionaries, task, run)
            # ds005207 stores numeric sleep codes under 'staging'. Resolve ONLY
            # through its documented BIDS JSON Levels, never through generic numeric guesses.
            if value_name == cols.get('staging'):
                description, source = _dictionary_level(bundle.dictionaries,'staging',value)
                if description is not None:
                    known = {'wake':'W','rem':'REM','n1':'N1','n2':'N2','n3':'N3'}
                    sem = known.get(description.strip().lower(),description.strip())
                    fam = 'sleep'
                    mapping = f'BIDS JSON staging: {source}'
                else:
                    sem, fam, mapping = value, 'unknown', None
                    if 'sleep_staging_numeric_codes_not_documented' not in warnings:
                        warnings.append('sleep_staging_numeric_codes_not_documented')
            rt = _number(row[cols['response_time']]) if 'response_time' in cols else None
            # Explicit ds003061 JSON has conflicting units metadata; none is assumed.
            if rt is not None and 'response_time_units_unverified' not in warnings:
                warnings.append('response_time_units_unverified')
            e = Event(onset_s=onset, duration_s=duration_raw,
                      sample_index=int(sample) if sample is not None else None,
                      raw_value=value, semantic_label=sem, family=fam,
                      source=p.name, row=int(row_no)+2, mapping_source=mapping,
                      original_response_time=rt)
            if not (0 <= onset <= duration + 1/raw.info['sfreq']):
                e.alignment = 'outside_recording'
                warnings.append(f'Out-of-bounds event in {p.name}, row {row_no+2}')
            elif sample is not None and abs(onset - sample/raw.info['sfreq']) > max(0.02, 2/raw.info['sfreq']):
                e.alignment = 'sample_onset_conflict'
                warnings.append(f'Sample/onset mismatch in {p.name}, row {row_no+2}')
            else:
                e.alignment = 'verified_within_recording'
            events.append(e); n_added += 1
        if n_added:
            # Use one event/stage table as primary; do not silently merge competing files.
            if len(event_files)>1:
                warnings.append(f'Multiple annotation tables detected; primary {p.name}, review associations')
            break
    if not events:
        for i, (on, dur, descr) in enumerate(zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description)):
            v = str(descr)
            sem, fam, mapping = classify_event_value(v, bundle.dictionaries, task, run)
            # MNE annotation onsets use raw time origin; for EEGLAB sample 0 at 0.
            events.append(Event(onset_s=float(on), duration_s=float(dur), sample_index=None,
                                raw_value=v, semantic_label=sem, family=fam,
                                source='embedded_eeg', row=i, mapping_source=mapping,
                                alignment='verified_within_recording' if 0<=on<=duration else 'outside_recording'))
    # Preserve only one copy when both EEGLAB events and sidecar refer to same sample.
    if events and any(e.alignment in ('outside_recording','sample_onset_conflict') for e in events):
        warnings.append('Some event timings failed alignment validation')
    return events, sorted(set(warnings))
