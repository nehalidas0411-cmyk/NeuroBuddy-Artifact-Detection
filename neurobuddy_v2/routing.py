"""Deterministic, auditable task decision intelligence."""
from dataclasses import dataclass, asdict

@dataclass
class Decision:
    task_family: str
    subtype: str | None
    evidence: list[str]
    confidence_tier: str
    analysis_readiness: dict[str, bool]
    status: str
    conflicts: list[str]

    def as_dict(self):
        return asdict(self)


def decide(bundle, events, warnings=None):
    warnings = warnings or []
    meta_task = bundle.entities.get('task','').lower()
    run = bundle.entities.get('run')
    kinds = {e.family for e in events if e.alignment == 'verified_within_recording'}
    sems = {e.semantic_label for e in events if e.alignment == 'verified_within_recording'}
    reasons, conflicts = [], list(bundle.conflicts)
    candidates = set()
    if 'p300' in meta_task or 'oddball' in meta_task:
        candidates.add('ODDBALL');reasons.append('BIDS task indicates P300/oddball')
    if any(x in meta_task for x in ('motion','motor','imagery','exercise')):
        candidates.add('MOTOR');reasons.append('BIDS task indicates motor paradigm')
    if 'sleep' in meta_task:
        candidates.add('SLEEP');reasons.append('BIDS task indicates sleep study')
    if {'target','standard'}.issubset(sems):
        candidates.add('ODDBALL');reasons.append('Documented target and standard events')
    if any(x in sems for x in ('left_hand','right_hand','both_hands','both_feet','motor_imagery')):
        candidates.add('MOTOR');reasons.append('Motor cue semantic labels')
    if {'W','N2'}.issubset(sems) or (kinds & {'sleep'} and len(sems & {'W','N1','N2','N3','REM'}) >= 2):
        candidates.add('SLEEP');reasons.append('Scored sleep-stage sequence')
    if len(candidates)>1:
        conflicts.append('Contradictory specialized task evidence: '+', '.join(sorted(candidates)))
    if any(e.alignment == 'sample_onset_conflict' for e in events):
        conflicts.append('Sample/onset mismatch in event table')
    if any(e.alignment == 'outside_recording' for e in events):
        conflicts.append('Out-of-bounds event times')
    if conflicts:
        family = sorted(candidates)[0] if len(candidates)==1 else 'UNKNOWN'
        return Decision(family, None, reasons, 'conflicting_evidence', {}, 'NEEDS_REVIEW', conflicts)
    family = next(iter(candidates)) if candidates else 'GENERAL'
    subtype = None
    if family == 'ODDBALL': subtype = 'auditory_oddball' if 'p300' in meta_task else 'oddball'
    if family == 'MOTOR':
        if meta_task == 'motion' and run and run.isdigit():
            n = int(run)
            subtype = 'imagery' if n in (4,6,8,10,12,14) else 'execution' if n in (3,5,7,9,11,13) else 'rest'
            reasons.append(f'Documented EEGMMI run {n}: {subtype}')
        else:
            subtype = 'imagery' if 'imagery' in meta_task else 'execution' if 'execution' in meta_task else 'unknown_motor_task'
    if family == 'SLEEP': subtype = 'scored_sleep' if 'sleep' in kinds else 'unscored_sleep'
    readiness = {
        'shared_eeg': True,
        'erp': family=='ODDBALL' and {'target','standard'}.issubset(sems),
        'reaction_time': family=='ODDBALL' and 'response' in sems and 'target' in sems,
        'motor_erd': family=='MOTOR' and len(sems & {'left_hand','right_hand','both_hands','both_feet'})>=1,
        'hypnogram': family=='SLEEP' and len(sems & {'W','N1','N2','N3','REM'})>=2,
    }
    tier = 'verified_by_annotations' if family != 'GENERAL' and any(e.family.lower()==family.lower() for e in events) else 'supported_by_task_metadata' if family != 'GENERAL' else 'fallback_general'
    return Decision(family,subtype,reasons,tier,readiness,'READY',[])
