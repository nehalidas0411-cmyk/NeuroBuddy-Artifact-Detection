"""File intake, BIDS-aware bundle discovery, and EEG reader.

Never infer task identity from extension presence. A recording is one primary signal
file plus its verified companions. File intake and scientific inference are separate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib
import json
import re
import zipfile

import mne

SIGNALS = (".set", ".edf", ".bdf", ".vhdr", ".fif", ".fif.gz")
ENTITIES = ("sub", "ses", "task", "acq", "run", "rec", "space", "desc")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bids_entities(p: Path) -> dict[str, str]:
    return {key: match.group(1) for key in ENTITIES
            if (match := re.search(rf'(?:^|_){key}-([A-Za-z0-9]+)(?:_|\.|$)', p.name))}


def file_kind(p: Path) -> str:
    name = p.name.lower()
    if name.endswith(('.set', '.edf', '.bdf', '.vhdr', '.fif', '.fif.gz')):
        return 'signal'
    if name.endswith('.fdt'):
        return 'data_companion'
    if name.endswith('.json'):
        return 'metadata'
    if name.endswith(('.tsv', '.csv')):
        if '_channels.' in name or '_electrodes.' in name:
            return 'channel_metadata'
        if 'scoring' in name or 'stages' in name or 'hypnogram' in name:
            return 'sleep_annotation'
        return 'possible_annotation'
    return 'other'


@dataclass
class RecordingBundle:
    signal: Path
    companion: Optional[Path] = None
    annotations: list[Path] = field(default_factory=list)
    dictionaries: list[Path] = field(default_factory=list)
    metadata: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    status: str = 'READY'

    @property
    def entities(self):
        return bids_entities(self.signal)

    @property
    def name(self) -> str:
        return self.signal.stem

    def all_files(self):
        return [self.signal] + ([self.companion] if self.companion else []) + self.annotations + self.dictionaries + self.metadata

    def manifest(self):
        return [{'name': f.name, 'role': file_kind(f), 'sha256': sha256(f), 'bytes': f.stat().st_size}
                for f in self.all_files()]


def _is_same_recording(candidate: Path, signal: Path) -> bool:
    """Require equal populated BIDS entities; accept fewer fields on inherited sidecars."""
    a, b = bids_entities(candidate), bids_entities(signal)
    for k in ('sub', 'ses', 'task', 'run'):
        if k in a and k in b and a[k].lower() != b[k].lower():
            return False
    if 'acq' in a and 'acq' in b and a['acq'].lower() != b['acq'].lower():
        # Scoring is sometimes a different BIDS acq than the PSG whose epochs it scores.
        if not ('scoring' in a['acq'].lower() and b['acq'].lower() == 'psg'):
            return False
    inherited_json = (candidate.suffix.lower() == '.json' and candidate.parent in signal.parents
                      and 'task' in a and a.get('task', '').lower() == b.get('task','').lower())
    return bool(a.get('sub') is not None and a.get('sub') == b.get('sub')
                or signal.parent == candidate.parent or inherited_json)


def _candidate_score(p: Path, sig: Path) -> int:
    a, b = bids_entities(p), bids_entities(sig)
    s = sum(3 for k in ('sub','ses','task','run') if k in a and a[k] == b.get(k))
    s += 3 if p.parent == sig.parent else 0
    s += 2 if '_events.' in p.name.lower() else 0
    s += 1 if p.name.lower().endswith('.tsv') else 0
    if 'acq' in a and a['acq'] == b.get('acq'):
        s += 3
    if 'scoring' in p.name.lower() and b.get('acq','').lower() == 'psg':
        s += 2
    return s


def _collect(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink())


def discover(source: str | Path) -> list[RecordingBundle]:
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    files = _collect(source) if source.is_dir() else _collect(source.parent)
    # Locate inherited event dictionaries in an actual BIDS root, not arbitrary parent paths.
    # BIDS permits task-specific JSON sidecars to live above sub/ses/eeg directories.
    for anc in list(source.parents)[:5]:
        if (anc/'dataset_description.json').is_file():
            files += [p for p in anc.glob('*.json') if p.is_file() and p not in files]
            break
    if source.is_file() and file_kind(source) != 'signal':
        # A single sidecar can be submitted from a folder containing EEG files.
        files = [source] + [f for f in files if f != source]
    signals = [p for p in files if file_kind(p) == 'signal']
    if source.is_file() and file_kind(source) == 'signal':
        signals = [source]
    bundles: list[RecordingBundle] = []
    for sig in signals:
        b = RecordingBundle(signal=sig)
        companions = [f for f in files if f.parent == sig.parent and f.suffix.lower() == '.fdt'
                      and f.stem == sig.stem]
        if companions:
            b.companion = companions[0]
        candidates = [f for f in files if f != sig and file_kind(f) in
                      {'possible_annotation','sleep_annotation','metadata','channel_metadata'}
                      and _is_same_recording(f, sig)]
        ann = [p for p in candidates if file_kind(p) in ('possible_annotation','sleep_annotation')]
        # Ignore generic metadata tables, but inspect any TSV/CSV with matching entity set.
        groups = {}
        for p in ann:
            canonical = re.sub(r'\(\d+\)(?=\.(?:tsv|csv)$)', '', p.name, flags=re.I)
            groups.setdefault(canonical, []).append(p)
        for _, group in groups.items():
            group.sort(key=lambda p: (re.search(r'\(\d+\)', p.name) is not None, -_candidate_score(p, sig)))
            best = group[0]
            b.annotations.append(best)
            for alt in group[1:]:
                if sha256(best) == sha256(alt):
                    b.warnings.append(f'Duplicate identical annotation ignored: {alt.name}')
                else:
                    b.conflicts.append(f'Conflicting annotation versions: {best.name} vs {alt.name}')
        b.annotations.sort(key=lambda p: (-int('scoring' in p.name.lower() and b.entities.get('task','').lower()=='sleep'), -_candidate_score(p, sig)))
        jsons = [p for p in candidates if file_kind(p) == 'metadata']
        b.dictionaries = [p for p in jsons if '_events.' in p.name.lower()]
        b.metadata = [p for p in jsons if p not in b.dictionaries]
        if b.conflicts:
            b.status = 'NEEDS_REVIEW'
        bundles.append(b)
    return bundles


def load_eeg(bundle: RecordingBundle):
    name = bundle.signal.name.lower()
    readers = {
        '.set': mne.io.read_raw_eeglab,
        '.edf': mne.io.read_raw_edf,
        '.bdf': mne.io.read_raw_bdf,
        '.vhdr': mne.io.read_raw_brainvision,
        '.fif': mne.io.read_raw_fif,
    }
    extension = '.fif' if name.endswith('.fif.gz') else bundle.signal.suffix.lower()
    if extension not in readers:
        raise ValueError(f'Unsupported recording: {bundle.signal.name}')
    try:
        raw = readers[extension](str(bundle.signal), preload=False, verbose='ERROR')
    except Exception as e:
        err = str(e)
        if 'fdt' in err.lower() or 'not found' in err.lower() or 'no such file' in err.lower():
            bundle.status = 'INCOMPLETE'
        else:
            bundle.status = 'FAILED'
        raise RuntimeError(f'{bundle.status}: could not load EEG {bundle.signal.name}: {err}') from e
    if raw.n_times < 2 or not any(t == 'eeg' for t in raw.get_channel_types()):
        bundle.status = 'FAILED'
        raise ValueError('No usable EEG signal channels')
    return raw


def safe_extract_zip(source: Path, destination: Path, max_uncompressed: int = 4_000_000_000):
    """Strict ZIP extraction with traversal, symlink and size checks."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        files = z.infolist()
        if len(files) > 20000 or sum(x.file_size for x in files) > max_uncompressed:
            raise ValueError('Archive exceeds configured file count or decompressed size limit')
        for info in files:
            path = Path(info.filename)
            if path.is_absolute() or '..' in path.parts or (info.external_attr >> 16 & 0o170000) == 0o120000:
                raise ValueError('Unsafe archive path or symlink')
            resolved = (destination / path).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError('Archive path traversal')
        z.extractall(destination)
    return destination
