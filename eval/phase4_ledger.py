"""Crash-safe bookkeeping for the unattended Phase 4 run.

A "unit" is one expensive step (a tuning trial, a model x feature-set evaluation, ...). A finished unit is a
JSON file written atomically (temp file + rename), so a killed process can never leave a half-written result
that a later run would trust. Re-running skips finished units, retries failed ones, and never retrains a
model whose artifact and result already exist.
"""
import atexit
import json
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and value != value:
        return None
    return value


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(clean(payload), indent=1, allow_nan=False))


class RunLocked(SystemExit):
    pass


def _alive(pid):
    """Linux-only liveness check by /proc cmdline (the run happens in the container); False when unverifiable."""
    cmd = Path(f'/proc/{pid}/cmdline')
    try:
        return 'eval.phase4' in cmd.read_text().replace('\0', ' ')
    except OSError:
        return False


def acquire_lock(out):
    lock = Path(out) / 'run.lock'
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = -1
        if pid != os.getpid() and pid > 0 and _alive(pid):
            raise RunLocked(f'Another Phase 4 run is already active (pid {pid}). Monitor it; do not start a second.')
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True) if lock.exists() and lock.read_text().strip() == str(os.getpid()) else None)


class Ledger:
    def __init__(self, out):
        self.out = Path(out)
        for d in ('logs', 'units', 'models', 'tuning_models', 'failures'):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.durations = defaultdict(list)
        self.expected = {}
        self.stage = 'starting'
        self.current = ''

    # ---- logging ---------------------------------------------------------------
    def log(self, message, level='INFO'):
        line = f'{datetime.now().isoformat(timespec="seconds")} {level} {message}'
        print(line, flush=True)
        with (self.out / 'logs' / 'run.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    def error(self, unit, text):
        with (self.out / 'logs' / 'errors.log').open('a', encoding='utf-8') as stream:
            stream.write(f'=== {datetime.now().isoformat(timespec="seconds")} {unit}\n{text}\n')

    # ---- units -----------------------------------------------------------------
    def path(self, kind, name):
        return self.out / 'units' / kind / f'{name}.json'

    def done(self, kind, name, required=()):
        return self.path(kind, name).exists() and all(Path(p).exists() for p in required)

    def load(self, kind, name):
        return json.loads(self.path(kind, name).read_text(encoding='utf-8'))

    def completed(self, kind):
        directory = self.out / 'units' / kind
        return sorted(p.stem for p in directory.glob('*.json')) if directory.exists() else []

    def failed(self):
        return sorted(p.stem for p in (self.out / 'failures').glob('*.json'))

    def run(self, kind, name, fn, required=()):
        """Return the unit's result, computing it if needed. Exceptions are recorded and swallowed (-> None)."""
        if self.done(kind, name, required):
            return self.load(kind, name)
        unit = f'{kind}/{name}'
        self.current = unit
        started = time.time()
        try:
            result = fn()
            atomic_write_json(self.path(kind, name), result)   # written last: its presence means "complete"
        except Exception as exc:                                # noqa: BLE001 - a failed unit must not kill the run
            text = traceback.format_exc()
            self.error(unit, text)
            atomic_write_json(self.out / 'failures' / f'{kind}__{name}.json',
                              {'unit': unit, 'error': f'{type(exc).__name__}: {exc}', 'at': time.time()})
            self.log(f'FAILED {unit}: {type(exc).__name__}: {exc} (see logs/errors.log; rerun retries it)', 'ERROR')
            self.write_progress()
            return None
        (self.out / 'failures' / f'{kind}__{name}.json').unlink(missing_ok=True)
        seconds = time.time() - started
        self.durations[kind].append(seconds)
        self.log(f'done {unit} ({seconds:.0f}s)')
        self.write_progress()
        return self.load(kind, name)

    # ---- progress ---------------------------------------------------------------
    def write_progress(self):
        done = {k: len(self.completed(k)) for k in self.expected}
        remaining = {k: max(0, total - done[k]) for k, total in self.expected.items()}
        everything = [d for ds in self.durations.values() for d in ds]
        fallback = sum(everything) / len(everything) if everything else None
        eta = None
        if fallback is not None:
            eta = sum(remaining[k] * (sum(self.durations[k]) / len(self.durations[k]) if self.durations[k] else fallback)
                      for k in remaining)
        info = {'stage': self.stage, 'current_unit': self.current, 'units_done': done, 'units_expected': self.expected,
                'failed_units': self.failed(), 'elapsed_minutes_this_session': round((time.time() - self.started) / 60, 1),
                'rough_eta_minutes': None if eta is None else round(eta / 60), 'updated': datetime.now().isoformat(timespec='seconds')}
        atomic_write_json(self.out / 'progress.json', info)
        total, finished = sum(self.expected.values()), sum(done.values())
        lines = [f'Phase 4 benchmark   stage: {self.stage}', f'updated: {info["updated"]}',
                 f'current: {self.current or "-"}', f'overall: {finished}/{total} units ({100 * finished / max(total, 1):.0f}%)',
                 f'rough time left: {"unknown yet" if eta is None else f"~{round(eta / 60)} min"} (CatBoost fits vary a lot)']
        lines += [f'  {k:12s} {done[k]}/{self.expected[k]}' for k in self.expected]
        if info['failed_units']:
            lines.append(f'FAILED units ({len(info["failed_units"])}): {", ".join(info["failed_units"][:8])}')
        atomic_write_text(self.out / 'PROGRESS.txt', '\n'.join(lines) + '\n')
