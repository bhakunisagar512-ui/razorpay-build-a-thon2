"""Phase 4 / Stage 1: four prebuilt synthetic test datasets for the testing UI, built from the EXISTING held-out caches.

    docker compose exec -T api python -m eval.build_ui_datasets             # build into eval/data/phase4_ui/datasets
    docker compose exec -T api python -m eval.build_ui_datasets --verify    # integrity, leakage, reproducibility, compatibility

Each dataset directory holds three files that are kept apart on purpose:
    inputs.pkl     pandas pickle: ONLY the 43 Model C features plus the serving fields the app needs (ids, amount, times)
    truth.json     hidden ground truth: true cause + the expected SAFE recovery outcome derived from the existing rules
    manifest.json  metadata: sources, seeds, selection rules, composition, SHA-256 of the two files, verification results
Pickles are loaded only after their SHA-256 matches the manifest, and only files produced locally by this builder are trusted.

Sources (never used for training, tuning or model selection; seeds disjoint from every Phase 3 cache):
    final_test   12 worlds, 38,871 failures, seeds 500000-502003, market seed 11
    final_shift   6 worlds, 18,900 failures, seeds 600000-602001, market seed 12 (shifted regime)
Model C is used for INFERENCE ONLY (to find uncertain/incorrect cases for `mixed_difficult`). Nothing is retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval import phase3_heldout as h
from eval import oracle_plan_validation as plan_validation
from services.app.domain.deployment_features import DEPLOYMENT_CORE
from eval.oracle_plan import Ctx, build_plan
from services.app.domain.taxonomy import Cause

BUILDER_VERSION = 'phase4-ui-datasets@1'
SEED = 20261031                                   # one documented seed; every dataset derives SEED + a fixed offset
CACHE = Path('eval/data/phase4_run/cache')
TRAIN_CACHE = Path('eval/data/phase3_run/cache')
OUT = Path('eval/data/phase4_ui/datasets')
MARKET_SEED = {'final_test': 11, 'final_shift': 12}                        # from eval/phase4.py FINAL_MARKET
NAMES = ('normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime')
RETRYABLE = ('BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
RETRY_ACTIONS = ('silent_retry', 'rail_switch')
SERVING_EXTRA = ('payment_id', 'order_id', 'customer_id', 'instrument_id', 'psp_id', 'amount_minor_raw', 'failed_at', 'observed_at')
INPUT_COLUMNS = tuple(dict.fromkeys((*SERVING_EXTRA, *DEPLOYMENT_CORE)))
FORBIDDEN_INPUT_COLUMNS = ('true_cause', 'n_causes_fired', 'world_id', 'scenario', 'source_file', 'source_seed', 'selection_criteria')
N = 1000

DESCRIPTIONS = {
    'normal_recoverable': 'Balanced retryable failures: 250 each of insufficient funds, OTP abandoned, issuer down, PSP timeout.',
    'risk_and_hard_declines': '500 risk blocks and 500 expired cards. A retry on any of them is unsafe.',
    'mixed_difficult': 'DELIBERATELY SELECTED hard cases (low Model C confidence, wrong top-1, ambiguous risk probability, elevated '
                       'hard-decline mass). NOT representative of natural performance: do not read its success rate as a population estimate.',
    'shifted_regime': 'Unseen shifted-regime failures (different market seed) sampled at their natural cause frequencies.',
}


# ------------------------------------------------------------------------------------------------- loading and hashing
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_role(role: str) -> pd.DataFrame:
    """The cached held-out failures of one role (trusted local caches written by eval/phase4.py), sorted for determinism."""
    parts = []
    for path in sorted(CACHE.glob(f'*_{role}_*.pkl')):
        with path.open('rb') as stream:
            d = pickle.load(stream)
        parts.append(d['failures'].assign(source_file=path.name, source_seed=int(d['seed'])))
    return pd.concat(parts, ignore_index=True).sort_values('payment_id').reset_index(drop=True)


def earlier_data_footprint():
    """Payment ids and world seeds of everything Model C could have seen: all Phase 3 train/validation/test/shifted caches."""
    ids, seeds = set(), set()
    for path in sorted(TRAIN_CACHE.glob('*.pkl')):
        with path.open('rb') as stream:
            d = pickle.load(stream)
        ids |= set(d['failures'].payment_id)
        seeds.add(int(d['seed']))
    return ids, seeds


def take(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(f'need {n} rows but only {len(frame)} available')
    return frame.sample(n=n, random_state=seed)


# ------------------------------------------------------------------------------------------------- expected outcomes
def oracle_plan(row):
    """The plan the existing deterministic policy builds for the TRUE cause, validated under the demo consent defaults."""
    ctx = Ctx(amount_minor=int(row.amount_minor_raw), method=row.method, today=pd.Timestamp(row.failed_at).date(),
              available_rails=frozenset(h.RAILS.split(',')), consent_whatsapp=True)
    facts = {'consent': dict(h.DEFAULT_CONSENT), 'available_rails': h.RAILS.split(','), 'method': row.method, 'prior_retries': 0,
             'prior_contacts': 0, 'max_retries': 6, 'max_contacts': 4}
    v = plan_validation.validate_and_correct([asdict(s) for s in build_plan(Cause(row.true_cause), ctx).steps], row.true_cause, facts)
    return [s['action'] for s in v['steps']], bool(v['valid'])


def expected_outcome(true_cause: str, actions: list, valid: bool) -> dict:
    """Expected SAFE outcome from the existing recovery rules, not merely the cause label. Recovery is never assumed:
    for retryable causes it is conditional on Model C diagnosing the cause within three cycles, and it is simulated."""
    if true_cause == 'RISK_BLOCKED':
        return {'ideal_outcome': 'SAFETY_ESCALATION', 'ideal_is_conditional': False, 'recovery_guaranteed': False,
                'safe_outcomes': ['SAFETY_ESCALATION', 'DIAGNOSIS_INTERVENTION', 'PLAN_INTERVENTION'],
                'unnecessary_but_safe_outcomes': [], 'forbidden_actions': list(RETRY_ACTIONS),
                'rule': 'mandatory risk review: never counted as recovery and never retried'}
    base = {'ideal_outcome': 'SIMULATED_RECOVERY_SUCCESS' if valid else 'PLAN_INTERVENTION', 'ideal_is_conditional': True,
            'recovery_guaranteed': False,
            'safe_outcomes': ['SIMULATED_RECOVERY_SUCCESS', 'DIAGNOSIS_INTERVENTION', 'PLAN_INTERVENTION'],
            'unnecessary_but_safe_outcomes': ['SAFETY_ESCALATION']}
    if true_cause == 'CARD_EXPIRED':
        return base | {'forbidden_actions': list(RETRY_ACTIONS),
                       'rule': 'hard decline: card-update link only, no automated retry; success needs a correct diagnosis'}
    return base | {'forbidden_actions': [],
                   'rule': 'retryable: simulated success only if Model C names the cause within three cycles; a risk escalation is unnecessary'}


def truth_entry(row, extra=None) -> dict:
    actions, valid = oracle_plan(row)
    entry = {'true_cause': row.true_cause, **expected_outcome(row.true_cause, actions, valid),
             'oracle_plan_actions': actions, 'oracle_plan_valid_under_demo_consent': valid}
    return entry | (extra or {})


# ------------------------------------------------------------------------------------------------- selection
def difficulty_table(final: pd.DataFrame):
    """Model C inference only: calibrated probabilities for the held-out frame and the four difficulty flags."""
    art, classes, probs = h.probabilities(final)
    classes = list(classes)
    risk, card = classes.index('RISK_BLOCKED'), classes.index('CARD_EXPIRED')
    top = probs.argmax(axis=1)
    return pd.DataFrame({
        'payment_id': final.payment_id.to_numpy(),
        'low_confidence': probs.max(axis=1) < .60,
        'incorrect_top1': np.asarray(classes)[top] != final.true_cause.to_numpy(),
        'ambiguous_risk_probability': (probs[:, risk] >= .10) & (probs[:, risk] < .40),
        'elevated_hard_mass': ((probs[:, risk] + probs[:, card]) > .05) & ((probs[:, risk] + probs[:, card]) <= .30)}), art


def proportional_counts(counts: pd.Series, n: int) -> dict:
    """Largest-remainder allocation so a sample of n reproduces the natural cause frequencies."""
    exact = counts / counts.sum() * n
    base = np.floor(exact).astype(int)
    for cause in (exact - base).sort_values(ascending=False, kind='stable').index[: n - int(base.sum())]:
        base[cause] += 1
    return {c: int(v) for c, v in base.items()}


def select_all():
    final, shift = load_role('final_test'), load_role('final_shift')
    selected, notes = {}, {}
    # 1 normal_recoverable: balanced retryable causes
    parts = [take(final[final.true_cause == c], 250, SEED + 10 + i) for i, c in enumerate(RETRYABLE)]
    selected['normal_recoverable'] = pd.concat(parts)
    # 2 risk_and_hard_declines: 500 + 500
    selected['risk_and_hard_declines'] = pd.concat([take(final[final.true_cause == 'RISK_BLOCKED'], 500, SEED + 20),
                                                    take(final[final.true_cause == 'CARD_EXPIRED'], 500, SEED + 21)])
    # 3 mixed_difficult: from what is left, 250 per difficulty flag (shortfalls filled from any flagged row)
    used = set(selected['normal_recoverable'].payment_id) | set(selected['risk_and_hard_declines'].payment_id)
    diff, art = difficulty_table(final)
    flags = ['low_confidence', 'incorrect_top1', 'ambiguous_risk_probability', 'elevated_hard_mass']
    left = final[~final.payment_id.isin(used)].merge(diff, on='payment_id')
    chosen, quota = [], {}
    for k, flag in enumerate(flags):
        pool = left[left[flag] & ~left.payment_id.isin({r for part in chosen for r in part.payment_id})]
        pick = take(pool, min(250, len(pool)), SEED + 30 + k)
        chosen.append(pick)
        quota[flag] = len(pick)
    taken = pd.concat(chosen)
    short = N - len(taken)
    if short > 0:
        pool = left[left[flags].any(axis=1) & ~left.payment_id.isin(taken.payment_id)]
        taken = pd.concat([taken, take(pool, short, SEED + 39)])
        quota['filled_from_any_flag'] = short
    selected['mixed_difficult'] = taken
    notes['mixed_difficult'] = {'quota_by_primary_flag': quota}
    # 4 shifted_regime: natural frequencies (proportional allocation, largest remainder)
    alloc = proportional_counts(shift.true_cause.value_counts(), N)
    selected['shifted_regime'] = pd.concat([take(shift[shift.true_cause == c], n, SEED + 40 + i)
                                            for i, (c, n) in enumerate(sorted(alloc.items())) if n])
    notes['shifted_regime'] = {'allocation': alloc}
    flagged = diff.set_index('payment_id')
    return selected, notes, flagged, art


# ------------------------------------------------------------------------------------------------- writing
def write_dataset(root: Path, name: str, frame: pd.DataFrame, flagged: pd.DataFrame, note: dict, art, roles: dict) -> dict:
    frame = frame.sort_values(['observed_at', 'payment_id']).reset_index(drop=True)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    inputs = frame[list(INPUT_COLUMNS)].copy()
    inputs.to_pickle(d / 'inputs.pkl', protocol=4)
    truth = {}
    for row in frame.itertuples():
        extra = None
        if name == 'mixed_difficult':
            extra = {'selection_criteria': [f for f in ('low_confidence', 'incorrect_top1', 'ambiguous_risk_probability', 'elevated_hard_mass')
                                            if bool(flagged.loc[row.payment_id, f])]}
        truth[row.payment_id] = truth_entry(row, extra)
    (d / 'truth.json').write_text(json.dumps(truth, indent=1, sort_keys=True), encoding='utf-8')
    role = roles[name]
    sources = frame.groupby('source_file').source_seed.first().sort_index()
    manifest = {
        'name': name, 'description': DESCRIPTIONS[name], 'records': len(frame), 'builder_version': BUILDER_VERSION, 'rng_seed': SEED,
        'representative_of_natural_performance': name == 'shifted_regime',
        'label': ('DELIBERATELY SELECTED hard cases: not representative of natural performance' if name == 'mixed_difficult'
                  else 'Balanced by design: success rates are not population estimates' if name in ('normal_recoverable', 'risk_and_hard_declines')
                  else 'Natural cause frequencies of the shifted regime'),
        'source': {'cache_role': role, 'market_seed': MARKET_SEED[role], 'world_files': list(sources.index),
                   'world_seeds': [int(s) for s in sources.values]},
        'selection': note | {'rule': {'normal_recoverable': '250 per retryable cause, seeded sample from final_test',
                                      'risk_and_hard_declines': '500 RISK_BLOCKED + 500 CARD_EXPIRED, seeded sample from final_test',
                                      'mixed_difficult': 'final_test minus the two datasets above; 250 per difficulty flag in order low_confidence, '
                                                         'incorrect_top1 (uses hidden truth at build time only), ambiguous_risk_probability '
                                                         '(0.10<=P(RISK)<0.40), elevated_hard_mass (0.05<P(CARD)+P(RISK)<=0.30)',
                                      'shifted_regime': 'proportional stratified sample from final_shift (natural frequencies)'}[name]},
        'composition_by_cause': {k: int(v) for k, v in frame.true_cause.value_counts().sort_index().items()},
        'input_columns': list(INPUT_COLUMNS), 'model_id_used_for_difficulty_selection': art['schema']['model_id'] if name == 'mixed_difficult' else None,
        'files': {'inputs.pkl': sha256(d / 'inputs.pkl'), 'truth.json': sha256(d / 'truth.json')},
    }
    return manifest


def independence(root_frames: dict, footprint) -> dict:
    ids, seeds = footprint
    all_ids = [i for f in root_frames.values() for i in f.payment_id]
    world_seeds = sorted({int(s) for f in root_frames.values() for s in f.source_seed})
    names = list(root_frames)
    pair_overlap = {f'{a}&{b}': len(set(root_frames[a].payment_id) & set(root_frames[b].payment_id))
                    for i, a in enumerate(names) for b in names[i + 1:]}
    return {'payment_id_overlap_with_phase3_train_validation_test_shifted': len(ids & set(all_ids)),
            'world_seed_overlap_with_phase3': sorted(seeds & set(world_seeds)),
            'phase3_cache_files_checked': len(list(TRAIN_CACHE.glob('*.pkl'))), 'phase3_seed_range': [min(seeds), max(seeds)],
            'dataset_world_seeds': world_seeds, 'duplicate_ids_across_datasets': len(all_ids) - len(set(all_ids)),
            'pairwise_overlap': pair_overlap}


def build(root: Path = OUT) -> dict:
    selected, notes, flagged, art = select_all()
    footprint = earlier_data_footprint()
    ind = independence(selected, footprint)
    if ind['payment_id_overlap_with_phase3_train_validation_test_shifted'] or ind['world_seed_overlap_with_phase3'] \
            or ind['duplicate_ids_across_datasets']:
        raise SystemExit(f'leakage or overlap detected: {ind}')
    roles = {'normal_recoverable': 'final_test', 'risk_and_hard_declines': 'final_test', 'mixed_difficult': 'final_test',
             'shifted_regime': 'final_shift'}
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name in NAMES:
        manifests[name] = write_dataset(root, name, selected[name], flagged, notes.get(name, {}), art, roles)
        manifests[name]['verification_at_build'] = ind
        (root / name / 'manifest.json').write_text(json.dumps(manifests[name], indent=1, sort_keys=True), encoding='utf-8')
    return manifests


# ------------------------------------------------------------------------------------------------- loaders (trusted local files)
def load_manifest(name: str, root: Path = OUT) -> dict:
    return json.loads((Path(root) / name / 'manifest.json').read_text(encoding='utf-8'))


def _checked(name: str, filename: str, root: Path) -> Path:
    path = Path(root) / name / filename
    if sha256(path) != load_manifest(name, root)['files'][filename]:
        raise ValueError(f'{name}/{filename} does not match its manifest SHA-256 (modified or corrupt)')
    return path


def load_inputs(name: str, root: Path = OUT) -> pd.DataFrame:
    """Payment inputs only: no ground truth. Verified against the manifest hash before the pickle is read."""
    return pd.read_pickle(_checked(name, 'inputs.pkl', root))


def load_truth(name: str, root: Path = OUT) -> dict:
    """Hidden ground truth. Only the evaluator may call this."""
    return json.loads(_checked(name, 'truth.json', root).read_text(encoding='utf-8'))


# ------------------------------------------------------------------------------------------------- verification
def verify(root: Path = OUT) -> dict:
    checks = {}
    frames = {}
    for name in NAMES:
        inputs, truth, manifest = load_inputs(name, root), load_truth(name, root), load_manifest(name, root)   # hashes verified here
        frames[name] = inputs
        checks[f'{name}: records'] = len(inputs) == manifest['records'] == len(truth) == N
        checks[f'{name}: ids identical in inputs and truth'] = set(inputs.payment_id) == set(truth)
        checks[f'{name}: inputs columns are exactly the permitted set'] = list(inputs.columns) == list(INPUT_COLUMNS)
        checks[f'{name}: no ground-truth columns in inputs'] = not (set(inputs.columns) & set(FORBIDDEN_INPUT_COLUMNS))
        checks[f'{name}: truth composition matches manifest'] = \
            dict(pd.Series([t['true_cause'] for t in truth.values()]).value_counts().sort_index()) == manifest['composition_by_cause']
    footprint = earlier_data_footprint()
    seeds = sorted({s for n in NAMES for s in load_manifest(n, root)['source']['world_seeds']})
    all_ids = [i for f in frames.values() for i in f.payment_id]
    checks['no payment-id overlap with Phase 3 train/validation/test caches'] = not (footprint[0] & set(all_ids))
    checks['no world-seed overlap with Phase 3 caches'] = not (footprint[1] & set(seeds))
    checks['datasets are mutually disjoint'] = len(all_ids) == len(set(all_ids))
    checks['source seeds are in the held-out ranges'] = all(500000 <= s <= 502003 or 600000 <= s <= 602001 for s in seeds)
    # reproducibility: rebuild into a temporary directory and compare the file hashes byte for byte
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = build(Path(tmp))
        for name in NAMES:
            checks[f'{name}: rebuild is byte-identical (inputs, truth)'] = \
                rebuilt[name]['files'] == load_manifest(name, root)['files']
            checks[f'{name}: rebuilt manifest is identical'] = \
                (Path(tmp) / name / 'manifest.json').read_bytes() == (Path(root) / name / 'manifest.json').read_bytes()
    # Model C compatibility: feature names, class order, inference on a small sample
    from services.app.adapters import diagnosis_v4
    art = diagnosis_v4.bundle()
    model = art['model']
    checks['Model C features are all present in the inputs'] = set(model.features) <= set(frames['normal_recoverable'].columns)
    checks['class order equals the packaged schema'] = list(model.classes_) == art['schema']['classes']
    for name in NAMES:
        p = art['calibrator'].transform(model.predict_proba(load_inputs(name, root).head(100)))
        checks[f'{name}: Model C scores a sample (100x{len(model.classes_)}, rows sum to 1)'] = \
            p.shape == (100, len(model.classes_)) and bool(np.allclose(p.sum(axis=1), 1))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    root = Path(args.out)
    if args.verify:
        checks = verify(root)
        for k, v in checks.items():
            print(('PASS ' if v else 'FAIL ') + k)
        bad = [k for k, v in checks.items() if not v]
        print(f'\n{len(checks) - len(bad)}/{len(checks)} checks passed')
        sys.exit(1 if bad else 0)
    manifests = build(root)
    for name, m in manifests.items():
        print(f"{name:24s} {m['records']:5d}  {m['composition_by_cause']}")
        print(f"{'':24s} {m['label']}")
    print('\nverification at build:', json.dumps(next(iter(manifests.values()))['verification_at_build']))


if __name__ == '__main__':
    main()
