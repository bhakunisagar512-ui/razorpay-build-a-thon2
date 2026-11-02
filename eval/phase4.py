"""Phase 4: unattended, resumable model benchmark on the Phase 3 sequences@v4 worlds.

Run (inside the api container, from /code; scripts/run_phase4.ps1 does this for you):
    python -m eval.phase4 all --output-dir eval/data/phase4_run           # full benchmark, resumable
    python -m eval.phase4 all --smoke --output-dir eval/data/phase4_smoke # tiny end-to-end check
Re-running the same command resumes: finished units are skipped, failed units are retried, trained models
are reloaded from disk instead of retrained. Nothing here touches the serving classifier, recovery workflows
or Phase 3 outputs.

Data roles (all separate worlds; Phase 3's cached worlds are reused read-only):
    train        Phase 3 train worlds      (4 per scenario)  fit models
    validation   Phase 3 validation worlds (2 per scenario)  tune hyperparameters, choose calibration, select the model
    calibration  Phase 3 *test* worlds     (3 per scenario)  fit sigmoid / isotonic calibrators only. Phase 3 scored
                 only its HGB feature sets on them; nothing was trained, tuned or selected on them.
    final_test   FRESH worlds, seeds 500000+ (4 per scenario) generated only after selection.json is frozen
    final_shift  FRESH worlds, seeds 600000+ (2 per scenario) different issuer/PSP market (market_seed 12)
Models train on the three scenarios pooled, because a deployed model does not know its regime.

Order of work (so an interrupted run loses as little as possible):
    1 tuning  2 deployment_core validation  3 FREEZE selection.json  4 remaining validation (ablations)
    5 fresh worlds + one-shot test  6 unseen-regime / latency / memory  7 report, package, share bundle, pytest

Frozen selection rule (see SELECTION_RULE / select_model): deployment_core, pooled validation, each model with
its best validation-log-loss calibration. Eligible: gated unsafe-retry rate <= 1.0% and ECE <= 0.05 (all models
if none qualify). Highest macro-F1 wins; within 0.005 is a tie; ties go to the lower log loss, but a simpler
family wins when within 0.01 log loss (HGB < Random Forest < XGBoost < CatBoost).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', str(min(6, os.cpu_count() or 1)))   # phase3 pins 1 thread; Phase 4 is capped at 6 cores

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from eval import phase3
from eval.phase3_quality import world_quality
from eval.phase4_confidence import calibration_block, classification_block, confidence_profile, full_report, safety_block, world_indices
from eval.phase4_ledger import Ledger, acquire_lock, atomic_write_json, atomic_write_text
from eval.phase4_models import FAMILIES, SIMPLICITY_RANK, TRIALS_PER_FAMILY, BenchModel, trial_params
from eval.probability_calibration import ProbabilityCalibrator
from eval.sequences_v4 import assemble_v4, generate_world, rebuild_features_at
from services.app.domain.deployment_features import (
    DEPLOYMENT_CORE, DEPLOYMENT_WITH_TIMING, DERIVED_FROM_HEADLINE, DURATION_FEATURES, HEADLINE_FEATURES, REASON_PAIR, without)
from services.app.domain.features_v4 import FEATURES_V4

SCENARIOS = phase3.SCENARIOS
ORDERS = 24000
FINAL_SEED = {'final_test': 500000, 'final_shift': 600000}
FINAL_COUNT = {'final_test': 4, 'final_shift': 2}
FINAL_MARKET = {'final_test': 11, 'final_shift': 12}
CALIBRATION_METHODS = ('uncalibrated', 'sigmoid', 'isotonic')
PHASE3_REPORT = Path('eval/data/phase3_run/ablation-and-selection.json')
PHASE3_CACHE = Path('eval/data/phase3_run/cache')

SELECTION_RULE = {
    'evaluated_on': 'pooled validation worlds, deployment_core feature set, per-model best-logloss calibration',
    'eligibility': {'max_gated_unsafe_retry_rate': 0.01, 'max_ece': 0.05, 'fallback': 'all models if none eligible'},
    'primary': 'highest macro-F1', 'tie_band_macro_f1': 0.005,
    'tie_break': 'lowest log-loss; a simpler family wins when within 0.01 log-loss',
    'simplicity_order': sorted(SIMPLICITY_RANK, key=SIMPLICITY_RANK.get),
    'tuning': f'{TRIALS_PER_FAMILY} trials per family (trial 0 = default, rest random), criterion = sigmoid-calibrated '
              'validation log-loss, tuned at deployment_core and reused for every other feature set',
    'gate': 'frozen Phase 2.5 control, confidence 0.60 / hard-decline mass 0.03, never retuned'}


@dataclass(frozen=True)
class RunConfig:
    name: str = 'full'
    orders: int = ORDERS
    trials: int = TRIALS_PER_FAMILY
    final_count: tuple = tuple(FINAL_COUNT.items())
    sample_rows: tuple = None            # (train, validation, calibration) row caps; smoke only
    configs: tuple = None                # restrict feature configurations; smoke only
    latency_repeats: int = 300
    run_pytest: bool = True


FULL = RunConfig()
SMOKE = RunConfig('smoke', orders=3000, trials=2, final_count=(('final_test', 1), ('final_shift', 1)),
                  sample_rows=(2500, 1500, 1500),
                  configs=('deployment_core', 'deployment_core-no_reason', 'full_57-no_headline3'),
                  latency_repeats=15, run_pytest=False)


def selected_42():
    return tuple(json.loads(PHASE3_REPORT.read_text())['selection']['selected_feature_set'])


def feature_configs(only=None):
    """Name -> feature tuple. Ablations remove the headline fields (and, separately, everything derived from them)."""
    s42 = selected_42()
    configs = {'full_57': FEATURES_V4, 'selected_42': s42, 'deployment_with_timing': DEPLOYMENT_WITH_TIMING,
               'deployment_core': DEPLOYMENT_CORE}
    for base in ('full_57', 'selected_42', 'deployment_with_timing'):
        f = configs[base]
        configs[f'{base}-no_reason'] = without(f, REASON_PAIR)
        configs[f'{base}-no_duration'] = without(f, DURATION_FEATURES)
        configs[f'{base}-no_headline3'] = without(f, HEADLINE_FEATURES)
        configs[f'{base}-no_headline3_or_derived'] = without(f, HEADLINE_FEATURES, DERIVED_FROM_HEADLINE)
    configs['deployment_core-no_reason'] = without(DEPLOYMENT_CORE, REASON_PAIR)
    configs['deployment_core-no_headline3_or_derived'] = without(DEPLOYMENT_CORE, HEADLINE_FEATURES, DERIVED_FROM_HEADLINE)
    return {k: v for k, v in configs.items() if only is None or k in only}


# --------------------------------------------------------------------------- data
GENERATOR_FILES = ('eval/sequences_v4.py', 'eval/world_config_v4.py', 'services/app/domain/features_v4.py',
                   'services/app/domain/features.py', 'services/app/domain/failure_vocab.py')


def generator_signature(orders=ORDERS):
    digest = hashlib.sha256(str(orders).encode())
    for name in GENERATOR_FILES:
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def verify_phase3_cache(cache):
    """The Phase 3 cache signature also covers phase3_quality.py (edited after generation, which cannot change worlds).
    Instead of trusting it, regenerate one world with the current code and require an identical frame."""
    marker = Path(cache) / 'phase4_generator_verified.json'
    signature = generator_signature()
    if marker.exists() and json.loads(marker.read_text()).get('generator_signature') == signature:
        return
    scenario, role, index = 'calm', 'validation', 0
    with (Path(cache) / f'{scenario}_{role}_{index}.pkl').open('rb') as stream:
        cached = pickle.load(stream)['failures']
    world = generate_world(ORDERS, phase3.world_seed(scenario, role, index), scenario=scenario,
                           market_seed=phase3.ROLE_MARKET[role])
    pd.testing.assert_frame_equal(assemble_v4(world).failures.reset_index(drop=True), cached.reset_index(drop=True))
    marker.write_text(json.dumps({'generator_signature': signature, 'verified_world': f'{scenario}_{role}_{index}',
                                  'method': 'regenerated with current code; frame identical'}))


def _tag(frame, scenario, role, index):
    frame = frame.copy()
    frame['world_index'], frame['scenario'] = index, scenario
    frame['world_key'] = f'{scenario}/{role}/{index}'
    return frame


def load_phase3_roles(cache=PHASE3_CACHE):
    if not (Path(cache) / 'calm_train_0.pkl').exists():
        raise FileNotFoundError(f'Phase 3 world cache not found in {cache}. Run `make phase3` first.')
    verify_phase3_cache(cache)
    data = {}
    for scenario in SCENARIOS:
        for role in ('train', 'validation', 'test'):
            for index in range(phase3.ROLE_COUNT[role]):
                with (Path(cache) / f'{scenario}_{role}_{index}.pkl').open('rb') as stream:
                    payload = pickle.load(stream)
                data.setdefault(('calibration' if role == 'test' else role, scenario), []).append(
                    _tag(payload['failures'], scenario, role, index))
    return data


def pooled(data, role, scenarios=SCENARIOS):
    return pd.concat([f for s in scenarios for f in data[(role, s)]], ignore_index=True)


def cap(frame, rows, seed=0):
    return frame if rows is None or len(frame) <= rows else frame.sample(rows, random_state=seed).reset_index(drop=True)


def final_job(job):
    scenario, role, index, cache, signature, orders = job
    path = Path(cache) / f'{scenario}_{role}_{index}.pkl'
    if path.exists():
        try:
            with path.open('rb') as stream:
                if pickle.load(stream)['signature'] == signature:
                    return str(path)
        except Exception:
            pass                                    # truncated by a kill: regenerate
    seed = FINAL_SEED[role] + 1000 * SCENARIOS.index(scenario) + index
    world = generate_world(orders, seed, scenario=scenario, market_seed=FINAL_MARKET[role])
    dataset = assemble_v4(world)
    quality = world_quality(dataset)
    if index == 0:
        sample = np.random.default_rng(seed).choice(dataset.failures.payment_id.to_numpy(), size=4, replace=False)
        quality['future_invariance'] = {'failures_checked': 4, 'feature_mismatches': int(sum(
            not phase3._same(dataset.failures[dataset.failures.payment_id == pid].iloc[0][k], v)
            for pid in sample for k, v in rebuild_features_at(world, pid).items()))}
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        pickle.dump({'signature': signature, 'failures': dataset.failures, 'quality': quality, 'role': role,
                     'scenario': scenario, 'seed': seed}, stream)
    os.replace(tmp, path)
    return str(path)


def load_final(cache, workers, cfg):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    signature = generator_signature(cfg.orders)
    counts = dict(cfg.final_count)
    jobs = [(s, r, i, str(cache), signature, cfg.orders) for s in SCENARIOS for r, c in counts.items() for i in range(c)]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(final_job, jobs))
    data, seeds, leak = {}, {}, []
    for scenario, role, index, *_ in jobs:
        with (cache / f'{scenario}_{role}_{index}.pkl').open('rb') as stream:
            payload = pickle.load(stream)
        data.setdefault((role, scenario), []).append(_tag(payload['failures'], scenario, role, index))
        seeds[f'{scenario}/{role}/{index}'] = payload['seed']
        if 'future_invariance' in payload['quality']:
            leak.append(payload['quality']['future_invariance']['feature_mismatches'])
    return data, seeds, leak


# --------------------------------------------------------------------------- fitting / evaluation
def fit_calibrators(model, calibration):
    p = model.predict_proba(calibration)
    y = calibration.true_cause.to_numpy()
    return {m: ProbabilityCalibrator(m, model.classes_).fit(p, y) for m in CALIBRATION_METHODS}


def headline(frame, model, p):
    y, labels = frame.true_cause.to_numpy(), np.asarray(model.classes_)
    cls, cal = classification_block(p, y, labels), calibration_block(p, y, labels)
    conf = confidence_profile(p, y, labels)
    safe = safety_block(frame, model, p)
    return {'macro_f1': cls['macro_f1'], 'accuracy': cls['accuracy'], **cal,
            'share_below_60': conf['share_below_60'], 'share_60_to_80': conf['share_60_to_80'],
            'share_above_80': conf['share_above_80'], 'accuracy_when_confident_ge_60': conf['accuracy_when_confident_ge_60'],
            'error_rate_when_confident_ge_60': conf['error_rate_when_confident_ge_60'],
            'unsafe_retries_gated': safe['unsafe_retry_gated']['count'], 'hard_declines': safe['hard_declines'],
            'unsafe_retry_rate_gated': safe['unsafe_retry_gated']['rate'],
            'retryable_recall_gated': safe['retryable_recall_gated'], 'abstention_rate': safe['abstention_rate'], 'n': len(y)}


def evaluate_partitions(model, calibrators, partitions, full_methods):
    """Headline numbers for every calibration method; the full report only for `full_methods`."""
    out = {}
    for pname, frame in partitions.items():
        raw = model.predict_proba(frame)
        worlds = world_indices(frame)
        out[pname] = {}
        for method, calibrator in calibrators.items():
            p = raw if method == 'uncalibrated' else calibrator.transform(raw)
            out[pname][method] = full_report(frame, model, p, worlds) if method in full_methods else headline(frame, model, p)
    return out


def validation_partitions(validation):
    parts = {s: validation[validation.scenario == s].reset_index(drop=True) for s in SCENARIOS}
    parts['pooled'] = validation
    return parts


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def select_model(summary):
    """summary: family -> {'macro_f1','logloss','ece','unsafe_rate'} at deployment_core (best calibration)."""
    rule = SELECTION_RULE
    limit_u, limit_e = rule['eligibility']['max_gated_unsafe_retry_rate'], rule['eligibility']['max_ece']
    eligible = {f: s for f, s in summary.items() if s['unsafe_rate'] <= limit_u and s['ece'] <= limit_e} or dict(summary)
    top = max(s['macro_f1'] for s in eligible.values())
    tied = {f: s for f, s in eligible.items() if top - s['macro_f1'] <= rule['tie_band_macro_f1'] + 1e-9}
    best_ll = min(s['logloss'] for s in tied.values())
    close = [f for f, s in tied.items() if s['logloss'] - best_ll <= .01 + 1e-9]
    winner = min(close, key=lambda f: SIMPLICITY_RANK[f])
    return winner, {'eligible': sorted(eligible), 'tied_on_macro_f1': sorted(tied), 'within_logloss_band': sorted(close)}


# --------------------------------------------------------------------------- latency / memory
def latency_profile(model, calibrator, frame, repeats=300):
    from threadpoolctl import threadpool_limits
    one = frame.iloc[[0]]
    times = []
    with threadpool_limits(1):
        model.set_inference_threads(1)
        for _ in range(min(20, repeats)):
            calibrator.transform(model.predict_proba(one))
        for _ in range(repeats):
            started = time.perf_counter()
            calibrator.transform(model.predict_proba(one))
            times.append((time.perf_counter() - started) * 1000)
        batch = frame.iloc[:2000]
        started = time.perf_counter()
        calibrator.transform(model.predict_proba(batch))
        rate = len(batch) / (time.perf_counter() - started)
    return {'single_request_ms_median': float(np.median(times)), 'single_request_ms_p95': float(np.quantile(times, .95)),
            'batch_rows_per_second_1thread': float(rate), 'threads': 1}


MEMORY_PROBE = r'''
import json, os, sys
import joblib, psutil
sys.path.insert(0, os.getcwd())
import eval.phase4_models                      # sklearn/pandas imported; family libs next so they are not billed to the model
for lib in ("xgboost", "catboost"):
    try: __import__(lib)
    except Exception: pass
proc = psutil.Process()
before = proc.memory_info().rss
bundle = joblib.load(sys.argv[1])
after = proc.memory_info().rss
print(json.dumps({"rss_mb_after_load": (after - before) / 2**20}))
'''


def memory_profile(path):
    result = subprocess.run([sys.executable, '-c', MEMORY_PROBE, str(path)], capture_output=True, text=True, cwd=os.getcwd())
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        info = {'rss_mb_after_load': None, 'error': result.stderr[-300:]}
    info['artifact_mb_on_disk'] = Path(path).stat().st_size / 2**20
    return info


# --------------------------------------------------------------------------- the run
def run_all(out, cfg=FULL, workers=6):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    acquire_lock(out)
    L = Ledger(out)
    configs = feature_configs(cfg.configs)
    if 'deployment_core' not in configs:
        raise SystemExit('deployment_core must be among the configurations')
    L.expected = {'tuning': cfg.trials * len(FAMILIES), 'validation': len(configs) * len(FAMILIES),
                  'test': len(configs) * len(FAMILIES), 'unseen': len(FAMILIES), 'perf': len(FAMILIES)}
    L.log(f'=== Phase 4 run "{cfg.name}" start; {len(configs)} feature sets x {len(FAMILIES)} models; output {out}')
    L.log(f'python {platform.python_version()}, cpus {os.cpu_count()}; resume = skip finished units, retry failed ones')
    write_environment(out)
    L.write_progress()

    L.stage = 'loading data'
    data = load_phase3_roles()
    rows = cfg.sample_rows or (None, None, None)
    train = cap(pooled(data, 'train'), rows[0])
    validation = cap(pooled(data, 'validation'), rows[1])
    calibration = cap(pooled(data, 'calibration'), rows[2])
    L.log(f'train {len(train)}, validation {len(validation)}, calibration {len(calibration)} failures')
    vparts = validation_partitions(validation)

    # ---- 1 tuning ------------------------------------------------------------------------
    L.stage = '1/7 hyperparameter tuning'

    def tune(family, i, params):
        model_path = out / 'tuning_models' / f'{family}__{i}.joblib'

        def fn():
            started = time.perf_counter()
            model = BenchModel(family, DEPLOYMENT_CORE, params).fit(train)
            fit_s = time.perf_counter() - started
            labels = np.asarray(model.classes_)
            raw = model.predict_proba(validation)
            sig = ProbabilityCalibrator('sigmoid', model.classes_).fit(model.predict_proba(calibration),
                                                                      calibration.true_cause.to_numpy())
            y = validation.true_cause.to_numpy()
            joblib.dump(model, model_path, compress=3)
            return {'trial': i, 'params': params, 'fit_seconds': fit_s,
                    'val_logloss_uncalibrated': float(log_loss(y, raw, labels=labels)),
                    'val_logloss_sigmoid': float(log_loss(y, sig.transform(raw), labels=labels)),
                    'val_macro_f1': float(f1_score(y, labels[raw.argmax(axis=1)], labels=labels, average='macro', zero_division=0))}
        return L.run('tuning', f'{family}__{i}', fn, required=[model_path])

    trials, best = {}, {}
    for family in FAMILIES:
        found = [tune(family, i, p) for i, p in enumerate(trial_params(family, cfg.trials))]
        found = [r for r in found if r is not None]
        if found:
            trials[family] = found
            best[family] = min(found, key=lambda r: r['val_logloss_sigmoid'])
    atomic_write_json(out / 'tuning.json', {'trials': trials, 'best_params': {f: b['params'] for f, b in best.items()}})
    live = [f for f in FAMILIES if f in best]
    if not live:
        L.log('No model family produced a tuned model; see logs/errors.log', 'ERROR')
        return finish(out, L, cfg, ok=False)
    best_params = {f: best[f]['params'] for f in live}

    # ---- 2 deployment_core validation, 3 freeze ----------------------------------------------
    def validation_unit(config, family):
        features = configs[config]
        bundle_path = out / 'models' / f'{config}__{family}.joblib'

        def fn():
            if bundle_path.exists():
                model = joblib.load(bundle_path)['model']
            elif config == 'deployment_core':
                model = joblib.load(out / 'tuning_models' / f'{family}__{best[family]["trial"]}.joblib')
            else:
                model = BenchModel(family, features, best_params[family]).fit(train)
            calibrators = fit_calibrators(model, calibration)
            joblib.dump({'model': model, 'calibrators': calibrators}, bundle_path, compress=3)
            return {'results': evaluate_partitions(model, calibrators, vparts, ('sigmoid',)),
                    'model_sha256': file_hash(bundle_path), 'n_features': len(features), 'params': best_params[family]}
        return L.run('validation', f'{config}__{family}', fn, required=[bundle_path])

    L.stage = '2/7 deployment_core validation'
    for family in live:
        validation_unit('deployment_core', family)
    selection = freeze_selection(out, L, live, best_params, configs)
    if selection is None:
        return finish(out, L, cfg, ok=False)
    method_of = selection['calibration_method_by_family']
    live = [f for f in live if f in method_of]

    L.stage = '4/7 feature-set and ablation validation'
    for config in configs:
        for family in live:
            validation_unit(config, family)

    # ---- 5 one-shot test ------------------------------------------------------------------------
    L.stage = '5/7 generating fresh test worlds'
    L.write_progress()
    try:
        fresh, seeds, leak = load_final(out / 'cache', workers, cfg)
    except Exception:                                   # noqa: BLE001
        L.error('fresh worlds', traceback.format_exc())
        L.log('Could not generate the fresh test worlds; rerun to retry (see logs/errors.log)', 'ERROR')
        return finish(out, L, cfg, ok=False)
    L.log(f'fresh worlds ready; future-leak mismatches: {leak}')
    parts = {}
    for scenario in SCENARIOS:
        parts[f'test/{scenario}'] = pooled(fresh, 'final_test', (scenario,))
        parts[f'shifted/{scenario}'] = pooled(fresh, 'final_shift', (scenario,))
    parts['test/pooled'] = pooled(fresh, 'final_test')
    parts['shifted/pooled'] = pooled(fresh, 'final_shift')
    main_sets = ('deployment_core', 'full_57', 'selected_42', 'deployment_with_timing')

    L.stage = '5/7 one-shot test evaluation'
    for config in configs:
        for family in live:
            def fn(config=config, family=family):
                path = out / 'models' / f'{config}__{family}.joblib'
                if not L.done('validation', f'{config}__{family}'):
                    raise RuntimeError('no validation result for this model; rerun to retry it')
                if file_hash(path) != L.load('validation', f'{config}__{family}')['model_sha256']:
                    raise RuntimeError(f'{path.name} changed after validation')
                bundle = joblib.load(path)
                full = ('sigmoid', 'isotonic', 'uncalibrated') if config in main_sets else (method_of[family],)
                return evaluate_partitions(bundle['model'], bundle['calibrators'], parts, full)
            L.run('test', f'{config}__{family}', fn)

    L.stage = '6/7 unseen regime, latency, memory'
    mild = ('calm', 'typical')
    train_mild = cap(pooled(data, 'train', mild), rows[0])
    calib_mild = cap(pooled(data, 'calibration', mild), rows[2])
    for family in live:
        def unseen(family=family):
            model = BenchModel(family, DEPLOYMENT_CORE, best_params[family]).fit(train_mild)
            cal = fit_calibrators(model, calib_mild)
            return evaluate_partitions(model, cal, {'stressed_unseen': parts['test/stressed'],
                                                    'stressed_shifted_unseen': parts['shifted/stressed']}, (method_of[family],))
        L.run('unseen', family, unseen)

        def perf(family=family):
            path = out / 'models' / f'deployment_core__{family}.joblib'
            bundle = joblib.load(path)
            cal = bundle['calibrators'][method_of[family]]
            return {**latency_profile(bundle['model'], cal, parts['test/typical'], cfg.latency_repeats), **memory_profile(path)}
        L.run('perf', family, perf)

    return finish(out, L, cfg, ok=True, selection=selection, live=live, parts=parts, seeds=seeds, leak=leak, configs=configs)


def freeze_selection(out, L, live, best_params, configs):
    path = out / 'selection.json'
    if path.exists():
        L.log('selection.json already frozen; reusing it')
        return json.loads(path.read_text())
    summary = {}
    for family in live:
        if not L.done('validation', f'deployment_core__{family}'):
            L.log(f'{family} has no deployment_core validation; excluded from selection', 'WARN')
            continue
        pooled_val = L.load('validation', f'deployment_core__{family}')['results']['pooled']
        scores = {m: (v['calibration']['logloss'] if 'calibration' in v else v['logloss']) for m, v in pooled_val.items()}
        method = min(scores, key=scores.get)
        v = pooled_val[method]
        flat = ({'ece': v['calibration']['ece_top_label_10_bins'], 'macro_f1': v['macro_f1'],
                 'unsafe_rate': v['safety']['unsafe_retry_gated']['rate']} if 'calibration' in v else
                {'ece': v['ece_top_label_10_bins'], 'macro_f1': v['macro_f1'], 'unsafe_rate': v['unsafe_retry_rate_gated']})
        summary[family] = {'calibration': method, 'logloss': scores[method], **flat}
    if not summary:
        L.log('Nothing to select from', 'ERROR')
        return None
    winner, why = select_model(summary)
    selection = {'frozen': True, 'frozen_at_unix': time.time(), 'rule': SELECTION_RULE,
                 'deployment_core_validation_summary': summary, 'winner': winner, 'selection_trace': why,
                 'calibration_method_by_family': {f: s['calibration'] for f, s in summary.items()},
                 'best_params': best_params,
                 'deployment_core_model_sha256': {f: L.load('validation', f'deployment_core__{f}')['model_sha256'] for f in summary},
                 'feature_configs': {k: list(v) for k, v in configs.items()},
                 'final_test_worlds_generated_before_freeze': False, 'final_test_seeds': FINAL_SEED}
    atomic_write_json(path, selection)
    L.log(f'SELECTION FROZEN on validation data: winner = {winner} ({why})')
    return selection


def write_environment(out):
    import sklearn
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'cpus': os.cpu_count(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}
    for lib in ('xgboost', 'catboost', 'matplotlib'):
        try:
            info[lib] = __import__(lib).__version__
        except Exception:
            info[lib] = 'MISSING'
    atomic_write_json(Path(out) / 'environment.json', info)


# --------------------------------------------------------------------------- aggregate, report, package, share
def aggregate(out, L, live, configs, parts, seeds, leak, selection):
    out = Path(out)
    results, tests, missing = {}, {}, []
    for config in configs:
        results[config], tests[config] = {}, {}
        for family in live:
            name = f'{config}__{family}'
            if L.done('validation', name):
                results[config][family] = L.load('validation', name)['results']
            else:
                missing.append(f'validation/{name}')
            if L.done('test', name):
                tests[config][family] = L.load('test', name)
            else:
                missing.append(f'test/{name}')
    atomic_write_json(out / 'validation.json', {'results': results})
    unseen = {f: L.load('unseen', f) for f in live if L.done('unseen', f)}
    perf = {f: L.load('perf', f) for f in live if L.done('perf', f)}
    missing += [f'unseen/{f}' for f in live if f not in unseen] + [f'perf/{f}' for f in live if f not in perf]
    report = {'protocol': {'selection': selection['rule'], 'winner_frozen_before_test': selection['winner'],
                           'final_seeds': seeds, 'leak_check_mismatches': leak,
                           'train_worlds': 'Phase 3 train, pooled over 3 scenarios',
                           'calibration_worlds': 'Phase 3 test worlds (unused by any fit or selection)',
                           'synthetic_only': True, 'serving_changed': False},
              'selection': {k: selection[k] for k in ('winner', 'selection_trace', 'deployment_core_validation_summary',
                                                      'calibration_method_by_family', 'best_params')},
              'results': tests, 'unseen_regime': unseen, 'performance': perf,
              'partition_sizes': {k: int(len(v)) for k, v in parts.items()}, 'missing_units': missing}
    atomic_write_json(out / 'test_report.json', report)
    return missing


def finish(out, L, cfg, ok, selection=None, live=None, parts=None, seeds=None, leak=None, configs=None):
    out = Path(out)
    if not ok:
        L.stage = 'STOPPED EARLY (see logs/errors.log); rerun the same command to resume'
        L.write_progress()
        return False
    L.stage = '7/7 report, package, share bundle'
    L.write_progress()
    missing = aggregate(out, L, live, configs, parts, seeds, leak, selection)
    if missing:
        L.log(f'{len(missing)} units missing/failed: {missing[:6]} ... report will show gaps; rerun to retry', 'WARN')
    steps = {}
    try:
        from eval import phase4_report
        phase4_report.build(out, out / 'report')
        steps['report'] = 'ok'
    except Exception:                                   # noqa: BLE001
        L.error('report', traceback.format_exc())
        steps['report'] = 'FAILED (logs/errors.log)'
    try:
        from eval import phase4_package
        phase4_package.package(out, (out / 'artifacts') if cfg.name != 'full' else Path('eval/artifacts/phase4'))
        steps['package'] = 'ok'
    except Exception:                                   # noqa: BLE001
        L.error('package', traceback.format_exc())
        steps['package'] = 'FAILED (logs/errors.log)'
    if cfg.run_pytest:
        started = time.time()
        try:
            proc = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider'],
                                  capture_output=True, text=True, timeout=3 * 3600)
            atomic_write_text(out / 'logs' / 'pytest.log', proc.stdout + '\n' + proc.stderr)
            steps['pytest'] = f'exit {proc.returncode}: ' + (proc.stdout.strip().splitlines() or ['no output'])[-1]
        except Exception as exc:                        # noqa: BLE001
            steps['pytest'] = f'FAILED to run: {exc}'
        L.log(f'pytest tests: {steps["pytest"]} ({time.time() - started:.0f}s)')
    share_bundle(out, L, selection, steps, missing)
    gaps = missing or any(v.startswith('FAILED') for v in steps.values())
    L.stage = 'COMPLETE WITH GAPS (see SUMMARY.md)' if gaps else 'COMPLETE'
    L.write_progress()
    L.log(f'=== {L.stage}. Share: {out / "share"}')
    return True


def share_bundle(out, L, selection, steps, missing):
    out = Path(out)
    share = out / 'share'
    share.mkdir(exist_ok=True)
    lines = [f'Phase 4 benchmark summary  (run dir: {out})', f'winner (frozen on validation): {selection["winner"]}',
             f'selection trace: {json.dumps(selection["selection_trace"])}', f'steps: {json.dumps(steps)}',
             f'failed units: {L.failed() or "none"}; missing in report: {missing or "none"}', '',
             'Open report/generated-tables.md for every table. Figures are in report/figures/.']
    tables = out / 'report' / 'generated-tables.md'
    if tables.exists():
        sections = tables.read_text(encoding='utf-8').split('\n### ')
        lines += ['', '--- headline tables ---'] + ['### ' + s for s in sections[1:3]]
    atomic_write_text(share / 'SUMMARY.md', '\n'.join(lines) + '\n')
    archive = share / 'phase4_results.zip'
    tmp = archive.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('selection.json', 'tuning.json', 'validation.json', 'test_report.json', 'environment.json',
                     'progress.json', 'PROGRESS.txt'):
            if (out / name).exists():
                z.write(out / name, name)
        for sub in ('report', 'logs'):
            if (out / sub).exists():
                for p in sorted((out / sub).rglob('*')):
                    if p.is_file():
                        z.write(p, p.relative_to(out).as_posix())
        z.write(share / 'SUMMARY.md', 'SUMMARY.md')
    os.replace(tmp, archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', nargs='?', default='all', choices=('all',))
    parser.add_argument('--output-dir', default='eval/data/phase4_run')
    parser.add_argument('--workers', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--smoke', action='store_true', help='tiny end-to-end check in a separate output dir')
    args = parser.parse_args()
    run_all(args.output_dir, SMOKE if args.smoke else FULL, args.workers)
