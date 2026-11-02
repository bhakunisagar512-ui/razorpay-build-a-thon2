"""Phase 3 experiment: sequences@v4 worlds, 5 vs 26 vs 57 features, group ablation, selection.

Reproduce (container):  python -m eval.phase3 --output-dir /tmp/phase3
Nothing here touches the serving classifier, recovery workflows, or earlier reports.

Protocol (fixed before looking at results):
  * Separate worlds (distinct seeds => distinct customers, outage calendars, noise) for
    train / validation / test, plus a shifted-market test (different issuer/PSP baselines).
  * All feature selection and ablation uses validation worlds only. The test worlds are
    scored once, after the selected set is frozen.
  * Same model family and hyperparameters as the existing offline classifier
    (HistGradientBoosting, max_iter=200, random_state=0); only the input columns differ.
  * The existing safety gate is applied unchanged (confidence 0.60, hard-mass 0.03).
    Its legacy error-tuple rules are keyed on the old vocabulary and never fire here.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder

from eval.phase3_quality import (aggregate_quality, lookup_ceilings, single_feature_probe, world_quality, write_json)
from eval.probability_calibration import ProbabilityCalibrator, calibration_metrics
from eval.safety_pipeline import apply_policy, metrics as gate_metrics, prepare
from eval.sequences_v4 import assemble_v4, generate_world, rebuild_features_at
from services.app.domain.diagnosis_safety import SafetyThresholds
from services.app.domain.features import BASELINE_FEATURES, FEATURES
from services.app.domain.features_v4 import (CATEGORICAL_FEATURES_V4, FEATURE_GROUPS_V4, FEATURES_V4,
                                             NEW_FEATURES_V4)

SCENARIOS = ('calm', 'typical', 'stressed')
ROLE_SEED = {'train': 100000, 'validation': 200000, 'test': 300000, 'shifted_test': 400000}
ROLE_COUNT = {'train': 4, 'validation': 2, 'test': 3, 'shifted_test': 2}
ROLE_MARKET = {'train': 11, 'validation': 11, 'test': 11, 'shifted_test': 12}
GATE = SafetyThresholds(.60, .03)  # frozen: the Phase 2.5 control gate, never retuned here
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
RETRYABLE_CAUSES = ('BD_INSUFFICIENT_FUNDS', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
# Predeclared selection rules.
GROUP_MIN_SCENARIOS = 2        # a group must help in this many of the 3 scenarios
FEATURE_LOFO_DLOGLOSS = .001   # leave-one-feature-out logloss increase that counts as useful
FEATURE_PERM_DLOGLOSS = .002   # permutation logloss increase that counts as useful
BOOTSTRAPS = 300


def world_seed(scenario, role, index):
    return ROLE_SEED[role] + 1000 * SCENARIOS.index(scenario) + index


def source_signature(n):
    files = ['eval/sequences_v4.py', 'eval/world_config_v4.py', 'eval/phase3_quality.py',
             'services/app/domain/features_v4.py', 'services/app/domain/features.py',
             'services/app/domain/failure_vocab.py']
    digest = hashlib.sha256(str(n).encode())
    for name in files:
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- worlds
def _same(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b or (isinstance(b, float) and abs(float(a) - b) <= 1e-9 * max(1., abs(b)))


def generate_job(job):
    scenario, role, index, n, cache, signature, leakage_checks = job
    path = Path(cache) / f'{scenario}_{role}_{index}.pkl'
    if path.exists():
        with path.open('rb') as stream:
            payload = pickle.load(stream)
        if payload['signature'] == signature:
            return str(path)
    seed = world_seed(scenario, role, index)
    started = time.perf_counter()
    world = generate_world(n, seed, scenario=scenario, market_seed=ROLE_MARKET[role])
    generated = time.perf_counter() - started
    dataset = assemble_v4(world)
    quality = world_quality(dataset)
    quality.update(seed=seed, generation_seconds=generated, total_seconds=time.perf_counter() - started)
    if leakage_checks and index == 0:
        rng = np.random.default_rng(seed)
        sample = rng.choice(dataset.failures.payment_id.to_numpy(), size=5, replace=False)
        mismatches = 0
        for pid in sample:
            rebuilt = rebuild_features_at(world, pid)
            stored = dataset.failures[dataset.failures.payment_id == pid].iloc[0]
            mismatches += sum(not _same(stored[k], v) for k, v in rebuilt.items())
        quality['future_invariance'] = {'failures_checked': len(sample), 'feature_mismatches': int(mismatches)}
    with path.open('wb') as stream:
        pickle.dump({'signature': signature, 'failures': dataset.failures, 'quality': quality, 'role': role,
                     'scenario': scenario, 'seed': seed}, stream)
    return str(path)


def load_scenario(cache, scenario):
    data, quality = {}, {}
    for role, count in ROLE_COUNT.items():
        frames, qualities = [], []
        for index in range(count):
            with (Path(cache) / f'{scenario}_{role}_{index}.pkl').open('rb') as stream:
                payload = pickle.load(stream)
            frame = payload['failures'].copy()
            frame['world_index'] = index
            frames.append(frame)
            qualities.append(payload['quality'])
        data[role], quality[role] = frames, qualities
    return data, quality


# --------------------------------------------------------------------------- models
class Phase3Model:
    """Existing offline recipe (one-hot categoricals + HGB, 200 iterations) on any column set."""

    def __init__(self, features):
        self.features = tuple(features)
        self.categorical = [f for f in self.features if f in CATEGORICAL_FEATURES_V4]
        self.numeric = [f for f in self.features if f not in CATEGORICAL_FEATURES_V4]
        parts = []
        if self.categorical:
            parts.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical))
        if self.numeric:
            parts.append(('numeric', 'passthrough', self.numeric))
        self.encoder = ColumnTransformer(parts)
        self.model = HistGradientBoostingClassifier(max_iter=200, random_state=0)

    def _prep(self, frame):
        data = frame[list(self.features)].copy()
        for name in self.categorical:
            data[name] = data[name].astype(str)
        for name in self.numeric:
            data[name] = pd.to_numeric(data[name], errors='coerce').astype(float)
        return data

    def fit(self, frame):
        self.model.fit(self.encoder.fit_transform(self._prep(frame)), frame.true_cause.to_numpy())
        self.classes_ = [str(c) for c in self.model.classes_]
        self.known_tuples = set(frame[['gw_error_code', 'gw_error_source', 'gw_error_step']]
                                .itertuples(index=False, name=None))
        return self

    def predict_proba(self, frame):
        return self.model.predict_proba(self.encoder.transform(self._prep(frame)))


def fast_macro_f1(y, pred, k):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(k), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(k), where=cm.sum(axis=1) > 0)
    denominator = precision + recall
    return float(np.divide(2 * precision * recall, denominator, out=np.zeros(k), where=denominator > 0).mean())


def fit_score(name, features, train, validation):
    model = Phase3Model(features).fit(train)
    probabilities = model.predict_proba(validation)
    labels = np.asarray(model.classes_)
    y = validation.true_cause.to_numpy()
    pred = labels[probabilities.argmax(axis=1)]
    return name, {'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
                  'accuracy': float(accuracy_score(y, pred)),
                  'logloss': float(log_loss(y, probabilities, labels=labels)), 'pred': pred}


def paired_bootstrap(y, pred_a, pred_b, labels, seed=0, draws=BOOTSTRAPS):
    """Distribution of macro-F1(a) - macro-F1(b) over resampled failures (same rows for both)."""
    index = {label: i for i, label in enumerate(labels)}
    y_i = np.array([index[v] for v in y])
    a_i, b_i = np.array([index[v] for v in pred_a]), np.array([index[v] for v in pred_b])
    rng = np.random.default_rng(seed)
    k = len(labels)
    deltas = []
    for _ in range(draws):
        s = rng.integers(0, len(y_i), len(y_i))
        deltas.append(fast_macro_f1(y_i[s], a_i[s], k) - fast_macro_f1(y_i[s], b_i[s], k))
    deltas = np.array(deltas)
    return {'delta_mean': float(deltas.mean()), 'lo95': float(np.quantile(deltas, .025)),
            'hi95': float(np.quantile(deltas, .975))}


def permutation_job(model, validation, feature, repeats, seed):
    y = validation.true_cause.to_numpy()
    labels = np.asarray(model.classes_)
    base = model.predict_proba(validation)
    base_loss = log_loss(y, base, labels=labels)
    base_f1 = f1_score(y, labels[base.argmax(axis=1)], labels=labels, average='macro', zero_division=0)
    losses, f1s = [], []
    for r in range(repeats):
        shuffled = validation.copy()
        shuffled[feature] = np.random.default_rng(seed + r).permutation(shuffled[feature].to_numpy())
        p = model.predict_proba(shuffled)
        losses.append(log_loss(y, p, labels=labels) - base_loss)
        f1s.append(base_f1 - f1_score(y, labels[p.argmax(axis=1)], labels=labels, average='macro', zero_division=0))
    return feature, float(np.mean(losses)), float(np.mean(f1s))


# --------------------------------------------------------------------------- pass 1: validation analyses
def validation_analysis(scenario, data, workers):
    train, validation = pd.concat(data['train'], ignore_index=True), pd.concat(data['validation'], ignore_index=True)
    variants = {'baseline_5': BASELINE_FEATURES, 'existing_26': FEATURES, 'full_57': FEATURES_V4}
    for group, names in FEATURE_GROUPS_V4.items():
        variants[f'drop:{group}'] = tuple(f for f in FEATURES_V4 if f not in names)
        variants[f'add:{group}'] = (*FEATURES, *names)
    for feature in NEW_FEATURES_V4:
        variants[f'lofo:{feature}'] = tuple(f for f in FEATURES_V4 if f != feature)
    results = dict(Parallel(n_jobs=workers)(delayed(fit_score)(name, features, train, validation)
                                            for name, features in variants.items()))
    labels = sorted(train.true_cause.unique())
    y = validation.true_cause.to_numpy()
    ablation = {}
    for group in FEATURE_GROUPS_V4:
        ablation[group] = {
            'drop_from_57': {**{k: results[f'drop:{group}'][k] for k in ('macro_f1', 'accuracy', 'logloss')},
                             'loss_vs_full_57': paired_bootstrap(y, results['full_57']['pred'], results[f'drop:{group}']['pred'], labels)},
            'add_to_26': {**{k: results[f'add:{group}'][k] for k in ('macro_f1', 'accuracy', 'logloss')},
                          'gain_vs_existing_26': paired_bootstrap(y, results[f'add:{group}']['pred'], results['existing_26']['pred'], labels)}}
    lofo = {f: {'dlogloss': results[f'lofo:{f}']['logloss'] - results['full_57']['logloss'],
                'dmacro_f1': results['full_57']['macro_f1'] - results[f'lofo:{f}']['macro_f1']} for f in NEW_FEATURES_V4}
    full_model = Phase3Model(FEATURES_V4).fit(train)
    permutation = {f: {'dlogloss': dl, 'dmacro_f1': df} for f, dl, df in Parallel(n_jobs=workers)(
        delayed(permutation_job)(full_model, validation, f, 3, 17) for f in FEATURES_V4)}
    return {'scenario': scenario, 'validation_failures': len(validation), 'train_failures': len(train),
            'sets': {k: {m: v[m] for m in ('macro_f1', 'accuracy', 'logloss')} for k, v in results.items()
                     if not k.startswith(('drop:', 'add:', 'lofo:'))},
            'ablation': ablation, 'leave_one_feature_out': lofo, 'permutation_importance': permutation}


def select_features(analyses):
    """Apply the predeclared rules. Test data is never consulted."""
    retained_groups, group_evidence = [], {}
    for group, names in FEATURE_GROUPS_V4.items():
        wins_drop = sum(a['ablation'][group]['drop_from_57']['loss_vs_full_57']['lo95'] > 0 for a in analyses.values())
        wins_add = sum(a['ablation'][group]['add_to_26']['gain_vs_existing_26']['lo95'] > 0 for a in analyses.values())
        keep = wins_drop >= GROUP_MIN_SCENARIOS or wins_add >= GROUP_MIN_SCENARIOS
        group_evidence[group] = {'scenarios_where_dropping_hurts_significantly': wins_drop,
                                 'scenarios_where_adding_helps_significantly': wins_add, 'retained': keep}
        if keep:
            retained_groups.append(group)
    selected, feature_evidence = [], {}
    for group in retained_groups:
        for feature in FEATURE_GROUPS_V4[group]:
            lofo_votes = sum(a['leave_one_feature_out'][feature]['dlogloss'] > FEATURE_LOFO_DLOGLOSS for a in analyses.values())
            perm_votes = sum(a['permutation_importance'][feature]['dlogloss'] > FEATURE_PERM_DLOGLOSS for a in analyses.values())
            keep = lofo_votes >= GROUP_MIN_SCENARIOS or perm_votes >= GROUP_MIN_SCENARIOS
            feature_evidence[feature] = {'group': group, 'lofo_votes': lofo_votes, 'permutation_votes': perm_votes,
                                         'retained': keep,
                                         'mean_lofo_dlogloss': float(np.mean([a['leave_one_feature_out'][feature]['dlogloss'] for a in analyses.values()])),
                                         'mean_permutation_dlogloss': float(np.mean([a['permutation_importance'][feature]['dlogloss'] for a in analyses.values()]))}
            if keep:
                selected.append(feature)
    dropped = [f for f in NEW_FEATURES_V4 if f not in selected]
    return {'rules': {'group': f'95% paired-bootstrap lower bound of macro-F1 effect > 0 in >= {GROUP_MIN_SCENARIOS} of 3 scenarios, '
                               'for drop-from-57 OR add-to-26',
                      'feature': f'within retained groups: leave-one-out dlogloss > {FEATURE_LOFO_DLOGLOSS} OR permutation dlogloss > '
                                 f'{FEATURE_PERM_DLOGLOSS}, in >= {GROUP_MIN_SCENARIOS} of 3 scenarios',
                      'scope': 'only the 31 new candidates are screened; the existing 26 are always kept'},
            'groups': group_evidence, 'features': feature_evidence,
            'selected_new_features': selected, 'dropped_new_features': dropped,
            'selected_feature_set': [*FEATURES, *selected]}


# --------------------------------------------------------------------------- pass 2: test evaluation
def upper95(k, n):
    return None if n == 0 else float(1.0 if k == n else beta.ppf(.95, k + 1, n - k))


def evaluate(model, calibrator, frames):
    labels = np.asarray(model.classes_)
    rows, ys, preds, raw_p, cal_p, unsafe_raw = [], [], [], [], [], 0
    gated = dict(unsafe=0, hard=0, retryable=0, retryable_allowed=0, abstain=0, n=0)
    for frame in frames:
        p = model.predict_proba(frame)
        pc = calibrator.transform(p)
        y, pred = frame.true_cause.to_numpy(), labels[p.argmax(axis=1)]
        data = prepare(frame, model, pc)
        gm = gate_metrics(data, apply_policy(data, GATE))
        hard = np.isin(y, HARD)
        raw_unsafe = int((hard & np.isin(pred, RETRYABLE_CAUSES)).sum())
        rows.append({'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
                     'accuracy': float(accuracy_score(y, pred)), 'gated_macro_f1': gm['macro_f1_all'],
                     'abstain_rate': gm['abstain_rate'], 'unsafe_retries_gated': gm['unsafe_retry_count'],
                     'unsafe_retries_raw': raw_unsafe, 'hard_declines': gm['hard_decline_count']})
        ys.append(y)
        preds.append(pred)
        raw_p.append(p)
        cal_p.append(pc)
        gated['unsafe'] += gm['unsafe_retry_count']
        gated['hard'] += gm['hard_decline_count']
        retryable = int(np.isin(y, RETRYABLE_CAUSES).sum())
        gated['retryable'] += retryable
        gated['retryable_allowed'] += round(gm['true_retryable_recall'] * retryable) if retryable else 0
        gated['abstain'] += round(gm['abstain_rate'] * len(y))
        gated['n'] += len(y)
    y, pred = np.concatenate(ys), np.concatenate(preds)
    p_raw, p_cal = np.vstack(raw_p), np.vstack(cal_p)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    matrix = pd.crosstab(pd.Categorical(y, categories=labels), pd.Categorical(pred, categories=labels), dropna=False)
    raw_unsafe_total = sum(r['unsafe_retries_raw'] for r in rows)
    return {
        'n_failures': int(len(y)), 'worlds': len(frames),
        'macro_f1': {'mean_over_worlds': float(np.mean([r['macro_f1'] for r in rows])),
                     'sd_over_worlds': float(np.std([r['macro_f1'] for r in rows], ddof=1)) if len(rows) > 1 else None,
                     'pooled': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0))},
        'accuracy': {'mean_over_worlds': float(np.mean([r['accuracy'] for r in rows])),
                     'pooled': float(accuracy_score(y, pred))},
        'per_class': {str(l): {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
                      for l, p, r, f, s in zip(labels, precision, recall, f1, support)},
        'confusion_matrix': {'labels': [str(l) for l in labels], 'rows_true_cols_predicted': matrix.to_numpy().tolist()},
        'calibration_uncalibrated': {**calibration_metrics(p_raw, y, list(labels)),
                                     'logloss': float(log_loss(y, p_raw, labels=labels))},
        'calibration_sigmoid': {**calibration_metrics(p_cal, y, list(labels)),
                                'logloss': float(log_loss(y, p_cal, labels=labels))},
        'unsafe_retry_raw': {'count': raw_unsafe_total, 'hard_declines': gated['hard'],
                             'rate': raw_unsafe_total / gated['hard'] if gated['hard'] else None,
                             'upper95': upper95(raw_unsafe_total, gated['hard'])},
        'unsafe_retry_gated': {'count': gated['unsafe'], 'hard_declines': gated['hard'],
                               'rate': gated['unsafe'] / gated['hard'] if gated['hard'] else None,
                               'upper95': upper95(gated['unsafe'], gated['hard']),
                               'per_world_counts': [r['unsafe_retries_gated'] for r in rows]},
        'gated_macro_f1_mean_over_worlds': float(np.mean([r['gated_macro_f1'] for r in rows])),
        'gate_abstention_rate': gated['abstain'] / gated['n'],
        'gate_true_retryable_recall': gated['retryable_allowed'] / gated['retryable'] if gated['retryable'] else None,
    }


def subset_metrics(model, frames, mask_column='is_generic_reason'):
    """Where the provider gave no usable reason (generic or missing): the cases ML must resolve."""
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame[mask_column] == 1]
    labels = np.asarray(model.classes_)
    pred = labels[model.predict_proba(frame).argmax(axis=1)]
    y = frame.true_cause.to_numpy()
    return {'n_failures': int(len(frame)), 'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, pred))}


def latency_profile(model, frame, repeats=200):
    one = frame.iloc[[0]]
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(one)
        times.append((time.perf_counter() - started) * 1000)
    batch = frame.iloc[:1000]
    started = time.perf_counter()
    model.predict_proba(batch)
    elapsed = time.perf_counter() - started
    return {'single_row_ms_median': float(np.median(times)), 'single_row_ms_p95': float(np.quantile(times, .95)),
            'batch_rows_per_second': float(len(batch) / elapsed)}


def fit_and_evaluate(sets, train, validation, test_by_name):
    """Train per feature set, calibrate on validation, evaluate on each named test partition."""
    out = {}
    for name, features in sets.items():
        model = Phase3Model(features).fit(train)
        calibrator = ProbabilityCalibrator('sigmoid', model.classes_).fit(model.predict_proba(validation),
                                                                         validation.true_cause.to_numpy())
        out[name] = {'n_features': len(features),
                     'evaluation': {part: evaluate(model, calibrator, frames) for part, frames in test_by_name.items()},
                     'generic_reason_subset': {part: subset_metrics(model, frames) for part, frames in test_by_name.items()},
                     'latency': latency_profile(model, next(iter(test_by_name.values()))[0])}
    return out


# --------------------------------------------------------------------------- orchestration
def build_quality(scenario_data, scenario_quality, skip_probe):
    quality = {s: aggregate_quality(s, scenario_quality[s]) for s in SCENARIOS}
    for s in SCENARIOS:
        invariance = [w.get('future_invariance') for ws in scenario_quality[s].values() for w in ws if w.get('future_invariance')]
        quality[s]['future_invariance'] = invariance
        quality[s]['future_leak_free'] = all(v['feature_mismatches'] == 0 for v in invariance) and bool(invariance)
        train_all = pd.concat(scenario_data[s]['train'], ignore_index=True)
        test_all = pd.concat(scenario_data[s]['test'], ignore_index=True)
        quality[s]['heldout_lookup_accuracy_train_to_test'] = lookup_ceilings(train_all, test_all)
        quality[s]['majority_class_accuracy_test'] = float(
            (test_all.true_cause == train_all.true_cause.value_counts().index[0]).mean())
        if not skip_probe:
            probe = single_feature_probe(train_all, test_all, FEATURES_V4)
            quality[s]['single_feature_heldout_accuracy_top10'] = dict(sorted(probe.items(), key=lambda kv: -kv[1])[:10])
    return quality


def run(output_dir, n, workers, skip_probe=False, quality_only=False):
    out = Path(output_dir)
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    signature = source_signature(n)
    started = time.perf_counter()
    if not quality_only:
        jobs = [(s, role, i, n, str(cache), signature, True) for s in SCENARIOS for role, c in ROLE_COUNT.items() for i in range(c)]
        print(f'Generating {len(jobs)} worlds with {workers} workers', flush=True)
        with ProcessPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(generate_job, jobs), 1):
                if i % 6 == 0 or i == len(jobs):
                    print(f'  worlds ready: {i}/{len(jobs)} ({time.perf_counter() - started:.0f}s)', flush=True)
    scenario_data, scenario_quality = {}, {}
    for s in SCENARIOS:
        scenario_data[s], scenario_quality[s] = load_scenario(cache, s)
    quality = build_quality(scenario_data, scenario_quality, skip_probe)
    write_json(out / 'data-quality.json', quality)
    print('Data-quality report written', flush=True)
    if quality_only:
        return quality

    print('Pass 1: validation-only ablations and selection', flush=True)
    analyses = {}
    for s in SCENARIOS:
        analyses[s] = validation_analysis(s, scenario_data[s], workers)
        print(f'  {s}: full_57 macro-F1 {analyses[s]["sets"]["full_57"]["macro_f1"]:.3f} '
              f'(26: {analyses[s]["sets"]["existing_26"]["macro_f1"]:.3f}, 5: {analyses[s]["sets"]["baseline_5"]["macro_f1"]:.3f})', flush=True)
    selection = select_features(analyses)
    write_json(out / 'ablation-and-selection.json', {'analyses': analyses, 'selection': selection})
    print(f'  retained {len(selection["selected_new_features"])} of 31 new features', flush=True)

    print('Pass 2: frozen-selection test evaluation', flush=True)
    sets = {'baseline_5': BASELINE_FEATURES, 'existing_26': FEATURES, 'expanded_57': FEATURES_V4,
            'selected': tuple(selection['selected_feature_set'])}
    results = {}
    for s in SCENARIOS:
        d = scenario_data[s]
        train, validation = pd.concat(d['train'], ignore_index=True), pd.concat(d['validation'], ignore_index=True)
        results[s] = fit_and_evaluate(sets, train, validation, {'test': d['test'], 'shifted_market_test': d['shifted_test']})
        print('  ' + s + ': ' + ', '.join(f'{k} F1={v["evaluation"]["test"]["macro_f1"]["mean_over_worlds"]:.3f}' for k, v in results[s].items()), flush=True)
    pooled_train = pd.concat([f for s in SCENARIOS for f in scenario_data[s]['train']], ignore_index=True)
    pooled_val = pd.concat([f for s in SCENARIOS for f in scenario_data[s]['validation']], ignore_index=True)
    pooled = fit_and_evaluate(sets, pooled_train, pooled_val, {s: scenario_data[s]['test'] for s in SCENARIOS})
    report = {
        'protocol': {'train_worlds': ROLE_COUNT['train'], 'validation_worlds': ROLE_COUNT['validation'],
                     'test_worlds': ROLE_COUNT['test'], 'shifted_market_test_worlds': ROLE_COUNT['shifted_test'],
                     'orders_per_world': n, 'scenarios': list(SCENARIOS), 'gate': {'confidence': .60, 'hard_mass': .03},
                     'model': 'HistGradientBoosting max_iter=200 random_state=0 (one-hot categoricals), unchanged recipe',
                     'selection_rules': selection['rules'], 'source_signature': signature,
                     'synthetic_only': True, 'serving_changed': False},
        'in_scenario': results, 'pooled_training_scenario_test': pooled,
        'selection': {k: selection[k] for k in ('selected_new_features', 'dropped_new_features', 'selected_feature_set')},
        'runtime_seconds': time.perf_counter() - started}
    write_json(out / 'report.json', report)
    print(f'Done in {report["runtime_seconds"]:.0f}s -> {out}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/phase3')
    parser.add_argument('--orders', type=int, default=24000)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--skip-probe', action='store_true')
    parser.add_argument('--quality-only', action='store_true', help='rebuild data-quality.json from cached worlds')
    args = parser.parse_args()
    run(args.output_dir, args.orders, args.workers, args.skip_probe, args.quality_only)
