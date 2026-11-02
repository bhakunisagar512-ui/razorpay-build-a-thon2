"""Phase 5A.2: three-way HGB comparison on the SAME held-out Phase 4 test records.

    A  original as-served 5-feature classifier (legacy-vocabulary training recipe from services.app.adapters.diagnosis._fit)
    B  the same 5-feature recipe retrained on the v4 train worlds (Phase3Model(BASELINE_FEATURES))   <- primary baseline
    C  the improved Phase 4 HGB, loaded from eval/artifacts/phase4 (43 deployment_core features), NOT retrained

Headline = B vs C (what feature engineering buys). A is reported separately (as-served behaviour, with a
legacy-vocabulary mismatch). Raw classifier metrics (argmax of probabilities) are kept apart from policy-adjusted
metrics (A: its native rules/thresholds; B and C: the frozen 0.60 confidence / 0.03 hard-decline-mass gate).

Run (container, from /code):   python -m eval.phase5_compare --out eval/data/phase5_comparison
Windows:                       powershell -ExecutionPolicy Bypass -File .\\scripts\\run_phase5_comparison.ps1
Read-only on Phase 3/4 artifacts: the fresh test worlds are loaded from the Phase 4 cache without regeneration;
nothing under eval/data/phase4_run or eval/artifacts/phase4 is written. Serving code is not touched.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')      # before numpy/sklearn; eval.phase3 would otherwise pin 1 thread

import argparse
import json
import pickle
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from eval import phase3, phase4
from eval.classifier import CONF_THRESHOLD, FEATURES as ORIGINAL_FEATURES, HARD_CAUSES, HARD_DECLINE_MASS_LIMIT, RULES
from eval.phase4_confidence import (GATE, HARD, RETRYABLE, calibration_block, classification_block, confidence_profile,
                                    safety_block, world_indices)
from eval.phase4_ledger import atomic_write_json, atomic_write_text
from services.app.domain.features import BASELINE_FEATURES

CACHE_P3 = Path('eval/data/phase3_run/cache')
CACHE_P4 = Path('eval/data/phase4_run/cache')
ARTIFACT = Path('eval/artifacts/phase4')
SCENARIOS = phase3.SCENARIOS
NAMES = {'A': 'A original as-served (legacy-trained)', 'B': 'B 5-feature retrained on v4', 'C': 'C Phase 4 HGB (43 features)'}


# --------------------------------------------------------------------------- data (read-only)
def check_fresh_cache(cache=CACHE_P4, orders=phase4.ORDERS, counts=None):
    """Fail loudly instead of regenerating: phase4.load_final would otherwise (re)write a stale or missing world."""
    counts = counts or phase4.FINAL_COUNT
    signature = phase4.generator_signature(orders)
    missing = []
    for role in ('final_test', 'final_shift'):        # load_final reads both roles, so both must already be cached
        for scenario in SCENARIOS:
            for index in range(counts[role]):
                path = Path(cache) / f'{scenario}_{role}_{index}.pkl'
                try:
                    with path.open('rb') as stream:
                        ok = pickle.load(stream)['signature'] == signature
                except Exception:
                    ok = False
                if not ok:
                    missing.append(path.name)
    if missing:
        raise FileNotFoundError(f'Phase 4 test worlds missing or built by different generator code in {cache}: {missing}. '
                                'This script never regenerates them (it must not touch Phase 4 artifacts).')


def load_test_frames(cache=CACHE_P4):
    check_fresh_cache(cache)
    fresh, seeds, leak = phase4.load_final(cache, 1, phase4.FULL)          # all cached: pure read
    test = phase4.pooled(fresh, 'final_test')
    return test, {k: v for k, v in seeds.items() if '/final_test/' in k}, leak


def load_training_b(cache=CACHE_P3):
    data = phase4.load_phase3_roles(cache)
    train = phase4.pooled(data, 'train')
    seeds = {f'{s}/train/{i}': phase3.world_seed(s, 'train', i) for s in SCENARIOS for i in range(phase3.ROLE_COUNT['train'])}
    return train, seeds


def build_a():
    """The exact serving recipe: services.app.adapters.diagnosis._fit (generate(20000, seed=42), hash split, fit)."""
    from eval.classifier import train_test_split_by_hash
    from eval.generate import generate
    from services.app.adapters import diagnosis
    clf = diagnosis._fit()
    df = generate(n=diagnosis.TRAIN_N, seed=diagnosis.TRAIN_SEED).drop_duplicates(subset=['payment_id'])
    train, own_test = train_test_split_by_hash(df, test_pct=30)
    return clf, train, own_test


def integrity(train_b, test, train_a=None, seeds_b=None, seeds_test=None):
    """Leakage checks between each model's training data and the held-out records."""
    out = {'test_rows': int(len(test)), 'test_payment_ids_unique': bool(test.payment_id.is_unique),
           'b_train_rows': int(len(train_b)),
           'b_train_vs_test_payment_id_overlap': int(len(set(train_b.payment_id) & set(test.payment_id))),
           'b_train_vs_test_world_overlap': sorted(set(train_b.world_key) & set(test.world_key))}
    if seeds_b and seeds_test:
        out['b_train_seed_range'] = [min(seeds_b.values()), max(seeds_b.values())]
        out['test_seed_range'] = [min(seeds_test.values()), max(seeds_test.values())]
        out['seed_ranges_disjoint'] = not (set(seeds_b.values()) & set(seeds_test.values()))
    if train_a is not None:
        out['a_train_rows'] = int(len(train_a))
        out['a_train_vs_test_payment_id_overlap'] = int(len(set(train_a.payment_id) & set(test.payment_id)))
    return out


def assert_aligned(frame, *models):
    """Every model scores the identical rows: same frame, unique ids, every required column present."""
    assert frame.payment_id.is_unique and len(frame) > 0
    for features in models:
        missing = [f for f in features if f not in frame.columns]
        assert not missing, f'columns missing from the shared records: {missing}'


# --------------------------------------------------------------------------- scoring
def to_labels(p, classes, labels):
    """Re-order/pad a probability matrix to the common label order."""
    out = np.zeros((len(p), len(labels)))
    for i, c in enumerate(classes):
        if c in labels:
            out[:, list(labels).index(c)] = p[:, i]
    return out


def vocabulary_mismatch(clf, frame):
    """How much of the v4 records the legacy-trained encoder has never seen."""
    out = {}
    for name, categories in zip(ORIGINAL_FEATURES, clf.enc.categories_):
        out[name] = {'seen_levels_in_training': sorted(map(str, categories)),
                     'share_of_test_rows_with_unseen_level': float((~frame[name].astype(str).isin(set(map(str, categories)))).mean())}
    known = set(zip(frame.gw_error_code, frame.gw_error_source, frame.gw_error_step)) & set(RULES)
    out['rows_matching_a_legacy_rule_tuple'] = float(np.mean([key in RULES for key in
                                                              zip(frame.gw_error_code, frame.gw_error_source, frame.gw_error_step)]))
    out['distinct_legacy_rule_tuples_present_in_test'] = len(known)
    return out


def raw_block(p, frame, labels):
    y = frame.true_cause.to_numpy()
    block = {**classification_block(p, y, labels), 'calibration': calibration_block(p, y, labels),
             'confidence': confidence_profile(p, y, labels)}
    scores = [classification_block(p[idx], y[idx], labels)['macro_f1'] for idx in world_indices(frame)]
    block['macro_f1_over_worlds'] = {'mean': float(np.mean(scores)), 'sd': float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                                     'n_worlds': len(scores)}
    return block


def native_a_block(clf, frame, labels):
    """A's own decision logic (rules, 0.60 confidence, 0.25 hard-decline mass, LLM tier off). AMBIGUOUS is a real output."""
    pred = clf.predict_df(frame[[*ORIGINAL_FEATURES, 'true_cause']])
    y = pred.true_cause.to_numpy()
    amb = (pred.pred_cause == 'AMBIGUOUS').to_numpy()
    decided = ~amb
    hard = np.isin(y, HARD)
    retryable = np.isin(y, RETRYABLE)
    retry_pred = np.isin(pred.pred_cause.to_numpy(), RETRYABLE)
    return {'policy': f'native: legacy rules + confidence<{CONF_THRESHOLD} + hard-decline mass>{HARD_DECLINE_MASS_LIMIT}; LLM tier off',
            'ambiguous_rate': float(amb.mean()), 'decided_rows': int(decided.sum()),
            'stage_mix': pred.pred_stage.value_counts(normalize=True).to_dict(),
            'decided_accuracy': float((pred.pred_cause.to_numpy()[decided] == y[decided]).mean()) if decided.any() else None,
            'decided_macro_f1': float(f1_score(y[decided], pred.pred_cause.to_numpy()[decided], labels=list(labels), average='macro',
                                               zero_division=0)) if decided.any() else None,
            'macro_f1_all_rows_ambiguous_counts_as_miss': float(f1_score(y, pred.pred_cause.to_numpy(), labels=list(labels),
                                                                         average='macro', zero_division=0)),
            'accuracy_all_rows_ambiguous_counts_as_miss': float((pred.pred_cause.to_numpy() == y).mean()),
            'unsafe_retries': int((hard & retry_pred).sum()), 'hard_declines': int(hard.sum()),
            'unsafe_per_hard_decline': float((hard & retry_pred).sum() / hard.sum()) if hard.any() else None,
            'retryable_recall': float((retryable & retry_pred).sum() / retryable.sum()) if retryable.any() else None,
            'confidence_returned_below_60_share': float((pred.pred_conf.to_numpy() < .60).mean())}


def policy_block(frame, model, p):
    """B and C under the identical frozen gate (confidence 0.60 / hard-decline mass 0.03; legacy tuple rules and the
    expired-instrument veto from eval.safety_pipeline). Not safety-policy v2."""
    s = safety_block(frame, model, p)
    return {'policy': 'frozen gate: confidence 0.60, hard-decline mass 0.03 (eval.safety_pipeline.apply_policy)',
            'abstention_rate': s['abstention_rate'], 'unsafe_retries': s['unsafe_retry_gated']['count'],
            'hard_declines': s['hard_declines'], 'unsafe_per_hard_decline': s['unsafe_retry_gated']['rate'],
            'unsafe_upper95': s['unsafe_retry_gated']['upper95'], 'retryable_recall': s['retryable_recall_gated'],
            'retry_allowed_share': s['recovery_eligibility']['retry_allowed_share'],
            'gated_macro_f1': s['gated_macro_f1'], 'gated_accuracy': s['gated_accuracy'], 'gate_decisions': s['gate_decisions']}


def latency(fn, frame, repeats=300, rows=200):
    """One request = one pandas row through preprocessing + model (+ calibrator for C), one thread, same rows for every model."""
    from threadpoolctl import threadpool_limits
    one = [frame.iloc[[i]] for i in range(min(rows, len(frame)))]
    with threadpool_limits(1):
        for r in one[:20]:
            fn(r)
        times = []
        for k in range(repeats):
            r = one[k % len(one)]
            started = time.perf_counter()
            fn(r)
            times.append((time.perf_counter() - started) * 1000)
        batch = frame.iloc[:2000]
        started = time.perf_counter()
        fn(batch)
        rate = len(batch) / (time.perf_counter() - started)
    return {'single_request_ms_median': float(np.median(times)), 'single_request_ms_p95': float(np.quantile(times, .95)),
            'batch_rows_per_second': float(rate), 'threads': 1, 'requests_timed': repeats}


# --------------------------------------------------------------------------- the comparison
def compare(test, train_b, clf_a, art_c, *, train_a=None, seeds_b=None, seeds_test=None, bootstrap=300, latency_repeats=300):
    """test: shared held-out frame. train_b: v4 training frame for B (fitted here, once). clf_a: fitted CauseClassifier.
    art_c: {'model', 'calibrator', 'schema'} as returned by diagnosis_v4.load_artifact. Nothing is written here."""
    started = time.perf_counter()
    model_c, calibrator_c = art_c['model'], art_c['calibrator']
    assert_aligned(test, ORIGINAL_FEATURES, BASELINE_FEATURES, model_c.features)
    labels = sorted(test.true_cause.unique())
    b_fit_started = time.perf_counter()
    model_b = phase3.Phase3Model(BASELINE_FEATURES).fit(train_b)
    b_fit_seconds = time.perf_counter() - b_fit_started
    assert list(model_b.classes_) == labels == list(model_c.classes_), 'B and C must share the label order used by the gate'

    def probs(frame):
        return {'A': to_labels(clf_a.model.predict_proba(clf_a.enc.transform(frame[list(ORIGINAL_FEATURES)]).toarray()), clf_a.classes_, labels),
                'B': to_labels(model_b.predict_proba(frame), model_b.classes_, labels),
                'C': to_labels(calibrator_c.transform(model_c.predict_proba(frame)), model_c.classes_, labels)}

    partitions = {'pooled': test, **{s: test[test.scenario == s].reset_index(drop=True) for s in SCENARIOS if (test.scenario == s).any()}}
    results, predictions = {}, {}
    for name, frame in partitions.items():
        p = probs(frame)
        predictions[name] = {k: np.asarray(labels)[v.argmax(axis=1)] for k, v in p.items()}
        results[name] = {
            'rows': int(len(frame)),
            'raw': {k: raw_block(v, frame, labels) for k, v in p.items()},
            'policy_adjusted': {'A_native': native_a_block(clf_a, frame, labels),
                                'B_frozen_gate': policy_block(frame, model_b, p['B']),
                                'C_frozen_gate': policy_block(frame, model_c, p['C'])}}
        y = frame.true_cause.to_numpy()
        results[name]['B_vs_C_macro_f1'] = {
            'delta_C_minus_B': float(results[name]['raw']['C']['macro_f1'] - results[name]['raw']['B']['macro_f1']),
            'paired_bootstrap_C_minus_B': phase3.paired_bootstrap(y, predictions[name]['C'], predictions[name]['B'], labels, draws=bootstrap)}
    one_frame = test
    lat = {
        'A': latency(lambda f: clf_a.model.predict_proba(clf_a.enc.transform(f[list(ORIGINAL_FEATURES)]).toarray()), one_frame, latency_repeats),
        'B': latency(lambda f: model_b.predict_proba(f), one_frame, latency_repeats),
        'C': latency(lambda f: calibrator_c.transform(model_c.predict_proba(f)), one_frame, latency_repeats)}
    lat['note'] = ('Same rows, one thread, one pandas row per request, preprocessing + model (+ calibrator for C). Feature CONSTRUCTION is '
                   'excluded: A and B read 5 fields of the event; C additionally needs 38 history/window features from a feature store.')
    meta = {
        'created_unix': time.time(), 'python': platform.python_version(), 'runtime_seconds': None,
        'test_rows': int(len(test)), 'test_worlds': sorted(set(test.world_key)),
        'models': {
            'A': {'description': NAMES['A'], 'features': list(ORIGINAL_FEATURES), 'n_features': len(ORIGINAL_FEATURES),
                  'one_hot_columns': int(len(clf_a.enc.get_feature_names_out())),
                  'training': 'eval.generate.generate(n=20000, seed=42) legacy profile, drop duplicate payment_id, hash split test_pct=30, fit on the 70% train part',
                  'train_rows': None if train_a is None else int(len(train_a)),
                  'hyperparameters': {'HistGradientBoosting': {'max_iter': 200, 'random_state': 0}}, 'classes': list(clf_a.classes_)},
            'B': {'description': NAMES['B'], 'features': list(BASELINE_FEATURES), 'n_features': len(BASELINE_FEATURES),
                  'training': 'Phase 3 sequences@v4 train worlds, calm+typical+stressed pooled (same rows C was trained on)',
                  'train_rows': int(len(train_b)), 'train_seeds': seeds_b, 'fit_seconds': b_fit_seconds,
                  'hyperparameters': {'HistGradientBoosting': {'max_iter': 200, 'random_state': 0}}, 'classes': list(model_b.classes_)},
            'C': {'description': NAMES['C'], 'features': list(model_c.features), 'n_features': len(model_c.features),
                  'artifact_model_id': art_c['schema'].get('model_id'), 'artifact_sha256': art_c['schema'].get('model_sha256'),
                  'calibration': art_c['schema'].get('calibration'), 'hyperparameters': art_c['schema'].get('hyperparameters'),
                  'training': 'Phase 3 sequences@v4 train worlds, pooled; loaded, not retrained', 'classes': list(model_c.classes_)}},
        'integrity': integrity(train_b, test, train_a, seeds_b, seeds_test),
        'a_vocabulary_vs_test': vocabulary_mismatch(clf_a, test),
        'labels': labels, 'bootstrap_draws': bootstrap}
    meta['runtime_seconds'] = time.perf_counter() - started
    return {'meta': meta, 'results': results, 'latency': lat}


# --------------------------------------------------------------------------- outputs
def confusion_files(metrics, out_dir):
    out_dir = Path(out_dir)
    labels = metrics['meta']['labels']
    raw = metrics['results']['pooled']['raw']
    for key in 'ABC':
        cm = pd.DataFrame(raw[key]['confusion_matrix']['rows_true_cols_predicted'], index=[f'true_{l}' for l in labels],
                          columns=[f'pred_{l}' for l in labels])
        cm.to_csv(out_dir / f'confusion_{key}_raw.csv')


def figures(metrics, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    labels = metrics['meta']['labels']
    short = [l.replace('BD_', '').replace('TD_', '').replace('_', '\n') for l in labels]
    raw = metrics['results']['pooled']['raw']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key in zip(axes, 'ABC'):
        cm = np.array(raw[key]['confusion_matrix']['rows_true_cols_predicted'], float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(norm):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > .55 else 'black')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short, fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
        ax.set_title(f'{NAMES[key]}\nmacro-F1 {raw[key]["macro_f1"]:.3f}', fontsize=9)
        ax.set_xlabel('predicted (argmax)')
        ax.set_ylabel('true')
    fig.suptitle('Raw classifier confusion matrices (row-normalised recall), same test records', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrices.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=False)
    edges = np.linspace(0, 1, 11)
    for ax, key in zip(axes, 'ABC'):
        h = raw[key]['confidence']['histogram_10_bins']
        ax.bar(edges[:-1] + .05, h['correct'], width=.09, color='#2a6f97', label='correct')
        ax.bar(edges[:-1] + .05, h['incorrect'], width=.09, bottom=h['correct'], color='#d1495b', label='wrong')
        ax.axvline(.6, color='#555', lw=.8, ls='--')
        ax.set_title(f'{NAMES[key]}\n<60%: {100 * raw[key]["confidence"]["share_below_60"]:.1f}%', fontsize=9)
        ax.set_xlabel('max predicted probability')
        if key == 'A':
            ax.set_ylabel('predictions')
            ax.legend(fontsize=8)
    fig.suptitle('Confidence distribution, correct vs wrong (dashed = 60%)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'confidence_distribution.png', dpi=130)
    plt.close(fig)


def pct(v, digits=1):
    return '-' if v is None else f'{100 * v:.{digits}f}%'


def f3(v):
    return '-' if v is None else f'{v:.3f}'


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|',
                      *['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]])


def render_markdown(m):
    meta, res = m['meta'], m['results']
    raw, pol = res['pooled']['raw'], res['pooled']['policy_adjusted']
    labels = meta['labels']
    boot = res['pooled']['B_vs_C_macro_f1']['paired_bootstrap_C_minus_B']
    mod = meta['models']
    ig = meta['integrity']
    out = ['# Phase 5A.2: HGB model comparison (A original, B 5-feature on v4, C Phase 4)', '',
           'Generated by `eval/phase5_compare.py`; do not edit by hand. Synthetic data only: these numbers measure fit to the simulator, '
           'not real Razorpay traffic.', '',
           '## 1. Setup', '',
           table(['Model', 'Features', 'Training data', 'Train rows', 'Hyperparameters'],
                 [['A (as-served)', 5, 'legacy generator, seed 42, 70% hash split', mod['A']['train_rows'], 'HGB max_iter=200, random_state=0'],
                  ['B (5-feature, v4)', 5, 'Phase 3 v4 train worlds (pooled, 12 worlds)', mod['B']['train_rows'], 'HGB max_iter=200, random_state=0'],
                  ['C (Phase 4)', mod['C']['n_features'], 'Phase 3 v4 train worlds (pooled), not retrained', '-',
                   json.dumps(mod['C']['hyperparameters'])]]),
           '', f'Held-out records: **{meta["test_rows"]:,} failures** from {len(meta["test_worlds"])} fresh Phase 4 `final_test` worlds '
                f'(seeds {ig.get("test_seed_range", ["?", "?"])[0]} to {ig.get("test_seed_range", ["?", "?"])[1]}), identical for A, B and C. '
                f'C artifact: `{mod["C"]["artifact_model_id"]}` (calibration: {mod["C"]["calibration"]}).', '',
           '**Leakage checks:** B train/test payment_id overlap = ' + str(ig['b_train_vs_test_payment_id_overlap']) +
           '; shared worlds = ' + str(ig['b_train_vs_test_world_overlap'] or 'none') +
           '; seed ranges disjoint = ' + str(ig.get('seed_ranges_disjoint')) +
           ('; A train/test payment_id overlap = ' + str(ig['a_train_vs_test_payment_id_overlap']) if 'a_train_vs_test_payment_id_overlap' in ig else '') + '.', '',
           '**Differences from the as-served model that you must keep in mind:**', '',
           '* **B is not the serving model.** It uses the same classifier recipe and the same five columns, but it is trained on v4 data '
           '(about 3x the rows, v4 vocabulary) with no legacy rules, no 0.25 hard-decline gate and no abstention logic. '
           'It isolates the value of the extra features, not the behaviour of the deployed system.',
           '* **A sees a vocabulary it was not trained on.** The legacy generator uses error sources/steps such as `bank` / `authorization`; '
           'sequences@v4 uses Razorpay-style values such as `issuer_bank` / `payment_authorization`. See section 6.',
           '* **Different decision policies are reported in separate tables:** A uses its native rules and thresholds; B and C use the frozen '
           '0.60 / 0.03 gate. Raw (argmax) metrics are policy-free and are the only like-for-like comparison.', '',
           '## 2. Headline: B vs C (raw classifier, policy-free)', '']
    rows = []
    for part in ('pooled', *[s for s in SCENARIOS if s in res]):
        b, c = res[part]['raw']['B'], res[part]['raw']['C']
        rows.append([part, f'{res[part]["rows"]:,}', f3(b['macro_f1']), f3(c['macro_f1']), f'{c["macro_f1"] - b["macro_f1"]:+.3f}',
                     f3(b['accuracy']), f3(c['accuracy']), pct(b['confidence']['share_below_60']), pct(c['confidence']['share_below_60'])])
    out += [table(['Partition', 'Failures', 'B macro-F1', 'C macro-F1', 'C - B', 'B acc', 'C acc', 'B <60%', 'C <60%'], rows), '',
            f'Paired bootstrap (pooled, {meta["bootstrap_draws"]} draws): macro-F1 gain C - B = {boot["delta_mean"]:+.3f} '
            f'(95% CI {boot["lo95"]:+.3f} to {boot["hi95"]:+.3f}). Across-world macro-F1 mean +/- sd: '
            f'B {raw["B"]["macro_f1_over_worlds"]["mean"]:.3f} +/- {raw["B"]["macro_f1_over_worlds"]["sd"] or 0:.3f}, '
            f'C {raw["C"]["macro_f1_over_worlds"]["mean"]:.3f} +/- {raw["C"]["macro_f1_over_worlds"]["sd"] or 0:.3f}.', '',
            '## 3. All three, raw classifier (pooled)', '']
    rows = []
    for key in 'ABC':
        r, cf, cal = raw[key], raw[key]['confidence'], raw[key]['calibration']
        rows.append([NAMES[key], f3(r['accuracy']), f3(r['macro_f1']), pct(cf['share_below_60']), pct(cf['share_60_to_80']),
                     pct(cf['share_above_80']), pct(cf['accuracy_when_confident_ge_60']), f3(cf['confidence_correct']['mean']),
                     f3(cf['confidence_incorrect']['mean']), f3(cal['ece_top_label_10_bins']), f3(cal['logloss']), f3(cal['brier_multiclass'])])
    out += [table(['Model', 'Acc', 'Macro-F1', '<60%', '60-80%', '>80%', 'Acc @>=60%', 'Mean conf (correct)', 'Mean conf (wrong)', 'ECE', 'Log loss', 'Brier'], rows), '',
            '![confidence](../../eval/data/phase5_comparison/confidence_distribution.png)', '',
            '## 4. Per-class precision / recall / F1 (pooled, raw)', '']
    rows = []
    for label in labels:
        row = [label, raw['A']['per_class'][label]['support']]
        for key in 'ABC':
            v = raw[key]['per_class'][label]
            row += [f'{v["precision"]:.3f} / {v["recall"]:.3f} / {v["f1"]:.3f}']
        rows.append(row)
    out += [table(['Class', 'Support', 'A  P / R / F1', 'B  P / R / F1', 'C  P / R / F1'], rows), '',
            'Confusion matrices: `confusion_{A,B,C}_raw.csv` and `confusion_matrices.png` in `eval/data/phase5_comparison/`.', '',
            '![confusion](../../eval/data/phase5_comparison/confusion_matrices.png)', '',
            '## 5. Policy-adjusted behaviour (different policies, do not compare A to B/C directly)', '']
    a = pol['A_native']
    out += [f'**A, native logic** ({a["policy"]}): AMBIGUOUS rate **{pct(a["ambiguous_rate"])}**; stage mix '
            + ', '.join(f'{k} {pct(v)}' for k, v in sorted(a['stage_mix'].items())) + f'. On decided rows: accuracy {f3(a["decided_accuracy"])}, '
            f'macro-F1 {f3(a["decided_macro_f1"])}. Counting AMBIGUOUS as a miss: accuracy {f3(a["accuracy_all_rows_ambiguous_counts_as_miss"])}, '
            f'macro-F1 {f3(a["macro_f1_all_rows_ambiguous_counts_as_miss"])}. Unsafe retries {a["unsafe_retries"]}/{a["hard_declines"]} '
            f'({pct(a["unsafe_per_hard_decline"], 2)}); retryable recall {pct(a["retryable_recall"])}.', '']
    rows = []
    for key in ('B_frozen_gate', 'C_frozen_gate'):
        p = pol[key]
        rows.append([key[0], pct(p['abstention_rate']), f'{p["unsafe_retries"]}/{p["hard_declines"]}', pct(p['unsafe_upper95'], 2),
                     pct(p['retryable_recall']), pct(p['retry_allowed_share']), f3(p['gated_macro_f1']), f3(p['gated_accuracy'])])
    out += [f'**B and C, same frozen gate** ({pol["B_frozen_gate"]["policy"]}):', '',
            table(['Model', 'Abstention', 'Unsafe retries / hard declines', 'Unsafe upper 95%', 'Retryable recall', 'Retry allowed', 'Gated macro-F1', 'Gated acc'], rows), '',
            'The Phase 4 follow-up policy (`safety-policy@v2`) is **not** applied here; it is not activated.', '',
            '## 6. A and the legacy vocabulary mismatch', '']
    vm = meta['a_vocabulary_vs_test']
    rows = [[name, pct(vm[name]['share_of_test_rows_with_unseen_level'])] for name in ORIGINAL_FEATURES]
    out += [table(['A input feature', 'Share of test rows with a level A never saw in training'], rows), '',
            f'Rows whose (code, source, step) matches one of A\'s legacy hard-coded rules: {pct(vm["rows_matching_a_legacy_rule_tuple"])} '
            f'(so its rule stage is largely inert on v4). Unseen categories become all-zero one-hot blocks, so part of A\'s weakness here is '
            f'data-vocabulary shift, not a lack of features. This is why B, not A, is the baseline for the value of feature engineering.', '',
            '## 7. Prediction latency (consistent method)', '']
    lat = m['latency']
    out += [table(['Model', 'Median ms', 'p95 ms', 'Rows/s (batch, 1 thread)'],
                  [[NAMES[k], f'{lat[k]["single_request_ms_median"]:.2f}', f'{lat[k]["single_request_ms_p95"]:.2f}',
                    f'{lat[k]["batch_rows_per_second"]:.0f}'] for k in 'ABC']), '', lat['note'], '',
            '## 8. Limitations', '',
            '* Synthetic sequences@v4 data only; the label is the simulator\'s cause and `error_reason` is emitted from that cause, so gains from '
            'reason-based features are partly a property of the simulator (see `docs/phase4/feature-availability-audit.md`).',
            '* C\'s extra features need point-in-time history (instrument, issuer, customer windows) that must come from a feature store; A and B do not.',
            '* B was fitted once with seed 0; the across-world spread and the bootstrap interval describe test-sampling noise, not training-seed variance.',
            '* A vs B/C confounds model quality with data-vocabulary shift; only B vs C isolates features.',
            '* Policy-adjusted rows use different policies by design; unsafe-retry counts are small and carry wide upper bounds.',
            '* Latency excludes feature construction and any network/serialization overhead.', '',
            f'Runtime of this comparison: {meta["runtime_seconds"]:.0f}s.']
    return '\n'.join(out) + '\n'


def write_outputs(metrics, out_dir, docs_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / 'metrics.json', metrics)
    confusion_files(metrics, out_dir)
    figures(metrics, out_dir)
    md = render_markdown(metrics)
    atomic_write_text(out_dir / 'model_comparison.md', md)
    if docs_dir:
        docs_dir = Path(docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(docs_dir / 'model_comparison.md', md)
    return md


def main(out_dir, docs_dir=None, bootstrap=300, latency_repeats=300):
    from services.app.adapters import diagnosis_v4
    started = time.perf_counter()
    print('Loading held-out Phase 4 test worlds (read-only)...', flush=True)
    test, seeds_test, leak = load_test_frames()
    print(f'  {len(test):,} failures; future-leak mismatches recorded by Phase 4: {leak}', flush=True)
    print('Loading B training worlds (Phase 3 train)...', flush=True)
    train_b, seeds_b = load_training_b()
    print('Reconstructing A with the serving recipe (legacy data, seed 42)...', flush=True)
    clf_a, train_a, _ = build_a()
    print('Loading C artifact (no retraining)...', flush=True)
    art_c = diagnosis_v4.load_artifact(ARTIFACT)
    print('Fitting B once and evaluating A, B, C...', flush=True)
    metrics = compare(test, train_b, clf_a, art_c, train_a=train_a, seeds_b=seeds_b, seeds_test=seeds_test,
                      bootstrap=bootstrap, latency_repeats=latency_repeats)
    metrics['meta']['future_leak_mismatches_phase4_check'] = leak
    write_outputs(metrics, out_dir, docs_dir)
    pooled = metrics['results']['pooled']
    for key in 'ABC':
        r = pooled['raw'][key]
        print(f'  {key}: macro-F1 {r["macro_f1"]:.3f}  acc {r["accuracy"]:.3f}  <60% {100 * r["confidence"]["share_below_60"]:.1f}%', flush=True)
    print(f'  A native AMBIGUOUS rate: {100 * pooled["policy_adjusted"]["A_native"]["ambiguous_rate"]:.1f}%', flush=True)
    print(f'Done in {time.perf_counter() - started:.0f}s -> {out_dir}', flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='eval/data/phase5_comparison')
    parser.add_argument('--docs-dir', default=None, help='also write model_comparison.md here (docs/ is not mounted in the container)')
    parser.add_argument('--bootstrap', type=int, default=300)
    parser.add_argument('--latency-repeats', type=int, default=300)
    args = parser.parse_args()
    main(args.out, args.docs_dir, args.bootstrap, args.latency_repeats)
