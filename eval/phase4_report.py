"""Render Phase 4 figures and generated tables from the run JSON.

    python -m eval.phase4_report --run-dir eval/data/phase4_run --out docs/phase4

Reads test_report.json / validation.json / tuning.json / selection.json; writes docs/phase4/figures/*.png and
docs/phase4/generated-tables.md. Numbers in the narrative docs are copied from these tables, never typed twice.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eval.phase4_models import FAMILIES, FAMILY_LABEL

COLOR = {'hgb': '#2a6f97', 'xgboost': '#d1495b', 'catboost': '#edae49', 'random_forest': '#66a182'}
SCEN = ('calm', 'typical', 'stressed')
MAIN = ('deployment_core', 'deployment_with_timing', 'selected_42', 'full_57')
PCT = lambda v: '-' if v is None else f'{100 * v:.1f}%'
F3 = lambda v: '-' if v is None else f'{v:.3f}'


def flat(report):
    """Uniform view over a full report (has nested blocks) or a headline dict."""
    if 'confidence' in report:
        c, k, s = report['confidence'], report['calibration'], report['safety']
        return {'macro_f1': report['macro_f1'], 'accuracy': report['accuracy'], 'ece': k['ece_top_label_10_bins'],
                'logloss': k['logloss'], 'brier': k['brier_multiclass'], 'lt60': c['share_below_60'],
                'mid': c['share_60_to_80'], 'gt80': c['share_above_80'], 'acc_ge60': c['accuracy_when_confident_ge_60'],
                'err_ge60': c['error_rate_when_confident_ge_60'], 'unsafe': s['unsafe_retry_gated']['count'],
                'hard': s['hard_declines'], 'unsafe_rate': s['unsafe_retry_gated']['rate'],
                'unsafe_upper': s['unsafe_retry_gated']['upper95'], 'recall': s['retryable_recall_gated'],
                'abstain': s['abstention_rate'], 'n': c['n']}
    return {'macro_f1': report['macro_f1'], 'accuracy': report['accuracy'], 'ece': report['ece_top_label_10_bins'],
            'logloss': report['logloss'], 'brier': report['brier_multiclass'], 'lt60': report['share_below_60'],
            'mid': report['share_60_to_80'], 'gt80': report['share_above_80'], 'acc_ge60': report['accuracy_when_confident_ge_60'],
            'err_ge60': report['error_rate_when_confident_ge_60'], 'unsafe': report['unsafe_retries_gated'],
            'hard': report['hard_declines'], 'unsafe_rate': report['unsafe_retry_rate_gated'], 'unsafe_upper': None,
            'recall': report['retryable_recall_gated'], 'abstain': report['abstention_rate'], 'n': report['n']}


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def build(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    test = json.loads((run_dir / 'test_report.json').read_text())
    selection = json.loads((run_dir / 'selection.json').read_text())
    tuning = json.loads((run_dir / 'tuning.json').read_text())
    validation = json.loads((run_dir / 'validation.json').read_text())['results']
    method_of = selection['calibration_method_by_family']
    R = test['results']
    perf = test['performance']
    sections = []

    def chosen(config, family, partition='test/pooled', method=None):
        return R[config][family][partition][method or method_of[family]]

    # 1 headline comparison -----------------------------------------------------------------
    for config in MAIN:
        if config not in R:
            continue
        rows = []
        for fam in FAMILIES:
            m = flat(chosen(config, fam))
            rows.append([FAMILY_LABEL[fam] + (' *' if fam == selection['winner'] and config == 'deployment_core' else ''),
                         method_of[fam], F3(m['macro_f1']), F3(m['accuracy']), F3(m['ece']), F3(m['logloss']), F3(m['brier']),
                         PCT(m['lt60']), PCT(m['mid']), PCT(m['gt80']), PCT(m['acc_ge60']), PCT(m['err_ge60']),
                         f'{m["unsafe"]}/{m["hard"]}', PCT(m['recall']), PCT(m['abstain'])])
        sections.append(f'### Test, all scenarios pooled, feature set `{config}`\n\n' + md_table(
            ['Model', 'Calibration', 'Macro-F1', 'Acc', 'ECE', 'Log loss', 'Brier', '<60%', '60-80%', '>80%',
             'Acc @>=60%', 'Err @>=60%', 'Unsafe retries (gated)', 'Retryable recall', 'Abstain'], rows))

    # 2 per scenario / shifted market ---------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        row = [FAMILY_LABEL[fam]]
        for part in [f'test/{s}' for s in SCEN] + [f'shifted/{s}' for s in SCEN]:
            m = flat(chosen('deployment_core', fam, part))
            row.append(f'{m["macro_f1"]:.3f} / {PCT(m["lt60"])} / {m["unsafe"]}')
        rows.append(row)
    sections.append('### Deployment-core model by scenario: macro-F1 / share below 60% / gated unsafe retries\n\n' + md_table(
        ['Model'] + [f'{s}' for s in SCEN] + [f'{s} (shifted market)' for s in SCEN], rows))

    # 3 feature sets x models ------------------------------------------------------------------
    configs = list(R)
    rows = []
    for config in configs:
        rows.append([config] + [F3(flat(chosen(config, f))['macro_f1']) for f in FAMILIES] +
                    [F3(flat(chosen(config, f))['logloss']) for f in FAMILIES])
    sections.append('### Feature configurations (test, pooled): macro-F1 then log loss, calibrated\n\n' + md_table(
        ['Config'] + [f'F1 {FAMILY_LABEL[f]}' for f in FAMILIES] + [f'LL {FAMILY_LABEL[f]}' for f in FAMILIES], rows))

    # 4 calibration comparison -----------------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        for method in ('uncalibrated', 'sigmoid', 'isotonic'):
            m = flat(R['deployment_core'][fam]['test/pooled'][method])
            rows.append([FAMILY_LABEL[fam], method, F3(m['macro_f1']), F3(m['ece']), F3(m['logloss']), F3(m['brier']),
                         PCT(m['lt60']), PCT(m['acc_ge60']), m['unsafe']])
    sections.append('### Calibration methods (deployment_core, test pooled)\n\n' + md_table(
        ['Model', 'Method', 'Macro-F1', 'ECE', 'Log loss', 'Brier', '<60%', 'Acc @>=60%', 'Gated unsafe'], rows))

    # 5 correct vs incorrect ---------------------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        for method in ('uncalibrated', method_of[fam]):
            c = R['deployment_core'][fam]['test/pooled'][method]['confidence']
            good, bad = c['confidence_correct'], c['confidence_incorrect']
            rows.append([FAMILY_LABEL[fam], method, F3(good['mean']), F3(good['quantiles']['p50']), PCT(good['share_below_60']),
                         F3(bad['mean']), F3(bad['quantiles']['p50']), F3(bad['quantiles']['p90']), PCT(bad['share_below_60'])])
    sections.append('### Max-probability for correct vs incorrect predictions (deployment_core, test pooled)\n\n' + md_table(
        ['Model', 'Method', 'Correct mean', 'Correct median', 'Correct <60%', 'Wrong mean', 'Wrong median', 'Wrong p90', 'Wrong <60%'], rows))

    # 6 subsets -----------------------------------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        subsets = chosen('deployment_core', fam)['subsets']
        for name, v in subsets.items():
            if 'accuracy' not in v:
                continue
            rows.append([FAMILY_LABEL[fam], name, v['n'], PCT(v['accuracy']), F3(v['macro_f1_present_classes']),
                         PCT(v['share_below_60']), PCT(v['accuracy_when_confident_ge_60']), f'{v["confident_errors_per_1000"]:.0f}', F3(v['ece'])])
    sections.append('### Hard subsets (deployment_core, test pooled, calibrated)\n\n' + md_table(
        ['Model', 'Subset', 'n', 'Acc', 'Macro-F1*', '<60%', 'Acc @>=60%', 'Confident errors /1000', 'ECE'], rows) +
        '\n\n*macro over the classes present in the subset.')

    # 7 per-class ------------------------------------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        pc = chosen('deployment_core', fam)['per_class']
        for cls, v in pc.items():
            rows.append([FAMILY_LABEL[fam], cls, v['support'], F3(v['precision']), F3(v['recall']), F3(v['f1'])])
    sections.append('### Per-class precision / recall (deployment_core, test pooled, calibrated)\n\n' + md_table(
        ['Model', 'Class', 'Support', 'Precision', 'Recall', 'F1'], rows))

    # 8 ablations ---------------------------------------------------------------------------------------
    rows = []
    for base in ('full_57', 'selected_42', 'deployment_with_timing', 'deployment_core'):
        if base not in R:
            continue
        parent = {f: flat(chosen(base, f)) for f in FAMILIES}
        rows.append([base, '(parent)'] + [f'{parent[f]["macro_f1"]:.3f} / {PCT(parent[f]["lt60"])}' for f in FAMILIES])
        for suffix in ('no_reason', 'no_duration', 'no_headline3', 'no_headline3_or_derived'):
            name = f'{base}-{suffix}'
            if name not in R:
                continue
            row = [name, suffix]
            for f in FAMILIES:
                m = flat(chosen(name, f))
                row.append(f'{m["macro_f1"]:.3f} ({m["macro_f1"] - parent[f]["macro_f1"]:+.3f}) / {PCT(m["lt60"])}')
            rows.append(row)
    sections.append('### Ablations: macro-F1 (change vs parent) / share below 60%, test pooled, calibrated\n\n' + md_table(
        ['Config', 'Removed'] + [FAMILY_LABEL[f] for f in FAMILIES], rows))

    # 9 unseen regime ---------------------------------------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        for part in ('stressed_unseen', 'stressed_shifted_unseen'):
            m = flat(test['unseen_regime'][fam][part][method_of[fam]])
            ref = flat(chosen('deployment_core', fam, 'test/stressed' if part == 'stressed_unseen' else 'shifted/stressed'))
            rows.append([FAMILY_LABEL[fam], part, F3(m['macro_f1']), F3(ref['macro_f1']), F3(m['ece']), F3(ref['ece']),
                         PCT(m['lt60']), f'{m["unsafe"]}/{m["hard"]}', f'{ref["unsafe"]}/{ref["hard"]}'])
    sections.append('### Trained on calm+typical only, scored on stressed (vs trained on all three)\n\n' + md_table(
        ['Model', 'Test', 'Macro-F1 unseen', 'Macro-F1 seen', 'ECE unseen', 'ECE seen', '<60% unseen', 'Unsafe unseen', 'Unsafe seen'], rows))

    # 10 performance / tuning -----------------------------------------------------------------------------
    rows = [[FAMILY_LABEL[f], f'{perf[f]["single_request_ms_median"]:.2f}', f'{perf[f]["single_request_ms_p95"]:.2f}',
             f'{perf[f]["batch_rows_per_second_1thread"]:.0f}', F3(perf[f].get('rss_mb_after_load')) if perf[f].get('rss_mb_after_load') is not None else '-',
             f'{perf[f]["artifact_mb_on_disk"]:.2f}'] for f in FAMILIES]
    sections.append('### Single-request inference (one thread, preprocessing + model + calibrator) and memory\n\n' + md_table(
        ['Model', 'Median ms', 'p95 ms', 'Rows/s batch', 'RSS after load (MB)', 'Artifact (MB)'], rows))
    rows = []
    for f in FAMILIES:
        for t in tuning['trials'][f]:
            rows.append([FAMILY_LABEL[f], t['trial'], json.dumps(t['params']), F3(t['val_logloss_sigmoid']),
                         F3(t['val_logloss_uncalibrated']), F3(t['val_macro_f1']), f'{t["fit_seconds"]:.0f}'])
    sections.append('### Hyperparameter search (validation; criterion = sigmoid-calibrated log loss)\n\n' + md_table(
        ['Model', 'Trial', 'Params', 'Val LL (sigmoid)', 'Val LL (raw)', 'Val macro-F1 (raw)', 'Fit s'], rows))
    rows = [[FAMILY_LABEL[f], v['calibration'], F3(v['macro_f1']), F3(v['logloss']), F3(v['ece']), PCT(v['unsafe_rate'])]
            for f, v in selection['deployment_core_validation_summary'].items()]
    sections.append(f'### Frozen selection (validation, deployment_core): winner = {FAMILY_LABEL[selection["winner"]]}\n\n' + md_table(
        ['Model', 'Best calibration', 'Macro-F1', 'Log loss', 'ECE', 'Gated unsafe rate'], rows) +
        f'\n\nTrace: `{json.dumps(selection["selection_trace"])}`')
    (out_dir / 'generated-tables.md').write_text('# Phase 4 generated tables\n\nProduced by `eval/phase4_report.py` from the run JSON; do not edit.\n\n' + '\n\n'.join(sections) + '\n', encoding='utf-8')

    figures(R, selection, method_of, test, fig_dir)


def figures(R, selection, method_of, test, fig_dir):
    labels = [c for c in next(iter(R['deployment_core'].values()))['test/pooled']['sigmoid']['confusion_matrix']['labels']]
    short = [l.replace('BD_', '').replace('TD_', '').replace('_', '\n') for l in labels]
    # reliability diagrams
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True)
    for j, fam in enumerate(FAMILIES):
        for i, method in enumerate(('uncalibrated', method_of[fam])):
            ax = axes[i, j]
            bins = [b for b in R['deployment_core'][fam]['test/pooled'][method]['confidence']['reliability_bins'] if b['n']]
            ax.plot([0, 1], [0, 1], color='#999', lw=1, ls='--')
            ax.plot([b['mean_confidence'] for b in bins], [b['accuracy'] for b in bins], marker='o', color=COLOR[fam], lw=2)
            ax.axvline(.6, color='#bbb', lw=.8)
            ax.set_title(f'{FAMILY_LABEL[fam]}\n{method}', fontsize=10)
            if j == 0:
                ax.set_ylabel('accuracy in bin')
            if i == 1:
                ax.set_xlabel('mean predicted confidence')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.2)
    fig.suptitle('Reliability diagrams, deployment_core, test (all scenarios). Grey line = 60% gate.', fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / 'reliability.png', dpi=130); plt.close(fig)
    # confidence histograms
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    edges = np.linspace(0, 1, 11)
    for j, fam in enumerate(FAMILIES):
        for i, method in enumerate(('uncalibrated', method_of[fam])):
            ax = axes[i, j]
            h = R['deployment_core'][fam]['test/pooled'][method]['confidence']['histogram_10_bins']
            ax.bar(edges[:-1] + .05, h['correct'], width=.09, color='#2a6f97', label='correct')
            ax.bar(edges[:-1] + .05, h['incorrect'], width=.09, bottom=h['correct'], color='#d1495b', label='wrong')
            ax.axvline(.6, color='#555', lw=.8, ls='--')
            ax.set_title(f'{FAMILY_LABEL[fam]}\n{method}', fontsize=10)
            if j == 0:
                ax.set_ylabel('predictions')
            if i == 1:
                ax.set_xlabel('max predicted probability')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Confidence distribution: correct vs wrong predictions (dashed = 60%)', fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / 'confidence-histograms.png', dpi=130); plt.close(fig)
    # confusion matrices (row-normalised)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, fam in zip(axes, FAMILIES):
        cm = np.array(R['deployment_core'][fam]['test/pooled'][method_of[fam]]['confusion_matrix']['rows_true_cols_predicted'], float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(norm):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > .55 else 'black')
        ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(short, fontsize=6); ax.set_yticklabels(short, fontsize=6)
        ax.set_title(FAMILY_LABEL[fam], fontsize=10); ax.set_xlabel('predicted'); ax.set_ylabel('true')
    fig.suptitle('Confusion matrices (row-normalised recall), deployment_core, test pooled', fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / 'confusion-matrices.png', dpi=130); plt.close(fig)
    # risk-coverage
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for fam in FAMILIES:
        rc = R['deployment_core'][fam]['test/pooled'][method_of[fam]]['confidence']['risk_coverage']
        xs = [rc[t]['coverage'] for t in rc]; ys = [rc[t]['accuracy'] for t in rc]
        ax.plot(xs, ys, marker='o', color=COLOR[fam], label=FAMILY_LABEL[fam])
    ax.set_xlabel('coverage (share answered at threshold .5 to .9)'); ax.set_ylabel('accuracy on answered'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax.set_title('Abstention trade-off, deployment_core, test pooled', fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / 'risk-coverage.png', dpi=130); plt.close(fig)
    # share below 60: uncal vs sigmoid vs isotonic
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = .26
    for k, method in enumerate(('uncalibrated', 'sigmoid', 'isotonic')):
        vals = [R['deployment_core'][f]['test/pooled'][method]['confidence']['share_below_60'] if 'confidence' in R['deployment_core'][f]['test/pooled'][method]
                else R['deployment_core'][f]['test/pooled'][method]['share_below_60'] for f in FAMILIES]
        ax.bar(np.arange(4) + (k - 1) * width, [100 * v for v in vals], width, label=method)
    ax.set_xticks(range(4)); ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=8)
    ax.set_ylabel('% of predictions with max prob < 60%'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    ax.set_title('Predictions below the 60% confidence gate, by calibration', fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / 'share-below-60.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='eval/data/phase4_run')
    parser.add_argument('--out', default='docs/phase4')
    args = parser.parse_args()
    build(args.run_dir, args.out)
