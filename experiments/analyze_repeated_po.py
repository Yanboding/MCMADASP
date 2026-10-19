"""Local-only batch inference. This module never imports or invokes an optimizer."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import geom, t

from experiments.repeated_po_schema import match_manifest, validate_batch, validate_group


def expected_tau(sampling):
    """Analytic mean under the exact clipped inverse-CDF law used by the code."""
    proposal = sampling['proposal']
    if (proposal['type'] != 'stratified_geometric'
            or sampling['tau_definition'] != 'arrival_length_plus_one'
            or sampling['weight_convention'] != 'stratum_mass_over_count'):
        raise ValueError('cannot establish the zero mean of the length control for this sampling law')
    q = float(proposal['discount_factor_proposal'])
    low, high = sampling['quantile_clip']
    if not 0 < q < 1 or not 0 < low < high < 1:
        raise ValueError('invalid geometric probability or clipping endpoints')
    lo, hi = int(geom.ppf(low, 1-q)), int(geom.ppf(high, 1-q))
    # E[clamp(L,lo,hi)] = lo + sum_{k=lo}^{hi-1} P(L>k).
    return 1. + lo + q**lo * (-math.expm1((hi-lo) * math.log(q))) / (1-q)


def batch_observation(batch):
    validate_batch(batch)
    anchor = np.asarray(batch['reference']['coefficients'], dtype=float)
    upper = batch['training']['objective_upper_bound']
    baseline = {'alp': 0., 'zero': 0.}
    raw = {'alp': 0., 'zero': 0.}
    substitutions = {'alp': 0, 'zero': 0}
    for path in batch['paths']:
        for name in baseline:
            record = path['ir'][name]
            # Even a numerically optimal solve retains its certified lower value.
            value = record['lower_bound']
            substitutions[name] += int(not record['exact'])
            if name == 'zero':
                value += float(np.asarray(path['policy_gradient']) @ anchor)
            baseline[name] += path['weight'] * value
            gap = path[f'raw_gap_fitted_minus_{name}']
            if gap is None:
                raw[name] = None
            elif raw[name] is not None:
                raw[name] += path['weight'] * gap
    mean_tau = expected_tau(batch['sampling'])
    return {'batch_id': batch['batch_id'], 'repetition': batch['repetition'],
            'R_alp': upper - baseline['alp'], 'R_zero': upper - baseline['zero'],
            'H': sum(p['weight'] * (p['termination_time'] - mean_tau) for p in batch['paths']),
            'expected_tau': mean_tau, 'raw_gain_alp': raw['alp'], 'raw_gain_zero': raw['zero'],
            'conservative_baselines': substitutions,
            'solver_absolute_gap': batch['training']['absolute_gap'],
            'termination_reason': batch['training']['termination_reason']}


def student_summary(values, half_width_threshold=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError('at least two finite independent batch values required')
    n, mean = len(values), float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    half = float(t.ppf(.975, n-1) * se)
    answer = {'batches': n, 'mean': mean, 'standard_error': se,
              'ci95': [mean-half, mean+half], 'half_width': half, 'width': 2*half,
              'one_sided_upper95': mean + float(t.ppf(.95, n-1)) * se}
    if half_width_threshold is not None:
        if half_width_threshold <= 0:
            raise ValueError('predeclared half-width threshold must be positive')
        answer['predeclared_half_width_threshold'] = half_width_threshold
        answer['threshold_met'] = half <= half_width_threshold
    return answer


def loo_adjust(values, control):
    values, control = np.asarray(values, dtype=float), np.asarray(control, dtype=float)
    if values.shape != control.shape or values.ndim != 1 or len(values) < 3:
        raise ValueError('LOO requires at least three paired batch records')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(control)):
        raise ValueError('nonfinite LOO input')
    slopes, fallback = np.zeros(len(values)), np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        keep = np.arange(len(values)) != i  # position, including bootstrap duplicates
        x, y = control[keep], values[keep]
        dx, dy = x-x.mean(), y-y.mean()
        denominator = float(dx @ dx)
        tolerance = np.finfo(float).eps * max(1., float(x @ x)) * 100
        if denominator <= tolerance:
            fallback[i] = True
        else:
            slopes[i] = float(dx @ dy) / denominator
    return values - slopes * control, slopes, fallback


def cv_summary(values, control, bootstrap_count=5000, seed=20261018):
    values, control = np.asarray(values, dtype=float), np.asarray(control, dtype=float)
    if values.shape != (len(control), 2) or bootstrap_count < 1:
        raise ValueError('expected two paired benchmark columns and positive bootstrap count')
    adjustments = [loo_adjust(values[:, j], control) for j in range(2)]
    estimates = np.array([a[0].mean() for a in adjustments])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(control), size=(bootstrap_count, len(control)))
    bootstrap = np.empty((bootstrap_count, 2))
    fallback_count = np.zeros(2, dtype=int)
    for i, draw in enumerate(indices):
        for j in range(2):
            adjusted, _, fallback = loo_adjust(values[draw, j], control[draw])
            bootstrap[i, j] = adjusted.mean()
            fallback_count[j] += fallback.sum()
    result = {'bootstrap_count': bootstrap_count, 'bootstrap_seed': seed,
              'resampling_unit': 'complete independent training batch',
              'selection_note': 'Exploratory if requested after viewing baseline width; retain both outputs.'}
    for j, name in enumerate(('alp', 'zero')):
        errors = bootstrap[:, j] - estimates[j]
        low, high = estimates[j]-np.quantile(errors, .975), estimates[j]-np.quantile(errors, .025)
        result[name] = {'mean': float(estimates[j]), 'ci95': [float(low), float(high)],
                        'one_sided_upper95': float(estimates[j]-np.quantile(errors, .05)),
                        'width': float(high-low), 'slopes': adjustments[j][1].tolist(),
                        'adjusted_batches': adjustments[j][0].tolist(),
                        'zero_variance_folds': int(adjustments[j][2].sum()),
                        'bootstrap_zero_variance_folds': int(fallback_count[j])}
    return result


def analyze(manifest, results_root, mode='baseline', half_width_threshold=None,
            bootstrap_count=5000, seed=20261018):
    declared = [c['config_id'] for c in manifest['configurations']]
    if not declared or len(set(declared)) != len(declared):
        raise ValueError('empty or duplicate manifest configurations')
    if {r['config_id'] for r in manifest['runs']} != set(declared):
        raise ValueError('manifest configurations do not match its runs')
    if {d['config_id'] for d in manifest['display_settings']} != set(declared):
        raise ValueError('display settings do not match manifest configurations')
    for identity in declared:
        runs = [r for r in manifest['runs'] if r['config_id'] == identity]
        if len(runs) != manifest['expected_batches']:
            raise ValueError(f'configuration {identity} has missing or duplicate runs')
    expected = {run['batch_id']: run for run in manifest['runs']}
    if len(expected) != len(manifest['runs']):
        raise ValueError('duplicate manifest batch IDs')
    found = {}
    for path in sorted(Path(results_root).rglob('batch.json')):
        batch = json.loads(path.read_text())
        identity = batch.get('batch_id')
        if identity not in expected:
            raise ValueError(f'unexpected result {path}; use the campaign-specific results directory')
        if identity in found:
            raise ValueError(f'duplicate result for {identity}')
        match_manifest(batch, expected[identity])
        found[identity] = batch
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError(f'missing {len(missing)} batches; no CI produced: {missing[:10]}')
    groups = defaultdict(list)
    for batch in found.values():
        groups[batch['config_id']].append(batch)
    output = {'mode': mode, 'configurations': {}, 'display_settings': manifest['display_settings'],
              'inference_note': ('The two-sided interval targets the expectation of a certified optimized SAA upper estimate, '
                                 'not the improvement itself. The one-sided limit bounds potential improvement approximately. '
                                 'Student-t and batch-bootstrap intervals do not guarantee finite-sample coverage. '
                                 'All comparisons concern the same all-period continuous relaxation.'),
              'independent_unit': 'training batch; path gaps are diagnostic only'}
    for identity, batches in sorted(groups.items()):
        batches = validate_group(batches, manifest['expected_batches'], manifest['expected_paths'])
        observations = [batch_observation(b) for b in batches]
        values = np.array([[r['R_alp'], r['R_zero']] for r in observations])
        baseline = {name: student_summary(values[:, j], half_width_threshold) for j, name in enumerate(('alp', 'zero'))}
        result = {'baseline': baseline, 'batches': observations, 'complete': True}
        if mode == 'cv':
            result['cv'] = cv_summary(values, [r['H'] for r in observations], bootstrap_count, seed)
            for name in ('alp', 'zero'):
                old, new = baseline[name]['width'], result['cv'][name]['width']
                result['cv'][name]['width_change_fraction'] = (new-old)/old if old > 0 else None
        output['configurations'][identity] = result
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('baseline', 'cv'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--half-width-threshold', type=float)
    parser.add_argument('--bootstrap-count', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args(argv)
    try:
        report = analyze(json.loads(args.manifest.read_text()), args.results_root, args.mode,
                         args.half_width_threshold, args.bootstrap_count, args.seed)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'Analysis stopped: {exc}\n')
    for identity, result in report['configurations'].items():
        for name in ('alp', 'zero'):
            r = result['baseline'][name]
            print(f'{identity} {name}: mean={r["mean"]:.6g}, CI={r["ci95"]}, upper95={r["one_sided_upper95"]:.6g}')
            if args.mode == 'cv':
                r = result['cv'][name]
                print(f'  CV: mean={r["mean"]:.6g}, CI={r["ci95"]}, upper95={r["one_sided_upper95"]:.6g}, width change={r["width_change_fraction"]}')


if __name__ == '__main__':
    main()
