"""Solver-free checks for repeated-PO result certificates."""
from collections import Counter
import json
import math

import numpy as np


def finite(value, name):
    if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
        raise ValueError(f'{name} must be finite')
    return float(value)


def close(left, right, name, atol=1e-5):
    if not math.isclose(left, right, rel_tol=1e-8, abs_tol=atol):
        raise ValueError(f'{name}: inconsistent values {left} and {right}')


def validate_batch(batch):
    """Fail closed; never silently drop failed or incomplete replicates."""
    try:
        if batch['schema_version'] != 1:
            raise ValueError('unsupported schema_version')
        if batch['decision_set'] != {'first': 'continuous', 'future': 'continuous'}:
            raise ValueError('all three IR comparisons must use continuous decisions')
        domain = batch['theta_domain']
        if (domain['coefficient_bound'] is not None or domain['alp_value_floor']
                or domain['regularization'] != 0 or domain['fixed_coefficients']):
            raise ValueError('coefficient domain differs from the unrestricted experiment')
        if batch['reference']['policy'] != 'alp':
            raise ValueError('reference policy must be fixed ALP')
        reference = np.asarray(batch['reference']['coefficients'], dtype=float)
        fitted = np.asarray(batch['training']['coefficients'], dtype=float)
        if reference.ndim != 1 or reference.size == 0 or fitted.shape != reference.shape:
            raise ValueError('invalid coefficient vectors')
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(fitted)):
            raise ValueError('nonfinite coefficients')
        paths, sampling, training = batch['paths'], batch['sampling'], batch['training']
        if len(paths) != sampling['sample_path_number']:
            raise ValueError('missing paths')
        if len({p['path_id'] for p in paths}) != len(paths):
            raise ValueError('duplicate path IDs')
        if len({p['initial_state_hash'] for p in paths}) != 1:
            raise ValueError('initial states differ within fixed-state experiment')
        proposal = sampling['proposal']
        if proposal['type'] != 'stratified_geometric':
            raise ValueError('this experiment requires stratified_geometric sampling')
        if sampling['weight_convention'] != 'stratum_mass_over_count':
            raise ValueError('unknown weight convention')
        if sampling['tau_definition'] != 'arrival_length_plus_one':
            raise ValueError('unknown termination-time convention')
        strata = Counter(p['stratum'] for p in paths)
        h = proposal['num_strata']
        if set(strata) != set(range(h)):
            raise ValueError('missing or unexpected strata')
        cert = training['master_certificate']
        if cert['sense'] != 'maximize' or not cert['valid']:
            raise ValueError('no valid MAX master certificate')
        upper = finite(training['objective_upper_bound'], 'master upper bound')
        close(upper, finite(cert['objective_bound'], 'master certificate'), 'master upper bound')
        incumbent_total = 0.
        all_incumbents = True
        lower_total = 0.
        for p in paths:
            weight = finite(p['weight'], 'weight')
            close(weight, 1. / (h * strata[p['stratum']]), 'stratum weight', atol=1e-12)
            if weight <= 0 or p['termination_time'] != p['arrival_length'] + 1 or p['arrival_length'] < 1:
                raise ValueError('invalid weight or termination time')
            if p['terminal'] != 'absorbed':
                raise ValueError('unexpected truncation')
            gradient = np.asarray(p['policy_gradient'], dtype=float)
            if gradient.shape != reference.shape or not np.all(np.isfinite(gradient)):
                raise ValueError('invalid policy gradient')
            correction = float(gradient @ (fitted - reference))
            for name in ('fitted', 'zero', 'alp'):
                ir = p['ir'][name]
                if not ir['bound_valid']:
                    raise ValueError(f'invalid MIN certificate for {name}')
                lower = finite(ir['lower_bound'], f'{name} lower bound')
                if ir['objective'] is not None:
                    objective = finite(ir['objective'], f'{name} incumbent')
                    if ir['solution_count'] < 1 or lower > objective + 1e-5 * max(1., abs(objective)):
                        raise ValueError(f'inconsistent MIN bounds for {name}')
                    close(objective, finite(ir['upper_bound'], 'MIN upper bound'), 'MIN incumbent')
                    if ir['exact']:
                        close(lower, objective, 'optimal MIN certificate')
                elif ir['exact']:
                    raise ValueError('exact result without incumbent')
            lower_total += weight * (p['ir']['fitted']['lower_bound'] - correction)
            if p['ir']['fitted']['objective'] is None:
                all_incumbents = False
            else:
                incumbent_total += weight * (p['ir']['fitted']['objective'] - correction)
            for baseline in ('zero', 'alp'):
                left, right = p['ir']['fitted']['objective'], p['ir'][baseline]['objective']
                gap = p[f'raw_gap_fitted_minus_{baseline}']
                if left is not None and right is not None:
                    close(finite(gap, 'raw path gap'), left - right, 'paired raw gap')
                elif gap is not None:
                    raise ValueError('raw gap requires both incumbents')
        close(sum(p['weight'] for p in paths), 1., 'total stratum mass', atol=1e-12)
        if all_incumbents:
            close(finite(training['full_objective'], 'full objective'), incumbent_total, 'control-variate constant')
        if upper < lower_total - 1e-5 * max(1., abs(lower_total)):
            raise ValueError('master upper bound below certified candidate value')
        return batch
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f'malformed batch: {exc}') from exc


def validate_group(batches, expected_batches=30, expected_paths=512):
    if len(batches) != expected_batches:
        raise ValueError(f'expected {expected_batches} batches, received {len(batches)}; no CI produced')
    if len({b['batch_id'] for b in batches}) != len(batches):
        raise ValueError('duplicate batch IDs')
    if {b['repetition'] for b in batches} != set(range(1, expected_batches + 1)):
        raise ValueError('missing or duplicate repetition numbers')
    keys = ('config_id', 'configuration', 'initial_state', 'decision_set', 'theta_domain', 'reference', 'sampling', 'solver')
    anchor = {key: batches[0][key] for key in keys}
    for b in batches:
        validate_batch(b)
        if b['sampling']['sample_path_number'] != expected_paths:
            raise ValueError('unexpected paths per batch')
        if {key: b[key] for key in keys} != anchor:
            raise ValueError('inconsistent model, reference, solver, or sampling metadata')
    for key in ('env', 'arrival', 'stop_time', 'initial_state'):
        if len({b['seeds'][key] for b in batches}) != len(batches):
            raise ValueError(f'reused {key} seeds across independent batches')
    commits = {b.get('runtime', {}).get('git_commit') for b in batches}
    if len(commits) > 1:
        raise ValueError('mixed code commits across repetitions')
    return sorted(batches, key=lambda b: b['repetition'])


def match_manifest(batch, run):
    """Compare downloaded result metadata to the saved, pre-run manifest."""
    params = run.get('params', run.get('parameters', run))
    for key in ('config_id', 'batch_id', 'repetition', 'configuration', 'initial_state',
                'decision_set', 'theta_domain', 'reference', 'sampling', 'solver', 'seeds'):
        if batch.get(key) != params.get(key):
            raise ValueError(f'{batch.get("batch_id")}: manifest mismatch for {key}')
