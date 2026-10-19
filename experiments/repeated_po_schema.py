"""Solver-free checks for repeated-PO result certificates."""
from collections import Counter
import copy
import math

import numpy as np

from experiments.po_certificates import BOUND_STATUSES, STATUS_NAMES


BOUND_STATUS_NAMES = {STATUS_NAMES[status] for status in BOUND_STATUSES}


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
        if (any('policy_gradient' in p for p in paths)
                and len({p['initial_state_hash'] for p in paths}) != 1):
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
        raw_total = 0.
        all_incumbents = True
        lower_total = 0.
        for p in paths:
            weight = finite(p['weight'], 'weight')
            close(weight, 1. / (h * strata[p['stratum']]), 'stratum weight', atol=1e-12)
            tau = finite(p['termination_time'], 'termination time')
            if weight <= 0 or tau < 2 or not tau.is_integer():
                raise ValueError('invalid weight or termination time')
            if 'policy_gradient' in p:
                if tau != p['arrival_length'] + 1 or p['arrival_length'] < 1:
                    raise ValueError('invalid termination time')
                if p['terminal'] != 'absorbed':
                    raise ValueError('unexpected truncation')
                gradient = np.asarray(p['policy_gradient'], dtype=float)
                if gradient.shape != reference.shape or not np.all(np.isfinite(gradient)):
                    raise ValueError('invalid policy gradient')
                correction = float(gradient @ (fitted - reference))
                reference_penalty = float(gradient @ reference)
            else:
                correction = finite(p['policy_control_variate'], 'policy control variate')
                reference_penalty = finite(p['policy_penalty_at_reference'], 'policy penalty at reference')
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
                raw_total += weight * p['ir']['fitted']['objective']
            fitted_cost = p['ir']['fitted']['objective']
            adjusted_cost = None if fitted_cost is None else fitted_cost - correction
            adjusted = {'cv_adjusted_fitted_ir_cost': adjusted_cost}
            for baseline in ('zero', 'alp'):
                left, right = p['ir']['fitted']['objective'], p['ir'][baseline]['objective']
                gap = p[f'raw_gap_fitted_minus_{baseline}']
                if left is not None and right is not None:
                    close(finite(gap, 'raw path gap'), left - right, 'paired raw gap')
                elif gap is not None:
                    raise ValueError('raw gap requires both incumbents')
                adjusted[f'cv_adjusted_gap_fitted_minus_{baseline}'] = (
                    None if adjusted_cost is None or right is None else
                    adjusted_cost - right - (reference_penalty if baseline == 'zero' else 0.))
            for field, value in adjusted.items():
                if field not in p:
                    continue
                if value is None:
                    if p[field] is not None:
                        raise ValueError(f'CV-adjusted {field} requires its incumbents')
                else:
                    close(finite(p[field], f'CV-adjusted {field}'), value, f'CV-adjusted {field}')
        close(sum(p['weight'] for p in paths), 1., 'total stratum mass', atol=1e-12)
        if all_incumbents:
            close(finite(training['full_objective'], 'full objective'), incumbent_total, 'control-variate constant')
        if 'tight_penalized_lower_bound' in training:
            close(finite(training['tight_penalized_lower_bound'], 'raw objective'), raw_total, 'raw objective mean')
            close(finite(training['objective_lower_bound'], 'candidate lower bound'), lower_total, 'candidate lower bound')
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
    # These experiments share a fixed initial state, so its unused seed may repeat.
    for key in ('env', 'arrival', 'stop_time'):
        if len({b['seeds'][key] for b in batches}) != len(batches):
            raise ValueError(f'reused {key} seeds across independent batches')
    return sorted(batches, key=lambda b: b['repetition'])


def match_manifest(batch, run):
    """Compare downloaded result metadata to the saved, pre-run manifest."""
    params = run if 'schema_version' in run else run.get('params', run.get('parameters', run))
    params = params.get('batch_metadata', params)
    for key in ('config_id', 'batch_id', 'repetition', 'configuration', 'initial_state',
                'decision_set', 'theta_domain', 'reference', 'sampling', 'solver', 'seeds'):
        if batch.get(key) != params.get(key):
            raise ValueError(f'{batch.get("batch_id")}: manifest mismatch for {key}')


def batch_from_training_record(record, run):
    """Join the ordinary training output to its external manifest using the saved UID."""
    parameters = run['parameters']
    uid = parameters.get('uid')
    if not isinstance(uid, str) or not uid or record.get('uid') != uid:
        raise ValueError(f"{run['batch_id']}: missing or mismatched training UID")
    compact = 'master_status' in record
    training_fields = ('coefficients', 'full_objective', 'objective_lower_bound',
                       'objective_upper_bound', 'termination_reason')
    if compact:
        training_fields += ('tight_penalized_lower_bound',)
        extra_fields = ('master_status', 'experiment_name', 'mutate_val', 'sample_path_number')
    else:
        training_fields += ('full_objective_exact', 'master_certificate', 'absolute_gap', 'relative_gap')
        extra_fields = ()
    missing = [key for key in ('paths', 'training_time_seconds', *training_fields, *extra_fields) if key not in record]
    if missing:
        raise ValueError(f"{run['batch_id']}: missing training output fields: {', '.join(missing)}")
    # Labels live only in the manifest; verify they describe the executed inputs
    # before using the proposal mean or the reference vector in statistical formulas.
    inner = parameters['agent_args']['agent_args']
    environment = dict(run['configuration'])
    for key, field in (('env', 'env_random_seed'), ('arrival', 'arrival_random_seed'),
                       ('stop_time', 'stop_time_random_seed')):
        environment[field] = run['seeds'][key]
    expected = dict(
        decision_set={'first': inner.get('current_decision_var_type'),
                      'future': inner.get('future_decision_var_type')},
        theta_domain={'coefficient_bound': inner.get('coefficient_bound'),
                      'alp_value_floor': bool(inner.get('alp_value_floor', False)),
                      'regularization': 0, 'fixed_coefficients': inner.get('fixed_coefficients') or {}},
        sampling={'sample_path_number': parameters['sample_path_number'],
                  'proposal': inner.get('sample_path_proposal', inner.get('sample_path_length_proposal')),
                  'tau_definition': 'arrival_length_plus_one',
                  'weight_convention': 'stratum_mass_over_count', 'quantile_clip': [1e-12, .999999999999]},
        solver={key: inner.get(key) for key in run['solver']})
    if (environment != parameters['env_args']
            or parameters['init_state_mode'] != 'shared'
            or parameters['init_state'] != run['initial_state']
            or parameters['init_state_seed'] != run['seeds']['initial_state']
            or any(run[key] != value for key, value in expected.items())
            or not inner.get('policy_centering') or inner.get('cap_policy') != 'alp'
            or inner.get('initial_coefficients') != run['reference']['coefficients']
            or inner.get('coefficient_bound_overrides') or not inner.get('save_path_certificates')):
        raise ValueError(f"{run['batch_id']}: manifest labels differ from training parameters")
    result = {key: copy.deepcopy(run[key]) for key in (
        'schema_version', 'config_id', 'batch_id', 'repetition', 'configuration',
        'initial_state', 'decision_set', 'theta_domain', 'seeds', 'reference', 'sampling', 'solver')}
    result['paths'] = copy.deepcopy(record['paths'])
    result['training'] = {key: copy.deepcopy(record[key]) for key in training_fields}
    result['training']['runtime_seconds'] = record['training_time_seconds']
    if compact:
        try:
            status = record['master_status']
            if status not in BOUND_STATUSES and status not in BOUND_STATUS_NAMES:
                raise ValueError(f'invalid MAX master status: {status}')
            training = result['training']
            upper = finite(training['objective_upper_bound'], 'master upper bound')
            lower = finite(training['objective_lower_bound'], 'candidate lower bound')
            training['master_certificate'] = dict(sense='maximize', status=status,
                                                 objective_bound=upper, valid=True)
            for path in result['paths']:
                for name in ('fitted', 'zero', 'alp'):
                    cost = path['ir'][name]
                    status = cost['status']
                    if status not in BOUND_STATUSES and status not in BOUND_STATUS_NAMES:
                        raise ValueError(f'invalid MIN status for {name}: {status}')
                    objective = finite(cost['objective'], f'{name} incumbent')
                    if 'lower_bound' not in cost:
                        if status not in (2, 'OPTIMAL'):
                            raise ValueError(f'nonoptimal MIN result for {name} requires a lower bound')
                        cost['lower_bound'] = objective
                    finite(cost['lower_bound'], f'{name} lower bound')
                    cost.update(upper_bound=objective, solution_count=1,
                                exact=status in (2, 'OPTIMAL'), bound_valid=True)
            training['full_objective_exact'] = all(path['ir']['fitted']['exact'] for path in result['paths'])
            training['absolute_gap'] = max(0., upper - lower)
            training['relative_gap'] = training['absolute_gap'] / max(1e-12, abs(lower))
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f'malformed compact training output: {exc}') from exc
    return validate_batch(result)
