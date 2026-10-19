"""Auditable repeated-PO driver. Importing this module never imports a solver."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from .po_certificates import min_certificate, max_certificate, centered_value, affine_cut, path_record


def atomic_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_params(p):
    if p['schema_version'] != 1:
        raise ValueError('Unsupported schema version')
    if p['decision_set'] != {'first': 'continuous', 'future': 'continuous'}:
        raise ValueError('Repeated PO requires all-period continuous decisions')
    if p['theta_domain'] != dict(coefficient_bound=None, alp_value_floor=False, regularization=0, fixed_coefficients={}):
        raise ValueError('Repeated PO requires the original unrestricted coefficient domain')
    if p['reference']['policy'] != 'alp':
        raise ValueError('The fixed reference policy must be ALP')
    if not 1 <= int(p['solver']['workers']) <= 32:
        raise ValueError('workers must be between 1 and 32')
    budget = os.environ.get('MCMADASP_CPU_BUDGET')
    if budget is not None and int(p['solver']['workers']) > int(budget):
        raise ValueError('workers exceeds allocated MCMADASP_CPU_BUDGET')
    if int(p['solver']['max_iterations']) < 1:
        raise ValueError('max_iterations must be positive')
    for key in ('absolute_gap_tolerance', 'subproblem_time_limit_seconds', 'master_time_limit_seconds'):
        value = p['solver'][key]
        if value is None and key != 'absolute_gap_tolerance':
            continue
        if not math.isfinite(float(value)) or not float(value) > 0:
            raise ValueError(f'{key} must be positive')


def run(p, job_id):
    validate_params(p)
    output = Path(p['output_dir']) / 'batch.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing result {output}')
    # All solver imports occur only after validation and explicit execution.
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB
    from experiments.experiment_config import get_config_by_type
    from importance_sampling import build_proposal
    from generating_function import AbsorptionALPPenaltyFunction
    from decision_maker import ApproxQAgent, ALPRowGenerationAgent
    from decision_maker.approximate_q_agent import dispose_agent_models
    started = time.monotonic()
    def progress(message):
        print(f"[{p['batch_id']}] {time.monotonic() - started:.1f}s {message}", flush=True)

    envs, models = [], []
    result = {k: copy.deepcopy(p[k]) for k in ('schema_version', 'config_id', 'batch_id', 'repetition',
              'configuration', 'initial_state', 'decision_set', 'theta_domain', 'seeds', 'reference', 'sampling', 'solver')}
    result['paths'] = []
    result['training'] = dict(coefficients=None, full_objective=None, objective_lower_bound=None,
        objective_upper_bound=None, master_certificate=dict(sense='maximize', status='NOT_SOLVED',
        objective=None, objective_bound=None, valid=False, reason='No primary master solve'),
        termination_reason='not_started', absolute_gap=None, relative_gap=None, runtime_seconds=0.)
    try:
        env_args = copy.deepcopy(p['configuration'])
        for key, field in [('env', 'env_random_seed'), ('arrival', 'arrival_random_seed'),
                           ('stop_time', 'stop_time_random_seed')]:
            env_args[field] = int(p['seeds'][key])
        environment = get_config_by_type('infinite_custom', args=env_args).env
        state = tuple(np.asarray(v, dtype=float) for v in p['initial_state'])
        anchor = np.asarray(p['reference']['coefficients'], dtype=float)
        generating = AbsorptionALPPenaltyFunction(environment, coefficients=anchor)
        if anchor.shape != (generating.number_of_coefficients,) or not np.isfinite(anchor).all():
            raise ValueError('Invalid reference coefficient vector')
        workers = int(p['solver']['workers'])
        progress(f'initializing {workers + 1} solver environments ({workers} subproblem workers)')
        for _ in range(workers + 1):
            license_env = gp.Env(empty=True)
            license_env.setParam('OutputFlag', 0)
            license_env.setParam('Threads', 1)
            if p['solver']['subproblem_time_limit_seconds'] is not None:
                license_env.setParam('TimeLimit', float(p['solver']['subproblem_time_limit_seconds']))
            envs.append(license_env)
            license_env.start()
        agent = ApproxQAgent(environment, environment.discount_factor, generating_function=generating,
            current_decision_var_type='continuous', future_decision_var_type='continuous',
            sample_path_number=int(p['sampling']['sample_path_number']),
            sample_path_proposal=build_proposal(p['sampling']['proposal']), grb_env=envs[0],
            subproblem_grb_envs=envs[1:])
        lengths = np.asarray([path.length for path in agent.sample_paths], dtype=int)
        progress(f'sampled {len(lengths)} paths; arrival lengths min={lengths.min()} '
                 f'mean={lengths.mean():.2f} max={lengths.max()}; computing fixed ALP policy gradients')
        # ALP is fixed at the anchor; no training coefficient ever changes its policy.
        policy = ALPRowGenerationAgent(environment, discount_factor=environment.discount_factor,
                                       grb_env=envs[0], coefficients=anchor)
        try:
            costs, gradients, saved_anchor = agent.policy_cost_centering(policy, state, anchor)
        finally:
            dispose_agent_models(policy)
        progress('fixed ALP policy gradients ready; building reusable path models')
        kappa = np.asarray(agent.sample_path_weights, dtype=float)
        if not np.isfinite(kappa).all() or np.any(kappa < 0):
            raise ValueError('Invalid path weights')
        # Only the feasible fixed-policy schedule supplies a global scenario cap.
        master, theta, xi = agent.train_master_builder_fn(coefficient_bound=GRB.INFINITY,
            scenario_upper_bound=GRB.INFINITY,
            policy_centering=(costs, gradients, saved_anchor))
        models.append(master)
        master.update()
        xi.UB = np.asarray(costs, dtype=float)
        if p['solver']['master_time_limit_seconds'] is not None:
            master.Params.TimeLimit = float(p['solver']['master_time_limit_seconds'])
        master.Params.Threads = 1
        master.Params.OutputFlag = 0
        built = []
        for sid in range(agent.sample_path_number):
            model, _, objective, gradient = agent.train_subproblem_builder_fn(envs[1 + sid % workers], sid,
                init_state=state, initial_coefficients=anchor, policy_centering=(costs, gradients, saved_anchor),
                cold_solve=False)
            models.append(model)
            if p['solver']['subproblem_time_limit_seconds'] is not None:
                model.Params.TimeLimit = float(p['solver']['subproblem_time_limit_seconds'])
            model.Params.Threads = 1
            built.append((model, objective, gradient))
            if (sid + 1) % 64 == 0 or sid + 1 == agent.sample_path_number:
                progress(f'built {sid + 1}/{agent.sample_path_number} path models')
        groups = [list(range(group, len(built), workers)) for group in range(workers)]

        def evaluate(candidate, add_cuts=True):
            def solve_group(ids):
                rows = []
                for sid in ids:
                    model, objective, gradient = built[sid]
                    objective(model, candidate)
                    model.optimize()
                    certificate = min_certificate(model)
                    # Same model stores centered values. Recover raw costs once.
                    shift = float(gradients[sid] @ (candidate - anchor))
                    raw = dict(certificate)
                    for key in ('objective', 'lower_bound', 'upper_bound'):
                        if raw[key] is not None:
                            raw[key] += shift
                    cut = None
                    if certificate['objective'] is not None:
                        cut = affine_cut(certificate['objective'], gradient(model, candidate), candidate)
                    rows.append((sid, certificate, raw, cut))
                return rows
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = [row for group in executor.map(solve_group, groups) for row in group]
            rows.sort(key=lambda row: row[0])
            if add_cuts:
                for sid, _, _, cut in rows:
                    if cut is not None:
                        intercept, slope = cut
                        master.addConstr(xi[sid] <= intercept + np.asarray(slope) @ theta)
                master.update()
            lower = (float(kappa @ [r[1]['lower_bound'] for r in rows])
                     if all(r[1]['bound_valid'] for r in rows) else None)
            full = (float(kappa @ [r[1]['objective'] for r in rows])
                    if all(r[1]['objective'] is not None for r in rows) else None)
            return dict(raw=[r[2] for r in rows], lower=lower, full=full, exact=all(r[1]['exact'] for r in rows))

        alp = evaluate(anchor)
        progress(f'ALP baseline complete; centered incumbent={alp["full"]} lower_bound={alp["lower"]}')
        zero = evaluate(np.zeros_like(anchor))
        progress(f'zero baseline complete; centered incumbent={zero["full"]} lower_bound={zero["lower"]}')
        candidates = [(anchor.copy(), alp), (np.zeros_like(anchor), zero)]
        valid_candidates = [(t, e) for t, e in candidates if e['lower'] is not None]
        if not valid_candidates:
            raise RuntimeError('Neither seed candidate has complete finite MIN certificates')
        fitted, best = max(valid_candidates, key=lambda pair: pair[1]['lower'])
        best_upper, best_certificate = None, None
        reason = 'max_iterations'
        for iteration in range(int(p['solver']['max_iterations'])):
            master.optimize()
            certificate = max_certificate(master)
            if certificate['valid'] and (best_upper is None or certificate['objective_bound'] < best_upper):
                best_upper, best_certificate = certificate['objective_bound'], certificate
            if int(master.SolCount) == 0:
                reason = 'master_no_incumbent'
                break
            candidate = np.asarray(theta.X, dtype=float)
            evaluation = evaluate(candidate)
            if evaluation['lower'] is not None and evaluation['lower'] > best['lower']:
                fitted, best = candidate.copy(), evaluation
            if iteration == 0 or (iteration + 1) % 10 == 0:
                current_gap = None if best_upper is None else best_upper - best['lower']
                progress(f'iteration {iteration + 1}; upper_bound={best_upper} '
                         f'lower_bound={best["lower"]} absolute_gap={current_gap}')
            if best_upper is not None:
                gap = best_upper - best['lower']
                if gap < -1e-6 * max(1., abs(best_upper), abs(best['lower'])):
                    raise RuntimeError('Certified lower bound exceeds master upper bound')
                if gap <= float(p['solver']['absolute_gap_tolerance']):
                    reason = 'absolute_gap_tolerance'
                    break
            if not certificate['valid']:
                reason = 'master_missing_certificate'
                break
        # Preserve the exact solver snapshots that established the selected candidate LB.
        # A later time-limited re-solve could weaken those certificates.
        fitted_evaluation = best
        gap = None if best_upper is None else max(0., best_upper - best['lower'])
        result['training'].update(coefficients=fitted.tolist(), full_objective=fitted_evaluation['full'],
            full_objective_exact=fitted_evaluation['exact'],
            objective_lower_bound=best['lower'], objective_upper_bound=best_upper,
            master_certificate=best_certificate or result['training']['master_certificate'],
            termination_reason=reason, absolute_gap=gap,
            relative_gap=None if gap is None else gap / max(1e-12, abs(best['lower'])))
        result['paths'] = [path_record(sid, path, state, agent.sample_path_strata[sid], kappa[sid],
            gradients[sid], dict(fitted=fitted_evaluation['raw'][sid], zero=zero['raw'][sid], alp=alp['raw'][sid]))
            for sid, path in enumerate(agent.sample_paths)]
        result['sampling_observed'] = agent._build_importance_sampling_info()
        from .repeated_po_schema import validate_batch
        validate_batch(result)
        progress(f'finished ({reason}); upper_bound={best_upper} '
                 f'lower_bound={best["lower"]} absolute_gap={gap}')
    except Exception as error:
        progress(f'failure: {type(error).__name__}: {error}')
        result['training'].update(termination_reason='failure', error=f'{type(error).__name__}: {error}')
        # Retain a partial audit record, but never let it masquerade as valid inference input.
        result['training']['master_certificate']['valid'] = False
        raise
    finally:
        result['training']['runtime_seconds'] = time.monotonic() - started
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        result['runtime'] = dict(git_commit=commit, job_id=job_id, numpy_version=np.__version__,
                                 gurobi_version='.'.join(map(str, gp.gurobi.version())))
        for model in reversed(models):
            model.dispose()
        for license_env in reversed(envs):
            license_env.dispose()
        atomic_result(output, result)
    return result
