import argparse
import copy
import json
import os
import shutil
import time

from scipy.stats import geom

import numpy as np

from experiments import get_config_by_type
from param_generation.caching import load_trained_coefficients_from_folder
from param_generation.command_files import write_command_file, write_grouped_command_file
from param_generation.datasets import (
    generate_penalty_coefficient_training_env,
    generate_test_paths_and_init_state,
    offset_sample_generation_seeds,
)
from param_generation.experiment_specs import build_variation_test_env
from param_generation.generating_functions import GENERATING_FUNCTION_CLASSES, build_generating_function, normalize_generating_function_spec
from param_generation.mutators import mutate_initial_state_congestion
from param_generation.registry import EXPERIMENT_SPECS
from param_generation.training import train_alp_coefficients
from decision_maker.alp_rg_agent import alp_expected_initial_state
from decision_maker.approximate_q_agent import NOISE_REMOVALS, TRAINING_OBJECTIVES, WORST_CASE_SCOPES
from utils import get_uid


def recipe_toy_eval_proposal_comparison(
    test_sample_path_num=4096,
    num_groups=500,
    dat_file='table.dat',
    penalty_coefficients_dir=os.path.join(
        'experiments', 'results', 'mixture_probability_toy_study'
    ),
):
    coefficients = load_trained_coefficients_from_folder(
        experiment_name='mixture_probability_toy_study',
        mutate_val=0.1,
        sample_path_number=256,
        folder_path=penalty_coefficients_dir,
    )
    if coefficients is None:
        raise RuntimeError(
            'No trained penalty coefficients (mutate_val=0.1, '
            f'sample_path_number=256) found in {penalty_coefficients_dir}.'
        )

    # 459 = 0.99-quantile of Geom(1 - 0.99). The fixed proposal samples
    # max_length = 458 arrivals so the rollout visits exactly 459 weighted
    # decision periods (see the spec: FixedLengthProposal(459) would raise in
    # _evaluation_period_weights).
    fixed_horizon = int(geom.ppf(0.99, 1 - 0.99))
    assert fixed_horizon == 459, fixed_horizon

    arms = [
        ('toy_eval_proposal_geometric_099',
         {'type': 'geometric', 'discount_factor_proposal': 0.99}),
        ('toy_eval_proposal_fixed_459',
         {'type': 'fixed', 'max_length': fixed_horizon - 1}),
        ('toy_eval_proposal_mixture_095_l01',
         {'type': 'mixture_geometric', 'target_discount_factor': 0.99,
          'discount_factor_proposal': 0.95, 'lambda_0': 0.1}),
    ]
    all_records = []
    for experiment_name, evaluation_proposal_spec in arms:
        test_envs = build_variation_test_env(EXPERIMENT_SPECS[experiment_name])
        all_records.extend(
            generate_test_paths_and_init_state(
                test_envs=test_envs,
                test_sample_path_num=test_sample_path_num,
                warm_up_periods=0,
                num_periods=None,
                dat_file=None,
                is_require_penalty_coefficients=True,
                is_random_initial_state=False,
                policy_ids=['row_gen_alp'],
                evaluation_proposal_spec=evaluation_proposal_spec,
                penalty_coefficients_dir=penalty_coefficients_dir,
            )
        )
    write_grouped_command_file(
        results=all_records, num_groups=num_groups, dat_file=dat_file
    )
    return all_records


def recipe_saure_ejor_case_study(
    test_sample_path_num=4096,
    warm_up_periods=750,
    evaluation_periods=750,
    num_groups=500,
    dat_file='table.dat',
    penalty_coefficients_dir=os.path.join(
        'experiments', 'results', 'case_study_099_mixture_geometric_proposal_095'
    ),
):
    shared_coefficients = load_trained_coefficients_from_folder(
        experiment_name='case_study_099_mixture_geometric_proposal_095',
        mutate_val=0.1,
        sample_path_number=256,
        folder_path=penalty_coefficients_dir,
    )

    def resolve_penalty_coefficients_dir(experiment_name):
        own_coefficients = load_trained_coefficients_from_folder(
            experiment_name=experiment_name,
            mutate_val=0.1,
            sample_path_number=256,
        )
        if own_coefficients is not None:
            print(f"Using penalty coefficients from {experiment_name}'s own results folder.")
            return None
        if shared_coefficients is None:
            raise RuntimeError(
                'No trained penalty coefficients (mutate_val=0.1, '
                f"sample_path_number=256) found in {experiment_name}'s results "
                f'folder or in {penalty_coefficients_dir}; train them for the '
                'overtime-100 case study first.'
            )
        print(f'Using shared penalty coefficients from {penalty_coefficients_dir} for {experiment_name}.')
        return penalty_coefficients_dir

    replication_penalty_dir = resolve_penalty_coefficients_dir('case_study_ejor_replication')
    steady_state_penalty_dir = resolve_penalty_coefficients_dir('case_study_ejor_alp_steady_state')

    replication_envs = build_variation_test_env(
        EXPERIMENT_SPECS['case_study_ejor_replication']
    )
    steady_state_envs = build_variation_test_env(
        EXPERIMENT_SPECS['case_study_ejor_alp_steady_state']
    )
    start = time.time()
    (replication_variant,) = replication_envs.values()
    (steady_state_variant,) = steady_state_envs.values()
    print(time.time() - start, 'seconds to build the test envs')
    if replication_variant['env_args'] != steady_state_variant['env_args']:
        raise RuntimeError(
            'The Saure EJOR specs must share identical env_args so the warm-up '
            'prefixes and the ALP can be shared.'
        )
    env_args = replication_variant['env_args']

    train_alp_coefficients(
        env_args=env_args, experiment_name='case_study_ejor_replication'
    )
    source_path = os.path.join(
        'experiments', 'results', 'case_study_ejor_replication', 'alp_train.jsonl'
    )
    target_path = os.path.join(
        'experiments', 'results', 'case_study_ejor_alp_steady_state', 'alp_train.jsonl'
    )
    if os.path.isfile(source_path) and not os.path.isfile(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copyfile(source_path, target_path)

    warm_up_env = get_config_by_type(
        'infinite_custom', args=offset_sample_generation_seeds(env_args, 3003)
    ).env
    warm_up_paths = [
        warm_up_env.reset_arrivals(stop_time=warm_up_periods)
        for _ in range(test_sample_path_num)
    ]

    replication_records = generate_test_paths_and_init_state(
        test_envs=replication_envs,
        test_sample_path_num=test_sample_path_num,
        warm_up_periods=warm_up_periods,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=True,
        is_random_initial_state=False,
        policy_ids=['row_gen_alp', 'myopic'],
        # max_length is one less than the evaluation horizon: the rollout visits
        # one trailing decision period after the last sampled arrival, so the
        # fixed proposal weights exactly ``evaluation_periods`` periods by
        # gamma**(t-1).
        evaluation_proposal_spec={'type': 'fixed', 'max_length': evaluation_periods - 1},
        penalty_coefficients_dir=replication_penalty_dir,
        warm_up_paths=warm_up_paths,
    )
    steady_state_records = generate_test_paths_and_init_state(
        test_envs=steady_state_envs,
        test_sample_path_num=test_sample_path_num,
        warm_up_periods=warm_up_periods,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=True,
        is_random_initial_state=False,
        policy_ids=['approx_penalized_hindsight', 'row_gen_alp', 'myopic'],
        warm_up_policy_id='row_gen_alp',
        penalty_coefficients_dir=steady_state_penalty_dir,
        warm_up_paths=warm_up_paths,
        sample_gen_seed_offset=2002,
    )
    # Experiment 1 reports policy costs only: its information-relaxation
    # bounds are ~750-period direct models (four per record) that dominate the
    # runtime and are not part of the replication comparison.
    for record in replication_records:
        record['skip_information_relaxation'] = True

    # Segmented layout in ONE dat file: the first num_groups//2 lines hold
    # experiment 1 only, the rest experiment 2 only, so cluster jobs can be
    # resourced per experiment (experiment 1 lines never build the
    # 256-scenario hindsight solver and thus need only one Gurobi token).
    replication_lines = write_grouped_command_file(
        results=replication_records, num_groups=num_groups // 2, dat_file=None
    )
    steady_state_lines = write_grouped_command_file(
        results=steady_state_records,
        num_groups=num_groups - num_groups // 2,
        dat_file=None,
        start_index=num_groups // 2 + 1,
    )
    lines = replication_lines + steady_state_lines
    with open(dat_file, 'w') as f:
        f.writelines(lines)
    print(f"Saved {len(lines)} commands to {dat_file}")
    return replication_records, steady_state_records


def build_parser():
    parser = argparse.ArgumentParser(
        prog='generate_params',
        description='Generate run.py parameter (.dat) files for MCMADASP experiments.')
    sub = parser.add_subparsers(dest='command', required=True)

    train = sub.add_parser(
        'train', help='Penalty-coefficient training commands (one per variant); '
                      "each variant's scenario count comes from its agent_args.")
    train.add_argument('experiment', choices=sorted(EXPERIMENT_SPECS))
    train.add_argument('--variants', default=None,
                       help='comma-separated mutate_val values to keep '
                            '(default: all variants of the experiment)')
    train.add_argument('--dat', default='table.dat')
    train.add_argument('--init-state-seed', type=int, default=12345)
    train.add_argument('--sample-paths-seed', type=int, default=42,
                       help='first sample-path seed; with --num-path-seeds M '
                            'the seeds seed, seed+1, ..., seed+M-1 are used')
    train.add_argument(
        '--num-init-states', type=int, default=0, metavar='K',
        help='emit K commands per variant, each training on ONE fixed initial '
             'state shared by all scenarios (K states drawn from the env\'s '
             'reference distribution with --init-state-seed); default 0 = '
             'runner draws one state per scenario')
    train.add_argument(
        '--reset-init-state', action='store_true',
        help="train from the variant's own reset initial state "
             "(env_args['reset_params']['init_state'], e.g. the fixed-occupancy "
             'state of the occupancy experiments) shared by all scenarios; '
             'mutually exclusive with --num-init-states')
    train.add_argument(
        '--num-path-seeds', type=int, default=1, metavar='M',
        help='emit M commands per variant (and per initial state), one per '
             'consecutive sample-path seed starting at --sample-paths-seed')
    train.add_argument(
        '--penalty-function', choices=sorted(GENERATING_FUNCTION_CLASSES), default=None,
        help='generating function of the penalty: linear_penalty (default, the '
             'legacy form) or absorption_linear_penalty (Brown-Haugh absorption-time '
             'form: survival-weighted terms, terminal expected term); stamps every '
             'generating-function spec of the variant and suffixes policy_id / '
             'experiment_name with _bh')
    train.add_argument(
        '--coefficient-bound', type=float, default=None, metavar='B',
        help='box constraint |theta_k| <= B on every penalty coefficient in the '
             'Benders master (default: unbounded); suffixes policy_id / '
             'experiment_name with _cb<B> so bounded runs train and store separately')
    train.add_argument(
        '--regularization', choices=['l1', 'l2'], default=None,
        help='add an L1 or L2 penalty on the coefficient vector to the Benders '
             'master (requires --regularization-lambda); one command per lambda, '
             "results land in experiments/results/<experiment>_<type>_<lambda>/")
    train.add_argument(
        '--regularization-lambda', default=None, metavar='L1[,L2,...]',
        help='comma-separated regularization weights (>= 0), one command each')
    train.add_argument(
        '--regularization-scale', choices=['feature_std', 'none'], default='feature_std',
        help='per-coefficient scale inside the penalty: feature_std (default) = '
             'scenario-weighted std of the build-time penalty features; none = raw')
    train.add_argument(
        '--expected-init-state', action='store_true',
        help='train from the ALP relevance-weighted expected initial state '
             '(E_u_alpha, E_v_alpha, E_w_alpha) shared by all scenarios; mutually '
             'exclusive with --reset-init-state and --num-init-states')
    train.add_argument(
        '--objective', choices=TRAINING_OBJECTIVES, default='mean',
        help='training criterion over the scenarios: mean (kappa-weighted sample '
             'average, default) or worst_case (kappa-weighted average over the initial '
             'states of the minimum pathwise value over that state\'s sample paths)')
    train.add_argument(
        '--worst-case-scope', choices=WORST_CASE_SCOPES, default='per_state',
        help='where the worst_case objective takes its minimum: per_state (default) = '
             'one minimum per initial state, averaged with the state-relevance weights; '
             'joint = one minimum across every state and path, over the pathwise value '
             'minus the value-function approximation at that scenario\'s initial state')
    train.add_argument(
        '--fix-block', action='append', default=None, metavar='BLOCK',
        help='pin a whole block of coefficients (intercept, regular, overtime, waitlist) at its '
             '--init-coefficients value, so training only moves the identified ones; repeatable')
    train.add_argument(
        '--min-norm-slack', type=float, default=0.0, metavar='EPS',
        help='after the cutting plane converges, return the smallest-norm coefficients whose '
             'sample-average objective is within EPS of the converged one, verified on the '
             'subproblems; 0 keeps the converged coefficients')
    train.add_argument(
        '--paths-per-state', type=int, default=None, metavar='K',
        help='generate mode only: draw sample_path_number / K initial states and give '
             'each of them K sample paths (the worst_case objective takes the minimum '
             'over those K paths); mutually exclusive with the shared-state options')
    train.add_argument(
        '--init-coefficients', default=None, metavar='JSONL',
        help='warm-start the Benders master at the coefficients of the first record '
             'with a coefficients list in this jsonl file (e.g. the ALP coefficients)')
    train.add_argument(
        '--name', default=None,
        help='experiment name for the generated command (requires a single variant)')
    train.add_argument(
        '--noise-removal', choices=NOISE_REMOVALS, default=None,
        help='remove the zero-mean, action-independent part of the penalty features (the '
             'constants of the feature expressions) inside every training subproblem: '
             'mean subtracts their scenario-weighted sample mean, so the sample-average '
             'objective has no slope along those directions and the reported bound is '
             'still the original penalty; pathwise subtracts each path own constant, '
             'which leaves the population problem unchanged and removes that noise '
             'entirely, and then the reported bound is the modified penalty')
    train.add_argument(
        '--intercept-bound', type=float, default=None, metavar='B',
        help='box |W_0| <= B for the intercept coefficient instead of --coefficient-bound '
             '(absorption_alp_penalty only); mutually exclusive with --fix-intercept')
    train.add_argument(
        '--fix-intercept', action='store_true',
        help='pin coefficient 0 (the intercept of absorption_alp_penalty) at 0 during '
             'training; its penalty term is action-independent with zero mean, so it '
             'only adds variance and drives the mean objective to the coefficient bound')

    def add_eval_arguments(sub_parser):
        sub_parser.add_argument('experiment', choices=sorted(EXPERIMENT_SPECS))
        sub_parser.add_argument('--variants', default=None,
                                help='comma-separated mutate_val values to keep '
                                     '(default: all variants of the experiment)')
        sub_parser.add_argument('--paths', type=int, default=4096)
        sub_parser.add_argument('--warm-up', type=int, default=0)
        sub_parser.add_argument('--groups', type=int, default=500)
        sub_parser.add_argument('--dat', default='table.dat')
        sub_parser.add_argument('--random-init', action='store_true')
        sub_parser.add_argument('--warm-up-policy', default=None)
        sub_parser.add_argument('--penalty-dir', default=None)
        sub_parser.add_argument('--seed-offset', type=int, default=1001)
        sub_parser.add_argument(
            '--eval-proposal', default=None, metavar='JSON',
            help='build_proposal-style spec for drawing the EVALUATION tails '
                 '(e.g. \'{"type": "geometric", "discount_factor_proposal": 0.99}\' '
                 'for the target horizon); default: the agent\'s own IS proposal')
        sub_parser.add_argument(
            '--init-occupancy', type=float, default=None, metavar='FRACTION',
            help='start every evaluation path from the fixed initial state '
                 'with this fraction of total (regular + overtime) capacity '
                 'booked on every day (mutate_initial_state_congestion); '
                 "default: the experiment's own reset initial state")
        sub_parser.add_argument('--skip-ir', action='store_true',
                                help='mark records skip_information_relaxation')
        sub_parser.add_argument(
            '--penalty-function', choices=sorted(GENERATING_FUNCTION_CLASSES), default=None,
            help='generating function of the bound and the policies: linear_penalty '
                 '(default, legacy) or absorption_linear_penalty (Brown-Haugh '
                 'absorption-time form; needs --eval-proposal so every record '
                 'carries survival weights and a terminal outcome); suffixes '
                 'policy_id / experiment_name with _bh')
        sub_parser.add_argument(
            '--policy-spec', default=None, metavar='JSON',
            help='JSON file mapping policy_id -> agent_args overrides, '
                 'deep-merged into every variant\'s agent_args before '
                 'generation (changes the variant uid; see --allow-retrain)')
        sub_parser.add_argument(
            '--allow-retrain', action='store_true',
            help='with --policy-spec: permit training penalty coefficients '
                 'when the overridden configuration has no cached ones')

    eval_parser = sub.add_parser('eval', help='Policy-evaluation sample paths.')
    add_eval_arguments(eval_parser)
    eval_parser.add_argument('--policies', required=True,
                             help='comma-separated policy ids')

    lower = sub.add_parser(
        'lowerbound', help='Information-relaxation lower bounds only (no policies).')
    add_eval_arguments(lower)
    lower.add_argument(
        '--penalty-ratios', default=None, metavar='T1,T2,...',
        help='evaluate the information-relaxation lower bound with the trained '
             'penalty coefficients scaled by each factor t of this comma-separated '
             'grid (0 and 1 are always included) on the evaluation paths; the '
             'coefficients must already exist under --penalty-dir (no training). '
             'See report_penalty_shrinkage for the overfitting diagnostic.')

    saure = sub.add_parser('saure-ejor', help='Saure EJOR case-study pair.')
    saure.add_argument('--paths', type=int, default=4096)
    saure.add_argument('--warm-up', type=int, default=750)
    saure.add_argument('--eval-periods', type=int, default=750)
    saure.add_argument('--groups', type=int, default=500)
    saure.add_argument('--dat', default='table.dat')

    comparison = sub.add_parser('eval-proposal-comparison',
                                help='Toy evaluation-proposal comparison arms.')
    comparison.add_argument('--paths', type=int, default=4096)
    comparison.add_argument('--groups', type=int, default=500)
    comparison.add_argument('--dat', default='table.dat')
    return parser


def _select_variants(test_envs, variants):
    if variants is None:
        return test_envs
    wanted = {value.strip() for value in variants.split(',') if value.strip()}
    selected = {key: variant for key, variant in test_envs.items()
                if str(key[2]) in wanted}
    missing = wanted - {str(key[2]) for key in test_envs}
    if missing:
        raise ValueError(
            f"--variants values not found in the experiment: {sorted(missing)}; "
            f"available: {sorted(str(key[2]) for key in test_envs)}")
    return selected


def _draw_fixed_init_states(env_args, init_state_seed, count):
    sampler_env_args = copy.deepcopy(env_args)
    sampler_env_args['env_random_seed'] = init_state_seed
    sampler_env = get_config_by_type('infinite_custom', args=sampler_env_args).env
    sampler_env.reset_random_seeds()
    return [
        tuple(np.array(component) for component in sampler_env.generate_initial_state())
        for _ in range(count)
    ]


def _regularization_tag(reg_type, lam):
    return f"{reg_type}_{lam:g}".replace('.', '_').replace('-', 'm')


def _parse_regularization(args):
    reg_type = getattr(args, 'regularization', None)
    lambdas_text = getattr(args, 'regularization_lambda', None)
    if reg_type is None and lambdas_text is None:
        return None, []
    if reg_type is None or lambdas_text is None:
        raise ValueError('--regularization and --regularization-lambda must be given together')
    lambdas = set()
    for item in lambdas_text.split(','):
        item = item.strip()
        if not item:
            raise ValueError(f"--regularization-lambda has an empty entry: {lambdas_text!r}")
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"--regularization-lambda entry is not a number: {item!r}") from exc
        if value < 0:
            raise ValueError(f"--regularization-lambda must be >= 0; got {item!r}")
        lambdas.add(value)
    return reg_type, sorted(lambdas)


def _apply_regularization(test_envs, reg_type, lambdas, scale):
    updated = {}
    for (_, experiment_name, mutate_val), variant in test_envs.items():
        for lam in lambdas:
            copy_variant = copy.deepcopy(variant)
            agent_args = copy_variant.setdefault('agent_args', {})
            agent_args.setdefault('agent_args', {})['regularization'] = {
                'type': reg_type, 'lambda': float(lam), 'scale': scale}
            tag = _regularization_tag(reg_type, lam)
            agent_args['policy_id'] = agent_args.get('policy_id', '') + '_' + tag
            uid = get_uid({'env_args': copy_variant['env_args'], 'agent_args': agent_args})
            updated[(uid, f"{experiment_name}_{tag}", mutate_val)] = copy_variant
    return updated


_PENALTY_SPEC_KEYS = (
    'generating_function_spec',
    'policy_generating_function_spec',
    'penalized_lowerbound_generating_function_spec',
    'training_generating_function_spec',
)
_PENALTY_FUNCTION_TAGS = {'absorption_linear_penalty': 'bh', 'absorption_alp_penalty': 'alp'}


def _apply_penalty_function(test_envs, name):
    if name is None:
        return test_envs
    if name not in GENERATING_FUNCTION_CLASSES:
        raise ValueError(f"unknown penalty function {name!r}; use one of {sorted(GENERATING_FUNCTION_CLASSES)}")
    tag = _PENALTY_FUNCTION_TAGS.get(name)
    updated = {}
    for (_, experiment_name, mutate_val), variant in test_envs.items():
        variant = copy.deepcopy(variant)
        agent_args = variant.setdefault('agent_args', {})
        inner = agent_args.setdefault('agent_args', {})
        for key in _PENALTY_SPEC_KEYS:
            spec = normalize_generating_function_spec(inner.get(key))
            spec['name'] = name
            inner[key] = spec
        if tag is not None:
            agent_args['policy_id'] = agent_args.get('policy_id', '') + '_' + tag
            experiment_name = f"{experiment_name}_{tag}"
        uid = get_uid({'env_args': variant['env_args'], 'agent_args': agent_args})
        updated[(uid, experiment_name, mutate_val)] = variant
    return updated


def _coefficient_bound_tag(bound):
    return f"cb{bound:g}".replace('.', '_').replace('+', '')


def _apply_coefficient_bound(test_envs, bound):
    if bound is None:
        return test_envs
    bound = float(bound)
    if not bound > 0:
        raise ValueError(f'--coefficient-bound must be positive; got {bound}')
    tag = _coefficient_bound_tag(bound)
    updated = {}
    for (_, experiment_name, mutate_val), variant in test_envs.items():
        variant = copy.deepcopy(variant)
        agent_args = variant.setdefault('agent_args', {})
        agent_args.setdefault('agent_args', {})['coefficient_bound'] = bound
        agent_args['policy_id'] = agent_args.get('policy_id', '') + '_' + tag
        uid = get_uid({'env_args': variant['env_args'], 'agent_args': agent_args})
        updated[(uid, f"{experiment_name}_{tag}", mutate_val)] = variant
    return updated


def _load_initial_coefficients(path):
    if path is None:
        return None, None
    path = os.path.normpath(path)
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line) if line.strip() else {}
            if isinstance(record.get('coefficients'), list):
                source = {'file': path, 'uid': record.get('uid')}
                if 'mutate_val' in record:
                    source['mutate_val'] = record['mutate_val']
                return [float(value) for value in record['coefficients']], source
    raise ValueError(f'{path} holds no record with a coefficients list')


def _training_generating_function(variant):
    spec = normalize_generating_function_spec(
        variant.get('agent_args', {}).get('agent_args', {}).get('training_generating_function_spec'))
    env = get_config_by_type('infinite_custom', args=variant['env_args']).env
    return build_generating_function(env, spec)


def _intercept_index(generating_function, flag):
    intercept_index = getattr(generating_function, 'intercept_index', None)
    if intercept_index is None:
        raise ValueError(f'{flag}: {generating_function.spec_name} has no intercept coefficient')
    return intercept_index


def _check_warm_start_inside_box(initial_coefficients, inner, intercept_index, intercept_bound):
    bound = inner.get('coefficient_bound')
    if bound is None and intercept_bound is None:
        return
    for index, value in enumerate(initial_coefficients):
        limit = intercept_bound if (index == intercept_index and intercept_bound is not None) else bound
        if limit is not None and abs(value) > limit:
            raise ValueError(
                f'--init-coefficients entry {index} = {value} lies outside the box +/-{limit}; '
                'raise --coefficient-bound or use --intercept-bound')


def _apply_training_objective(test_envs, objective, initial_coefficients, source, fix_intercept=False,
                              noise_removal=None, intercept_bound=None, paths_per_state=None,
                              min_norm_slack=0.0, fix_blocks=None,
                              worst_case_scope='per_state'):
    if fix_intercept and intercept_bound is not None:
        raise ValueError('--fix-intercept and --intercept-bound are mutually exclusive')
    if paths_per_state is not None and paths_per_state < 2:
        raise ValueError(f'--paths-per-state must be at least 2; got {paths_per_state}')
    if worst_case_scope != 'per_state' and objective != 'worst_case':
        raise ValueError('--worst-case-scope needs --objective worst_case')
    if intercept_bound is not None and not intercept_bound > 0:
        raise ValueError(f'--intercept-bound must be positive; got {intercept_bound}')
    updated = {}
    for (_, experiment_name, mutate_val), variant in test_envs.items():
        variant = copy.deepcopy(variant)
        inner = variant.setdefault('agent_args', {}).setdefault('agent_args', {})
        if objective != 'mean':
            inner['training_objective'] = objective
        if worst_case_scope != 'per_state':
            inner['worst_case_scope'] = worst_case_scope
        if paths_per_state is not None:
            sample_path_number = inner.get('sample_path_number', 256)
            if sample_path_number % paths_per_state:
                raise ValueError(
                    f'--paths-per-state {paths_per_state} does not divide sample_path_number {sample_path_number}')
            inner['paths_per_state'] = paths_per_state
        generating_function = None
        if initial_coefficients is not None or fix_intercept or intercept_bound is not None or fix_blocks:
            generating_function = _training_generating_function(variant)
        if fix_blocks:
            if initial_coefficients is None:
                raise ValueError('--fix-block needs --init-coefficients to pin the block at')
            blocks = generating_function.coefficient_blocks()
            unknown = [name for name in fix_blocks if name not in blocks]
            if unknown:
                raise ValueError(f'--fix-block: unknown block(s) {unknown}; choose from {sorted(blocks)}')
            inner['fixed_coefficients'] = {str(index): float(initial_coefficients[index])
                                           for name in fix_blocks for index in blocks[name]}
        intercept_index = None
        if intercept_bound is not None:
            intercept_index = _intercept_index(generating_function, '--intercept-bound')
            inner['coefficient_bound_overrides'] = {str(intercept_index): float(intercept_bound)}
        if initial_coefficients is not None:
            if len(initial_coefficients) != generating_function.number_of_coefficients:
                raise ValueError(
                    f"--init-coefficients has {len(initial_coefficients)} entries but "
                    f"{generating_function.spec_name} has {generating_function.number_of_coefficients}")
            if 'mutate_val' in source and source['mutate_val'] != mutate_val:
                raise ValueError(
                    f"--init-coefficients record has mutate_val {source['mutate_val']} but the variant is {mutate_val}")
            checked = list(initial_coefficients)
            if fix_intercept:
                checked[_intercept_index(generating_function, '--fix-intercept')] = 0.0
            _check_warm_start_inside_box(checked, inner, intercept_index, intercept_bound)
            inner['initial_coefficients'] = initial_coefficients
            inner['initial_coefficients_source'] = source
        if fix_intercept:
            inner['fixed_coefficients'] = {str(_intercept_index(generating_function, '--fix-intercept')): 0.0}
        if noise_removal is not None:
            inner['noise_removal'] = noise_removal
        if min_norm_slack:
            inner['min_norm_slack'] = float(min_norm_slack)
        uid = get_uid({'env_args': variant['env_args'], 'agent_args': variant['agent_args']})
        updated[(uid, experiment_name, mutate_val)] = variant
    return updated


def _run_train(args):
    test_envs = _select_variants(
        build_variation_test_env(EXPERIMENT_SPECS[args.experiment]), args.variants)
    test_envs = _apply_penalty_function(test_envs, args.penalty_function)
    test_envs = _apply_coefficient_bound(test_envs, args.coefficient_bound)
    reg_type, lambdas = _parse_regularization(args)
    if reg_type is not None:
        test_envs = _apply_regularization(test_envs, reg_type, lambdas, args.regularization_scale)
    initial_coefficients, initial_coefficients_source = _load_initial_coefficients(args.init_coefficients)
    test_envs = _apply_training_objective(
        test_envs, args.objective, initial_coefficients, initial_coefficients_source, args.fix_intercept,
        args.noise_removal, args.intercept_bound, args.paths_per_state, args.min_norm_slack,
        args.fix_block, args.worst_case_scope)
    if args.name is not None:
        if len(test_envs) != 1:
            raise ValueError(f'--name needs exactly one variant; got {len(test_envs)}')
        test_envs = {(key[0], args.name, key[2]): variant for key, variant in test_envs.items()}
    path_seeds = [args.sample_paths_seed + offset for offset in range(args.num_path_seeds)]
    if args.reset_init_state and args.num_init_states > 0:
        raise ValueError('--reset-init-state and --num-init-states are mutually exclusive')
    if args.expected_init_state and (args.reset_init_state or args.num_init_states > 0):
        raise ValueError('--expected-init-state is mutually exclusive with --reset-init-state and --num-init-states')
    if args.paths_per_state is not None and (args.expected_init_state or args.reset_init_state or args.num_init_states > 0):
        raise ValueError('--paths-per-state needs generate mode; drop --expected-init-state, --reset-init-state and --num-init-states')
    records = []
    for key, variant in test_envs.items():
        # The record-level scenario count overrides the agent's at run time
        # (run.py), so it must FOLLOW each variant's own agent_args.
        sample_path_number = (variant.get('agent_args', {})
                              .get('agent_args', {})
                              .get('sample_path_number', 256))
        init_states = [None]
        if args.num_init_states > 0:
            init_states = _draw_fixed_init_states(
                variant['env_args'], args.init_state_seed, args.num_init_states)
        elif args.reset_init_state:
            reset_state = variant['env_args'].get('reset_params', {}).get('init_state')
            if reset_state is None:
                raise ValueError(
                    f"--reset-init-state: variant {key} has no "
                    "env_args['reset_params']['init_state']")
            init_states = [tuple(np.array(component) for component in reset_state)]
        elif args.expected_init_state:
            env = get_config_by_type('infinite_custom', args=variant['env_args']).env
            init_states = [alp_expected_initial_state(env)]
        for init_state in init_states:
            for path_seed in path_seeds:
                records.extend(generate_penalty_coefficient_training_env(
                    {key: variant},
                    dat_file=None,
                    init_state=init_state,
                    sample_path_number=sample_path_number,
                    init_state_seed=args.init_state_seed,
                    sample_paths_seed=path_seed,
                ))
    write_command_file(records, args.dat)
    return records


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _apply_init_occupancy(test_envs, occupancy):
    updated = {}
    for (_, experiment_name, mutate_val), variant in test_envs.items():
        variant = copy.deepcopy(variant)
        mutate_initial_state_congestion(variant['env_args'], occupancy)
        group_uid = get_uid({'env_args': variant['env_args'],
                             'agent_args': variant['agent_args']})
        updated[(group_uid, experiment_name, mutate_val)] = variant
    return updated


def _apply_policy_spec(test_envs, policy_spec_path, policy_ids):
    with open(policy_spec_path, encoding='utf-8') as handle:
        spec = json.load(handle)
    unknown = sorted(set(spec) - set(policy_ids))
    if unknown:
        raise ValueError(
            f"--policy-spec names policy ids not in --policies: {unknown}")
    merged, sources = {}, {}
    for policy_id, overrides in spec.items():
        for key, value in overrides.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"--policy-spec conflict on agent_args['{key}']: "
                    f"{sources[key]} vs {policy_id}")
            merged[key], sources[key] = value, policy_id
    if not merged:
        return test_envs
    updated = {}
    for key, variant in test_envs.items():
        variant = copy.deepcopy(variant)
        _deep_update(variant.setdefault('agent_args', {}).setdefault('agent_args', {}), merged)
        updated[key] = variant
    return updated


def _guard_against_retrain(args, test_envs):
    if args.allow_retrain:
        return
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        sample_path_number = variant['agent_args']['agent_args'].get('sample_path_number')
        cached = load_trained_coefficients_from_folder(
            experiment_name=experiment_name,
            mutate_val=mutate_val,
            sample_path_number=sample_path_number,
            folder_path=args.penalty_dir,
        )
        if cached is None:
            raise SystemExit(
                f"No cached penalty coefficients for experiment={experiment_name}, "
                f"mutate_val={mutate_val}, sample_path_number={sample_path_number} "
                "under the --policy-spec overrides; generation would trigger a "
                "full Benders training run. Re-run with --allow-retrain to accept.")


def _parse_penalty_ratios(text):
    if text is None:
        return None
    ratios = set()
    for item in text.split(','):
        item = item.strip()
        if not item:
            raise ValueError(f"--penalty-ratios has an empty entry: {text!r}")
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"--penalty-ratios entry is not a number: {item!r}") from exc
        if value < 0:
            raise ValueError(f"--penalty-ratios must be >= 0; got {item!r}")
        ratios.add(value)
    return sorted(ratios)


def _run_eval(args, policy_ids):
    penalty_ratios = _parse_penalty_ratios(getattr(args, 'penalty_ratios', None))
    test_envs = _select_variants(
        build_variation_test_env(EXPERIMENT_SPECS[args.experiment]), args.variants)
    if args.init_occupancy is not None:
        test_envs = _apply_init_occupancy(test_envs, args.init_occupancy)
    if args.policy_spec:
        test_envs = _apply_policy_spec(test_envs, args.policy_spec, policy_ids)
    test_envs = _apply_penalty_function(test_envs, args.penalty_function)
    if args.policy_spec:
        _guard_against_retrain(args, test_envs)
    records = generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=args.paths,
        warm_up_periods=args.warm_up,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=True,
        is_random_initial_state=args.random_init,
        policy_ids=policy_ids,
        warm_up_policy_id=args.warm_up_policy,
        penalty_coefficients_dir=args.penalty_dir,
        sample_gen_seed_offset=args.seed_offset,
        evaluation_proposal_spec=(
            json.loads(args.eval_proposal) if args.eval_proposal else None),
        penalty_ratios=penalty_ratios,
    )
    if args.skip_ir:
        for record in records:
            record['skip_information_relaxation'] = True
    write_grouped_command_file(results=records, num_groups=args.groups,
                               dat_file=args.dat)
    return records


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'train':
        return _run_train(args)
    if args.command == 'eval':
        policy_ids = [p for p in args.policies.split(',') if p]
        return _run_eval(args, policy_ids)
    if args.command == 'lowerbound':
        return _run_eval(args, [])
    if args.command == 'saure-ejor':
        return recipe_saure_ejor_case_study(
            test_sample_path_num=args.paths, warm_up_periods=args.warm_up,
            evaluation_periods=args.eval_periods, num_groups=args.groups,
            dat_file=args.dat)
    if args.command == 'eval-proposal-comparison':
        return recipe_toy_eval_proposal_comparison(
            test_sample_path_num=args.paths, num_groups=args.groups,
            dat_file=args.dat)
    raise ValueError(f'Unknown command: {args.command}')
