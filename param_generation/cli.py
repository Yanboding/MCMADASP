"""Command-line interface for generating experiment parameter files.

``main`` exposes argparse subcommands (``train`` / ``eval`` / ``lowerbound``
plus the two multi-step orchestration recipes); each thin handler drives the
generation entry points in :mod:`param_generation.datasets` and writes a
``run.py`` job-array ``.dat`` file.
"""

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
from param_generation.mutators import mutate_initial_state_congestion
from param_generation.registry import EXPERIMENT_SPECS
from param_generation.training import train_alp_coefficients
from utils import get_uid


def recipe_toy_eval_proposal_comparison(
    test_sample_path_num=4096,
    num_groups=500,
    dat_file='table.dat',
    penalty_coefficients_dir=os.path.join(
        'experiments', 'results', 'mixture_probability_toy_study'
    ),
):
    """Evaluation-proposal comparison on the toy env (spec:
    docs/superpowers/specs/2026-07-29-eval-proposal-comparison-design.md).

    ONE fixed approx_penalized_hindsight policy (internal mixture IS proposal
    lambda_0=0.1, trained coefficients reused from
    ``mixture_probability_toy_study``) is evaluated on tails drawn from three
    different length proposals; each record carries the matching
    ``period_weights`` so ``run.py`` stays unbiased per arm. All
    3 x ``test_sample_path_num`` records go into ONE grouped dat file.
    """
    # Fail fast if the trained coefficients are missing: silently retraining
    # per arm would be slow and would break the fixed-policy comparison.
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
    """Emit all evaluation commands for the Saure EJOR case-study pair (spec:
    docs/superpowers/specs/2026-08-02-saure-ejor-case-study-design.md).

    Experiment 1 (``case_study_ejor_replication``):
    ALP and myopic on ``test_sample_path_num`` fixed-length paths
    (``warm_up_periods`` warm-up + ``evaluation_periods`` evaluation periods).
    Each policy warms itself up; the evaluation tail is weighted by
    gamma**(t-1) via a fixed-length proposal, i.e. the discounted tail cost.

    Experiment 2 (``case_study_ejor_alp_steady_state``): penalized hindsight, ALP and myopic on paths
    whose warm-up prefixes are IDENTICAL to experiment 1's. ALP alone rolls the
    warm-up (``warm_up_policy_id``) and every policy starts from its per-path
    steady state. Tail lengths come from the same mixture-geometric proposal the
    penalty coefficients were trained with (lambda_0=0.1, q=0.95, target 0.99),
    with unbiased per-period importance weights.

    Both experiments reuse the trained overtime-100 penalty coefficients and one
    shared ALP. Warm-up prefixes are drawn once from a dedicated seed stream
    (offset 3003); the experiments' tails use offsets 1001/2002, so tails are
    independent across experiments while warm-ups coincide. All ``2 * test_sample_path_num`` records are written into ONE grouped
    ``dat_file`` (default ``table.dat``); ``run.py`` routes each record to its
    own experiment results folder via ``experiment_name``. The file is
    SEGMENTED: lines ``1..num_groups//2`` hold experiment 1 only (with
    ``skip_information_relaxation=True`` — policy costs only), the remaining
    lines experiment 2 only, so cluster jobs can be resourced per experiment.

    For a quick local smoke run pass a small ``test_sample_path_num``
    explicitly (e.g. ``recipe_saure_ejor_case_study(test_sample_path_num=4,
    dat_file='table_smoke.dat')``) instead of editing the default: a bare
    ``python generate_params.py`` must keep producing the full 4096-path
    deliverable.
    """
    shared_coefficients = load_trained_coefficients_from_folder(
        experiment_name='case_study_099_mixture_geometric_proposal_095',
        mutate_val=0.1,
        sample_path_number=256,
        folder_path=penalty_coefficients_dir,
    )

    def resolve_penalty_coefficients_dir(experiment_name):
        """Prefer trained coefficients already present in the experiment's OWN
        results folder (e.g. a per-experiment ``penalty_coefficients.jsonl``);
        fall back to the shared overtime-100 training folder. Returning ``None``
        makes ``generate_test_paths_and_init_state`` load from the own folder."""
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

    # Resolve both experiments up front so a missing-coefficients error fires
    # before any ALP training or path generation.
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

    # Train (or load) the shared ALP once under the replication experiment,
    # then copy the cache so the steady-state experiment embeds the same
    # coefficients instead of retraining.
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

    # Both experiments must start every path from the SAME warm-up arrivals, so
    # the prefixes are drawn once from a dedicated stream and passed to both
    # generation calls.
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
        # No evaluation_proposal_spec: tails fall back to the agent's own
        # mixture-geometric proposal, matching the trained coefficients.
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
    """Keep only the variants whose ``mutate_val`` is listed in ``variants``
    (a comma-separated CLI string; values are compared as strings so numeric
    and non-numeric mutate_vals both work). ``None`` keeps everything."""
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
    """Draw ``count`` initial states from the env's reference distribution,
    reproducibly seeded by ``init_state_seed`` (same sampler as run.py's
    per-scenario draw, so state k here equals scenario k's state there)."""
    sampler_env_args = copy.deepcopy(env_args)
    sampler_env_args['env_random_seed'] = init_state_seed
    sampler_env = get_config_by_type('infinite_custom', args=sampler_env_args).env
    sampler_env.reset_random_seeds()
    return [
        tuple(np.array(component) for component in sampler_env.generate_initial_state())
        for _ in range(count)
    ]


def _run_train(args):
    test_envs = _select_variants(
        build_variation_test_env(EXPERIMENT_SPECS[args.experiment]), args.variants)
    path_seeds = [args.sample_paths_seed + offset for offset in range(args.num_path_seeds)]
    if args.reset_init_state and args.num_init_states > 0:
        raise ValueError('--reset-init-state and --num-init-states are mutually exclusive')
    records = []
    for key, variant in test_envs.items():
        # The record-level scenario count overrides the agent's at run time
        # (run.py), so it must FOLLOW each variant's own agent_args.
        sample_path_number = (variant.get('agent_args', {})
                              .get('agent_args', {})
                              .get('sample_path_number', 256))
        # None -> the runner draws one state per scenario (default); otherwise
        # each fixed state yields its own command, shared by all scenarios.
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
    """Pin every variant's reset initial state to ``occupancy`` of total
    capacity (see ``mutate_initial_state_congestion``) and re-key the variants
    on the resulting env uid, so ``group_id`` in the emitted records reflects
    the changed initial state."""
    updated = {}
    for (_, experiment_name, mutate_val), variant in test_envs.items():
        variant = copy.deepcopy(variant)
        mutate_initial_state_congestion(variant['env_args'], occupancy)
        group_uid = get_uid({'env_args': variant['env_args'],
                             'agent_args': variant['agent_args']})
        updated[(group_uid, experiment_name, mutate_val)] = variant
    return updated


def _apply_policy_spec(test_envs, policy_spec_path, policy_ids):
    """Deep-merge per-policy ``agent_args`` overrides into each variant.

    ``policy_spec_path`` is a JSON object ``{policy_id: {agent_args...}}``.
    Only ids in ``policy_ids`` are honoured (unknown ids raise). The variant's
    single ``agent_args`` block feeds every penalty-family policy, so overrides
    for different policy ids are merged into the same block; conflicting keys
    across ids raise instead of silently winning by order.
    """
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
    """With --policy-spec the overridden config may have no cached penalty
    coefficients; refuse to fall through to a full Benders training run
    unless --allow-retrain was passed."""
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


def _run_eval(args, policy_ids):
    test_envs = _select_variants(
        build_variation_test_env(EXPERIMENT_SPECS[args.experiment]), args.variants)
    if args.init_occupancy is not None:
        test_envs = _apply_init_occupancy(test_envs, args.init_occupancy)
    if args.policy_spec:
        test_envs = _apply_policy_spec(test_envs, args.policy_spec, policy_ids)
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
