from unittest import mock

import gurobipy as gp
import numpy as np
import pytest

import run
from experiments import get_config_by_type
from generating_function import AbsorptionALPPenaltyFunction
from param_generation import cli, datasets
from test.test_eval_proposal_generation import make_test_envs


def test_lowerbound_flag_is_opt_in_and_passed_to_generation():
    for flags, expected in (([], False), (['--relax-first-decision'], True)):
        with mock.patch.object(cli, 'generate_test_paths_and_init_state', return_value=[]) as generator, \
             mock.patch.object(cli, 'write_grouped_command_file'):
            cli.main(['lowerbound', 'base_toy_study', '--paths', '1', *flags])
        assert generator.call_args.kwargs['relax_first_decision'] is expected
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ['eval', 'base_toy_study', '--policies', 'myopic', '--relax-first-decision'])


def test_relaxed_records_have_distinct_uids_without_changing_legacy_records():
    def generate(**extra):
        return datasets.generate_test_paths_and_init_state(
            test_envs=make_test_envs(), test_sample_path_num=1,
            is_require_penalty_coefficients=False, policy_ids=['myopic'],
            evaluation_proposal_spec={'type': 'fixed', 'max_length': 1}, **extra)[0]

    legacy = generate()
    # Captured from the pre-flag generator with these deterministic inputs.
    assert legacy['uid'] == '6998d4dc2a2755f859efb83006028b7a'
    assert 'relax_first_decision' not in legacy
    assert generate(relax_first_decision=False) == legacy
    relaxed = generate(relax_first_decision=True)
    assert relaxed['relax_first_decision'] is True
    assert relaxed['uid'] != legacy['uid']
    assert {k: v for k, v in relaxed.items() if k not in ('uid', 'relax_first_decision')} == {
        k: v for k, v in legacy.items() if k != 'uid'}


@pytest.fixture(scope='module')
def grb_env():
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', 1)
        env.start()
        yield env


def test_continuous_first_decision_changes_the_actual_ir_feasible_set(grb_env):
    env = get_config_by_type('toy').env
    state = (np.zeros(env.planning_horizon), np.zeros(env.planning_horizon), np.array([0.5, 1.5]))
    args = (env, {'name': 'absorption_alp_penalty'}, [0.0], state,
            np.zeros((0, env.num_types)), [1.0], 'absorbed', grb_env, [])
    legacy = run.information_relaxation_bounds(*args)[0.0]
    relaxed = run.information_relaxation_bounds(*args, relax_first_decision=True)[0.0]
    assert relaxed < legacy - 1e-6
    assert run.information_relaxation_bounds(*args, relax_first_decision=False)[0.0] == pytest.approx(legacy)


def _evaluation_args(grb_env):
    config = get_config_by_type('toy')
    env = config.env
    anchor = np.linspace(-5.0, 5.0, AbsorptionALPPenaltyFunction(env).number_of_coefficients).tolist()
    return dict(
        uid='relax-first-test', experiment_name='relax-first-test', mutate_val=0.5,
        init_state=(np.array([2.] * (env.planning_horizon - 1) + [0.]),
                    np.zeros(env.planning_horizon), np.array([1., 2.])),
        sample_path=[[1., 2.]], warm_up_periods=0, env_args=config.args,
        policy_specs=[], group_id='g', grb_env=grb_env, grb_sub_envs=[], job_id='test',
        generating_function_spec={'name': 'absorption_alp_penalty', 'coefficients': anchor},
        period_weights=[1.0, env.discount_factor], terminal='absorbed')


@pytest.mark.parametrize('centering', [None, 'anchor', 'policy'])
@pytest.mark.parametrize('relax_first_decision', [False, True])
def test_evaluation_relaxes_all_ir_bounds_but_keeps_fixed_policy_integer(
        grb_env, tmp_path, monkeypatch, centering, relax_first_decision):
    monkeypatch.chdir(tmp_path)
    args = _evaluation_args(grb_env)
    if centering == 'anchor':
        args['alp_coefficients'] = args['generating_function_spec']['coefficients']
    elif centering == 'policy':
        args['policy_centering'] = True
    with mock.patch.object(run, 'ApproxQAgent', wraps=run.ApproxQAgent) as agent:
        (record,) = run.evaluate_policy_costs_with_information_relaxation(
            **args, relax_first_decision=relax_first_decision)
    current = 'continuous' if relax_first_decision else 'integer'
    expected = [(current, 'continuous')] * 2
    if centering is not None:
        expected += [('integer', 'continuous')]
    if centering == 'anchor':
        expected += [(current, 'continuous')]
        assert record['center_potential_gap'] == pytest.approx(0.0, abs=1e-8)
    assert [(call.kwargs['current_decision_var_type'], call.kwargs['future_decision_var_type'])
            for call in agent.call_args_list] == expected
    if relax_first_decision:
        assert record['relax_first_decision'] is True
        assert record['information_relaxation_decision_types'] == {
            'current': 'continuous', 'future': 'continuous'}
    else:
        assert 'relax_first_decision' not in record
        assert 'information_relaxation_decision_types' not in record


def test_policy_branch_relaxes_bounds_without_modifying_policy_arguments(grb_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _evaluation_args(grb_env)
    args['policy_specs'] = [{'policy_id': 'myopic', 'agent_name': 'myopic', 'agent_args': {}}]
    policy_result = {'policy_id': 'myopic', 'agent_name': 'myopic',
                     'total_cost': 100.0, 'penalized_cost': 110.0, 'total_penalty': 10.0}
    with mock.patch.object(run, 'ApproxQAgent', wraps=run.ApproxQAgent) as agent, \
         mock.patch.object(run, 'calculate_policy_costs_with_penalty',
                           return_value=policy_result) as policy:
        (summary,) = run.evaluate_policy_costs_with_information_relaxation(
            **args, relax_first_decision=True)
    assert summary['policy_id'] == 'myopic'
    assert len(agent.call_args_list) == 2
    assert all(call.kwargs['current_decision_var_type'] == 'continuous'
               and call.kwargs['future_decision_var_type'] == 'continuous'
               for call in agent.call_args_list)
    assert 'relax_first_decision' not in policy.call_args.kwargs
    assert policy.call_args.kwargs['agent_args'] == {}
