"""``ApproxQAgent.benders_decomposition_train`` with L1/L2 master regularization (toy env).

Run from the repo root:  python -m test.test_agent_regularization
"""
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import LinearPenaltyFunction


def train(regularization):
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    state, _ = env.reset(**config.reset_params)
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=4,
                         generating_function=LinearPenaltyFunction(env))
    obj, coefficients, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, regularization=regularization)
    return agent, float(obj), np.asarray(coefficients, dtype=float), info


def test_regularized_training_reports_unregularized_bound():
    _, obj_free, theta_free, info_free = train(None)
    assert 'regularization' not in info_free
    n = theta_free.size
    agent, obj, theta, info = train({'type': 'l2', 'lambda': 1e-3})
    report = info['regularization']
    assert report['type'] == 'l2' and report['lambda'] == 1e-3 and report['scale_mode'] == 'feature_std'
    assert len(report['scale']) == n and all(s > 0 for s in report['scale'])
    assert obj == report['saa_objective'] and obj <= obj_free + 1e-6
    values, mean = agent.coefficient_model.evaluate_action(theta, parallel=False)
    assert np.isclose(mean, report['saa_objective'])
    assert report['regularized_objective'] <= report['saa_objective'] + 1e-9
    # The two objectives differ by exactly the regularization term at theta*:
    # saa (no regularization) - regularized = lambda * ||S theta*||^2.
    scaled = np.asarray(report['scale']) * theta
    assert np.isclose(report['saa_objective'] - report['regularized_objective'],
                      report['lambda'] * float(scaled @ scaled), rtol=1e-6, atol=1e-6)


def test_scale_none_is_ones_and_huge_l1_zeroes_coefficients():
    agent, obj, theta, info = train({'type': 'l1', 'lambda': 1e6, 'scale': 'none'})
    assert np.allclose(info['regularization']['scale'], 1.0)
    assert np.max(np.abs(theta)) < 1e-6, theta
    bound_at_zero = float(np.asarray(agent.sample_path_weights) @ np.array(
        [agent.subproblem_initial_cuts[s][0] for s in range(agent.sample_path_number)]))
    assert abs(obj - bound_at_zero) < 1e-6 * max(1.0, abs(bound_at_zero)), (obj, bound_at_zero)


def test_master_builder_rejects_bad_regularization():
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=4,
                         generating_function=LinearPenaltyFunction(env))
    for bad in ({'type': 'l3', 'lambda': 1.0}, {'type': 'l1', 'lambda': -1.0}):
        try:
            agent.train_master_builder_fn(regularization=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    n = agent.generating_function.number_of_coefficients
    try:
        agent.train_master_builder_fn(regularization={'type': 'l1', 'lambda': 1.0},
                                      regularization_scale=np.zeros(n))
    except ValueError:
        pass
    else:
        raise AssertionError('non-positive scale must raise')
    master, coefficient_vars, theta_vars = agent.train_master_builder_fn(
        regularization={'type': 'l1', 'lambda': 1.0})
    assert master.NumVars == 2 * n + theta_vars.shape[0]  # coefficients + coefficient_abs + theta


if __name__ == '__main__':
    test_regularized_training_reports_unregularized_bound()
    test_scale_none_is_ones_and_huge_l1_zeroes_coefficients()
    test_master_builder_rejects_bad_regularization()
    print('All agent regularization tests passed.')
