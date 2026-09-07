"""Penalty forms: weight tables, zero mean, legacy conventions (toy env).

Run from the repo root:  python -m test.test_penalty_forms
"""
import numpy as np

from decision_maker import MyopicAgent
from experiments import get_config_by_type
from generating_function import AbsorptionForm, AbsorptionLinearPenaltyFunction, LegacyForm
from importance_sampling import FixedLengthProposal, GeometricLengthProposal, SamplePath, Terminal
from importance_sampling.proposals import MixtureGeometricStratifiedQMCProposal
from utils import acquire_grb_env


def path_of(length, terminal, survival_weights=None, likelihood_ratios=None):
    return SamplePath(np.zeros((length, 2)), terminal, survival_weights, likelihood_ratios)


def test_absorption_term_weights():
    form = AbsorptionForm()
    gamma = 0.6
    for tau in (1, 2, 5):
        W = np.array([1.0] + [0.5 + 0.1 * k for k in range(1, tau)])
        for terminal in (Terminal.ABSORBED, Terminal.TRUNCATED):
            for s in range(tau):
                expected_weight, realized_weight = form.term_weights(s, tau, W, gamma, terminal)
                if s < tau - 1:
                    assert (expected_weight, realized_weight) == (gamma * W[s], W[s + 1])
                elif terminal is Terminal.ABSORBED:
                    assert (expected_weight, realized_weight) == (gamma * W[s], 0.0)
                else:
                    assert (expected_weight, realized_weight) == (0.0, 0.0)
    # Weights come from the path; missing law or outcome is refused.
    path = path_of(2, Terminal.ABSORBED, [1.0, 0.7, 0.6])
    assert np.allclose(form.period_weights(path, gamma), [1.0, 0.7, 0.6])
    for bad in (path_of(2, Terminal.ABSORBED), path_of(2, Terminal.UNSPECIFIED, [1.0, 0.7, 0.6])):
        try:
            form.period_weights(bad, gamma)
        except ValueError:
            pass
        else:
            raise AssertionError('AbsorptionForm accepted a path without an absorption law')
    assert form.hindsight_scenario_weights == 'kappa'


def test_absorption_zero_mean_identity_under_survival_weights():
    """sum_s [gamma P(L >= s-1) w_s - P(L >= s) w_{s+1}] = 0 term by term."""
    gamma = 0.9
    for proposal in (GeometricLengthProposal(0.8), MixtureGeometricStratifiedQMCProposal(gamma, 0.7, 0.4)):
        S = 30
        periods = np.arange(1, S + 2)
        survival = np.concatenate(([1.0], proposal.survival_probability(periods[:-1])))  # P(L >= s-1), s = 1..S+1
        w = gamma ** (periods - 1) / survival
        expected = gamma * survival[:-1] * w[:-1]
        realized = survival[1:] * w[1:]
        assert np.allclose(expected - realized, 0.0, atol=1e-12)


def test_legacy_modes_reproduce_the_old_conventions():
    gamma = 0.9
    u = np.array([1.0, 1.2, 1.5])
    with_ratios = path_of(3, Terminal.UNSPECIFIED, likelihood_ratios=u)
    assert np.allclose(LegacyForm('training').period_weights(with_ratios, gamma), [1.0, *u])
    assert np.allclose(LegacyForm('hindsight').period_weights(with_ratios, gamma), [1.0, *(gamma * u)])
    without = path_of(3, Terminal.UNSPECIFIED)
    assert np.allclose(LegacyForm('training').period_weights(without, gamma), 1.0)
    assert np.allclose(LegacyForm('hindsight').period_weights(without, gamma), [1.0, gamma, gamma, gamma])
    # Evaluation reads the record's weights; a record without them means ones.
    record = path_of(3, Terminal.UNSPECIFIED, survival_weights=[1.0, 0.9, 0.8, 0.7])
    assert np.allclose(LegacyForm('evaluation').period_weights(record, gamma), [1.0, 0.9, 0.8, 0.7])
    assert np.allclose(LegacyForm('evaluation').period_weights(without, gamma), 1.0)
    W = np.array([1.0, 2.0, 3.0, 4.0])
    for tau in (1, 2, 4):
        for mode in LegacyForm.MODES:
            form = LegacyForm(mode)
            for s in range(tau):
                weights = form.term_weights(s, tau, W, gamma, Terminal.UNSPECIFIED)
                if s == tau - 1:
                    assert weights == (0.0, 0.0)
                else:
                    k = s if mode == 'evaluation' else s + 1
                    assert weights == (W[k], W[k])
    for mode in LegacyForm.MODES:
        assert LegacyForm(mode).hindsight_scenario_weights == 'uniform'
    try:
        LegacyForm('bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('LegacyForm accepted an unknown mode')


def test_combine_matches_manual_sum():
    gamma = 0.9
    expected_terms = [10.0, 20.0, 30.0]
    realized_terms = [1.0, 2.0]
    path = path_of(2, Terminal.ABSORBED, survival_weights=[1.0, 0.8, 0.5], likelihood_ratios=[1.1, 1.3])
    absorbed = AbsorptionForm().combine(path, gamma, expected_terms, realized_terms)
    manual = (gamma * 1.0 * 10 - 0.8 * 1) + (gamma * 0.8 * 20 - 0.5 * 2) + gamma * 0.5 * 30
    assert np.isclose(absorbed, manual)
    truncated = path_of(2, Terminal.TRUNCATED, survival_weights=[1.0, 0.8, 0.5])
    assert np.isclose(AbsorptionForm().combine(truncated, gamma, expected_terms, realized_terms), manual - gamma * 0.5 * 30)
    evaluation = LegacyForm('evaluation').combine(path, gamma, expected_terms, realized_terms)
    assert np.isclose(evaluation, 1.0 * (10 - 1) + 0.8 * (20 - 2))
    training = LegacyForm('training').combine(path, gamma, expected_terms, realized_terms)
    assert np.isclose(training, 1.1 * (10 - 1) + 1.3 * (20 - 2))
    try:
        AbsorptionForm().combine(path, gamma, expected_terms[:2], realized_terms)
    except ValueError:
        pass
    else:
        raise AssertionError('combine accepted mismatched term lists')


def rollout_terms(env, agent, gf, theta, state, path):
    expected_terms, realized_terms = [], []
    for s in range(path.periods):
        _, action, _ = agent.solve(state, t=s + 1)
        action = tuple(np.array(c, dtype=float) for c in action)
        expected_terms.append(gf.expected_value(theta, state, action))
        if s < path.periods - 1:
            realized_terms.append(gf.value(theta, state, action, path.arrivals[s]))
            state = tuple(np.array(c, dtype=float) for c in env.get_next_state(state, action, path.arrivals[s]))
    return expected_terms, realized_terms


def test_monte_carlo_zero_mean_along_myopic_rollouts():
    """theta . Phi has mean zero under the absorption form for a fixed
    (nonanticipative) policy: the length law, the survival weights and the
    gamma given to the form all use gamma_test = 0.6 (plan C35)."""
    gamma_test = 0.6
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    grb_env = acquire_grb_env({'Threads': 1}, verbose=False)
    agent = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env)
    rng = np.random.default_rng(5)
    gf = AbsorptionLinearPenaltyFunction(env)
    theta = rng.normal(size=gf.number_of_coefficients) * 3.0
    init_state, _ = env.reset(**config.reset_params)
    init_state = tuple(np.array(c, dtype=float) for c in init_state)
    form = AbsorptionForm()
    for proposal, count in ((GeometricLengthProposal(gamma_test), 300), (FixedLengthProposal(3), 120)):
        paths = proposal.sample_paths(env.arrival_generator, count, target_discount_factor=gamma_test)
        values = []
        for path in paths:
            expected_terms, realized_terms = rollout_terms(env, agent, gf, theta, init_state, path)
            values.append(form.combine(path, gamma_test, expected_terms, realized_terms))
        values = np.array(values)
        half_width = 3.0 * values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean()) <= half_width + 1e-9, (type(proposal).__name__, values.mean(), half_width)
        assert values.std() > 0.0


if __name__ == '__main__':
    test_absorption_term_weights()
    test_absorption_zero_mean_identity_under_survival_weights()
    test_legacy_modes_reproduce_the_old_conventions()
    test_combine_matches_manual_sum()
    test_monte_carlo_zero_mean_along_myopic_rollouts()
    print('All penalty-form tests passed.')
