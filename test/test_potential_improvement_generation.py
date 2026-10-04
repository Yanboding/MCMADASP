import json
import numpy as np

from experiments import get_config_by_type
from importance_sampling import build_proposal
from param_generation.potential_improvement import default_prefix_periods, draw_replications
from test.toy_potential_improvement import TOY_ENV_ARGS

GEOMETRIC = {"type": "geometric", "discount_factor_proposal": 0.99}
STRATIFIED = {"type": "stratified_geometric", "discount_factor_proposal": 0.99, "num_strata": 4}


def _draw(spec, size=16, T=5):
    env = get_config_by_type('infinite_custom', args=TOY_ENV_ARGS).env
    return draw_replications(env, build_proposal(spec), size, T, True, None)


def test_default_prefix_is_the_998_quantile():
    assert default_prefix_periods(0.99) == 619


def test_draws_are_deterministic_and_cover_both_horizons():
    first, second = _draw(GEOMETRIC), _draw(GEOMETRIC)
    assert first == second
    for replication in first:
        stream = np.asarray(replication['arrival_stream'])
        assert len(stream) == max(replication['baseline_length'], 5 + replication['continuation_length'])
        assert len(replication['baseline_period_weights']) == replication['baseline_length'] + 1
        assert len(replication['continuation_period_weights']) == replication['continuation_length'] + 1
        assert replication['baseline_period_weights'][0] == 1.0
        assert replication['continuation_period_weights'][0] == 1.0
        assert replication['baseline_terminal'] == 'absorbed'


def test_iid_weights_are_uniform_and_stratified_weights_follow_the_proposal():
    iid = _draw(GEOMETRIC)
    assert {r['path_weight'] for r in iid} == {1.0 / 16} and {r['path_stratum'] for r in iid} == {0}
    stratified = _draw(STRATIFIED)
    proposal = build_proposal(STRATIFIED)
    np.testing.assert_allclose([r['path_weight'] for r in stratified], proposal.path_weights(16))
    assert [r['path_stratum'] for r in stratified] == list(proposal.path_strata(16))


def test_continuation_lengths_are_shuffled_away_from_the_stratum_order():
    stratified = _draw(STRATIFIED, size=64)
    strata = np.array([r['path_stratum'] for r in stratified])
    baseline = np.array([r['baseline_length'] for r in stratified])
    continuation = np.array([r['continuation_length'] for r in stratified])
    for k in range(3):
        assert baseline[strata == k].max() <= baseline[strata == k + 1].min()
    assert any(continuation[strata == k].max() > continuation[strata == k + 1].min() for k in range(3))


def test_init_occupancy_fixes_every_initial_state(tmp_path):
    from param_generation.cli import main
    records = main(['improvement', 'toy_stratified_099_scenario_1024', '--variants', '0.5', '--paths', '4',
                    '--groups', '4', '--init-occupancy', '0.9', '--penalty-dir',
                    'experiments/results/toy_alp_train', '--eval-proposal', json.dumps(GEOMETRIC),
                    '--prefix-periods', '5', '--name', 'occupancy_test', '--dat', str(tmp_path / 'x.dat')])
    expected = [[7, 7, 7, 7, 7, 7, 0], [2, 2, 2, 2, 2, 2, 0], [1, 2]]
    assert all(record['init_state'] == expected for record in records)
