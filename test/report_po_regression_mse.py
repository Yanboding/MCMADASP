"""Report the PO regression fit quality on the SAME initial states and sample
paths used for the Benders (pathwise-optimization) training.

Scenarios are grouped by shared initial state (``paths_per_init_state``
consecutive sample paths per state): X = the per-group initial states, Y = the
group-average information-relaxation costs at the trained coefficients, so
|X| = |Y| = sample_path_number / paths_per_init_state and mean(Y) equals the
PO lower bound. Averaging shrinks the target noise by 1/paths_per_init_state.
"""
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import MulticlassLinearPenaltyFunction, MulticlassQuadraticPenaltyFunction


def main():
    config = get_config_by_type('toy')
    env = config.env
    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=2048,
                         paths_per_init_state=32,
                         generating_function=MulticlassLinearPenaltyFunction(env),
                         solver_name='approx_Q')

    value_generating_function = MulticlassQuadraticPenaltyFunction(env)

    # Stage 1: Benders pathwise optimization on the agent's sample paths.
    po_lower_bound, _, _ = agent.benders_decomposition_train(parallel=True, verbose=False)

    # Stage 2: extract the regression dataset (group-averaged IR costs on the
    # SAME initial states/sample paths used in training).
    X, Y = agent.extract_po_regression_targets()
    print(f"|X| = |Y| = {len(X)} ({agent.paths_per_init_state} paths per state); "
          f"mean(Y) = {np.mean(Y):.4f} (PO bound {po_lower_bound:.4f})")

    # Stage 3: fit the quadratic value function on (X, Y).
    agent.generating_function = value_generating_function
    value_coefficients, training_mse = agent.regression_train(X, Y, regularization=1e-6)

    targets = np.asarray(Y, dtype=float)
    predictions = np.array([value_generating_function.calculate_state_value(state) for state in X])
    residuals = predictions - targets
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    target_std = float(np.std(targets))
    r_squared = 1.0 - mse / float(np.var(targets))

    # Cross-check: exact UNREGULARIZED least squares on the explicit design
    # matrix gives the best training MSE any fit of this basis can achieve.
    n_coeff = value_generating_function.number_of_coefficients
    design_matrix = np.zeros((len(X), n_coeff))
    for k in range(n_coeff):
        unit = np.zeros(n_coeff)
        unit[k] = 1.0
        blocks = value_generating_function.get_coefficients(unit)
        for i, state in enumerate(X):
            design_matrix[i, k] = value_generating_function.calculate_state_value(state, coefficients=blocks)
    kappa_lstsq, *_ = np.linalg.lstsq(design_matrix, targets, rcond=None)
    best_possible_mse = float(np.mean((design_matrix @ kappa_lstsq - targets) ** 2))

    print("=" * 60)
    print(f"PO lower bound:                {po_lower_bound:.4f}")
    print(f"sample paths:                  {agent.sample_path_number}")
    print(f"paths per initial state:       {agent.paths_per_init_state}")
    print(f"regression samples (states):   {len(Y)}")
    print(f"quadratic basis coefficients:  {len(value_coefficients)}")
    print(f"target mean / std:             {targets.mean():.4f} / {target_std:.4f}")
    print(f"target min / max:              {targets.min():.4f} / {targets.max():.4f}")
    print(f"train MSE (from solver):       {training_mse:.6e}")
    print(f"train MSE (recomputed):        {mse:.6e}")
    print(f"best possible MSE (lstsq):     {best_possible_mse:.6e}")
    print(f"train RMSE:                    {rmse:.4f}")
    print(f"R^2:                           {r_squared:.4f}")
    print(f"mean |relative error|:         {np.mean(np.abs(residuals) / np.maximum(np.abs(targets), 1.0)):.4%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
