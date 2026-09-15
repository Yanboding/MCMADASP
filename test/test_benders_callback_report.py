import contextlib
import io
import os
import re
import sys
import unittest

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_maker import ApproxQAgent
from generating_function import AbsorptionLinearPenaltyFunction
from importance_sampling.proposals import MixtureGeometricStratifiedQMCProposal
from metaheuristic_algorithm.benders_decomposition_solver import incumbent_value
from test.record_legacy_golden import MIXTURE, fresh_env, thetas
from utils import acquire_grb_env


class IncumbentValueTest(unittest.TestCase):
    def test_matches_expression_value_at_the_final_incumbent(self):
        grb = acquire_grb_env({'Threads': 1}, verbose=False)
        model = gp.Model('incumbent', env=grb)
        model.Params.OutputFlag = 0
        x = model.addMVar(3, lb=0, ub=5, vtype=GRB.INTEGER)
        expression = np.array([2.0, 3.0, 4.0]) @ x + 1.5
        model.addConstr(x.sum() >= 4)
        model.setObjective(expression, GRB.MINIMIZE)
        seen = []

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                seen.append(incumbent_value(m, expression))

        model.optimize(callback)
        self.assertTrue(seen)
        self.assertAlmostEqual(seen[-1], model.ObjVal)
        model.dispose()


class CallbackReportTest(unittest.TestCase):
    def test_first_stage_is_the_one_time_cost_under_kappa_weights(self):
        grb = acquire_grb_env({'Threads': 1}, verbose=False)
        config, env = fresh_env()
        state = tuple(np.array(c, dtype=float) for c in env.reset(**config.reset_params)[0])
        gf = AbsorptionLinearPenaltyFunction(env, coefficients=thetas(env)['a'])
        agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=5, generating_function=gf,
                             sample_path_proposal=MixtureGeometricStratifiedQMCProposal(**MIXTURE),
                             grb_env=grb, subproblem_grb_envs=[grb])
        kappa = np.asarray(agent.sample_path_weights)
        self.assertFalse(np.allclose(kappa, kappa[0]))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            objective, action, _ = agent.solve(state, t=1, parallel=False)
        lines = [line for line in buffer.getvalue().splitlines() if line.startswith('Callback iter')]
        self.assertTrue(lines)
        last = lines[-1]
        first_stage = float(re.search(r'first-stage ([-\d.]+)', last).group(1))
        cost_to_go = float(re.search(r'cost-to-go ([-\d.]+)', last).group(1))
        evaluated = float(re.search(r'evaluated obj ([-\d.]+)', last).group(1))
        gap = float(re.search(r'MIP gap ([-\d.]+)', last).group(1))
        self.assertAlmostEqual(first_stage + cost_to_go, evaluated, places=3)
        self.assertLess(gap, 1e-3)
        self.assertAlmostEqual(evaluated, objective, places=3)
        self.assertAlmostEqual(first_stage, float(env.cost_fn(state, action)), places=3)


if __name__ == '__main__':
    unittest.main()
