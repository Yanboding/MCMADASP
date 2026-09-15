import os
import sys
import unittest

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_maker import ApproxQAgent
from generating_function import AbsorptionLinearPenaltyFunction
from importance_sampling.proposals import MixtureGeometricStratifiedQMCProposal
from metaheuristic_algorithm import CutPool
from test.record_legacy_golden import MIXTURE, fresh_env, thetas
from utils import acquire_grb_env


def make_agent(env, grb, reuse_cuts, coefficients=None):
    gf = AbsorptionLinearPenaltyFunction(env, coefficients=coefficients or thetas(env)['a'])
    return ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=4, generating_function=gf,
                        sample_path_proposal=MixtureGeometricStratifiedQMCProposal(**MIXTURE),
                        grb_env=grb, subproblem_grb_envs=[grb], reuse_cuts=reuse_cuts)


def trajectory(env, config, periods):
    state, _ = env.reset(**config.reset_params)
    state = tuple(np.array(c, dtype=float) for c in state)
    arrivals = env.reset_arrivals(stop_time=periods)
    env.reset(init_state=state, t=1, new_arrivals=arrivals)
    return state


class CutPoolTest(unittest.TestCase):
    def test_purge_keeps_active_cuts_and_fresh_records(self):
        grb = acquire_grb_env({'Threads': 1}, verbose=False)
        pool = CutPool()
        pool.begin(np.array([1.0]))
        pool.record(0, 1.0, np.array([0.0]), np.array([0.0]), np.array([0.0]))
        pool.record(0, -5.0, np.array([0.0]), np.array([0.0]), np.array([0.0]))
        pool.purge([])
        self.assertEqual(len(pool.entries), 2)
        model = gp.Model('pool', env=grb)
        model.Params.OutputFlag = 0
        theta = model.addMVar(1, lb=-GRB.INFINITY, name='theta')
        x = model.addMVar(1, lb=0.0, ub=1.0, name='x')
        pool.begin(np.array([2.0]))
        added = pool.add_to_master(model, theta, x)
        model.setObjective(theta.sum(), GRB.MINIMIZE)
        model.optimize()
        self.assertAlmostEqual(model.ObjVal, 1.0)
        pool.record(0, 3.0, np.array([0.5]), np.array([0.0]), np.array([1.0]))
        pool.purge(added)
        self.assertEqual([entry.value for entry in pool.entries], [1.0, 3.0])
        model.dispose()

    def test_shifted_constant_uses_state_duals(self):
        grb = acquire_grb_env({'Threads': 1}, verbose=False)
        pool = CutPool()
        pool.begin(np.array([1.0, 2.0]))
        pool.record(0, 10.0, np.array([0.0]), np.array([3.0, -1.0]), np.array([2.0]))
        pool.purge([])
        model = gp.Model('shift', env=grb)
        model.Params.OutputFlag = 0
        theta = model.addMVar(1, lb=-GRB.INFINITY)
        x = model.addMVar(1, lb=0.0, ub=0.0)
        pool.begin(np.array([2.0, 5.0]))
        pool.add_to_master(model, theta, x)
        model.setObjective(theta.sum(), GRB.MINIMIZE)
        model.optimize()
        self.assertAlmostEqual(model.ObjVal, 10.0 + 3.0 * 1.0 - 1.0 * 3.0)
        model.dispose()


class CutReuseAgentTest(unittest.TestCase):
    def setUp(self):
        self.grb = acquire_grb_env({'Threads': 1}, verbose=False)

    def test_reuse_matches_rebuild_along_a_trajectory(self):
        config, env = fresh_env()
        state = trajectory(env, config, periods=4)
        rebuild = make_agent(env, self.grb, reuse_cuts=False)
        reuse = make_agent(env, self.grb, reuse_cuts=True)
        reused_counts = []
        for k in range(4):
            obj_rebuild, action_rebuild, info_rebuild = rebuild.solve(state, t=k + 1, parallel=False)
            obj_reuse, action_reuse, info_reuse = reuse.solve(state, t=k + 1, parallel=False)
            self.assertAlmostEqual(obj_rebuild, obj_reuse, places=6)
            for a, b in zip(action_rebuild, action_reuse):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(info_rebuild['reused_cuts'], 0)
            reused_counts.append(info_reuse['reused_cuts'])
            next_state, _, _, _ = env.step(tuple(np.asarray(c) for c in action_rebuild))
            state = tuple(np.array(c, dtype=float) for c in next_state)
        self.assertEqual(reused_counts[0], 0)
        self.assertTrue(all(count > 0 for count in reused_counts[1:]), reused_counts)

    def test_pool_is_reset_when_the_penalty_changes(self):
        config, env = fresh_env()
        state = trajectory(env, config, periods=2)
        agent = make_agent(env, self.grb, reuse_cuts=True)
        agent.solve(state, t=1, parallel=False)
        _, _, info = agent.solve(state, t=2, parallel=False)
        self.assertGreater(info['reused_cuts'], 0)
        agent.generating_function.set_coefficients(thetas(env)['b'])
        _, _, info = agent.solve(state, t=3, parallel=False)
        self.assertEqual(info['reused_cuts'], 0)


if __name__ == '__main__':
    unittest.main()
