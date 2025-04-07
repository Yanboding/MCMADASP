import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple
from utils.running_stat import RunningStat


class PolicyEvaluator:

    def __init__(self, env, agent, discount_factor, V=None):
        self.env = env
        self.agent = agent
        self.discount_factor = discount_factor
        if V is None:
            self.V = {}

    def evaluate(self, state, t):
        state_tuple = iter_to_tuple(state)
        action = self.agent.policy(state, t)
        q = 0
        for prob, next_state, cost, done in self.env.transition_dynamic(state, action, t):
            next_state_tuple = iter_to_tuple(next_state)
            if (next_state_tuple, t + 1) in self.V:
                next_state_val = self.V[(next_state_tuple, t + 1)]
            elif done:
                next_state_val = 0
            else:
                next_state_val = self.evaluate(next_state, t + 1)
            q += prob * (cost + self.discount_factor * next_state_val)
        self.V[(state_tuple, t)] = q
        return self.V[(state_tuple, t)]
    '''
    def solve(self):
        """
        Build and solve the MIP model using GurobiPy.

        Parameters:
        - N (int): Number of time periods.
        - I (int): Number of items.
        - gamma (float): Discount factor per period.
        - w (list): Waiting cost weights w[i] for i = 0 to I-1.
        - b1 (list of lists): Backlog parameter b1[t][i] for t = 0 to N-1, i = 0 to I-1.
        - delta (list of lists): Demand delta[t][i] for t = 0 to N-1, i = 0 to I-1.
        - r (list): Resource requirements r[i] for i = 0 to I-1.
        - C (float): Capacity limit per period.
        - O (float): Overtime cost coefficient.

        Returns:
        - solution_a (dict): Optimal values of a[tau, t, i].
        - solution_y (list): Optimal values of y[t].
        - obj_val (float): Optimal objective value.
        - Returns (None, None, None) if no optimal solution is found.
        """
        N = self.env.decision_epoch
        I = self.env.num_types
        gamma = self.discount_factor
        w = self.env.holding_cost
        b1 = self.env.future_first_appts
        delta = self.env.new_arrivals
        r = self.env.treatment_pattern
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        # Create the model
        model = gp.Model("Scheduling_MIP")

        # Add decision variables
        # a[tau, t, i] for t in range(N) (periods 1 to N), tau in range(N-t), i in range(I)
        a = model.addVars(
            [(tau, t, i) for t in range(N) for tau in range(N - t) for i in range(I)],
            vtype=GRB.INTEGER,
            lb=0,
            name="a"
        )

        # y[t] for t in range(N)
        y = model.addVars(
            N,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="y"
        )

        # Set objective function
        # Minimize sum_{t=1}^N gamma^(t-1) [ sum_{i=1}^I w_i ( sum_{j=0}^{N-t} b^1_{t+j-1, i} + sum_{k=0}^{t-1} a^{t-k}_{k, i} ) + O y^t ]
        # In code, t ranges from 0 to N-1, so gamma^(t-1) becomes gamma^t
        obj = gp.quicksum(
            gamma**t * (
                gp.quicksum(
                    w[i] * (
                        # sum_{j=0}^{N-t} b^1_{t+j-1, i}, ensuring t+j-1 is between 0 and N-1
                        gp.quicksum(
                            b1[t + j - 1][i] for j in range(N - t + 1) if 0 <= t + j - 1 < N
                        )
                        + # sum_{k=0}^{t-1} a^{t-k}_{k, i}, which is a[k, t-k, i] in code
                        gp.quicksum(
                            a[k, t - k, i] for k in range(t)
                        )
                    )
                    for i in range(I)
                )
                + O * y[t]
            )
            for t in range(N)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        # Add constraints

        # 1. Demand constraints: sum_{tau=0}^{N-t} a[tau, t, i] == delta[t][i]
        for t in range(N):
            for i in range(I):
                model.addConstr(
                    gp.quicksum(a[tau, t, i] for tau in range(N - t)) == delta[t][i],
                    name=f"demand_t{t}_i{i}"
                )

        # 2. Overtime constraints: y[t] >= sum_{i=1}^I (b^1_{t-1,i} + sum_{k=0}^{t-1} a^{t-k}_{k,i}) r_i - C
        for t in range(N):
            workload = gp.quicksum(
                (
                    (b1[t - 1][i] if t >= 1 else 0) +  # b^1_{t-1,i}, no backlog for t=0 (period 1)
                    gp.quicksum(a[k, t - k, i] for k in range(t))  # sum_{k=0}^{t-1} a^{t-k}_{k,i}
                ) * r[i, 0]
                for i in range(I)
            )
            model.addConstr(
                y[t] >= workload - C,
                name=f"overtime_t{t}"
            )
        model.optimize()

        # Check if optimal solution is found
        if model.status == GRB.OPTIMAL:
            solution_a = {(tau, t, i): a[tau, t, i].x for tau, t, i in a.keys()}
            solution_y = [y[t].x for t in range(N)]
            return solution_a, solution_y, model.objVal
        else:
            return None, None, None

    def simulation_evaluate(self, state, replication):
        for _ in range(replication):
            self.env.reset(state, 1)
            solution_a, solution_y, objVal = self.solve()
            self.value_function_stats.record(objVal)
        mean = self.value_function_stats.mean()
        half_window = self.value_function_stats.half_window(.95)
        print('half_window')
        print(half_window)
        return self.value_function_stats.mean(), mean-half_window, mean+half_window
    '''

if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv
    from environment.utility import get_system_dynamic
    env_params = {
        'treatment_pattern': [[2],
                              [1]],
        'decision_epoch': 7,
        'system_dynamic': get_system_dynamic(3, 4, [0.5, 0.5]),
        'holding_cost': [10, 5],
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 2,
        'discount_factor': 0.99
    }
    env = AdvanceSchedulingEnv(**env_params)
    evaluator = PolicyEvaluator(env, None, env.discount_factor)
    print(evaluator.simulation_evaluate(1000))


