import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from utils import iter_to_tuple


class MyopicAllocationAdvanceAgent:

    def __init__(self, env, discount_factor, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        if self.env.problem_type == 'allocation':
            self.policy = self.allocation_policy
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}

    def solve(self, state, t, tau):
        N = self.env.decision_epoch
        I = self.env.num_types
        w = self.env.holding_cost
        bookings, w_bar, b = state
        r = self.env.treatment_pattern
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        # Create the model
        model = gp.Model("MyopicAllocation_MIP")
        # Suppress Gurobi output completely
        model.setParam('OutputFlag', 0)
        model.setParam('LogToConsole', 0)
        # Decision Variables
        q_t = model.addVars(I, vtype=GRB.INTEGER, name="q_t")
        y_t = model.addVar(lb=0, name="y_t")
        # Set objective function
        # Minimize sum_{t=1}^N gamma^(t-1) [ sum_{i=1}^I w_i ( sum_{j=0}^{N-t} b^1_{t+j-1, i} + sum_{k=0}^{t-1} a^{t-k}_{k, i} ) + O y^t ]
        # In code, t ranges from 0 to N-1, so gamma^(t-1) becomes gamma^t
        obj = gp.quicksum(
            w[i] * (w_bar[i] + sum(b[j, i] for j in range(tau, N - t + 1))) for i in range(I)) + O * y_t

        model.setObjective(obj, GRB.MINIMIZE)

        # Add constraints
        # (cons1) Ensure arrivals match scheduled "starts":
        #    sum_{k=0}^{N - tau} a[tau, k, i] = delta[tau, i]   for tau=t..N, i=1..I
        for i in range(I):
            model.addConstr(q_t[i] <= w_bar[i], name=f"ValidAllocation[i={i+1}]")

        # (cons2) Overtime capacity constraints:
        #    y[tau] >= sum_i ( b[tau-t,i] + sum_{k=0}^{tau-t} a[tau-k, k,i] ) * r_i - C
        #    for tau=t..N
        workload = gp.quicksum((b[tau, i] + q_t[i]) * r[0, i] for i in range(I))
        model.addConstr(workload - y_t <= C, name=f"Overtime[tau={tau}]")
        model.optimize()
        # Check if optimal solution is found
        if model.status == GRB.OPTIMAL:
            q_t_solution = model.getAttr('X', q_t)
            y_t_solution = y_t.X
            solution_q = np.zeros(I)
            for i, value in q_t_solution.items():
                solution_q[i] = value
            solution_q = solution_q.astype(int)
            obj_value = model.objVal
            return solution_q, y_t_solution, obj_value
        else:
            return None, None, None

    def policy(self, state, t):
        '''
        Sample paths starting from
        :param state:
        :param t:
        :return:
        '''
        bookings, remaining_arrivals, future_schedule = state
        action = np.zeros_like(future_schedule)
        for tau in range(len(future_schedule)):
            if t+tau == self.env.decision_epoch:
                action[tau] = remaining_arrivals
            else:
                allocation_action, solution_y, obj_value = self.solve((bookings, remaining_arrivals, future_schedule), t, tau)
                action[tau] = allocation_action
                remaining_arrivals -= allocation_action
        return action

    def allocation_policy(self, state, t):
        if t == self.env.decision_epoch:
            action = state[1]
        else:
            action, solution_y, obj_value = self.solve(state, t, 0)
        return action

if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv
    from environment.utility import get_system_dynamic

    decision_epoch = 5
    bookings = np.array([0])
    future_schedule = np.array([[0, 0] for i in range(decision_epoch)])
    new_arrival = np.array([4, 4])
    state = (bookings, new_arrival, future_schedule)
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': decision_epoch,
        'system_dynamic': get_system_dynamic(3, 4, [0.5, 0.5]),
        'holding_cost': [10, 5],
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 3,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }
    env = AdvanceSchedulingEnv(**env_params)
    state, info = env.reset(state, 1)
    print(state)
    agent = MyopicAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])

    print("final answer:", agent.policy(state, 2))