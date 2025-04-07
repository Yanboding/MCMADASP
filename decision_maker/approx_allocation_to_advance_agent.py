import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from utils import iter_to_tuple
class ApproxAllocationAdvanceAgent:

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

    def solve(self, state, t, x, num_sample_paths):
        '''
        we need a probability of the sample path
        :param state:
        :param t:
        :param tau:
        :param num_sample_paths:
        :return:
        '''
        N = self.env.decision_epoch
        I = self.env.num_types
        gamma = self.discount_factor
        w = self.env.holding_cost
        bookings, w_bar, b = state
        r = self.env.treatment_pattern
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        M = num_sample_paths
        delta = np.zeros((M, N-t-x, I))
        P = np.zeros((M,))
        for omega in range(num_sample_paths):
            #new_arrivals, sample_path_probability = self.env.reset_arrivals(t+x+1)
            new_arrivals = self.env.reset_arrivals(t + x + 1)
            delta[omega] = new_arrivals
            #P[omega] = sample_path_probability

        model = gp.Model('StochasticAllocation')
        # Suppress Gurobi output completely
        #model.setParam('OutputFlag', 0)
        #model.setParam('LogToConsole', 0)
        # Decision Variables
        q_t = model.addVars(I, vtype=GRB.INTEGER, name="q_t")
        y_t = model.addVar(lb=0, name="y_t")

        q_future = model.addVars(M, N - t - x, I, vtype=GRB.INTEGER, name="q_future")
        y_future = model.addVars(M, N - t - x, lb=0, name="y_future")

        # Objective Function
        immediate_cost = gp.quicksum(
            w[i] * (w_bar[i] + sum(b[j, i] for j in range(x, N - t + 1))) for i in range(I)) + O * y_t
        # delta = [t+1, ..., N]
        # t+x
        future_cost = gp.quicksum(
             gp.quicksum(
                gamma ** tau * (
                        gp.quicksum(
                            w[i] * (
                                    w_bar[i] - q_t[i] + delta[omega, 0, i]
                                    + gp.quicksum(b[x + tau + j, i] for j in range(N - t - x - tau + 1))
                                    + gp.quicksum(delta[omega, k-1, i] - q_future[omega, k - 1, i] for k in range(2, tau + 1))
                            ) for i in range(I)
                        ) + O * y_future[omega, tau-1]
                ) for tau in range(1, N - t - x + 1)
            ) for omega in range(M)
        )/M

        model.setObjective(immediate_cost + future_cost, GRB.MINIMIZE)
        # Constraints
        # Constraint 1
        model.addConstrs(q_t[i] <= w_bar[i] for i in range(I))
        # Constraint 2
        model.addConstr(gp.quicksum((q_t[i] + b[x, i]) * r[0, i] for i in range(I)) - y_t <= C)

        # Constraint 3 implicitly handled by variable definition (y_t >=0)

        # Constraint 4
        for omega in range(M):
            for tau in range(1, N - t - x + 1):
                model.addConstrs(
                    q_t[i] + gp.quicksum(q_future[omega, k - 1, i] for k in range(1, tau + 1))
                    <= w_bar[i] + gp.quicksum(delta[omega, j-1, i] for j in range(1, tau + 1))
                    for i in range(I)
                )

        # Constraint 5
        for omega in range(M):
            for tau in range(1, N - t - x + 1):
                model.addConstr(
                    gp.quicksum((q_future[omega, tau - 1, i] + b[x + tau, i]) * r[0, i] for i in range(I)) - y_future[
                        omega, tau - 1] <= C
                )

        # Constraint 6 implicitly handled by variable definition (y_future >=0)

        # Optimize model
        model.optimize()

        # Retrieve solution (example)
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
            print("No optimal solution found")
        return

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
            if t + tau == self.env.decision_epoch:
                action[tau] = remaining_arrivals
            else:
                allocation_action, solution_y, obj_value = self.solve((bookings, remaining_arrivals, future_schedule), t, tau, 3000)
                action[tau] = allocation_action
                remaining_arrivals -= allocation_action
        return action

    def allocation_policy(self, state, t):
        if t == self.env.decision_epoch:
            action = state[1]
        else:
            action, solution_y, obj_value = self.solve(state, t, 0, 3000)
        return action

if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv
    from environment.utility import get_system_dynamic

    decision_epoch = 4
    bookings = np.array([0])
    future_schedule = np.array([[0, 0] for i in range(decision_epoch)])
    new_arrival = np.array([5, 5])
    state = (bookings, new_arrival, future_schedule)
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': decision_epoch,
        'arrival_generator': None,
        'system_dynamic': get_system_dynamic(3, 4, [0.5, 0.5]),
        'holding_cost': [10, 5],
        'overtime_cost': 40,
        'duration': 1,
        'regular_capacity': 3,
        'discount_factor': 0.99
    }
    env = AdvanceSchedulingEnv(**env_params)
    agent = ApproxAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
    #agent.solve(state, 1,1, 1)
    action = agent.allocation_policy(state, 1)
    print(action)
    '''
    state, info = env.reset(state, 1)
    print(state)
    agent = ApproxAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])

    print("final answer:", agent.policy(state, 2))
    '''