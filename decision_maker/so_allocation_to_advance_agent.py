import copy

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from utils import iter_to_tuple
class SAAllocationAdvanceAgent:

    def __init__(self, env, discount_factor, sample_path_number=3000, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        self.sample_path_number = sample_path_number
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
        if self.env.problem_type == 'allocation':
            self.policy = self.allocation_policy
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}

    def solve(self, state, t, x, action=None):
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
        model = gp.Model('StochasticAllocation')
        # Suppress Gurobi output completely
        model.setParam('OutputFlag', 0)
        model.setParam('LogToConsole', 0)
        # Decision Variables
        q_t = model.addVars(I, vtype=GRB.INTEGER, name="q_t")
        y_t = model.addVar(lb=0, name="y_t")
        if action != None:
            for i in range(I):
                q_t[i].lb = action[i]
                q_t[i].ub = action[i]

        q_future = model.addVars(self.sample_path_number, range(1, N - t - x + 1), I, vtype=GRB.INTEGER, name="q_future")
        y_future = model.addVars(self.sample_path_number, range(1, N - t - x + 1), lb=0, name="y_future")

        # Objective Function
        # u^{t+x}_{i} = w_bar[i] + sum(b[j, i] for j in range(x, N - t + 1))
        immediate_cost = gp.quicksum(
            w[i] * (w_bar[i] + sum(b[j, i] for j in range(x, N - t + 1))) for i in range(I)) + O * y_t
        # delta = [t+1, ..., N]
        # current_time = t+x
        future_cost = gp.quicksum(gp.quicksum(gamma ** tau * (gp.quicksum(
                            w[i] * (
                                    w_bar[i] - q_t[i]
                                    + gp.quicksum(self.delta[omega, k, i] for k in range(x + 1, x + tau + 1))
                                    - gp.quicksum(q_future[omega, k, i] for k in range(1, tau))
                                    + gp.quicksum(b[x + tau + j, i] for j in range(N - t - x - tau + 1))
                            ) for i in range(I)
                        ) + O * y_future[omega, tau]) for tau in range(1, N - t - x + 1)
            ) for omega in range(self.sample_path_number)
        )/self.sample_path_number
        model.setObjective(immediate_cost + future_cost, GRB.MINIMIZE)
        # Constraints
        # Constraint 1
        model.addConstrs(q_t[i] <= w_bar[i] for i in range(I))
        # Constraint 2
        model.addConstr(gp.quicksum((q_t[i] + b[x, i]) * r[0, i] for i in range(I)) - y_t <= C)

        # Constraint 3 implicitly handled by variable definition (y_t >=0)

        # Constraint: the number of allocation is no more than the number of outstanding treatments in each period
        for tau in range(1, N - t - x):
            for omega in range(self.sample_path_number):
                model.addConstrs(
                    q_t[i] + gp.quicksum(q_future[omega, k, i] for k in range(1, tau + 1))
                    <= w_bar[i] + gp.quicksum(self.delta[omega, j, i] for j in range(1 + x, x + tau + 1))
                    for i in range(I)
                )
        # Constraint: the number of allocation equals to the number of outstanding treatments in the last period
        if N - t - x == 0:
            model.addConstrs(q_t[i] == w_bar[i] for i in range(I))
        else:
            for omega in range(self.sample_path_number):
                model.addConstrs(
                    q_t[i] + gp.quicksum(q_future[omega, k, i] for k in range(1, N - t - x + 1))
                    == w_bar[i] + gp.quicksum(self.delta[omega, j, i] for j in range(1 + x, N - t + 1))
                    for i in range(I)
                )

        # Constraint: overtime constraint at period t+tau in scenario omega
        for tau in range(1, N - t - x + 1):
            for omega in range(self.sample_path_number):
                model.addConstr(
                    gp.quicksum((q_future[omega, tau, i] + b[x + tau, i]) * r[0, i] for i in range(I)) - y_future[
                        omega, tau] <= C
                )

        # Constraint 6 implicitly handled by variable definition (y_future >=0)

        # Optimize model
        model.optimize()
        #print(immediate_cost.getValue())
        #print(future_cost.getValue())
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

    def exact_solve(self, state, t, x, action=None):
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
        P, delta = self.env.arrival_generator.get_sample_paths_with_prob(N - t - x)
        sample_path_number = len(delta)

        model = gp.Model('StochasticAllocation')
        # Suppress Gurobi output completely
        model.setParam('OutputFlag', 0)
        model.setParam('LogToConsole', 0)
        # Decision Variables
        q_t = model.addVars(I, vtype=GRB.INTEGER, name="q_t")
        y_t = model.addVar(lb=0, name="y_t")
        if action != None:
            for i in range(I):
                q_t[i].lb = action[i]
                q_t[i].ub = action[i]

        q_future = model.addVars(sample_path_number, range(1, N - t - x + 1), I, vtype=GRB.INTEGER, name="q_future")
        y_future = model.addVars(sample_path_number, range(1, N - t - x + 1), lb=0, name="y_future")

        # Objective Function
        # u^{t+x}_{i} = w_bar[i] + sum(b[j, i] for j in range(x, N - t + 1))
        immediate_cost = gp.quicksum(
            w[i] * (w_bar[i] + sum(b[j, i] for j in range(x, N - t + 1))) for i in range(I)) + O * y_t
        # delta = [t+1, ..., N]
        # current_time = t+x
        future_cost = gp.quicksum(P[omega]*gp.quicksum(gamma ** tau * (gp.quicksum(
                            w[i] * (
                                    w_bar[i] - q_t[i]
                                    + gp.quicksum(delta[omega, k, i] for k in range(x+1, x+tau+1))
                                    - gp.quicksum(q_future[omega, k, i] for k in range(1, tau))
                                    + gp.quicksum(b[x + tau + j, i] for j in range(N - t - x - tau + 1))
                            ) for i in range(I)
                        ) + O * y_future[omega, tau]) for tau in range(1, N - t - x + 1)
            ) for omega in range(sample_path_number)
        )
        model.setObjective(immediate_cost + future_cost, GRB.MINIMIZE)
        # Constraints
        # Constraint 1
        model.addConstrs(q_t[i] <= w_bar[i] for i in range(I))
        # Constraint 2
        model.addConstr(gp.quicksum((q_t[i] + b[x, i]) * r[0, i] for i in range(I)) - y_t <= C)

        # Constraint 3 implicitly handled by variable definition (y_t >=0)

        # Constraint: the number of allocation is no more than the number of outstanding treatments in each period
        for tau in range(1, N - t - x):
            for omega in range(sample_path_number):
                model.addConstrs(
                    q_t[i] + gp.quicksum(q_future[omega, k, i] for k in range(1, tau + 1))
                    <= w_bar[i] + gp.quicksum(delta[omega, j, i] for j in range(1 + x, x + tau + 1))
                    for i in range(I)
                )
        # Constraint: the number of allocation equals to the number of outstanding treatments in the last period
        if N - t - x == 0:
            model.addConstrs(q_t[i] == w_bar[i] for i in range(I))
        else:
            for omega in range(sample_path_number):
                model.addConstrs(
                    q_t[i] + gp.quicksum(q_future[omega, k, i] for k in range(1, N - t - x + 1))
                    == w_bar[i] + gp.quicksum(delta[omega, j, i] for j in range(1 + x, N - t + 1))
                    for i in range(I)
                )

        # Constraint 5
        for tau in range(1, N - t - x + 1):
            for omega in range(sample_path_number):
                model.addConstr(
                    gp.quicksum((q_future[omega, tau, i] + b[x + tau, i]) * r[0, i] for i in range(I)) - y_future[
                        omega, tau] <= C
                )

        # Constraint 6 implicitly handled by variable definition (y_future >=0)

        # Optimize model
        model.optimize()
        #print(immediate_cost.getValue())
        #print(future_cost.getValue())
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
        bookings, remaining_arrivals, future_schedule = copy.deepcopy(state)
        action = np.zeros_like(future_schedule)
        for tau in range(len(future_schedule)):
            allocation_action, solution_y, obj_value = self.solve((bookings, remaining_arrivals, future_schedule), t, tau)
            action[tau] = allocation_action
            remaining_arrivals -= allocation_action
        return action

    def allocation_policy(self, state, t):
        action, solution_y, obj_value = self.solve(state, t, 0)
        return action

if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv, MultiClassPoissonArrivalGenerator
    from environment.utility import get_system_dynamic

    decision_epoch = 3
    class_number = 2
    bookings = np.array([0])
    future_schedule = np.array([[0]*class_number for i in range(decision_epoch)])
    new_arrival = np.array([5, 6])
    state = (bookings, new_arrival, future_schedule)
    arrival_generator = MultiClassPoissonArrivalGenerator(3, 2, [1 / class_number] * class_number)
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': decision_epoch,
        'arrival_generator': arrival_generator,
        'holding_cost': [10, 5],
        'overtime_cost': 40,
        'duration': 1,
        'regular_capacity': 5,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }
    env = AdvanceSchedulingEnv(**env_params)

    agent = SAAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'], sample_path_number=6000)

    #agent.solve(state, 1,1, 1)
    solution_a, y_t_solution, approx_obj_value = agent.solve(state, 1, 0)
    exact_solution_a, exact_y_t_solution, exact_obj_value = agent.exact_solve(state, 1, 0)
    print(approx_obj_value, exact_obj_value)
    print(solution_a, exact_solution_a)
    '''
    state, info = env.reset(state, 1)
    print(state)
    agent = ApproxAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])

    print("final answer:", agent.policy(state, 2))
    '''