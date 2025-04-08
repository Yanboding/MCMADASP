from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SAAdvanceAgent:

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
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}

    def solve(self, state, t, x=None):
        N = self.env.decision_epoch
        I = self.env.num_types
        gamma = self.discount_factor
        w = self.env.holding_cost
        bookings, delta_t, b = state
        r = self.env.treatment_pattern
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        model = gp.Model("SA_Advance")
        model.setParam('OutputFlag', 0)
        model.setParam('LogToConsole', 0)

        # ----------------------
        # 1. Define the sets
        # ----------------------
        # i in {0,...,I-1}  -- for convenience we'll shift Python indexing to 0..I-1
        # j in {0,...,N-t}
        # tau in {1,...,N-t}
        # omega in {0,...,M-1} in Python indexing
        # (Adjust as needed if your data is 1-indexed.)

        # ----------------------
        # 2. Add decision variables
        # ----------------------

        # a^t_{j,i}: # of new treatments allocated in period t + j (for each i)
        #   j = 0..(N - t), i = 0..(I-1)
        #   must be nonnegative integers
        a_t = model.addVars(N-t+1, I, vtype=GRB.INTEGER, name="a_t")

        # a^{t+tau}_{j,i,ω}: # of new treatments allocated in period (t + tau + j) in scenario ω
        #   tau = 1..(N - t)
        #   j = 0..(N - t - tau)
        #   i = 0..(I-1)
        #   ω = 0..(M - 1)
        a_future = {}
        for tau in range(1, N - t + 1):
            a_t_tau_omega = model.addVars(self.sample_path_number, N - t - tau + 1, I, vtype=GRB.INTEGER, name="a_t_j_tau_omega")
            a_future[tau] = a_t_tau_omega

        # Overtime in period t
        y_t = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_t")

        # Overtime in period (t + tau) under scenario ω
        y_future = model.addVars(self.sample_path_number, range(1, N - t + 1), lb=0, vtype=GRB.CONTINUOUS, name="y_future")
        b_bar_t = model.addVars(N-t+1, I, vtype=GRB.INTEGER, name="b_bar_t")
        for j in range(N-t+1):
            for i in range(I):
                b_bar_t[j, i] = b[j, i] + a_t[j, i]
        # New schedule in period (t+\tau) under scenario ω
        b_bar_future = {}
        for tau in range(1, N - t + 1):
            b_bar_t_tau_omega = model.addVars(self.sample_path_number, N - t - tau + 1, I, vtype=GRB.INTEGER,
                                              name="b_bar_t_j_tau_omega")
            b_bar_future[tau] = b_bar_t_tau_omega
            for omega in range(self.sample_path_number):
                for j in range(N - t - tau + 1):
                    for i in range(I):
                        b_bar_future[tau][omega, j, i] = b[tau + j, i] + a_t[tau + j, i] + gp.quicksum(
                            a_future[k][omega, tau + j - k, i] for k in range(1, tau + 1))

        # immediate_cost: sum_{i,j} w_i [ b^t_{j,i} + a^t_{j,i} ]
        immediate_cost = gp.quicksum(w[i] * gp.quicksum(b_bar_t[j, i] for j in range(N-t+1))
                                     for i in range(I)) + O * y_t

        # future_cost: (1/M)* sum_{omega} sum_{tau} gamma^tau [
        #                sum_{i,j} w_i * ( b^t_{\tau+j,i} + a^t_{\tau+j,i}
        #                    + sum_{k=1}^tau a^{t+k}_{\tau+j-k, i, ω} )
        #                + O * y^{t+tau}_ω
        #              ]
        # \bar{b}_{j,i, \omega}^{t+\tau} = gp.quicksum(b[tau+j, i] + a_t[tau+j, i]
        #             + gp.quicksum(a_future[k][omega, tau+j-k, i] for k in range(1, tau+1))
        #             for j in range(N-t- tau + 1))
        future_cost = gp.quicksum(gp.quicksum(gamma ** tau * (gp.quicksum(w[i] * gp.quicksum(b_bar_future[tau][omega, j, i]
                                  for j in range(N-t- tau + 1))
                                  for i in range(I)) + O * y_future[omega, tau])
                                  for tau in range(1, N-t+1))
                                  for omega in range(self.sample_path_number))/self.sample_path_number

        model.setObjective(immediate_cost + future_cost, GRB.MINIMIZE)

        # ----------------------
        # 3. Define the constraints
        # ----------------------

        #
        # (cons3)  sum_{j=0..N-t} a^t_{j,i} = delta_i^t
        #
        # for i in 1..I
        #
        for i in range(I):
            model.addConstr(
                gp.quicksum(a_t[j, i] for j in range(N - t + 1)) == delta_t[i],
                name=f"cons3_delta_t_{i}"
            )

        #
        # (cons4)  sum_{j=0..N-t-τ} a^{t+τ}_{j,i,ω} = delta_i^{t+τ}
        #
        # for τ = 1..(N-t), i=1..I, ω=1..M
        # --> Each scenario must meet the same demand for category i in period t+τ
        #     (You need to have delta_i^{t+τ} for each τ, presumably the same across scenarios;
        #      if each scenario has a *different* demand, adapt accordingly.)
        #
        # Suppose you have a data structure delta_t_tau[tau][i] that gives delta_i^{t+τ}.
        #
        for tau in range(1, N - t + 1):
            for i in range(I):
                for omega in range(self.sample_path_number):
                    model.addConstr(gp.quicksum(a_future[tau][omega, j, i] for j in range(N - t - tau + 1)) == self.delta[omega, tau, i], name=f"cons4_delta_{tau}_{i}_om{omega}")

        #
        # (cons5) and (cons6):
        #   y^t >= sum_{i=1}^I (b^t_{0,i} + a^t_{0,i}) * r_i - C
        #   and y^t >= 0 (already handled by lb=0)
        #
        # If b_t[j,i] is given by b_t[j][i], then for j=0:
        model.addConstr(
            y_t >= gp.quicksum(b_bar_t[0, i] * r[0, i] for i in range(I)) - C,
            name="cons5"
        )
        #
        # (cons7) and (cons8):
        #   for τ=1..N−t, ω=1..M:
        #     y^{t+τ}_ω >= sum_{i=1}^I [ b^t_{\tau+0,i} + a^t_{\tau+0,i} + sum_{k=1}^τ a^{t+k}_{\tau+0−k,i,ω} ] * r_i  - C
        #   and y^{t+τ}_ω >= 0 (again lb=0 handles the non-negativity).
        #
        # Here we need the “0th” job in period t+τ, that is j=0 in the expression \bar{b}^{t+τ}_{0,i,ω}.
        # Notice that j=0 => \bar{b}^{t+τ}_{0,i,ω} = b^t_{\tau,i} + a^t_{\tau,i} + sum_{k=1}^τ a^{t+k}_{\tau−k,i,ω}.
        #
        for tau in range(1, N - t + 1):
            for omega in range(self.sample_path_number):
                model.addConstr(
                    y_future[omega, tau] >= gp.quicksum(b_bar_future[tau][omega, 0, i] * r[0, i] for i in range(I)) - C,
                    name=f"cons7_tau{tau}_om{omega}")

        model.optimize()
        # Retrieve solution (example)
        if model.status == GRB.OPTIMAL:
            a_t_solution = model.getAttr('X', a_t)
            y_t_solution = y_t.X
            solution_a = np.zeros((N-t+1, I))
            for (t, i), value in a_t_solution.items():
                solution_a[t, i] = value
            solution_a = solution_a.astype(int)
            obj_value = model.objVal
            return solution_a, y_t_solution, obj_value
        else:
            print("No optimal solution found")
        return

    def policy(self, state, t):
        action, overtime, obj_value = self.solve(state, t)
        return action

if __name__ =="__main__":
    from environment import AdvanceSchedulingEnv, MultiClassPoissonArrivalGenerator
    from environment.utility import get_system_dynamic

    decision_epoch = 20
    class_number = 2
    bookings = np.array([0])
    future_schedule = np.array([[0]*class_number for i in range(decision_epoch)])
    new_arrival = np.array([5, 6])
    state = (bookings, new_arrival, future_schedule)
    arrival_generator = MultiClassPoissonArrivalGenerator(3, 1, [1 / class_number] * class_number)
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': decision_epoch,
        'arrival_generator': arrival_generator,
        'holding_cost': [10, 5],
        'overtime_cost': 40,
        'duration': 1,
        'regular_capacity': 5,
        'discount_factor': 0.99
    }
    # opt: 200.9459842557252
    env = AdvanceSchedulingEnv(**env_params)
    agent = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'], sample_path_number=3000)
    print(state)
    solution_a, y_t_solution, obj_value = agent.solve(state, 1, 0)
    print(solution_a)
    print(obj_value)
    # [[5, 0], [0, 5], [0, 1]]
    # Total cost: 292.20212499999997
    # Immediate cost: 160
    # Future cost: 132.20212499999997

    # Total cost: 294.375
    # Immediate cost: 160
    # Future cost: 134.375

    # 290.4041666665768
