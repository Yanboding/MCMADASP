import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from utils import iter_to_tuple


class HindsightApproximationAgent:

    def __init__(self, env, discount_factor, num_sample_paths, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.num_sample_paths = num_sample_paths
        self.V = V
        self.Q = Q
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}

    def solve(self, state, t):
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
        bookings, new_arrival, b = state
        delta = self.env.new_arrivals
        r = self.env.treatment_pattern
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        # Create the model
        model = gp.Model("Scheduling_MIP")

        # Add decision variables
        # a[tau, t, i] for t in range(N) (periods 1 to N), tau in range(N-t), i in range(I)
        a = {}
        for tau in range(t, N + 1):
            for k in range(N - tau + 1):
                for i_type in range(1, I + 1):
                    a[tau, k, i_type] = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=0,
                        name=f"a[{tau},{k},{i_type}]"
                    )
        # y[t] for t in range(N)
        y = {}
        for tau in range(t, N + 1):
            y[tau] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                name=f"y[{tau}]"
            )

        # Set objective function
        # Minimize sum_{t=1}^N gamma^(t-1) [ sum_{i=1}^I w_i ( sum_{j=0}^{N-t} b^1_{t+j-1, i} + sum_{k=0}^{t-1} a^{t-k}_{k, i} ) + O y^t ]
        # In code, t ranges from 0 to N-1, so gamma^(t-1) becomes gamma^t
        obj = gp.quicksum(
            gamma ** (tau - 1) * (
                    gp.quicksum(
                        w[i_type-1] * (
                                gp.quicksum(b[tau - t + j, i_type-1] for j in range(N - tau + 1))
                                +
                                gp.quicksum(
                                    a[tau - k, k + j, i_type]
                                    for k in range(tau - t + 1)
                                    for j in range(N - tau + 1)
                                    # If needed, guard with: if (k+j) <= N - (tau - k)
                                )
                        )
                        for i_type in range(1, I + 1)
                    )
                    + O * y[tau]
            )
            for tau in range(t, N + 1)
        )

        model.setObjective(obj, GRB.MINIMIZE)

        # Add constraints
        # (cons1) Ensure arrivals match scheduled "starts":
        #    sum_{k=0}^{N - tau} a[tau, k, i] = delta[tau, i]   for tau=t..N, i=1..I
        for tau in range(t, N + 1):
            print('delta:', delta[tau - 1])
            for i_type in range(1, I + 1):

                model.addConstr(
                    gp.quicksum(a[tau, k, i_type] for k in range(N - tau + 1)) == delta[tau-1, i_type-1],
                    name=f"ArrivalBalance[tau={tau},i={i_type}]"
                )

        # (cons2) Overtime capacity constraints:
        #    y[tau] >= sum_i ( b[tau-t,i] + sum_{k=0}^{tau-t} a[tau-k, k,i] ) * r_i - C
        #    for tau=t..N
        for tau in range(t, N + 1):

            workload = gp.quicksum(
                (
                        b[tau - t, i_type-1] +
                        gp.quicksum(a[tau - k, k, i_type] for k in range(tau - t + 1))
                ) * r[0, i_type-1]
                for i_type in range(1, I + 1)
            )
            model.addConstr(
                y[tau] >= workload - C,
                name=f"Overtime[tau={tau}]"
            )
        model.optimize()

        # Check if optimal solution is found
        if model.status == GRB.OPTIMAL:
            solution_a = {(tau, k, i): a[tau, k, i].x for tau, k, i in a.keys()}
            solution_y = [y[tau].x for tau in range(t, N+1)]
            obj_value = model.objVal
            return solution_a, solution_y, obj_value
        else:
            return None, None, None

    def policy(self, state, t):
        '''
        Sample paths starting from
        :param state:
        :param t:
        :return:
        '''
        state_tuple = iter_to_tuple(state)
        for i in range(self.num_sample_paths):
            self.env.reset_arrivals()
            for action in self.env.valid_actions(state, t):
                action_tuple = iter_to_tuple(action)
                one_time_cost = self.env.cost_fn(state, action, t)
                next_state = self.env.post_advance_scheduling_state(state, action)
                solution_a, _, obj_value = self.solve(next_state, t+1)
                hindsight_Q = one_time_cost + obj_value
                self.Q[state_tuple, t][action_tuple] += (hindsight_Q - self.Q[state_tuple, t][action_tuple])/(i+1)
        best_action = self.get_best_action(state, t)
        return best_action

    def get_best_action(self, state, t):
        # All actions that available in the given state
        state_tuple = iter_to_tuple(state)
        minValue = float('inf')
        bestAction = None
        for action, qValue in self.Q[(state_tuple, t)].items():
            if qValue < minValue or (qValue == minValue and action > bestAction):
                minValue = qValue
                bestAction = action
        return np.array(bestAction)


if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv
    from environment.utility import get_system_dynamic
    bookings = np.array([0])
    future_schedule = np.array([[0, 0], [1, 1]])
    new_arrival = np.array([2, 1])
    state = (bookings, new_arrival, future_schedule)
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': 7,
        'system_dynamic': get_system_dynamic(3, 4, [0.5, 0.5]),
        'holding_cost': [10, 5],
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 2,
        'discount_factor': 0.99
    }
    env = AdvanceSchedulingEnv(**env_params)
    state, info = env.reset(state, 1)
    print(state)
    agent = HindsightApproximationAgent(env, discount_factor=env_params['discount_factor'])

    print("final answer:", agent.policy(state, 6))