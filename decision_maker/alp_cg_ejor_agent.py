import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import ColumnGenerationSolver
from experiments import get_config_by_type
from utils import get_solution_value, solve_and_handle_errors, clean_value


class ALPEJORColumnGenerationAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False, verbose=True):
        super().__init__(env, discount_factor, V, Q)
        self.is_trained = False
        # simulate multiple sample path
        # apply myopic policy to estimate the expected value of each component
        self.E_u_alpha = [self.env.regular_capacity * 0.95 ** (i) for i in range(self.env.planning_horizon)]
        self.E_u_alpha[-1] = 0
        self.E_v_alpha = [self.env.overtime_capacity * 0.1 ** (i+1) for i in range(self.env.planning_horizon)]
        self.E_v_alpha[-1] = 0
        self.E_w_alpha = self.env.arrival_generator.mean_by_type
        if coefficients is not None:
            final_duals = coefficients
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        if pretrain:
            self.train(debug=False,verbose=verbose)

    def train(self, tol=1e-6, phase1_max_iter=3000, phase2_max_iter=30000, use_barrier=True, debug=False, verbose=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_columns(tol=tol, max_iter=phase1_max_iter, use_barrier=use_barrier, verbose=verbose)
        self.cg_solver = ColumnGenerationSolver(master_builder=lambda: self.master_builder(use_barrier=use_barrier),
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=self.get_obj_coefficient)
        print('Training Coeffecients')
        self.cg_solver.solve(tol=tol, max_iter=phase2_max_iter, verbose=verbose)
        print('Finished Training Coeffecients!')
        final_duals = [clean_value(c.Pi, 1e-8) for c in self.cg_solver.master_model.getConstrs()]
        self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        self.is_trained = True
        return final_duals

    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = float(next(it))
        U = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        V = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        W = np.array([float(next(it)) for _ in range(self.env.num_types)])
        return W_0, U, V, W

    def get_approx_value_fn(self, state, W_0, U, V, W):
        regular_bookings, overtimes, waitlist = state
        return W_0 + np.dot(U, regular_bookings) + np.dot(V, overtimes) + np.dot(W, waitlist)

    def get_candidate(self, state_var, action_var):
        regular_booking_vars, overtime_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        regular_bookings = get_solution_value(regular_booking_vars).astype(float)
        overtime = get_solution_value(overtime_vars).astype(float)
        waitlist = get_solution_value(waitlist_vars).astype(int)
        advance_scheduling_decision = get_solution_value(advance_scheduling_decision_vars).astype(int)
        overtime_decision = get_solution_value(overtime_decision_vars).astype(float)
        return ((regular_bookings, overtime, waitlist), (advance_scheduling_decision, overtime_decision))

    def master_builder(self, use_barrier=False):
        '''
        Total number of constraints: N + N+l-1 + N+l -2 ... + l + N*I
        '''
        master_model = gp.Model("MasterRMP")
        master_model.ModelSense = GRB.MINIMIZE
        master_model.setParam('OutputFlag', 0)
        master_model.setParam('DualReductions', 0)
        if use_barrier:
            master_model.setParam("Method", 2)      # barrier
            master_model.setParam("Crossover", 0)   # no simplex crossover
        else:
            master_model.setParam("Method", 1) 
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-8)
        master_model.setParam("OptimalityTol", 1e-8)
        master_model.addConstr(
            (
                    gp.LinExpr() == 1
            ),
            name="constr_W_0")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_u_alpha[j]
                for j in range(self.env.planning_horizon)
            ),
            name="constr_U")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_v_alpha[j]
                for j in range(self.env.planning_horizon)
            ),
            name="constr_V")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_w_alpha[i]
                for i in range(self.env.num_types)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def pricing_callback(self, duals, max_attempts=20):
        """
        Solves the pricing subproblem by iterating through each time period.

        This improved approach solves N smaller, independent optimization problems,
        one for each period, instead of one large MIP. This is more efficient
        and directly calculates the minimum reduced cost for each period.
        """
        # --- 1. Initial Setup (Parameters and Duals) ---
        W_0, U, V, W = self.get_coefficients(duals)

        pricing_model = gp.Model(f"Pricing_Problem", env=self.grb_env)
        pricing_model.setParam("MultiObjPre", 0)
        pricing_model.setParam("Method", 2)      # barrier
        pricing_model.setParam("Crossover", 0)   # no simplex crossover
        pricing_model.setParam("MIPGap", 1e-9)    # if MILP pricing, but keep very tight
        pricing_model.setParam("Threads", 1)      # stable and reproducible reduced costs
        pricing_model.setParam("Presolve", 2)     # aggressive presolve speeds up pricing
        pricing_model.setParam("Heuristics", 0.5)
        #pricing_model.setParam("MIPFocus", 1)      # feasibility emphasis
        # modest global limits as a safety net (you can tune these)
        #pricing_model.setParam("TimeLimit", 2.0)
        #pricing_model.setParam("NodeLimit", 50000)
        pricing_model.setParam("OutputFlag", 0)

        state_var = self.get_state_var(pricing_model)
        action_var = self.get_action_var(pricing_model, advance_scheduling_type=GRB.INTEGER)
        self.add_action_space_constraints(pricing_model, state_var, action_var)

        next_state_var = self.get_next_state(pricing_model, state_var, action_var, self.env.arrival_generator.mean_by_type)

        # --- Objective ---
        candidate_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        approx_V = self.get_approx_value_fn(state_var, W_0, U, V, W)
        reduced_cost = candidate_cost + self.env.discount_factor * self.get_approx_value_fn(next_state_var, W_0, U, V, W) - approx_V
        pricing_model.setObjective(reduced_cost, GRB.MINIMIZE)
        for attempt in range(max_attempts):
            start = time.time()
            print('dual:',duals)
            if solve_and_handle_errors(pricing_model):
                print('pricing problem causes:', time.time()-start)
                candidate = self.get_candidate(state_var, action_var)
                yield candidate, pricing_model.ObjVal
            (x_vars, y_var) = action_var
            _,(x_vals,_) = candidate
            candidate_vars = self.tuple_of_arrays_to_list(x_vars)
            candidate_vals = self.tuple_of_arrays_to_list(x_vals)
            self.eliminate_one_candidate(pricing_model, candidate_vars, candidate_vals, f'no_good_cut_{attempt}')
            print('eliminate_one:', candidate)

    def tuple_of_arrays_to_list(self, obj):
        result = []
        if isinstance(obj, np.ndarray):
            result.extend(obj.flatten().tolist())
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                result.extend(self.tuple_of_arrays_to_list(item))
        else:
            # in case you have scalars mixed in
            result.append(obj)
        return result

    def eliminate_one_candidate(self, model, vars, vals, name):
        delta_list = []
        for i, (var, val) in enumerate(zip(vars, vals)):
            delta_le = model.addVar(vtype=GRB.BINARY, name=f"delta_le_{i}")
            delta_ge = model.addVar(vtype=GRB.BINARY, name=f"delta_ge_{i}")
            model.addGenConstrIndicator(delta_le, True, var <= val - 1)
            model.addGenConstrIndicator(delta_ge, True, var >= val + 1)
            delta_list.extend([delta_le, delta_ge])
        model.addConstr(gp.quicksum(delta_list) >= 1, name=name)

    def generate_initial_state_action_pairs(self, verbose=False):
        for i in range(self.env.num_types):
            with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
                init_columns_model.setParam('DualReductions', 0)
                init_columns_model.setParam("MultiObjPre", 0)
                state_var = self.get_state_var(init_columns_model)
                action_var = self.get_action_var(init_columns_model, advance_scheduling_type=GRB.INTEGER)
                self.add_action_space_constraints(init_columns_model, state_var, action_var)
                next_state_var = self.get_next_state(init_columns_model, state_var, action_var,
                                                     self.env.arrival_generator.mean_by_type)
                regular_booking_vars, overtime_booking_vars, waitlist_vars = state_var
                advance_scheduling_decision_vars, overtime_decision_vars = action_var
                next_regular_booking_vars, next_overtime_booking_vars, next_waitlist_vars = next_state_var
                bookings_diff_var = regular_booking_vars - self.env.discount_factor * next_regular_booking_vars
                maximum_difference_var = init_columns_model.addVar(lb=-GRB.INFINITY, name='maximum_difference')
                init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                init_columns_model.addConstrs(
                    (
                        maximum_difference_var <= bookings_diff_var[j]
                        for j in range(self.env.planning_horizon)
                    ),
                    name="maximum_difference",
                )
                init_columns_model.addConstrs(
                    (
                        overtime_booking_vars[j] == 0
                        for j in range(self.env.planning_horizon)
                    ),
                    name="zero_overtime_bookings",
                )
                init_columns_model.addConstrs(
                    (
                        overtime_decision_vars[j] == 0
                        for j in range(self.env.planning_horizon)
                    ),
                    name="zero_overtime_decisions",
                )
                init_columns_model.addConstrs(
                    (
                        waitlist_vars[k] == (self.env.arrival_generator.maximum_arrival if k == i else 0)
                        for k in range(self.env.num_types)
                    ),
                    name="waitlist_initialization",
                )
                if not solve_and_handle_errors(init_columns_model, verbose=verbose):
                    raise ValueError("initial set of columns is infeasible.")
                candidate = self.get_candidate(state_var, action_var)
                yield candidate

    def generate_initial_columns(self, tol=1e-6, max_iter=3000, use_barrier=False, debug=False,  verbose=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_state_action_pairs()
        self.cg_solver = ColumnGenerationSolver(master_builder=lambda: self.initial_columns_builder(use_barrier=use_barrier),
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=None)
        print('Generating inital columns')
        initial_columns = self.cg_solver.initial_columns_solve(tol=tol, max_iter=max_iter,verbose=verbose)
        print('Found feasible columns!')
        return initial_columns

    def initial_columns_builder(self, use_barrier=False):
        master_model = gp.Model("InitMasterRMP")
        master_model.setParam('OutputFlag', 0)
        master_model.setParam('DualReductions', 0)
        if use_barrier:
            master_model.setParam("Method", 2)      # barrier
            master_model.setParam("Crossover", 0)   # no simplex crossover
        else:
            master_model.setParam("Method", 1)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-8)
        master_model.setParam("OptimalityTol", 1e-8)
        # Artificial variable for W_0 constraint
        s_W0 = master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='art_W0')
        # Artificial variables for U constraints (planning horizon)
        s_U = master_model.addVars(
            self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=0, name="art_U"
        )
        # Artificial variables for V constraints (planning horizon)
        s_V = master_model.addVars(
            self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=0, name="art_V"
        )
        # Artificial variables for W constraints (types)
        s_W = master_model.addVars(
            self.env.num_types, vtype=GRB.CONTINUOUS, lb=0, name="art_W"
        )
        M  = 1e4
        # Objective: minimize sum of artificials
        master_model.setObjective(
            M * (s_W0 + s_U.sum() + s_V.sum() + s_W.sum()),
            GRB.MINIMIZE
        )
        # Add constraints, each backed up by its own artificial
        master_model.addConstr(
            gp.LinExpr() + s_W0 == 1,
            name="constr_W_0"
        )
        master_model.addConstrs(
            (s_U[j] >= self.E_u_alpha[j] for j in range(self.env.planning_horizon)),
            name="constr_U"
        )
        master_model.addConstrs(
            (s_V[j] >= self.E_v_alpha[j] for j in range(self.env.planning_horizon)),
            name="constr_V"
        )
        master_model.addConstrs(
            (s_W[i] >= self.E_w_alpha[i] for i in range(self.env.num_types)),
            name="constr_W"
        )
        master_model.update()
        return master_model

    def get_constr_coefficients(self, candidate):
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action = candidate
        regular_bookings, overtimes, waitlist = state
        new_regular_bookings, new_overtimes, new_waitlist = self.env.get_next_state(state, action, mu, is_var=False)
        # W_0 coefficient
        W_0 = 1 - gamma
        # Z coefficients
        U = (regular_bookings - gamma * new_regular_bookings).tolist()
        V = (overtimes - gamma * new_overtimes).tolist()
        # W_i coefficients
        W = (waitlist - gamma * new_waitlist).tolist()
        coefficients = [W_0] + U + V + W
        coefficients = [clean_value(c, 1e-12) for c in coefficients]
        return coefficients

    def get_obj_coefficient(self, candidate):
        state, action = candidate
        return self.env.cost_fn(state, action, is_var=False)

    def coeff_C(self, i, n):
        part1 = sum((self.discount_factor ** k) * self.env.holding_cost(k, i) for k in range(n + 1))
        part2 = sum(self.discount_factor * self.env.treatment_pattern[k+1-n, i] * self.U[k] for k in range(n-1,n-1+self.env.num_sessions))
        part3 = self.env.postponing_cost(i) + self.discount_factor * self.W[i]
        return part1 + part2 - part3
    
    def coeff_H(self, m):
        if m == 0:
            return self.env.overtime_cost(m)
        else:
            return self.discount_factor ** m * self.env.overtime_cost(m) + self.discount_factor * (self.V[m-1] - self.U[m-1])

    def generate_all_columns(self):
        for column in self.env.generate_state_action_pairs():
            yield column

    def solve(self, state, t, action=None, verbose=False):
        # ---------- shortcuts ----------
        with (gp.Model("ALP_policy", env=self.grb_env) as policy_model):
            policy_model.setParam("MultiObjPre", 0)
            policy_model.setParam('DualReductions', 0)
            policy_model.setParam("FeasibilityTol", 1e-8)
            policy_model.setParam("OptimalityTol", 1e-8)
            # m.setParam("OutputFlag", 0)
            # m.setParam("LogToConsole", 0)
            # m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            action_var = self.get_action_var(policy_model, advance_scheduling_type=GRB.INTEGER)
            self.add_action_space_constraints(policy_model, state, action_var)
            if action is not None:
                self.set_action(action_var=action_var, action=action)
            x_var, y_var = action_var
            flatten_x_var = x_var.reshape(-1)
            # ---------- 1. objective ----------
            C = [self.coeff_C(i, n) for i in range(self.env.num_types) for n in range(self.env.booking_window_size)]
            H = [self.coeff_H(m) for m in range(self.env.planning_horizon)]
            policy_model.setObjective(np.dot(C, flatten_x_var) + np.dot(H, y_var), GRB.MINIMIZE)

            if not solve_and_handle_errors(policy_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            # ---------- 8. return ----------
            action = self.get_solution(action_var, is_final=True)
            return action, policy_model.ObjVal, {}
    
    def direct_solve(self, state, t, action=None, verbose=False):
        with (gp.Model("ALP_policy", env=self.grb_env) as policy_model):
            policy_model.setParam("MultiObjPre", 0)
            policy_model.setParam('DualReductions', 0)
            policy_model.setParam("FeasibilityTol", 1e-8)
            policy_model.setParam("OptimalityTol", 1e-8)
            # m.setParam("OutputFlag", 0)
            # m.setParam("LogToConsole", 0)
            # m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            action_var = self.get_action_var(policy_model, advance_scheduling_type=GRB.INTEGER)
            self.add_action_space_constraints(policy_model, state, action_var)
            if action is not None:
                self.set_action(action_var=action_var, action=action)
            imm_cost = self.env.cost_fn(state, action_var, is_var=True)
            new_state_var = self.get_next_state(model=policy_model,
                                                state=state,
                                                action=action_var,
                                                new_arrival=self.env.arrival_generator.mean_by_type)
            fut_cost = self.discount_factor * self.get_approx_value_fn(state=new_state_var,
                                                                            W_0=self.W_0,
                                                                            U=self.U,
                                                                            V=self.V,
                                                                            W=self.W)
            policy_model.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            if not solve_and_handle_errors(policy_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            # ---------- 8. return ----------
            action = self.get_solution(action_var, is_final=True)
            return action, policy_model.ObjVal, {}


if "__main__" == __name__:
    config = get_config_by_type('ejor')
    env = config.env
    init_state = config.init_state
    coefficients = [-195284.271323261, 123.537317331, 122.301944158, 121.078924717, 119.868135469, 118.669454115, 117.482759574, 116.307931978, 115.144852658, 113.993404131, 112.85347009, 111.724935389, 110.607686035, 109.501609175, 108.406593083, 107.322527152, 106.249301881, 105.186808862, 104.134940773, 103.093591366, 102.062655452, 101.042028898, 100.031608609, 99.031292522, 98.040979597, 97.060569801, 96.089964103, 95.129064462, 94.177773818, 93.235996079, 92.303636119, 91.380599757, 90.46679376, 89.562125822, 88.666504564, 87.779839518, 86.902041123, 86.033020712, 85.172690505, 84.3209636, 83.477753964, 82.642976424, 81.81654666, 80.998381193, 80.188397381, 79.386513408, 78.592648274, 77.806721791, 77.028654573, 76.258368027, 75.495784347, 74.740826503, 73.993418238, 73.253484056, 72.520949215, 71.795739723, 71.077782326, 70.367004503, 69.663334458, 68.966701113, 68.277034102, 67.594263761, 66.918321123, 66.249137912, 65.586646533, 64.930780068, 64.281472267, 63.638657544, 63.002270969, 62.372248259, 61.748525777, 61.131040519, 60.519730114, 59.914532813, 59.315387484, 58.72223361, 58.135011273, 57.553661161, 56.978124549, 56.408343304, 55.844259871, 55.285817272, 54.732959099, 54.185629508, 53.643773213, 53.107335481, 0.0, 23.537317331, 23.301944158, 23.068924717, 22.838235469, 22.609853115, 22.383754584, 22.159917038, 21.938317867, 21.718934689, 21.501745342, 21.286727888, 21.073860609, 20.863122003, 20.654490783, 20.447945875, 20.243466417, 20.041031753, 19.840621435, 19.642215221, 19.445793068, 19.251335138, 19.058821786, 18.868233569, 18.679551233, 18.492755721, 18.307828163, 18.124749882, 17.943502383, 17.764067359, 17.586426685, 17.410562419, 17.236456794, 17.064092226, 16.893451304, 16.724516791, 16.557271623, 16.391698907, 16.227781918, 16.065504099, 15.904849058, 15.745800567, 15.588342562, 15.432459136, 15.278134545, 15.125353199, 14.974099667, 14.82435867, 14.676115084, 14.529353933, 14.384060393, 14.24021979, 14.097817592, 13.956839416, 13.817271021, 13.679098311, 13.542307328, 13.406884255, 13.272815412, 13.140087258, 2e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 728.993093121, 247.074634663, 610.323639007, 1789.204068776, 2263.926831342, 3465.554667463, 237.338908229, 700.26785657, 1253.362150343, 586.274452439, 1779.349320377, 118.669454115, 2279.557885455, 7157.009484284, 3298.362359767, 3617.528300023, 3699.344846683, 3298.362359767]
    coefficients = [clean_value(coefficient, 1e-8) for coefficient in coefficients]
    agent = ALPEJORColumnGenerationAgent(env=env, discount_factor=0.99, coefficients=coefficients, pretrain=False)
    state = (
       np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.]), 
       np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.]), 
       np.array([0, 1, 0, 0, 3, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    action = (np.array([[0, 1, 0, 0, 3, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.]))
    
    action, obj, info = agent.direct_solve(state=state, t=1)
    x, y = action
    '''
    action, obj, info = agent.solve(state=state, t=1)
    x, y = action
    print('simplifyed:')
    print(action)
    print(obj)
    '''
    
    #print(list(agent.generate_initial_state_action_pairs()))
    #agent.train(verbose=False)
    #print(agent.coeff_C(0,1))

    # master obj: 25250

