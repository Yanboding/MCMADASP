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
        decay_factor = 0.95
        required_bookings = [(self.env.regular_capacity + self.env.overtime_capacity) * decay_factor**(j+1) for j in range(self.env.planning_horizon)]
        required_bookings[-1] = 0
        required_bookings = np.array(required_bookings)
        self.E_u_alpha = np.minimum(required_bookings, self.env.regular_capacity)
        self.E_v_alpha = np.minimum(np.maximum(required_bookings - self.env.regular_capacity, 0), self.env.regular_capacity)
        self.E_w_alpha = self.env.arrival_generator.mean_by_type
        if coefficients is not None:
            final_duals = coefficients
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        if pretrain:
            self.train(debug=False,verbose=verbose, use_barrier=True)

    def train(self, tol=1e-6, phase1_max_iter=3000, phase2_max_iter=30000, use_barrier=True, debug=False, verbose=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_columns(tol=tol, max_iter=phase1_max_iter, use_barrier=use_barrier, verbose=verbose)
        self.cg_solver = ColumnGenerationSolver(master_builder=lambda: self.master_builder(use_barrier=use_barrier),
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=self.get_obj_coefficient,
                                                dual_regularization_penalty=0.1)
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
        regular_bookings_cost = gp.quicksum(U[i] * regular_bookings[i] for i in range(len(U)))
        overtime_cost = gp.quicksum(V[j] * overtimes[j] for j in range(len(V)))
        waitlist_cost = gp.quicksum(W[k] * waitlist[k] for k in range(len(W)))
        return W_0 + regular_bookings_cost + overtime_cost + waitlist_cost
    
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
        #pricing_model.setParam("Method", 2)      # barrier
        #pricing_model.setParam("Crossover", 0)   # no simplex crossover
        pricing_model.setParam("MIPGap", 1e-9)    # if MILP pricing, but keep very tight
        pricing_model.setParam("Threads", 1)      # stable and reproducible reduced costs
        #pricing_model.setParam("Presolve", 1)     # aggressive presolve speeds up pricing
        #pricing_model.setParam("Heuristics", 0.2)
        #pricing_model.setParam("MIPFocus", 1)      # feasibility emphasis
        # modest global limits as a safety net (you can tune these)
        #pricing_model.setParam("TimeLimit", 2.0)
        #pricing_model.setParam("NodeLimit", 50000)
        pricing_model.setParam("OutputFlag", 0)
        pricing_model.setParam('DualReductions', 0)
        #pricing_model.setParam("FeasibilityTol", 1e-9)
        #pricing_model.setParam("OptimalityTol", 1e-9)

        state_var = self.get_state_var(pricing_model)
        action_var = self.get_action_var(pricing_model, advance_scheduling_type=GRB.INTEGER)
        self.add_action_space_constraints(pricing_model, state_var, action_var, is_pricing=True)

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
                init_columns_model.setParam("FeasibilityTol", 1e-9)
                init_columns_model.setParam("OptimalityTol", 1e-9)
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
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
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
        #coefficients = [c for c in coefficients]
        return coefficients

    def get_obj_coefficient(self, candidate):
        state, action = candidate
        return self.env.cost_fn(state, action, is_var=False)

    def coeff_C(self, i, n):
        part1 = sum((self.discount_factor ** k) * self.env.holding_cost(k, i) for k in range(n)) - self.env.postponing_cost(i)
        part2 = self.discount_factor * sum(self.env.treatment_pattern[k+1-n, i] * self.U[k] for k in range(max(n-1,0),n+self.env.num_sessions-1))
        part3 =  self.discount_factor * self.W[i]
        cin = part1 + part2 - part3
        return clean_value(cin, 1e-8)
    
    def coeff_H(self, m):
        hm = self.discount_factor ** m * self.env.overtime_cost(m)
        if m > 0:
            hm += self.discount_factor * (self.V[m-1] - self.U[m-1])
        return clean_value(hm, 1e-8)
    
    def myopic_coeff_C(self, i, n):
        cin = sum((self.discount_factor ** k) * self.env.holding_cost(k, i) for k in range(n))
        g_i = self.env.postponing_cost(i)
        return (cin - g_i)
    
    def myopic_coeff_H(self, m):
        return self.discount_factor ** m * self.env.overtime_cost(m)

    def generate_all_columns(self):
        for column in self.env.generate_state_action_pairs():
            yield column

    def paper_solve(self, state, t, action=None, verbose=False):
        # ---------- shortcuts ----------
        with (gp.Model("ALP_policy", env=self.grb_env) as policy_model):
            policy_model.setParam("MultiObjPre", 0)
            policy_model.setParam('DualReductions', 0)
            #policy_model.setParam("FeasibilityTol", 1e-8)
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
            advance_scheduling_cost = np.dot(C, flatten_x_var)
            overtime_cost = np.dot(H, y_var)
            policy_model.setObjective(advance_scheduling_cost + overtime_cost, GRB.MINIMIZE)

            if not solve_and_handle_errors(policy_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            # ---------- 8. return ----------
            action = self.get_solution(action_var, is_final=True)
            return action, policy_model.ObjVal, {}
    
    def solve(self, state, t, action=None, verbose=False):
        with (gp.Model("ALP_policy", env=self.grb_env) as policy_model):
            policy_model.setParam("MultiObjPre", 0)
            policy_model.setParam('DualReductions', 0)
            policy_model.setParam("FeasibilityTol", 1e-9)
            policy_model.setParam("OptimalityTol", 1e-9)
            policy_model.setParam("OutputFlag", 0)
            policy_model.setParam("LogToConsole", 0)
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
    train_args = {"uid": "0360ea73787bbe07b431cee558f64d77", "result": {"agent_name": "alp", "args": {"coefficients": [-55667.655958531, 88.638487172, 89.533825426, 88.638487172, 89.533825426, 88.638487172, 89.533825426, 88.638487172, 89.533825426, 88.638487172, 89.533825426, 88.638487172, 89.533825426, 88.638487172, 87.7521023, 86.874581277, 86.005835464, 85.145777109, 84.294319338, 83.451376145, 82.616862384, 81.79069376, 80.972786822, 80.163058954, 79.361428364, 78.567814081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 178.172312597, 178.172312597, 178.172312597, 178.172312597, 178.172312597]}}}

    sample_path = [[2, 5, 4, 8, 6], [4, 3, 3, 5, 8], [2, 4, 5, 8, 4], [3, 4, 3, 10, 3], [4, 4, 7, 12, 10], [2, 10, 4, 6, 5], [4, 5, 4, 8, 6], [4, 5, 4, 6, 4], [5, 3, 2, 11, 6], [2, 1, 6, 6, 4], [2, 1, 6, 4, 4], [1, 3, 5, 12, 9], [3, 4, 1, 3, 3], [5, 5, 3, 7, 16], [1, 5, 6, 4, 14], [1, 4, 7, 8, 9], [3, 3, 1, 3, 7], [2, 4, 6, 6, 3], [7, 6, 4, 6, 7], [1, 7, 5, 2, 9], [5, 5, 9, 2, 5], [2, 1, 7, 3, 6], [3, 4, 4, 5, 5], [3, 5, 5, 5, 7], [4, 5, 5, 7, 7], [1, 4, 7, 7, 6], [3, 8, 2, 4, 2], [5, 6, 5, 4, 11], [3, 1, 5, 2, 9], [1, 4, 7, 5, 7], [6, 6, 5, 4, 5], [2, 7, 4, 5, 5], [3, 2, 10, 4, 5], [4, 7, 5, 6, 9], [5, 2, 4, 7, 6], [3, 4, 4, 8, 6], [6, 5, 10, 9, 5], [3, 2, 2, 10, 5], [6, 5, 6, 8, 6], [2, 3, 4, 6, 8], [3, 0, 4, 7, 8], [6, 5, 4, 11, 5], [4, 2, 7, 3, 2], [2, 6, 5, 11, 7], [4, 2, 7, 8, 9], [5, 8, 6, 6, 6], [2, 1, 5, 4, 5], [2, 5, 3, 6, 5], [3, 3, 7, 10, 6], [1, 2, 5, 10, 6], [1, 2, 2, 3, 5], [3, 4, 3, 9, 8], [2, 5, 6, 3, 13], [3, 8, 9, 9, 6], [1, 5, 3, 4, 7], [1, 6, 8, 3, 4], [7, 2, 1, 9, 11], [4, 2, 2, 7, 3], [3, 3, 5, 6, 5], [1, 4, 6, 4, 9], [1, 1, 4, 4, 14], [3, 6, 2, 9, 5], [2, 0, 7, 9, 13], [3, 3, 7, 6, 8], [5, 5, 6, 6, 5], [2, 7, 5, 2, 10], [1, 6, 4, 1, 9], [1, 8, 3, 5, 7], [5, 1, 8, 11, 12], [1, 5, 9, 9, 8], [2, 5, 4, 8, 10], [2, 4, 6, 4, 9], [7, 6, 8, 4, 6], [4, 5, 9, 5, 4], [3, 2, 7, 8, 4], [4, 4, 1, 12, 4], [4, 4, 4, 5, 4], [5, 3, 3, 6, 9], [1, 4, 6, 5, 6], [3, 3, 11, 4, 8], [6, 3, 8, 4, 9], [5, 6, 5, 11, 11], [4, 7, 3, 7, 9], [0, 5, 1, 7, 4], [4, 7, 5, 3, 9], [4, 1, 4, 8, 12], [4, 2, 7, 7, 11], [2, 7, 5, 9, 2], [4, 3, 7, 6, 5], [2, 3, 5, 6, 9], [5, 5, 1, 4, 5], [2, 8, 7, 8, 9], [6, 8, 2, 7, 14], [1, 7, 3, 4, 7], [3, 4, 5, 9, 13], [5, 2, 2, 5, 7], [2, 4, 4, 4, 4], [2, 2, 5, 8, 11], [5, 1, 6, 9, 8], [1, 3, 6, 5, 6], [3, 9, 6, 8, 8], [2, 5, 4, 6, 7], [2, 3, 3, 5, 8], [3, 4, 3, 6, 6], [4, 3, 4, 5, 3], [4, 3, 4, 7, 11], [2, 3, 3, 9, 5], [5, 2, 5, 12, 3], [5, 2, 3, 5, 9], [0, 5, 6, 1, 7], [5, 3, 4, 7, 7], [2, 8, 5, 6, 14], [2, 2, 4, 3, 9], [4, 7, 6, 8, 9], [2, 2, 6, 5, 4], [4, 4, 5, 6, 7], [2, 4, 3, 4, 7], [2, 9, 5, 11, 12], [2, 2, 5, 8, 8], [5, 2, 6, 12, 6], [3, 4, 6, 4, 11], [6, 2, 5, 4, 7], [3, 1, 3, 8, 7], [2, 5, 7, 9, 8], [5, 6, 6, 6, 8], [4, 3, 5, 7, 10], [2, 4, 7, 11, 5], [2, 5, 3, 3, 4], [2, 2, 6, 4, 7], [2, 0, 6, 8, 10], [1, 2, 5, 5, 5], [3, 3, 4, 7, 9], [3, 4, 8, 8, 10], [1, 6, 4, 5, 7], [3, 6, 6, 4, 8], [3, 3, 6, 7, 2], [7, 5, 4, 7, 6], [6, 4, 4, 7, 10], [3, 4, 6, 6, 10], [3, 2, 6, 5, 8], [3, 2, 5, 6, 7], [4, 1, 3, 4, 6], [3, 4, 7, 3, 13], [3, 5, 9, 10, 5], [5, 6, 8, 6, 1], [4, 2, 9, 6, 8], [0, 2, 5, 7, 7], [7, 5, 7, 2, 7], [2, 5, 6, 3, 6], [4, 3, 4, 10, 7], [1, 4, 0, 6, 3], [4, 3, 4, 5, 7], [2, 4, 5, 5, 5], [1, 2, 7, 3, 4], [2, 3, 7, 4, 6], [4, 4, 5, 8, 11], [1, 5, 9, 4, 3], [2, 3, 4, 6, 4], [1, 2, 8, 7, 9], [5, 7, 3, 8, 6], [4, 4, 4, 5, 6], [4, 2, 8, 2, 16], [2, 5, 8, 8, 4], [2, 0, 9, 4, 10], [5, 3, 9, 4, 10], [6, 2, 6, 9, 6], [1, 4, 5, 10, 6], [3, 6, 7, 3, 3], [4, 4, 4, 10, 3], [2, 4, 5, 5, 10], [3, 4, 9, 8, 7], [4, 7, 7, 10, 5], [1, 2, 6, 4, 5], [0, 3, 4, 9, 9], [6, 4, 6, 7, 10], [2, 1, 11, 4, 6], [4, 5, 9, 6, 4], [3, 3, 8, 2, 5], [4, 5, 3, 11, 8], [1, 3, 4, 5, 7], [2, 5, 7, 4, 4], [2, 2, 1, 8, 8], [3, 5, 3, 4, 5], [3, 2, 8, 9, 5], [3, 6, 8, 3, 10], [3, 2, 4, 6, 5], [2, 5, 5, 1, 8], [1, 8, 3, 8, 4], [3, 3, 4, 6, 6], [3, 3, 8, 9, 4], [0, 5, 7, 4, 3], [3, 3, 4, 11, 5], [1, 5, 6, 6, 7], [2, 6, 3, 4, 4], [3, 5, 5, 6, 5], [6, 4, 6, 2, 5], [2, 3, 5, 3, 5], [2, 4, 3, 6, 8], [4, 3, 4, 9, 7], [6, 5, 7, 4, 9], [5, 5, 2, 9, 13], [5, 3, 4, 8, 8], [1, 4, 6, 4, 9], [2, 1, 4, 8, 5], [3, 6, 6, 5, 7], [4, 5, 5, 6, 6], [6, 3, 7, 8, 6], [5, 3, 5, 5, 4], [3, 3, 5, 8, 8], [2, 4, 4, 4, 5], [5, 2, 5, 4, 6], [5, 5, 2, 8, 4], [4, 4, 4, 7, 10], [3, 5, 5, 4, 6], [4, 4, 6, 5, 7], [4, 4, 8, 8, 13], [4, 5, 10, 6, 8], [1, 4, 6, 7, 7], [4, 4, 3, 4, 6], [0, 6, 2, 8, 10], [3, 6, 4, 6, 10], [6, 4, 3, 8, 5], [1, 8, 4, 9, 6], [2, 6, 2, 3, 8], [4, 3, 2, 7, 6], [4, 4, 0, 7, 5], [2, 3, 2, 4, 6], [1, 5, 4, 10, 4], [1, 6, 5, 8, 9], [1, 5, 3, 5, 9], [3, 5, 10, 4, 12], [3, 4, 3, 5, 8], [1, 7, 6, 6, 3], [4, 4, 6, 9, 7], [3, 7, 8, 1, 8], [4, 4, 2, 5, 10], [4, 3, 3, 8, 3], [3, 5, 3, 4, 3], [1, 4, 6, 3, 6], [2, 4, 6, 2, 6], [3, 2, 6, 5, 8], [3, 4, 6, 5, 5], [3, 5, 2, 6, 9], [4, 3, 5, 6, 4], [5, 3, 10, 5, 4], [2, 8, 5, 9, 9], [3, 6, 6, 7, 7], [4, 6, 3, 5, 8], [3, 2, 4, 9, 4], [4, 3, 5, 4, 13], [5, 2, 5, 7, 11], [4, 1, 2, 2, 10], [7, 4, 5, 6, 8], [2, 5, 8, 8, 9], [4, 5, 4, 4, 4], [1, 2, 4, 4, 9], [3, 5, 4, 9, 7], [4, 4, 2, 4, 11], [2, 4, 7, 3, 2], [0, 8, 9, 8, 1], [5, 3, 5, 12, 5], [5, 2, 5, 8, 8], [3, 3, 4, 2, 5], [1, 2, 3, 7, 4], [1, 7, 5, 6, 6], [2, 2, 9, 8, 6], [5, 5, 9, 5, 8], [2, 3, 5, 6, 12], [5, 7, 4, 6, 5], [7, 5, 3, 8, 8], [6, 5, 5, 7, 1], [2, 4, 5, 5, 4], [4, 1, 4, 5, 5], [1, 4, 3, 8, 7], [1, 6, 9, 9, 6], [1, 5, 4, 5, 10], [6, 2, 4, 2, 3], [5, 7, 4, 10, 6], [6, 7, 4, 9, 4], [3, 2, 4, 5, 3], [0, 5, 6, 6, 12], [2, 2, 5, 3, 8], [4, 3, 2, 4, 4], [4, 6, 6, 3, 5], [5, 7, 4, 4, 8], [1, 4, 4, 4, 9], [3, 5, 3, 8, 5], [4, 2, 7, 11, 6], [1, 5, 2, 4, 7], [5, 4, 5, 6, 7], [2, 4, 6, 3, 7], [1, 3, 4, 3, 7], [3, 4, 5, 7, 10], [4, 4, 5, 7, 7], [7, 1, 3, 7, 6], [2, 4, 4, 8, 6], [3, 2, 8, 2, 8], [5, 4, 8, 2, 8], [2, 2, 0, 4, 6], [5, 3, 0, 5, 6], [3, 5, 3, 5, 2], [1, 4, 3, 7, 4], [2, 3, 7, 9, 5], [2, 4, 6, 4, 8], [4, 4, 7, 7, 5], [3, 2, 4, 5, 11], [4, 6, 7, 7, 5], [4, 6, 8, 3, 7], [2, 3, 1, 5, 5], [4, 3, 1, 6, 4], [3, 4, 5, 3, 12], [6, 4, 6, 3, 11], [7, 6, 8, 4, 5], [3, 3, 4, 4, 4], [3, 2, 8, 4, 9], [0, 6, 4, 4, 6], [4, 4, 4, 6, 4], [5, 4, 7, 8, 6], [3, 6, 6, 5, 6], [4, 9, 6, 4, 7], [2, 4, 7, 5, 8], [1, 6, 7, 8, 10], [5, 4, 3, 4, 6], [2, 1, 6, 5, 11], [3, 12, 6, 7, 5], [3, 2, 5, 5, 14], [3, 3, 2, 4, 5], [2, 5, 1, 7, 6]]
    config = get_config_by_type('ejor_default')
    env = config.env
    init_state = config.init_state
    coefficients = train_args['result']['args']['coefficients']
    print("coefficients:", coefficients)
    agent = ALPEJORColumnGenerationAgent(env=env, discount_factor=0.99, pretrain=False)
    #print('ALP')
    #print([agent.coeff_C(i,n) for n in range(agent.env.booking_window_size) for i in range(agent.env.num_types)])
    #print('Myopic')
    #print([agent.myopic_coeff_C(i,n) for n in range(agent.env.booking_window_size) for i in range(agent.env.num_types)])
    #print('ALP')
    #print([agent.coeff_H(m) for m in range(agent.env.planning_horizon)])
    #print('Myopic')
    #print([agent.myopic_coeff_H(m) for m in range(agent.env.planning_horizon)])
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

