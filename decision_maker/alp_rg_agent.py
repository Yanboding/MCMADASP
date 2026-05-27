import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import RowGenerationSolver
from utils import get_solution_value, solve_and_handle_errors, clean_value

class ALPRowGenerationAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False, decay_factor=0.95, grb_env=None):
        super().__init__(env, discount_factor, V, Q, grb_env=grb_env)
        self.is_trained = False
        self.decay_factor = decay_factor
        required_bookings = [(self.env.regular_capacity + self.env.overtime_capacity) * self.decay_factor**(j) for j in range(self.env.planning_horizon)]
        required_bookings[-1] = 0
        required_bookings = np.array(required_bookings)
        self.E_u_alpha = np.minimum(required_bookings, self.env.regular_capacity)
        self.E_v_alpha = required_bookings - self.E_u_alpha
        self.E_w_alpha = self.env.arrival_generator.mean_by_type
        if coefficients is not None:
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(coefficients)
        if pretrain:
            self.train(False)

    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = float(next(it))
        U = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        V = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        W = np.array([float(next(it)) for _ in range(self.env.num_types)])
        return W_0, U, V, W

    def get_approx_value_fn(self, state, W_0, U, V, W):
        regular_bookings, overtimes, waitlist = state
        return W_0 + U @ regular_bookings + V @ overtimes + W @ waitlist

    def get_candidate(self, state_var, action_var):
        regular_booking_vars, overtime_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        
        regular_bookings = regular_booking_vars.X
        overtime = overtime_vars.X
        waitlist = waitlist_vars.X
        advance_scheduling_decision = np.round(advance_scheduling_decision_vars.X).astype(int)
        overtime_decision = overtime_decision_vars.X

        return ((regular_bookings, overtime, waitlist), (advance_scheduling_decision, overtime_decision))

    def generate_all_candidates(self):
        for column in self.env.generate_state_action_pairs():
            yield column

    def get_constraint_data(self, model, candidate):
        state, action = candidate
        candidate_cost = self.env.cost_fn(state, action)
        approx_V_t = self.get_approx_value_fn(state=state,
                                              W_0=self.W_0_var,
                                              U=self.U_vars,
                                              V=self.V_vars,
                                              W=self.W_vars)
        new_state = self.env.get_next_state(state, action, self.env.arrival_generator.mean_by_type,
                                            is_var=False)
        approx_V_next = self.get_approx_value_fn(state=new_state,
                                                 W_0=self.W_0_var,
                                                 U=self.U_vars,
                                                 V=self.V_vars,
                                                 W=self.W_vars)
        return approx_V_t - self.env.discount_factor * approx_V_next <= candidate_cost

    def generate_initial_candidates(self, verbose=False):
        for i in range(self.env.num_types):
            with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
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

    def master_builder(self):
        master_model = gp.Model('MasterRMP')
        master_model.setParam('DualReductions', 0)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        master_model.setParam('OutputFlag', 0)
        BigM = 1e4
        self.W_0_var = master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BigM, name=f"W_0")
        self.U_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=BigM, name=f"U_{j}") for j in range(self.env.planning_horizon)])
        self.V_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=BigM, name=f"V_{j}") for j in range(self.env.planning_horizon)])
        self.W_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=BigM, name=f"W_{i}") for i in range(1, self.env.num_types + 1)])
        obj = (
            self.W_0_var
            + gp.quicksum(self.U_vars[j] * self.E_u_alpha[j] for j in range(self.env.planning_horizon))
            + gp.quicksum(self.V_vars[j] * self.E_v_alpha[j] for j in range(self.env.planning_horizon))
            + gp.quicksum(self.W_vars[i] * self.E_w_alpha[i] for i in range(self.env.num_types))
        )

        master_model.setObjective(obj, GRB.MAXIMIZE)
        master_model.update()
        coefficient_vars = [self.W_0_var] + self.U_vars.tolist() + self.V_vars.tolist() + self.W_vars.tolist()
        return master_model, obj, coefficient_vars

    def separation_callback(self, solution):
        W_0, U, V, W = self.get_coefficients(solution)
        separation_model = gp.Model(f"Separation_Problem", env=self.grb_env)
        separation_model.setParam('OutputFlag', 0)
        state_var = self.get_state_var(separation_model)
        action_var = self.get_action_var(separation_model, advance_scheduling_type=GRB.INTEGER)
        self.add_action_space_constraints(separation_model, state_var, action_var)
        # --- Objective ---
        candidate_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        approx_V = self.get_approx_value_fn(state=state_var,
                                            W_0=W_0,
                                            U=U,
                                            V=V,
                                            W=W)
        new_state_var = self.get_next_state(model=separation_model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=self.env.arrival_generator.mean_by_type)
        approx_V_next = self.get_approx_value_fn(state=new_state_var,
                                                 W_0=W_0,
                                                 U=U,
                                                 V=V,
                                                 W=W)
        dual_cost = approx_V - self.env.discount_factor * approx_V_next
        violation = dual_cost - candidate_cost
        separation_model.setObjective(violation, GRB.MAXIMIZE)
        if solve_and_handle_errors(separation_model):
            candidate = self.get_candidate(state_var, action_var)
            yield candidate, separation_model.ObjVal

    def train(self, debug, tol=1e-6, max_iter=20000, verbose=False):
        if debug == True:
            initial_candidates = self.generate_all_candidates()
        else:
            initial_candidates = self.generate_initial_candidates(verbose=verbose)
        self.rg_solver = RowGenerationSolver(master_builder=self.master_builder,
                                             separation_callback=self.separation_callback,
                                             get_constraint_data=self.get_constraint_data,
                                             initial_candidates=initial_candidates)
        self.rg_solver.solve(tol=tol, max_iter=max_iter)
        final_coefficients = [v.X for v in self.rg_solver.coefficient_vars]
        self.W_0, self.U, self.V, self.W = self.get_coefficients(final_coefficients)
        self.is_trained = True
        print("Training completed. Objective value:", self.rg_solver.master_model.ObjVal)
        master_obj = self.rg_solver.master_model.ObjVal
        return master_obj, final_coefficients
    
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
            return policy_model.ObjVal, action, {}

if "__main__" == __name__:
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    init_state = config.init_state
    agent = ALPRowGenerationAgent(env=env, discount_factor=env.discount_factor)
    print(agent.train(debug=False))
    #duals = [84.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # [10000.0, 5919.593918987857, 5860.397979797979, 5801.794, 0.0, 4951.015202530357, 4901.505050505051, 4852.490000000002, 0.0, 10000.0, 10000.0]
    #print(list(agent.separation_callback(duals)))
    # (10874.249099999995, [7236.880199999996, 98.9999999999999, 99.0, 98.00999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 197.99999999999997, 293.0498999999999])