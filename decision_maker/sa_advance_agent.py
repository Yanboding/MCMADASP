import time
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import get_solution_value, RunningStats, solve_and_handle_errors


class SAAdvanceAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None, sample_path_number=500, current_decision_var_type='integer', future_decision_var_type='continuous', is_myopic=False):
        self.env = env
        self.discount_factor = discount_factor
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type =='integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type =='integer' else GRB.CONTINUOUS
        self.is_myopic = is_myopic
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
        self.V = V
        self.Q = Q
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}
        self.grb_env = self._acquire_grb_env()

    def set_sample_paths(self, sample_path_number):
        self.sample_path_number = sample_path_number
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
        print(self.delta)

    def set_real_sample_paths(self, sample_paths):
        self.sample_path_number = len(sample_paths)
        self.delta = np.array(sample_paths)

    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])
    # ------------------------------------------------------------------
    # Helper: wait‑until‑token‑free loop
    # ------------------------------------------------------------------
    def _acquire_grb_env(self, silent=True, wait=TOKEN_WAIT):
        """
        Try to create and start a gp.Env.  If all tokens are in use,
        wait <wait> seconds and retry indefinitely.
        """
        while True:
            try:
                grb_env = gp.Env(empty=True)    # no token yet
                if silent:
                    grb_env.setParam("OutputFlag", 0)
                grb_env.start()                 # tries to grab ONE token
                print('Get one token...')
                return grb_env                  # success
            except gp.GurobiError as e:
                if "All tokens currently in use" in str(e):
                    print('Waiting...')
                    time.sleep(wait)            # back‑off and try again
                else:
                    raise                       # some other licence error

    def add_action_space_constraints(self, model, state_var, action_var, t, tau):
        P = self.env.planning_horizon - t - tau
        I = self.env.num_types
        u_var, w_var = state_var
        x_var, y_var = action_var
        model.addConstrs(
            (x_var[:, i].sum() == w_var[i]
             for i in range(I)),
            name=f"valid_advance_scheduling_{t+tau}",
        )
        post_action_regular_bookings, post_action_waitlist = self.env.post_action_state(state_var, action_var, is_var=True)
        model.addConstrs(
            (
                post_action_regular_bookings[m] <= self.env.regular_capacity
                for m in range(P + 1)
            ),
            name=f"valid_post_action_regular_bookings_{t+tau}",
        )
        return model

    def get_action_var(self, model, t, tau, advance_scheduling_type):
        H = self.env.decision_epoch - t - tau
        P = self.env.planning_horizon - t - tau
        I = self.env.num_types
        x_var_t = np.array([
            [model.addVar(vtype=advance_scheduling_type, lb=0, name=f"x^{t+tau},{j},{i}") for i in range(I)]
            for j in range(H + 1)
        ])
        y_var_t = np.array(
            [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y^{t+tau},{j}") for j in range(P + 1)]
        )
        return (x_var_t, y_var_t)

    def get_solution(self, action_var):
        x_var, y_var = action_var
        x = get_solution_value(x_var).astype(int)
        y = get_solution_value(y_var).astype(float)
        return (x, y)

    def set_action(self, action_var, action):
        x_var, y_var = action_var
        x, y = action
        for i, row in enumerate(x_var):
            for j, var in enumerate(row):
                var.lb = var.ub = x[i][j]
        for j, var in enumerate(y_var):
            var.lb = var.ub = y[j]

    def direct_solve(self, state, t, action=None):
        H = self.env.decision_epoch - t
        # ---------- model ----------
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            # ---------- 1. today’s increments ----------
            action_var_t = self.get_action_var(model=m, t=t, tau=0, advance_scheduling_type=GRB.INTEGER)
            if action is not None:
                self.set_action(action_var=action_var_t, action=action)
            # add action constraint
            self.add_action_space_constraints(model=m, state_var=state, action_var=action_var_t, t=t, tau=0)
            # ---------- 1. objective ----------
            imm_cost = self.env.cost_fn(state, action_var_t, t)
            fut_cost = 0
            state_t_tau_vars = []
            action_t_tau_vars = []
            if not self.is_myopic:
                prev_state_scenario = [state for _ in range(self.sample_path_number)]
                prev_action_scenario = [action_var_t for _ in range(self.sample_path_number)]
                for tau in range(1, H+1):
                    state_scenario = []
                    action_scenario = []
                    for omega in range(self.sample_path_number):
                        state_t_tau = self.env.get_next_state(state=prev_state_scenario[omega],
                                                              action=prev_action_scenario[omega],
                                                              new_arrival=self.delta[omega, tau], is_var=True)
                        action_var_t_tau = self.get_action_var(model=m, t=t, tau=tau, advance_scheduling_type=GRB.CONTINUOUS)
                        self.add_action_space_constraints(model=m, state_var=state_t_tau, action_var=action_var_t_tau, t=t, tau=tau)
                        fut_cost += self.env.discount_factor ** tau * self.env.cost_fn(state_t_tau, action_var_t_tau, t+tau)
                        state_scenario.append(state_t_tau)
                        action_scenario.append(action_var_t_tau)
                    state_t_tau_vars.append(state_scenario)
                    action_t_tau_vars.append(action_scenario)
                    prev_state_scenario = state_scenario
                    prev_action_scenario = action_scenario
                fut_cost = fut_cost / self.sample_path_number
            m.setObjective(imm_cost+fut_cost, GRB.MINIMIZE)
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            '''
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
            '''
            print('imm_cost:', imm_cost.getValue())
            if t >= self.env.decision_epoch or self.is_myopic:
                print('future_cost:', fut_cost)
            else:
                print('future_cost:', fut_cost.getValue())
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                action = self.get_solution(action_var_t)
                return action, m.ObjVal, {}
            else:
                m.write('not_optimal.lp')
                raise RuntimeError("Optimal solution not found")

    def policy(self, state, t):
        action, obj_value, info = self.solve(state, t)
        return action

    def build_linking_constraints(self, model, action_t_var):
        x_t_var, y_t_var = action_t_var
        linking_constraints = []
        for j, row in enumerate(x_t_var):
            for i, var in enumerate(row):
                constraint = model.addConstr(var == 0.0, name=f'link_x_{j},{i}')
                linking_constraints.append(constraint)
        for j, var in enumerate(y_t_var):
            constraint = model.addConstr(var == 0.0, name=f'link_y_{j}')
            linking_constraints.append(constraint)
        return linking_constraints

    def subproblem_builder(self, state, t, scenario_id):
        H = self.env.decision_epoch - t
        P = self.env.planning_horizon - t
        I = self.env.num_types
        # model = Model(HiGHS.Optimizer)
        # set_silent(model)
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=self.grb_env)
        sub_model.setParam('InfUnbdInfo', 1)
        sub_model.setParam('DualReductions', 0)
        # get u^t+1 and linking constrs
        action_t_var = self.get_action_var(model=sub_model, t=t, tau=0, advance_scheduling_type=GRB.CONTINUOUS)
        linking_constraints = self.build_linking_constraints(sub_model, action_t_var)
        # Initialize scenario state and action like in direct solution
        u_t_tau_var = self.env.get_next_regular_bookings(state, action_t_var, is_var=True)
        fut_cost = 0
        for tau in range(1, H + 1):
            state_t_tau_var = (u_t_tau_var, self.delta[scenario_id, tau])
            action_t_tau_var = (x_t_tau_var, y_t_tau_var)= self.get_action_var(model=sub_model, t=t, tau=tau,
                                                   advance_scheduling_type=GRB.CONTINUOUS)
            # action constraints in period t+tau
            sub_model.addConstrs((x_t_tau_var[:, i].sum() == self.delta[scenario_id, tau, i] for i in range(I)), name=f"valid_advance_scheduling_{t + tau}", )
            bar_u_t_tau_var, _ = self.env.post_action_state(state_t_tau_var, action_t_tau_var, is_var=True)
            sub_model.addConstrs((bar_u_t_tau_var[m] <= self.env.regular_capacity for m in range(P - tau + 1) ), name=f"valid_post_action_regular_bookings_{t + tau}", )
            fut_cost += self.env.discount_factor ** tau * self.env.cost_fn(state_t_tau_var, action_t_tau_var, t + tau)
            u_t_tau_var = np.array( [sub_model.addVar(vtype=GRB.CONTINUOUS, name=f"u^{t + tau}_{j}") for j in range(P - tau)])
            sub_model.addConstrs((u_t_tau_var[j] == bar_u_t_tau_var[j + 1] for j in range(P - tau)), name=f"transit{t + tau}" )
        sub_model.setObjective(fut_cost, GRB.MINIMIZE)
        return sub_model, linking_constraints

    def set_linking_constraints_rhs(self, sub_model, linking_constraints, action_t):
        # Flatten in the same order as self.flatten(action)
        action_flat = self.flatten(action_t)
        for i, constr in enumerate(linking_constraints):
            constr.setAttr("RHS", action_flat[i])
        sub_model.update()

    def solve_subproblem(self, sub_model, linking_constraints, action_t, verbose=False):
        # @variable(model, x[i in 1:n, j in 1:n] == x_bar[i, j])
        self.set_linking_constraints_rhs(sub_model, linking_constraints, action_t)
        if solve_and_handle_errors(sub_model, verbose=verbose):
            v = sub_model.ObjVal
            duals = np.array([linking_constraints[j].Pi for j in range(len(linking_constraints))])
            return True, v, duals
        else:
            #print(self.get_solution(action_t_var), action_t)
            # dual objective value
            v = sum(c.FarkasDual * c.RHS for c in sub_model.getConstrs())
            # Use Farkas duals on LINKING rows only
            ray = np.array([c.FarkasDual for c in linking_constraints], dtype=float)
            return False, v, ray

    def master_problem(self, state, t):
        master_model = gp.Model(f"SA_Advance_Master", env=self.grb_env)
        master_model.setParam('DualReductions', 0)
        # create action variables in period t
        action_t_var = self.get_action_var(model=master_model, t=t, tau=0, advance_scheduling_type=GRB.INTEGER)
        # add action constraint
        self.add_action_space_constraints(model=master_model, state_var=state, action_var=action_t_var, t=t, tau=0)
        # set imm_cost and a cost to go lb
        theta_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(len(self.delta))])
        imm_cost = self.env.cost_fn(state, action_t_var, t)
        z = imm_cost + theta_vars.sum()/self.sample_path_number
        master_model.setObjective(z, GRB.MINIMIZE)
        master_model.update()
        return master_model, imm_cost, theta_vars, action_t_var

    def flatten(self, action):
        x, y = action
        return np.append(x.reshape(-1), y)

    def solve(self, state, t, action=None, tol=1e-6, max_iter=1000, verbose=False):
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        master_model, imm_cost, theta_vars, action_t_var = self.master_problem(state, t)
        if action is not None:
            self.set_action(action_var=action_t_var, action=action)
        sub_models = [self.subproblem_builder(state=state, t=t, scenario_id=scenario_id) for scenario_id in range(self.sample_path_number)]
        # for k in 1:MAXIMUM_ITERATIONS
        for iteration in range(max_iter):
            print('Iteration:', iteration+1, 'Start')
            # optimize!(model)
            # assert_is_solved_and_feasible(model)
            if not solve_and_handle_errors(master_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            # lower_bound = objective_value(model)
            lower_bound = master_model.ObjVal
            # x_k = value.(x)
            action_t = self.get_solution(action_t_var)
            flat_action_t_var = self.flatten(action_t_var)
            flat_action_t = self.flatten(action_t)
            cost_to_go_estimation = 0
            opt_cuts = []
            for scenario_id in range(len(self.delta)):
                sub_model, linking_constraints = sub_models[scenario_id]
                is_feasible, v, duals = self.solve_subproblem(sub_model=sub_model, linking_constraints=linking_constraints, action_t=action_t, verbose=verbose)
                if not is_feasible:
                    cut_expr = v + np.dot(duals, flat_action_t_var - flat_action_t)

                    master_model.addConstr(cut_expr >= 0,
                                           name=f"feasible_cut_{iteration}_{scenario_id}")
                    # An infeasible scenario invalidates the whole solution, so we break and re-solve master
                    break
                # If feasible, generate the strengthened cut using the dynamic method
                #duals = self.generate_dynamic_cut_duals(sub_model, linking_constraints, flat_action_t, core_point_flat, mu)
                cost_to_go_estimation += v
                # cut = @constraint(model, θ >= ret.obj + sum(ret.π .* (x .- x_k)))
                cut_rhs = v + np.dot(duals, flat_action_t_var - flat_action_t)
                opt_cuts.append(theta_vars[scenario_id] >= cut_rhs)
            else:
                master_model.addConstrs((opt_cuts[i] for i in range(len(opt_cuts))), name="opt_cut_")
                cost_to_go_estimation = cost_to_go_estimation / self.sample_path_number
                upper_bound = imm_cost.getValue() + cost_to_go_estimation
                # Average the future cost across scenarios like in direct solution
                if abs(upper_bound - lower_bound)/abs(upper_bound) < tol:
                    return action_t, master_model.ObjVal, {}
            master_model.update()
            print('upper_bound:', upper_bound)
            print('lower_bound:', lower_bound)

            print('-'*20)



if __name__ =="__main__":
    from experiments import get_config_by_type

    config = get_config_by_type('adv_default')
    env = config.env
    discount_factor = env.discount_factor
    agent = SAAdvanceAgent(env, discount_factor, **{'sample_path_number': 50, 'is_myopic':False})
    print('Init State:', config.init_state)
    print('Future arrivals:', agent.delta[0])

    start = time.time()
    #action = (np.array([[3, 1], [0, 2]]), np.array([2,0]))
    action=None
    action, obj_value, info= agent.solve(config.init_state, 1, action=action)
    print(time.time() - start)
    print('bender_decomposition:')
    print(obj_value)
    print(action)


    start = time.time()
    #action = (np.array([[3, 1], [0, 2]]), np.array([2,0]))
    action = None
    action, obj_value, info = agent.direct_solve(config.init_state, 1, action=action)
    print(time.time() - start)
    print('direct solve:')
    print(obj_value)
    print(action)
    # 251490.33719727152



