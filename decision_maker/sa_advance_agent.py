import time
from collections import defaultdict
import concurrent.futures

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import get_solution_value, solve_and_handle_errors, clean_value, acquire_grb_env

# -----------------------------
# Persistent scenario worker
# -----------------------------
class SubproblemWorker:
    """
    One worker per scenario. Owns its own gp.Env and gp.Model.
    Build once, then call solve(action) repeatedly.
    """
    def __init__(self, builder_fn, builder_args, thread_id:int, verbose:bool=True):
        self.env = acquire_grb_env({"Threads": 1}, verbose=verbose)
        if not verbose:
            self.env.setParam("OutputFlag", 0)
        # Important: avoid core oversubscription; parallelism is at outer level.
        self.env.setParam("Threads", 1)
        self.env.start()

        # Build the model and linking constraints inside THIS env.
        self.model, self.link_rows = builder_fn(env=self.env, **builder_args)
        self.thread_id = thread_id
        # Stash action^t variable tuple for quick flattening of variables when building cuts
        # (the builder must set this)
        assert hasattr(self.model, "_action_t_var"), "Builder must set model._action_t_var"
        self._flat_action_vars_cache = None

    def _flat_action_vars(self):
        """
        Returns a flat list of gurobi Var objects corresponding to action at period t.
        Cached after first call.
        """
        if self._flat_action_vars_cache is None:
            x_t, y_t = self.model._action_t_var
            # x_t is a 2D array-like of Vars; y_t is 1D array-like
            flat_x = [v for row in x_t for v in row]
            flat_y = list(y_t)
            self._flat_action_vars_cache = flat_x + flat_y
        return self._flat_action_vars_cache

    def _flatten_action_values(self, action):
        """
        Flattens an action=(x,y) value tuple into a 1D numpy array of numbers
        using the same order as _flat_action_vars().
        """
        x, y = action
        return np.concatenate([x.reshape(-1), y])

    def set_link_rhs(self, action_flat_values):
        """
        Update RHS of linking constraints so they enforce: (action vars) == (action values).
        Assumes link rows were built as equality rows var == 0 initially.
        """
        for i, constr in enumerate(self.link_rows):
            constr.setAttr("RHS", float(action_flat_values[i]))
        # No explicit update() needed; optimize() will sync.

    def solve(self, action, verbose:bool=False):
        """
        Set links to the candidate master action and optimize the subproblem.
        Return (is_feasible, objective_value, dual_vector_or_ray_on_link_rows).
        """
        action_flat = self._flatten_action_values(action)
        self.set_link_rhs(action_flat)

        if verbose:
            self.model.Params.OutputFlag = 1
        else:
            self.model.Params.OutputFlag = 0

        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            v = self.model.ObjVal
            duals = np.array([c.Pi for c in self.link_rows], dtype=float)
            return True, v, duals
        else:
            # Infeasible: use Farkas duals / ray
            v = sum(c.FarkasDual * c.RHS for c in self.model.getConstrs())
            ray = np.array([c.FarkasDual for c in self.link_rows], dtype=float)
            return False, v, ray

    def dispose(self):
        try:
            self.model.dispose()
        finally:
            self.env.dispose()

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
        self.grb_env = acquire_grb_env({"Threads": 1}, verbose=False, wait=SAAdvanceAgent.TOKEN_WAIT)

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

    def get_solution(self, action_var, is_final=False):
        x_var, y_var = action_var
        if is_final:
            x = np.array([[clean_value(var.Xn, tolerance=1e-6) for var in row] for row in x_var]).astype(int)
        else:
            x = get_solution_value(x_var).astype(float)
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
            m.setParam('DualReductions', 0)
            m.setParam("MultiObjPre", 0)
            m.setParam('MIPFocus', 1)
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
                action = self.get_solution(action_var_t, is_final=True)
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
        sub_model.setParam("MultiObjPre", 0)
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

    # ---------------------------------------------------
    # Subproblem builder used by workers
    # ---------------------------------------------------
    def _subproblem_build_for_worker(self, env, state, t, scenario_id):
        H = self.env.decision_epoch - t
        P = self.env.planning_horizon - t
        I = self.env.num_types

        sub = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
        sub.Params.InfUnbdInfo = 1
        sub.Params.DualReductions = 0
        sub.Params.MultiObjPre = 0
        sub.Params.OutputFlag = 0

        # action at t (continuous here; we only link its value)
        action_t_var = self.get_action_var(model=sub, t=t, tau=0, advance_scheduling_type=GRB.CONTINUOUS)
        link_rows = self.build_linking_constraints(sub, action_t_var)

        # Build future chain for this scenario (fixed arrivals = self.delta[scenario_id, tau])
        u_t1_var = self.env.get_next_regular_bookings(state, action_t_var, is_var=True)
        fut_cost = 0.0
        for tau in range(1, H + 1):
            s_tau = (u_t1_var, self.delta[scenario_id, tau])
            x_tau, y_tau = self.get_action_var(model=sub, t=t, tau=tau, advance_scheduling_type=GRB.CONTINUOUS)

            # action feasibility at t+tau; here arrivals are fixed numbers
            sub.addConstrs(
                (gp.quicksum(x_tau[j, i] for j in range(x_tau.shape[0])) == self.delta[scenario_id, tau, i]
                 for i in range(I)),
                name=f"valid_advance_scheduling_{t + tau}"
            )

            bar_u_tau, _ = self.env.post_action_state(s_tau, (x_tau, y_tau), is_var=True)
            sub.addConstrs(
                (bar_u_tau[m] <= self.env.regular_capacity for m in range(P - tau + 1)),
                name=f"valid_post_action_regular_bookings_{t + tau}"
            )

            fut_cost += (self.env.discount_factor ** tau) * self.env.cost_fn(s_tau, (x_tau, y_tau), t + tau)

            # shift bookings: u^{t+tau} := bar_u^{t+tau}[1:]
            if P - tau > 0:
                u_t1_var = np.array(
                    [sub.addVar(vtype=GRB.CONTINUOUS, name=f"u^{t + tau}_{j}") for j in range(P - tau)])
                sub.addConstrs(
                    (u_t1_var[j] == bar_u_tau[j + 1] for j in range(P - tau)),
                    name=f"transit{t + tau}"
                )
            else:
                u_t1_var = np.array([], dtype=object)

        sub.setObjective(fut_cost, GRB.MINIMIZE)

        # Let worker cache these for fast cut construction
        sub._action_t_var = action_t_var
        return sub, link_rows

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
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam('MIPFocus', 1)
        # create action variables in period t
        action_t_var = self.get_action_var(model=master_model, t=t, tau=0, advance_scheduling_type=GRB.INTEGER)
        # add action constraint
        self.add_action_space_constraints(model=master_model, state_var=state, action_var=action_t_var, t=t, tau=0)
        # set imm_cost and a cost to go lb
        theta_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(len(self.delta))])
        imm_cost = self.env.cost_fn(state, action_t_var, t)
        z = imm_cost + theta_vars.sum()/self.sample_path_number
        master_model.setObjective(z, GRB.MINIMIZE)
        return master_model, imm_cost, theta_vars, action_t_var

    def flatten(self, action):
        x, y = action
        return np.append(x.reshape(-1), y)

    def solve(self, state, t, action=None, tol=1e-6, max_iter=3000, verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, t, action=action)
            return action, obj_value, info
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        master_model, imm_cost, theta_vars, action_t_var = self.master_problem(state, t)
        flat_action_t_var = self.flatten(action_t_var)
        if action is not None:
            self.set_action(action_var=action_t_var, action=action)
        # Build one worker per scenario once, then reuse
        builder_args = [{"state": state, "t": t, "scenario_id": sid} for sid in range(self.sample_path_number)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.sample_path_number) as ex:
            workers = [
                ex.submit(SubproblemWorker, self._subproblem_build_for_worker, builder_args[sid], sid, verbose).result()
                for sid in range(self.sample_path_number)
            ]

            try:
                for iteration in range(1, max_iter + 1):
                    if not solve_and_handle_errors(master_model, verbose=verbose):
                        raise RuntimeError("Master model optimal solution not found")
                    action_t = self.get_solution(action_t_var)
                    flat_action_t = self.flatten(action_t)

                    lower_bound = master_model.ObjVal

                    # Ask all workers to solve for this action
                    #futures = [ex.submit(w.solve, action_t, verbose) for w in workers]
                    futures = {
                        ex.submit(w.solve, action_t, verbose): w.thread_id for w in workers
                    }
                    feasibility_cuts = []
                    optimality_cuts = []
                    cost_to_go_estimation = 0.0
                    all_feasible = True
                    for future in concurrent.futures.as_completed(futures):
                        scenario_id = futures[future]
                        is_feasible, v, duals = future.result()
                        if not is_feasible:
                            print(f"Iteration {iteration}, scenario {scenario_id} infeasible; adding feasibility cut")
                            all_feasible = False
                            # Add feasibility cut to master
                            cut_expr = v + np.dot(duals, flat_action_t_var - flat_action_t)
                            feasibility_cuts.append(cut_expr >= 0)
                        else:
                            # If feasible, generate the strengthened cut using the dynamic method
                            cost_to_go_estimation += v
                            # cut = @constraint(model, θ >= ret.obj + sum(ret.π .* (x .- x_k)))
                            cut_rhs = v + np.dot(duals, flat_action_t_var - flat_action_t)
                            optimality_cuts.append(theta_vars[scenario_id] >= cut_rhs)
                    if not all_feasible:
                        print(f"Iteration {iteration}, adding {len(feasibility_cuts)} feasibility cuts")
                        # Some scenario infeasible: add feasibility cuts and repeat
                        master_model.addConstrs((feasibility_cuts[i] for i in range(len(feasibility_cuts))), name="feas_cut_")
                    else:
                        print(f"Iteration {iteration}, adding {len(optimality_cuts)} optimality cuts")
                        # All scenarios feasible: add optimality cuts and continue
                        master_model.addConstrs((optimality_cuts[i] for i in range(len(optimality_cuts))), name="opt_cut_")
                        cost_to_go_estimation = cost_to_go_estimation / self.sample_path_number
                        upper_bound = imm_cost.getValue() + cost_to_go_estimation
                        # Average the future cost across scenarios like in direct solution
                        if abs(upper_bound - lower_bound) < tol:
                            action_t = self.get_solution(action_t_var, is_final=True)
                            return action_t, upper_bound, {}
                    #master_model.update()
                    print('upper_bound:', upper_bound)
                    print('lower_bound:', lower_bound)

                    print('-' * 20)
                print('Max iterations reached')
                return action_t, upper_bound, {}
            finally:
                for w in workers:
                    w.dispose()



if __name__ =="__main__":
    from experiments import get_config_by_type

    config = get_config_by_type('base_case')
    env = config.env
    discount_factor = env.discount_factor
    agent = SAAdvanceAgent(env, discount_factor, **{'sample_path_number': 2, 'is_myopic':False})
    print('Init State:', config.init_state)
    print('Future arrivals:', agent.delta[0])
    start = time.time()
    #action = (np.array([[3, 1], [0, 2]]), np.array([2,0]))
    action=None
    action, obj_value, info= agent.solve(config.init_state, 1, action=action)
    print(time.time() - start)
    print('bender_decomposition:')
    print(obj_value) # 423492.46229695214
    print(action)


    start = time.time()
    action = None
    action, obj_value, info = agent.direct_solve(config.init_state, 1, action=action)
    print(time.time() - start)
    print('direct solve:')
    print(obj_value) # 423493.53852545697
    print(action)



