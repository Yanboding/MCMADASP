import copy
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from importance_sampling.proposals import ArrivalGeneratorSamplePathProposal
from metaheuristic_algorithm import SubproblemWorker, BendersDecompositionSolver
from utils import solve_and_handle_errors, flatten, set_link_rhs, acquire_grb_env, encode, get_status_string, is_single_init_state

class ApproxQAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, 
                 sample_path_number=100, 
                 current_decision_var_type='integer', 
                 future_decision_var_type='continuous',
                 penalty_ratio=1,
                 generating_function=None,
                 sample_path_proposal=None,
                 solver_name='approx_Q',
                 is_trained=False,
                 grb_env=None,
                 subproblem_grb_envs=None,
                 verbose=False):
        super().__init__(env, discount_factor, V=V, Q=Q, grb_env=grb_env)
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.arrival_generator = copy.deepcopy(self.env.arrival_generator)
        self.sample_path_proposal = sample_path_proposal or ArrivalGeneratorSamplePathProposal()
        self.delta, self.period_likelihood_ratios = self._initialize_sample_paths(sample_path_number)
        self.sample_path_number = len(self.delta)
        self.penalty_ratio = penalty_ratio
        self.generating_function = generating_function
        self.decision_model, self.state_linking_constraints, self.action_t_var = None, None, None
        self.is_trained = is_trained
        self.coefficient_model, self.coefficients = None, None
        self.workers = None
        self.solver_name = solver_name
        self.subproblem_grb_envs = subproblem_grb_envs
        self.verbose = verbose
        self.subproblem_cold_solve_seconds = {}
        # Per-scenario build-time Benders cut (v0, g0) from the cold solve at
        # coefficients a = 0; consumed by the solver to seed the master.
        self.subproblem_initial_cuts = {}
    
    def _require_generating_function(self):
        if self.generating_function is None:
            raise ValueError("generating_function is required for penalized SAA operations.")
        return self.generating_function

    def _get_subproblem_env(self, scenario_id, default_params, parallel):
        """Return the Gurobi env to use for the given subproblem.

        If a pool of pre-created environments (tokens) was supplied via
        ``subproblem_grb_envs`` it is reused. The pool may be SMALLER than the
        number of subproblems: subproblems are mapped onto the pool round-robin
        (``scenario_id % pool_size``) so that the number of tokens held equals
        the number of concurrent worker threads (typically the CPU count)
        rather than one token per subproblem. Subproblems sharing an env are
        solved sequentially by the Benders solver (grouped by env) to respect
        Gurobi's lack of thread-safety for concurrent optimization.

        Otherwise fall back to the previous behaviour: acquire a fresh token
        when running in parallel, or share the agent's main env when serial.
        """
        if self.subproblem_grb_envs is not None:
            if len(self.subproblem_grb_envs) == 0:
                raise ValueError("subproblem_grb_envs was supplied but is empty.")
            return self.subproblem_grb_envs[scenario_id % len(self.subproblem_grb_envs)]
        if parallel:
            return acquire_grb_env(default_params, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)
        return self.grb_env
    
    def _initialize_sample_paths(self, sample_path_number):
        proposal = self.sample_path_proposal
        if proposal is None:
            raise ValueError("sample_path_proposal is required for sample-path sampling.")
        delta, lengths = proposal.sample_arrival_paths(
            arrival_generator=self.arrival_generator,
            size=sample_path_number,
        )
        period_likelihood_ratios = proposal.period_likelihood_ratios(
            target_discount_factor=self.discount_factor,
            lengths=lengths,
        )
        return delta, period_likelihood_ratios
    
    def _get_period_likelihood_ratio(self, scenario_id, zero_based_period_index):
        if not self.period_likelihood_ratios:
            return 1.0
        return float(self.period_likelihood_ratios[scenario_id][zero_based_period_index])

    def _build_importance_sampling_info(self):
        proposal = self.sample_path_proposal
        info = {
            'proposal_type': type(proposal).__name__,
            'target_discount_factor': float(self.discount_factor),
        }
        proposal_discount_factor = getattr(proposal, 'discount_factor_proposal', None)
        if proposal_discount_factor is not None:
            info['proposal_discount_factor'] = float(proposal_discount_factor)
        lengths = np.asarray([len(path) for path in self.delta], dtype=int)
        if lengths.size > 0:
            info['sample_path_length_stats'] = {
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'mean': float(lengths.mean()),
            }
        return info
    
    def regression_train(self, X, Y, regularization=1e-8, coefficient_bound=GRB.INFINITY, verbose=False):
        """Fit the coefficients theta of V_theta(s) = sum_k theta_k * phi_k(s).

        Solves the (ridge-regularized) least-squares problem

            min_theta (1/N) * sum_i (V_theta(s_i) - y_i)^2 + regularization * ||theta||^2

        as a Gurobi QP, where ``X`` is an iterable of states and ``Y`` the
        corresponding value-function targets V(s_i). The fitted coefficients
        are written into ``self.generating_function`` so that subsequent calls
        to ``approx_Q_solve`` use the trained value-function approximation.
        The small ridge term keeps the QP well-posed and pins basis weights
        with zero features (e.g. action blocks) to zero.

        Returns ``(coefficients, training_mse)``.
        """
        generating_function = self._require_generating_function()
        Y = np.asarray(Y, dtype=float).reshape(-1)
        if len(X) != len(Y):
            raise ValueError(f"X and Y must have the same length, got {len(X)} and {len(Y)}.")
        if len(Y) == 0:
            raise ValueError("The training dataset is empty.")
        model = gp.Model("Value_Function_Regression", env=self.grb_env)
        model.setParam("OutputFlag", 1 if verbose else 0)
        theta_vars = generating_function.get_coefficient_var(model=model, coefficient_bound=coefficient_bound)
        coefficient_blocks = generating_function.get_coefficients(theta_vars)
        residual_vars = model.addMVar(shape=len(Y), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="residual")
        for i, state in enumerate(X):
            prediction = generating_function.calculate_state_value(state, is_var=True, coefficients=coefficient_blocks)
            model.addConstr(residual_vars[i].item() == prediction - Y[i], name=f"fit_{i}")
        objective = (residual_vars @ residual_vars) / len(Y)
        if regularization > 0:
            objective = objective + regularization * (theta_vars @ theta_vars)
        model.setObjective(objective, GRB.MINIMIZE)
        if not solve_and_handle_errors(model, verbose=verbose):
            raise RuntimeError("Value function regression failed to solve.")
        self.coefficients = np.asarray(theta_vars.X).tolist()
        training_mse = float(np.mean(np.square(np.asarray(residual_vars.X))))
        model.dispose()
        generating_function.set_coefficients(self.coefficients)
        self.is_trained = True
        # Invalidate any previously built decision model so it is rebuilt with
        # the freshly trained coefficients.
        self.decision_model, self.state_linking_constraints, self.action_var = None, None, None
        return self.coefficients, training_mse
    
    def build_state_linking_constraints(self, model, state_var):
        u_var, v_var, w_var = state_var
        linking_constraints = []
        for j, uj_var in enumerate(u_var):
            constraint = model.addConstr(uj_var == 0.0, name=f'link_u_{j}')
            linking_constraints.append(constraint)
        for j, vj_var in enumerate(v_var):
            constraint = model.addConstr(vj_var == 0.0, name=f'link_v_{j}')
            linking_constraints.append(constraint)
        for i, wi_var in enumerate(w_var):
            constraint = model.addConstr(wi_var == 0.0, name=f'link_w_{i}')
            linking_constraints.append(constraint)
        return linking_constraints
    
    def build_action_linking_constraints(self, model, action_var):
        x_var, y_var = action_var
        linking_constraints = []
        for j, row in enumerate(x_var):
            for i, var in enumerate(row):
                constraint = model.addConstr(var == 0.0, name=f'link_x_{j},{i}')
                linking_constraints.append(constraint)
        for j, var in enumerate(y_var):
            constraint = model.addConstr(var == 0.0, name=f'link_y_{j}')
            linking_constraints.append(constraint)
        return linking_constraints

    def decision_model_builder_fn(self):
        model = gp.Model(f"Decision_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        model.setParam("NonConvex", 2)
        generating_function = self._require_generating_function()
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        future_cost = self.penalty_ratio * generating_function.calculate_expected_continuation_value(state_var, action_var,is_var=True)
        model.setObjective(imm_cost + self.discount_factor * future_cost, GRB.MINIMIZE)
        return model, state_linking_constraints, action_var, {}
    
    def approx_Q_solve(self, state, t, action=None, verbose=False):
        if self.decision_model is None:
            self.decision_model, self.state_linking_constraints, self.action_var, self.info = self.decision_model_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        if action is not None:
            self.set_action(action_var=self.action_var, action=action)
        # Clean solution before resolving
        self.decision_model.reset()
        if not solve_and_handle_errors(self.decision_model, verbose=verbose):
            raise RuntimeError("Direct model optimal solution not found")
        # ---------- 8. return ----------
        action = self.get_solution(self.action_var, is_final=True)
        return self.decision_model.ObjVal, action, self.info
    
    def train_master_builder_fn(self, coefficient_bound=GRB.INFINITY):
        master_model = gp.Model(f"SAA_train_Master", env=self.grb_env)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        master_model.setParam("Method", 1)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        # theta_vars = np.array(
        #     [master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e10, name=f"eta_{omega}") for omega in range(len(self.delta))])
        theta_vars = master_model.addMVar(shape=self.sample_path_number, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e8, name="theta")
        z = theta_vars.sum() / self.sample_path_number
        coefficient_vars = self.generating_function.get_coefficient_var(model=master_model, coefficient_bound=coefficient_bound)
        master_model.setObjective(z, GRB.MAXIMIZE)
        # Gurobi finalizes objective sense on update; do this before Benders reads ModelSense.
        master_model.update()
        return master_model, coefficient_vars, theta_vars

    def train_subproblem_builder_fn(self, env, scenario_id, init_state = None):
        # This builder is thread-safe: it never mutates shared state on
        # self.generating_function. The penalty coefficients are kept in a LOCAL
        # variable (coefficient_blocks) and passed explicitly into
        # calculate_penalty, so concurrent builds on distinct Gurobi envs do not
        # clobber each other. (Scenarios sharing an env are still serialized by
        # _build_training_workers, since a Gurobi env is not thread-safe.)
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
        # TRUE-LP FORMULATION. For fixed penalty coefficients this subproblem is
        # a pure LP: the env cost is linear and the penalty is
        # `coefficient . (linear function of the decisions)` (see
        # LinearPenaltyFunction.calculate_penalty). We therefore build it with
        # the coefficients as NUMERIC CONSTANTS (no coefficient variables, no
        # coefficient linking constraints), so Gurobi sees a genuine LP instead
        # of the bilinear `coefficient_var * decision_var` QP that the old
        # builder produced. The master action (the coefficient vector) enters
        # only through the OBJECTIVE, rebuilt each Benders iteration by
        # `objective_builder_fn`; the Benders cut gradient is the penalty feature
        # vector phi(x*) returned by `cut_gradient_fn`. That gradient is exactly
        # the dual of the old `coefficients == action` linking constraint
        # (envelope theorem: d ObjVal / d action_k = phi_k(x*)), so the cut is
        # identical to the previous formulation -- but the LP can be solved and
        # warm-started by simplex, which the QP could not.
        #
        # Method strategy: solve the cold (build-time) model with barrier +
        # crossover, which is fast on a large LP and yields a simplex BASIS;
        # then switch to primal simplex for every warm re-solve. Each iteration
        # only changes the objective, so the retained basis stays primal-feasible
        # and primal simplex re-optimizes in a few pivots (LPWarmStart=2).
        sub_model.setParam("Method", 2)       # barrier for the one cold solve
        sub_model.setParam("Crossover", 1)    # crossover -> usable simplex basis
        sub_model.setParam("LPWarmStart", 2)  # retain basis across re-solves
        sub_model.setParam('InfUnbdInfo', 1)
        sub_model.setParam('NumericFocus', 2)
        # Forbid Gurobi from reporting the ambiguous INF_OR_UNBD status.
        sub_model.setParam('DualReductions', 0)
        sub_model.setParam("MultiObjPre", 0)
        sub_model.setParam("FeasibilityTol", 1e-9)
        sub_model.setParam("OptimalityTol", 1e-9)
        generating_function = self._require_generating_function()
        number_of_coefficients = generating_function.number_of_coefficients
        state = self.env.generate_initial_state() if init_state is None else init_state
        state_var = self.get_state_var(sub_model)
        state_linking_constraints = self.build_state_linking_constraints(sub_model, state_var)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
        self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
        # Build the LP once and pin the AFFINE penalty features to auxiliary
        # variables so that every Benders iteration touches only the objective
        # (no expression rebuild). The penalty for period t is
        #     lr_t * theta . phi_t(x),   phi_t(x) = tad_t * [post_regular;
        #         post_overtime; post_waitlist; advance_scheduling_flat;
        #         overtime_decision]
        # which is AFFINE in the decisions (see
        # LinearPenaltyFunction.calculate_gradient). Summed over periods,
        #     Phi(x) = sum_t lr_t * phi_t(x)
        # is accumulated symbolically into ``feature_vec`` (an MLinExpr) and then
        # bound to feature variables ``feature_var == Phi(x)`` below. The penalty
        # is then ``theta . feature_var``, so the per-iteration objective update
        # and Benders cut each touch only |coeff| entries -- as cheap as reading
        # the old QP's constraint .Pi, but warm-startable by primal simplex.
        T, K, W = self.env.planning_horizon, self.env.num_types, self.env.booking_window_size
        block_offsets = (0, T, 2 * T, 2 * T + K, 2 * T + K + W * K, number_of_coefficients)
        mean_by_type = self.env.arrival_generator.mean_by_type
        feature_vec = gp.MLinExpr.zeros(number_of_coefficients)

        cost_part = self.env.cost_fn(state_var, action_var, is_var=True)
        for period_index, new_arrival in enumerate(self.delta[scenario_id]):
            likelihood_ratio = self._get_period_likelihood_ratio(scenario_id, period_index)
            # phi_t enters Phi scaled by lr_t * total_arrival_difference (both
            # constants); accumulate it block-by-block into the feature vector.
            scale = likelihood_ratio * float(np.sum(mean_by_type - new_arrival))
            if scale != 0.0:
                post_rb, post_ot, post_wl = self.env.post_action_state(state_var, action_var, is_var=True)
                advance_scheduling_decision, overtime_decision = action_var
                feature_vec[block_offsets[0]:block_offsets[1]] += scale * post_rb
                feature_vec[block_offsets[1]:block_offsets[2]] += scale * post_ot
                feature_vec[block_offsets[2]:block_offsets[3]] += scale * post_wl
                feature_vec[block_offsets[3]:block_offsets[4]] += scale * advance_scheduling_decision.reshape(-1)
                feature_vec[block_offsets[4]:block_offsets[5]] += scale * overtime_decision
            state_var = self.get_next_state(model=sub_model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
            cost_part = cost_part + likelihood_ratio * self.env.cost_fn(state_var, action_var, is_var=True)

        # Introduce one auxiliary feature variable per coefficient, pinned to the
        # affine period-summed penalty feature:
        #     feature_var[k] == Phi_k(x) == (sum_t lr_t * phi_t(x))[k].
        # The penalty is then simply  theta . feature_var, so the action-dependent
        # part of the objective lives entirely on these |coeff| columns. Each
        # Benders iteration therefore only:
        #   * writes the |coeff| objective coefficients of feature_var, and
        #   * reads the |coeff| primal values feature_var.X == Phi(x*),
        # both independent of the ~n_vars LP size -- exactly as cheap as the old
        # QP's |coeff|-entry constraint ``.Pi`` read.
        feature_var = sub_model.addMVar(
            number_of_coefficients, lb=-GRB.INFINITY, name="penalty_feature")
        sub_model.addConstr(feature_var == feature_vec, name="penalty_feature_link")
        sub_model.update()

        def objective_builder_fn(model, action_values):
            """Set the LP objective for master coefficients ``action_values``.

            The action-independent decision cost (``cost_part``) is fixed at build
            time; the penalty is ``theta . feature_var``, so only the |coeff|
            objective entries of the feature variables change. One bulk ``Obj``
            write -- no expression rebuild and no n_vars-length update. The basis
            is retained, so primal simplex warm-starts in a few pivots."""
            feature_var.Obj = np.asarray(action_values, dtype=float).reshape(-1)

        def cut_gradient_fn(model, action_values):
            """Return the Benders subgradient phi(x*) = Phi(x*) = feature_var.X.

            Because feature_var == Phi(x) by construction, at the optimal primal
            solution feature_var.X is the period-summed penalty feature, i.e. the
            envelope-theorem gradient of ObjVal w.r.t. the action -- the exact
            value the old QP read from the linking constraint ``.Pi``, now a
            direct |coeff|-length primal read independent of the LP size."""
            return np.asarray(feature_var.X, dtype=float)

        # Cold objective corresponds to all-zero coefficients (penalty == 0),
        # which is exactly where the master starts, so the cold basis is a
        # near-perfect warm start for iteration 1.
        sub_model.setObjective(cost_part, GRB.MINIMIZE)

        # Pre-solve once at build time (barrier + crossover) so each worker
        # carries an initial simplex basis into the Benders loop. This solve and
        # the construction above run in parallel across env groups via
        # _build_training_workers.
        sub_model.Params.OutputFlag = 0
        cold_solve_start = time.time()
        sub_model.optimize()
        cold_solve_seconds = time.time() - cold_solve_start
        self.subproblem_cold_solve_seconds[scenario_id] = cold_solve_seconds

        # The cold solve ran at coefficients a = 0 (penalty == 0), so it already
        # yields this scenario's build-time Benders data for FREE:
        #   v0 = Q_s(0) = min cost             (subproblem value at a = 0)
        #   g0 = phi_s(x*(0)) = feature_var.X  (subgradient dQ_s/da at a = 0)
        # Because Q_s is concave in a, theta_s <= v0 + g0 . a is a globally valid
        # optimality cut. Caching it lets benders_decomposition_train SEED the
        # master with one cut per scenario before the first master solve, so the
        # solver never spends an iteration re-deriving the a = 0 cuts and the
        # first master action is gradient-informed rather than arbitrary.
        if sub_model.Status == GRB.OPTIMAL:
            self.subproblem_initial_cuts[scenario_id] = (
                float(sub_model.ObjVal),
                cut_gradient_fn(sub_model, np.zeros(number_of_coefficients)),
            )

        # Switch to primal simplex for all subsequent Benders re-solves. Each
        # iteration changes only the objective, so the retained crossover basis
        # is still primal-feasible and primal simplex warm-starts in a handful of
        # pivots -- the LP equivalent of what the bilinear QP could never do.
        sub_model.setParam("Method", 0)
        if self.verbose:
            print(
                f"Subproblem {scenario_id}: LP build, cold solve "
                f"{cold_solve_seconds:.2f}s; warm re-solves use primal simplex"
            )

        return sub_model, None, objective_builder_fn, cut_gradient_fn

    def _resolve_init_state_per_scenario(self, init_state):
        """Return a resolver ``sid -> initial state (or None)`` for the workers.

        ``init_state`` may be one of:
          * ``None`` -- every scenario draws its own initial state from the env;
          * a single ``(regular, overtime, waitlist)`` state -- shared by all
            scenarios;
          * a list with one state per sample path
            (``len == sample_path_number``) -- scenario ``sid`` uses
            ``init_state[sid]``.

        A per-scenario list is told apart from a single shared state by
        structure (a single state is a length-3 sequence of flat numeric
        vectors), so the two forms stay unambiguous even when
        ``sample_path_number == 3``.
        """
        is_per_scenario_list = (
            isinstance(init_state, (list, tuple))
            and not is_single_init_state(init_state)
        )
        if not is_per_scenario_list:
            return lambda sid: init_state
        if len(init_state) != self.sample_path_number:
            raise ValueError(
                "A per-scenario init_state list must have length "
                f"sample_path_number ({self.sample_path_number}); got "
                f"{len(init_state)}."
            )
        return lambda sid: init_state[sid]

    def _build_training_workers(self, init_state=None, parallel=True, verbose=False):
        """Build one ``SubproblemWorker`` per sample path for Benders training.

        Both the model construction and the one-shot cold barrier+crossover
        solve run in parallel: ``train_subproblem_builder_fn`` is thread-safe
        (it touches no shared mutable state). The only restriction is that a
        Gurobi ``Env`` is not thread-safe for concurrent use, so scenarios
        assigned to the same env are built sequentially within a group while
        different env groups run concurrently. Workers are returned ordered by
        ``scenario_id``.

        ``init_state`` selects each scenario's starting state -- shared
        (``None`` or a single state) or one per sample path; see
        :meth:`_resolve_init_state_per_scenario`. Resolving it up front keeps the
        parallel build deterministic (no RNG calls during the threaded
        construction).
        """
        envs = {sid: self._get_subproblem_env(sid, {"Threads": 1}, parallel)
                for sid in range(self.sample_path_number)}
        init_state_for = self._resolve_init_state_per_scenario(init_state)

        # Group scenarios by their Gurobi env (None -> own private group).
        groups = defaultdict(list)
        for sid, env in envs.items():
            groups[id(env) if env is not None else sid].append(sid)

        def build_one(sid):
            start = time.time()
            model, link_rows, objective_builder_fn, cut_gradient_fn = self.train_subproblem_builder_fn(
                env=envs[sid], scenario_id=sid, init_state=init_state_for(sid))
            worker = SubproblemWorker(
                model=model, link_rows=link_rows, state_linking_constraints=None,
                subproblem_id=sid, objective_builder_fn=objective_builder_fn,
                cut_gradient_fn=cut_gradient_fn, verbose=verbose, grb_env=envs[sid],
                initial_cut=self.subproblem_initial_cuts.get(sid))
            print(f'Finished build {sid} with sample path length '
                  f'{len(self.delta[sid])} in {time.time() - start} seconds')
            return worker

        def build_group(sids):
            return {sid: build_one(sid) for sid in sids}

        workers = {}
        if parallel and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=len(groups)) as executor:
                for future in [executor.submit(build_group, sids) for sids in groups.values()]:
                    workers.update(future.result())
        else:
            for sids in groups.values():
                workers.update(build_group(sids))

        return [workers[sid] for sid in range(self.sample_path_number)]

    def benders_decomposition_train(self,
                                    coefficient_bound=GRB.INFINITY,
                                    init_state = None,
                                    parallel=True,
                                    verbose=False,
                                    checkpoint_path=None,
                                    resume_checkpoint_path=None):
        # This function can be implemented to train the coefficients using Benders decomposition, which can potentially handle larger sample sizes more efficiently.
        # ``init_state`` may be None (each scenario draws its own state), a single
        # shared state, or one state per sample path (a list of length
        # sample_path_number); see ``_resolve_init_state_per_scenario``.
        overall_start = time.time()
        master_time = None
        workers_time = None
        coefficient_vars = None

        if self.coefficient_model is None:
            # Time master model building
            master_start = time.time()
            master_model, coefficient_vars, theta_vars = self.train_master_builder_fn(coefficient_bound)
            master_time = time.time() - master_start
            if verbose:
                print(f"[TIMING] Master model building: {master_time:.2f}s")
            
            # Time worker building
            workers_start = time.time()
            workers = self._build_training_workers(init_state=init_state, parallel=parallel, verbose=verbose)
            workers_time = time.time() - workers_start
            if verbose:
                print(f"[TIMING] Workers building (parallel={parallel}): {workers_time:.2f}s")
            
            self.coefficient_model  = BendersDecompositionSolver(master_model=master_model,
                                                                 workers=workers,
                                                                 imm_cost=None,
                                                                 theta_vars=theta_vars,
                                                                 action_vars=coefficient_vars)
        else:
            coefficient_vars = self.coefficient_model.action_vars
        # init_solution = [0] * coefficient_vars.shape[0]
        init_solution = None
        # init_solution = [30.936873875657064, 18.797811208402102, 32.55294582735802, 9.771833520893624, 29.930188363086305, -33.43114011845465, 20.429547473135614, 46.134539415653634, -4.1252674656013415, 13.548626019564873, 2.0323452690382933, 4.948082094437374, 57.93083271225687, -34.489745014299935, -22.752348538218, 43.54071439788093, 31.466967896430372, 13.378259245251227, 19.08776734394266, 59.071042428052856, 25.288221363707603, -85.09869832626279, 47.38011323828022, 13.324993447555471, 120.27618769746321, -146.4427825660298, 53.81124295809613, 62.20144296020482, 52.47460135187192, -20.605921066845603, -70.34995701361875, 163.51281510396765, 6.60533255004948, -113.00288841418705, 33.181652159801615, -138.30844267307205, 204.2166397682823, -108.06934994312908, 223.1596242294453, 54.34585807530416, -114.68916798053644, 53.779549516198095, 62.6890753579888, -132.9607322609606, -101.87111132543069, 188.38237324416775, -185.36068477897135, 160.72134253398778, -34.87783019457539, -165.09582122268893, 309.6723857352444, -281.3573729867282, 20.634822124277154, -4.330038407101594, -545.9479495163265, -9429.039169810098, 3.5409097151355686, 42.72230049696095, 4.050174429738009, 35.670251097571274, 36.98752627610553, -14.842276278996323, 49.913767857586976, 54.66824591965885, -52.530284161674125, -28.823009401464518, -57.652652547646206, 19.212696907575882, 76.85986613224023, 5.160918430492718, 39.50172476926792, 25.50910600866467, 7.7759334133560944, 2.0089986019809536, -30.58188582023919, 65.38994859008166, 36.671374472605386, -90.26366448958886, 105.78083415346698, -0.5648379682072561, 128.25828155090278, -156.0041086314322, 38.635648126459955, 20.38702052987313, 80.93143512215346, 3.815132126224707, -92.46317941391825, 197.96664881356114, -25.68888667004868, -12.328857914966688, 26.87900972002935, -129.8723021445483, 169.5594313556061, -73.65427536077355, 275.1945927950061, 124.6267210774115, -115.38995593402508, -0.8009833975152213, 170.7246728785445, -182.7657259493173, -38.18085368994055, 194.01171012567798, -94.2747443869524, 130.44227756106247, -66.84024405114174, -236.27371833168309, 78.66270113673475, -290.8761487468625, 559.6819883801721, 10.752569463662397, -568.3666354164685, 460.2991655053739, 158.37588443883902, -193.9216145621594, -66.55045841285988, -207.6933204550911, 884.6387736746859, 834.2162153685704, 103.98741130638292, 188.55833726572166, -148.4154591226984, 458.8053411903486, 97.82009269273522, -683.2441095608922, 249.08070077390957, 1555.0920357621142, 499.3936371428386, 636.4734869789855, 1159.886874217052, 1223.5386895740824, -1367.153207606466, -1595.6422893171573, -888.3440243358016, -422.17810791305544, -194.8461339958101, -318.43054212783505, 58.16660782382519, 945.9108164251008, 419.2529470909686, -75.06257225194535, -170.04139570753057, 1393.419184961684, 336.99120549569545, 243.60365082069922, 63.47063277731668, -2548.2971870188553, -56.59358071192403, -3898.125694572405, -119.945980678244, 1594.5858996684306, -1826.83311836072, 511.656664330189, 285.070548812395, 15.302270514760435, 783.1727251367164, -169.97055458293417, 30.50868250841404, -1419.3633857400755, 1978.7188485120134, -1237.17936359673, -299.7016084181448, 1133.7911125485473, 1626.7039614341609, -2791.7136436981596, 4932.110023795547, -1621.1935010611166, 1614.62808053739, -2684.6843105273883, -1834.1556419946737, -265.66105234726945, -1100.7602200001638, -43.19070738122594, -1015.8758240434228, -340.1657922735167, -1056.001519347583, -17.93079478161714, 1418.8189946868968, -1178.9843979360733, 533.1787579539204, -654.9609687922073, 309.5191379943023, -556.9154454676293, 3584.4264961308263, 1196.1117601494518, 664.956737746279, -802.0049723400815, -1152.795533244306, -310.3956584546715, -736.2714576859572, 211.31980942983816, -127.2607825414034, -494.4078690542375, -525.3473411965666, -732.8989818387079, 1481.9590664519344, 634.6806703577009, 134.6879987539506, 68.46968101891441, 3077.8622619915873, -1322.1970354819734, 512.0970780186783, 1245.7729699397166, -719.9155689609499, -1621.7511109563402, -750.1924173015896, -113.46683923530412, -169.61181143274712, -849.4577694557283, -202.98673572341264, -100.73746581099192, -145.70199903807247, -1877.1485745988907, 1424.9783136572046, -558.9686778396147, 1013.5386877376707, 429.5499314608378, -339.5011345735725, -422.5436391726017, -673.4762743015218, 2012.8666975530616, 978.2589217291377, 3172.189889204137, -1950.8414941347573, -177.7446453964339, -179.30227820592762, 33.51722879684361, -96.3454538689637, -256.65201206859786, -797.7019125809245, 200.6187091872159, 1008.0988339893294, -184.7277839675159, 210.7837975453668, -1666.6314832114995, 708.2979510251037, 298.1014440671508, 1362.881208995362, -1777.4202962087054, -602.6618168905517, 246.82127551917725, 2733.56739243182, -614.8762690171258, -562.0176945355186, -1840.366957509147, 6.300787588450724, -168.43747666850498, -107.4507641143353, 67.16014554401575, -1040.0810126884169, 1176.994223922966, -365.19203949432733, 1476.6907010629748, -622.3393092792348, 201.3979366409967, 867.4210477363565, -1947.1795033660965, -2124.9113409569673, -1348.1163150065315, -3049.156120283863, -64.26144623664425, -884.6054174034084, -747.6023216699457, -162.44191382329703, -336.3415576460618, -517.8905707558905, -1074.5696036563556, -656.891088093221, -942.1367976911298, 577.3087909632856, -1181.118838679084, -1502.0435839333647, 671.5980199618674, 193.16061819399056, -2786.47090227951, -1306.9089548468223, 852.8301044086406, -1302.224904385813, -365.3136254213454, 271.4708918756479, 803.9348108757275, 135.4272666347038, -195.44298380793145, 455.11977924363435, -88.76208815919365, -386.26451193944445, -1710.604457714806, 504.8046630280259, 436.7521130815965, -859.4393271359357, 243.42496847226232, 571.5925307498603, -368.76125584879134, -664.5101243587413, -2589.091548645905, -3156.5659348645954, -131.34181113562016, 593.5046967988571, -811.8911866374549, 51.22674394003911, -339.02008735168585, -245.10335194903865, 880.9973039628915, 1197.0678380355494, 1802.016595852356, -252.52625963731356, 1012.2671320087128, -204.94160854563643, -1089.7973361867819, -172.74467306160798, -2651.0334619461887, -6839.4918004310075, 784.4196371874947, -1170.2771159834117, -203.9784203434857, 655.6102521142936, -401.94938753332497, -309.32746400678286, 101.80668997192593, 448.39204659667644, 1111.3468105597963, 1319.0966574381764, -1208.9092976254644, -573.5167751989869, -908.1930329607058, -208.38052990026665, -1125.64513683742, 591.0173587930984, 2664.903442325281, -966.8707469666517, -3031.325021884568, -838.709739557322, -446.2868835568282, -850.6388486996685, -94.31674084351106, -19.070580089660293, -265.5034240234173, -793.4095053904985, 1510.6828589792303, 1618.531138143973, 673.3463076083483, 149.7474004104515, 1700.7353455994787, -107.45090930475713, -2917.3333534106296, -1896.7789996216147, -2672.789033777113, -181.71781053904746, 1124.4340396787939, -352.4730923468821, -83.94181618165813, 146.14487566285993, -54.73584703764347, -22.885619323205535, 367.8729495157054, 779.9052998295781, -934.3122290713162, -687.7701320571028, -1650.1609102906584, -1600.2096744658957, 2710.080368242038, 717.4665226361295, -40.71421466925588, 562.8942748669122, -1729.7452923672754, 480.29923516327557, 1865.6243513864613, 293.55308263106957, -433.0955032895382, 284.68396336058413, -576.146963705008, -447.8695286347708, 214.03040936304097, 2480.8180772126143, 55.251309514590616, 134.4546293320938, -1722.7339101804428, 820.8970772405369, 960.29222542825, -275.70321569254907, -1045.0418175086384, 1581.2753046832595, -6346.06265857456, -569.753195867397, 401.41532765688913, -2736.6430023314583, -184.37446659832415, 645.5012988466895, -787.6709263240222, 331.68374100012795, -0.6479521442944248, -1196.8429168055236, 1361.3667957557611, -439.9260662439809, 471.2435814815261, -1524.9493334236724, 306.73192480926315, 1405.3053110351905, -225.88895756671423, -1073.7983932649356, -3697.3755736504518, 1313.1719915872338, -1787.9433822421738, -1055.9662384214976, -344.2674554572283, 480.5497867772467, -401.63469274181057, -132.08782016089646, -113.06792199816887, -610.7462850622754, -570.706251138544, 134.04439385377407, -242.50379867228986, 1141.7066124539663, 726.2978033330783, -400.71446355254614, 990.5246738019534, -682.3572707952825, -2424.0201933059475, 385.29855549654735, -1452.2697167758374, -1234.3986673705163, -276.1457507214158, -180.90707045581195, -207.51250002460756, 474.33340429386413, -366.0105736851623, -156.65846087580408, 1761.139037081204, 1625.4558084790838, -2845.9189743448687, 866.9814741806542, -300.03082629473744, 40.90097763890692, 80.60506930927909, -657.3485269278951, -7502.349234869973, 2839.9554927345425, -343.1129155745543, -466.1732043276838, -259.48020006358223, 624.959438293353, 295.3300819742684, -324.2295717318382, -205.09570365338058, 216.8319159316121, -111.4475703152785, -2961.30246950438, 3.5970806512180378, 618.4868963159754, -172.65190086188164, -214.32370733063183, -1135.856103802185, 1607.3014365248878, 390.2826688005668, 622.9732840424609, 3225.2467499447466, -1785.4950347054846, -289.6105431865559, 228.58093926455103, -486.66798859216874, 331.9423895905381, -177.42511342302527, -580.4450372731009, 942.236750348567, -291.5343134274648, -2758.3542748332698, 1084.6285795089295, 1223.8005160593561, -184.147889323466, -3.0606979733181303, 709.3650861598024, -1467.0983744899718, -224.8782698194575, 3072.627086509512, -2608.3349615407533, -142.42374746249786, 507.75710013283634, -489.29113819279746, 366.563873607009, 249.3278768026298, -543.9822293316092, 988.2370033731124, -1150.624935985006, -884.4461500883978, -705.7819892536902, 67.02536691707697, -622.1388670131668, 9370.354244628512, 10000.0, -1284.387766343003, -43.83498594066923, -336.64051409717797, 76.80023632317157, 48.03185988238591, -97.959637054034, -20.066850268537664, 7.2169875455739305, -22.592328370140326, -16.302748965983394, 74.9041244429616, 134.32948935945322, 24.26064427462712, -71.579036855488, -34.67686324771993, -81.22386745965369, 34.97724202853562, 23.86161316037356, 78.48970745918116, 30.681211865345162, -51.54466078376435, 44.85999301534014, 54.94733072019185, -76.71202199176457, 9.631051612768688, 23.09136082112512, 61.60971403449678, 29.1867274453751, 110.7271508084791, -31.6304959353005, 119.9573825603302, -74.78749537832353, -65.5523035943056, -65.49342029477253, -102.44050265334715, -39.29540855231676, -124.12120150830569, -47.3855370243699, -106.74659128220623, -24.687965332567313, 108.83072031276096, -43.66847842195264, 285.6193246053477, -121.11212898998315, 178.66430447794028, 30.557917272950732, -146.216172243583, -225.0740034784603, 69.71566959773938, 340.3924192121032, 215.46725584117667, 237.22917603930162, -67.4859346202273, -610.9477512907654, 324.88956279581816, 601.4716391165779, -10000.0]
        # Time solver
        solver_start = time.time()
        upper_bound, info = self.coefficient_model.solve(init_solution=init_solution, is_hard_bound=True, max_iter=1500, parallel=parallel, verbose=verbose, checkpoint_path=checkpoint_path, resume_checkpoint_path=resume_checkpoint_path, min_norm_action=True)
        solver_time = time.time() - solver_start
        if verbose:
            print(f"[TIMING] Solver execution (parallel={parallel}): {solver_time:.2f}s")
        
        info = dict(info or {})
        info['importance_sampling'] = self._build_importance_sampling_info()
        
        # Add timing info
        overall_time = time.time() - overall_start
        info['timing'] = {
            'overall': overall_time,
            'master_model': master_time,
            'workers_building': workers_time,
            'solver': solver_time,
            'parallel': parallel
        }
        if verbose:
            print(f"[TIMING] Total training time: {overall_time:.2f}s (parallel={parallel})")
        
        self.coefficients = np.asarray(coefficient_vars.X).tolist()
        self.is_trained = True
        generating_function = self._require_generating_function()
        generating_function.set_coefficients(self.coefficients)
        return upper_bound, self.coefficients, info

    def calculate_information_relaxation_cost(self, state, sample_path, verbose=False, period_weights=None):
        model = gp.Model(f"IR_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        # add action constraint
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        # ---------- 1. objective ----------
        # Rolling the relaxation over ``sample_path`` visits ``len(sample_path) + 1``
        # decision periods. ``period_weights[t - 1]`` reweights period ``t`` by
        # ``gamma ** (t - 1) / P_proposal(L >= t)`` so this bound stays consistent
        # with the (identically reweighted) executed-policy cost when the sample
        # path was drawn from an importance-sampling proposal instead of the
        # target geometric horizon. ``None`` leaves every weight at 1, recovering
        # the plain relaxation.
        if period_weights is None:
            weights = np.ones(len(sample_path) + 1)
        else:
            weights = np.asarray(period_weights, dtype=float)
        imm_cost = weights[0] * self.env.cost_fn(state_var, action_var, is_var=True)
        future_cost = 0
        generating_function = self._require_generating_function()
        # for every sample path
        for period_index, new_arrival in enumerate(sample_path):
            penalty = self.penalty_ratio * generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
            state_var = self.get_next_state(model=model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
            future_cost += weights[period_index] * penalty + weights[period_index + 1] * one_time_cost
        model.setObjective(imm_cost + future_cost, GRB.MINIMIZE)
        if not solve_and_handle_errors(model, verbose=verbose):
            raise RuntimeError("Direct model optimal solution not found")
        return model.ObjVal
    
    def hindsight_master_builder_fn(self):
        model = gp.Model("Penalized_Hindsight_Master", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        theta_vars = model.addMVar(shape=self.sample_path_number, vtype=GRB.CONTINUOUS, lb=-1e10, name="theta")
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        model.setObjective(imm_cost + theta_vars.sum() / self.sample_path_number, GRB.MINIMIZE)
        model.update()
        return model, imm_cost, theta_vars, action_var, state_linking_constraints
    
    def hindsight_subproblem_builder_fn(self, env, scenario_id):
        model = gp.Model(f"Penalized_Hindsight_Subproblem_{scenario_id}", env=env)
        # IR-style parameters (same fidelity knobs as
        # calculate_information_relaxation_cost). The Benders optimality-cut
        # gradient is the link-row duals (.Pi), and Pi is available after ANY
        # optimal LP solve -- no InfUnbdInfo/DualReductions/NumericFocus needed.
        # Farkas rays for feasibility cuts are extracted LAZILY by
        # SubproblemWorker._get_feasibility_ray_for_link_rows, which flips
        # InfUnbdInfo/DualReductions/Method on the model only when a subproblem
        # actually comes back non-optimal, so we don't pay for ray bookkeeping
        # on every solve.
        #
        # Method=1 (dual simplex) is kept: with Threads=1 sub-envs the default
        # concurrent method degenerates to a single simplex anyway (benchmarked:
        # cold barrier+crossover is 10-25x slower here), and it guarantees
        # dual-simplex warm starts from the persisted basis on the RHS-only
        # re-solves between Benders iterations / decision epochs.
        model.setParam("Method", 1)
        model.setParam("MultiObjPre", 0)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        action_linking_constraints = self.build_action_linking_constraints(model, action_var)

        cost = 0
        generating_function = self._require_generating_function()
        for period_index, new_arrival in enumerate(self.delta[scenario_id]):
            likelihood_ratio = self._get_period_likelihood_ratio(scenario_id, period_index)
            penalty = self.penalty_ratio * generating_function.calculate_penalty(
                state_var,
                action_var,
                new_arrival,
                is_var=True,
            )
            cost += likelihood_ratio * penalty
            state_var = self.get_next_state(
                model=model,
                state=state_var,
                action=action_var,
                new_arrival=new_arrival,
            )
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            cost += likelihood_ratio * self.env.cost_fn(state_var, action_var, is_var=True)
        model.setObjective(self.discount_factor * cost, GRB.MINIMIZE)
        return model, action_linking_constraints, state_linking_constraints
    
    def hindsight_solve(self, state, t,
                        action=None,
                        tol=1e-9,
                        max_iterations=1000,
                        use_pareto_cuts=False,
                        pareto_epsilon=1e-4,
                        core_alpha=None,
                        parallel=False,
                        max_workers=None,
                        verbose=False):
        self._require_generating_function()
        master_model, imm_cost, theta_vars, action_vars, state_linking_constraints = self.hindsight_master_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        flatten_action_vars = flatten(action_vars)
        if action is not None:
            self.set_action(action_var=action_vars, action=action)

        if self.workers is None:
            self.workers = []
            for omega in range(self.sample_path_number):
                start = time.time()
                worker_env = self._get_subproblem_env(omega, {"Threads": 1}, parallel)
                sub_model, action_linking_constraints, worker_state_linking_constraints = self.hindsight_subproblem_builder_fn(
                    env=worker_env,
                    scenario_id=omega,
                )
                self.workers.append(SubproblemWorker(
                    model=sub_model,
                    link_rows=action_linking_constraints,
                    state_linking_constraints=worker_state_linking_constraints,
                    subproblem_id=omega,
                    grb_env=worker_env,
                ))
                print(f'Finished build {omega} with sample path length '
                  f'{len(self.delta[omega])} in {time.time() - start} seconds')

        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            # worker.model.reset()

        benders_solver = BendersDecompositionSolver(
            master_model=master_model,
            workers=self.workers,
            imm_cost=imm_cost,
            theta_vars=theta_vars,
            action_vars=flatten_action_vars,
        )
        obj, info = benders_solver.solve_with_callback(
            tol=tol,
            max_iter=max_iterations,
            use_pareto_cuts=use_pareto_cuts,
            pareto_epsilon=pareto_epsilon,
            max_workers=max_workers,
            parallel=parallel,
            verbose=verbose,
        )

        action_t = self.get_solution(action_vars, is_final=True)
        if 'debug' in info or 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta),
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return obj, action_t, info
    
    def solve(self, state, t,
                    action=None,
                    tol=1e-9,
                    max_iterations=1000,
                    use_pareto_cuts=False,
                    pareto_epsilon=1e-4,
                    core_alpha=None,
                    parallel=True,
                    max_workers=None,
                    verbose=False):
        if self.solver_name == 'approx_Q':
            return self.approx_Q_solve(state, t, action=action, verbose=verbose)
        elif self.solver_name == 'approx_penalized_hindsight':
            return self.hindsight_solve(state, t, action=action, tol=tol, max_iterations=max_iterations, use_pareto_cuts=use_pareto_cuts, pareto_epsilon=pareto_epsilon, core_alpha=core_alpha, parallel=parallel, max_workers=max_workers, verbose=verbose)
        else:
            raise ValueError(f"Unsupported solver_name: {self.solver_name}")
    

if __name__ == '__main__':
    from experiments import get_config_by_type
    from generating_function import MulticlassLinearPenaltyFunction, LinearPenaltyFunction
    config = get_config_by_type('toy')
    env = config.env
    test_state = (np.array([5, 5, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1, 2]))
    test_action = (np.array([[0, 0],
       [0, 0],
       [1, 1],
       [0, 1],
       [0, 0],
       [0, 0],
       [0, 0]]), np.array([0, 0, 0, 0, 0, 0, 0]))
    coefficients = [-8.506673230703488, 20.803823305410205, 8.949007725191958, 7.035934826472743, 6.404644810547097, 6.152425741039451, 215.93249503160055, 1.6207846766501177, 8.822177032253531, 6.391172301334453, 5.765390772891202, 5.50902974495165, 5.495551566445247, -50.39648401261288, 0.9401940738773069, 17.37900433117047, 7.64933715141955, 6.100169675595545, 5.77057773706476, 5.652617245825845, 215.58932932336387, 0.7393353464647373, 7.139718662697062, 5.313030831788522, 5.010546356203956, 4.977777252780282, 4.9990414864448605, -50.7353596798275, -1821.3953403927717, 343.8795537537197, 37.55879181629673, 39.203762515666966, -1882.1410231036018, 420.5404093482836, -1877.8480092709772, 418.5805478627768, -1855.2227173131294, 419.05603024125645, -1856.1932945867225, 419.5402047398881, -1856.684018334899, 419.7162383810243, -1856.6242232883421, 419.7373714931838, -2485.5936935288996, 0.0, 11.917602076210022, -111.58132849100905, 12.07483450156613, -112.0223378787287, 18.392953186410587, -111.92085801812323, 20.67530043490731, -111.77122539770485, 21.239612150222747, -111.73123837761058, 20.45200112265013, -111.7523714897918, 190.27622164613697, 0.0, -5.72171663290405e-12, 1.4248189742238095, 0.2407531112679777, 0.15812277618748655, 0.1165364293994906, 0.10596987331830707, 0.0, -1.234773977816752e-12, 0.6824583695475585, 0.20696116719471824, 0.17714940526803324, 0.13121802252884512, 0.14178457863045593, 0.0]
    # coefficients = None
    # coefficients = [0]*len(coefficients)
    generating_function = MulticlassLinearPenaltyFunction(env=env, coefficients=coefficients)
    # generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    agent = ApproxQAgent(
            env=env,
            discount_factor=env.discount_factor,
            sample_path_number=256,
            generating_function=generating_function,
            sample_path_proposal=ArrivalGeneratorSamplePathProposal(),
            solver_name='approx_penalized_hindsight',
            is_trained=True
        )
    obj, coefficients, info = agent.benders_decomposition_train(verbose=False)
    # print("Trained coefficients:", coefficients) # 28123
    obj, action, info = agent.solve(test_state, t=1, action=test_action, verbose=False)
    print("Objective from Decision model:", obj)
    print('Action:', action)

    # information_relaxation_cost = agent.calculate_information_relaxation_cost(test_state, sample_path)
    # print("Information relaxation cost for sample path 0:", information_relaxation_cost)
    coefficients = [4.031106747948797, 3.731703603754795, 3.5110029706811474, 3.495309016916508, 3.4469374308773695, 3.620388023893611, 3.468970454934606, 3.4121752858627588, 3.420954726074342, 3.2911472532541666, 3.341244596389515, 3.153996981016462, 3.068661661893202, 2.8007719608922343, 2.6406469670509978, 2.5585743444116815, 2.714828543497788, 2.7906901430833386, 2.8381822968840424, 2.723075370537117, 2.7392035106768162, 2.8880411064365035, 2.8867249052855186, 2.8363971914968715, 2.5830434439212695, 2.5868757260868733, 2.4595544891471945, 2.616520659839807, 2.4821093774589826, 2.653224437046447, 2.182940076545492, 2.1995155948316096, 2.1037601654952596, 2.3654658480754733, 2.179703557096218, 2.5953502390566427, 2.165565702569438, 2.538463441982458, 2.8668958956350252, 2.4901683595107897, 2.6469799405986123, 2.3391017301619286, 2.483527936605242, 2.291399239364182, 2.3631739372613083, 2.3386523583067174, 2.4906574536507833, 2.339416753686237, 2.094345377136051, 2.4008308323791425, 2.718716807768942, 1.8077437358206225, 2.233975229284624, 2.525210771469574, 2.5985016563154204, -9673.529416648953, 2.5308210327457346, 2.4386721134233085, 2.5035348846759007, 2.465905006032699, 2.2962559039679036, 2.2896813510888023, 2.29080695259654, 2.220695191350387, 2.3091382091533887, 2.1536156958427455, 2.2343206542136613, 2.179743618498833, 2.1237630768155213, 1.9084598247227404, 1.8268965265760926, 1.8212737177746021, 2.110110685669497, 2.2948762379764958, 2.2916681453243655, 2.26883242119402, 2.207594017016163, 2.408146495348774, 2.393804850315064, 2.3308995020979637, 2.1867258567090175, 2.161072911658266, 2.180878039136587, 2.3192141203053325, 2.2227018308622064, 2.372184628755349, 1.9321579069455765, 1.9078986434506078, 1.8080094154065591, 2.187342514276679, 2.033164145253977, 2.472464460746778, 1.9513293452873768, 2.381388498668457, 2.7310687290973874, 2.4015202777864033, 2.5609463688579126, 2.2280847468464344, 2.5168502971664566, 2.4845325235928613, 2.5511188776763447, 2.5361634474120365, 2.733940758958852, 2.5491626283328515, 2.2550018805213767, 2.716197768395432, 3.0756269104658713, 2.179398483625846, 2.6157409906918474, 2.9031534201003524, 2.9433251026257494, 326.4308899441294, -306.80538922482265, -271.34200787771806, 366.8716962296621, 226.66343282684102, -256.36646528334677, 310.2671867390345, 215.7647398459776, -299.21558893725523, 292.9005647060494, 195.288815855778, -247.67218549613608, -344.21798125278656, 359.57431000450015, 401.8398527932586, 429.0719770921296, 346.5934142514343, 400.27383599266614, 514.9903043363447, -321.19610240285147, -276.34784729290004, 306.53825015476104, 194.19673952677476, -319.5605341423852, 224.79726636833402, 243.51460727998528, -319.8182811593342, 255.35870432894444, 200.1455209779324, -330.4558924765697, -369.519723909847, 295.445431954633, 222.9500956398424, 337.09888618023797, 296.5907140823456, 323.50446294334324, 314.9410519515495, -315.04802201804887, -281.6993863469015, 302.87866300521637, 194.41066068788496, -319.77796321651294, 225.16246610386042, 243.4663673411833, -320.0286379604695, 255.20314053410584, 200.01957353819307, -330.2293614733444, -369.53006847918004, 295.9340112723312, 224.2521708508757, 337.424256905706, 296.2411124725786, 323.02630061138007, 315.52310211153053, -313.62652754385635, -279.1328114943153, 301.94401440875845, 194.403795138669, -319.90004113772557, 225.38926890546827, 243.47361005955827, -319.9847143590632, 255.3263252766901, 199.994535362237, -329.8431731280798, -369.52644352539755, 296.0032945728999, 224.03461499156947, 337.99265968068175, 296.5647097922374, 323.26389182335515, 316.3730561977518, -313.293839431637, -279.40854062177095, 304.20183496928803, 194.49189713542364, -319.9903672654127, 225.62956586954897, 243.47361612210625, -319.96749077516324, 255.39325517434736, 200.0052266039802, -329.81174472187377, -369.52644055889687, 296.0271376404562, 224.33045232565928, 338.2909773619267, 296.920891578342, 323.3417336613584, 316.6813963227287, -312.5992670605574, -267.6591213466545, 305.12374750304843, 194.7079890811583, -320.0217092821167, 226.21855906336896, 243.57036051538307, -319.9058833485815, 255.67989138535995, 200.08981610724368, -329.7794359344207, -369.47806870610475, 296.29248874615405, 224.32954316912583, 339.11427102080415, 297.01625229783167, 323.3950273138471, 317.395570241064, -296.4742044035793, -251.51421795260285, 301.3966235484586, 194.71005551365306, -320.18115179712913, 226.1380782247943, 247.00384597050652, -318.47496133618915, 257.0892958925524, 201.3298353320679, -328.3751560088731, -373.04526024964434, 296.0316491440517, 223.9284189948903, 339.00830978644444, 297.6663251065329, 323.87314817754304, 316.84994252048637, -286.9825202348329, -323.5874077764347, 326.0737452740432, 194.64729985142003, -319.3790552675782, 225.68477402877033, 236.93739353427918, -320.50575257072524, 254.33190433829623, 199.09015429542706, -333.78975092384644, -359.29479063927283, 290.45605939650886, 221.6634928059666, 338.94583584764587, 297.9445008352086, 324.2857086475178, 316.71199901598084, -286.9760745140047, -337.8457599217927, 328.7436070755866, 194.97180798874797, -318.5770489578863, 226.43426138248105, 250.36875429684733, -321.55600483197304, 254.31932519830298, 201.72617795624683, -336.5621312063631, -363.36936440524005, 291.06341342431006, 220.7643250887013, 339.62778859725586, 298.8643347384277, 324.9218083351807, 317.4330357370545, -381.3747833066809, -214.71075970372476, 341.39101112372737, 195.40247657977307, -317.6716749417992, 227.39689191692923, 254.1991179843244, -317.13873712091663, 246.4219520056813, 205.0636013170315, -334.27806961802344, -391.0987734102637, 291.28688832782063, 220.32554616225207, 340.192070078152, 299.7124180712708, 326.22940624381226, 317.9637809752094, -263.21989921435124, -364.5592522165971, 221.75111963142263, 196.22970132951377, -316.7663692346614, 229.0876393500621, 217.17140838707564, -327.0128312789129, 244.70621275575104, 206.09965119390472, -337.1873816681, -389.7237290090816, 290.071149491514, 220.93346152421327, 341.3461336492219, 300.76514443407905, 327.1090414152786, 319.1624998651205, -260.2320600776875, -383.6721827797446, 359.3127802076815, 195.93982791944654, -316.97538512966275, 228.07363407186494, 261.91027474658404, -332.47096504574074, 260.25525944732544, 191.24437550603216, -344.0327691953935, -407.7562698064903, 303.070054119029, 219.8234398836721, 340.9802148615199, 300.38517771716397, 326.6927602781998, 318.54855783526, -241.9695610826293, -384.89181920379633, 367.7103762800107, 196.02403430671438, -316.6995142943906, 227.1132363986526, 200.15328237481663, -307.67933870334673, 261.98938845597513, 222.40827022492886, -347.6656775502779, -334.8277043240305, 281.40661339901635, 219.31275381491287, 341.0762225915205, 300.49053786450713, 326.3128154913302, 316.8009959659903, -232.8958174931886, -409.17682832794526, 375.42505230670395, 196.27488152448677, -316.3589655277174, 226.47041703675495, 269.0427701724766, -345.92387035948013, 233.62915652334596, 228.13370844021483, -353.5089173365395, -329.2852277204511, 297.89702513326665, 219.53755577327138, 341.2403286918179, 300.8828832749696, 326.8542833558331, 316.6696704965234, -222.7347069045045, -414.51269301807224, 156.71284135427777, 196.19442903777417, -315.85601334673265, 226.20195846200477, 185.4418087260783, -298.60056107239143, 255.22715798943682, 233.86673612580125, -320.38939234294594, -323.70743305860196, 274.77330563059695, 221.18097587186458, 340.84113927591534, 300.6380872915306, 326.2993145929304, 317.3159851294695, -446.610866509056, -424.40218811597697, 155.9224306880078, 196.2295893043938, -315.2170392238477, 226.11728776426753, 179.04063637673062, -353.40530485697855, 275.889496667176, 236.4045986375786, -356.0234755268375, -430.84254484893245, 325.4932924707846, 218.81161629350936, 340.4755132597875, 300.2415676907858, 326.15426899145496, 316.25261381471137, -459.1059613307116, -158.21721310038993, 143.31468210990715, 195.92866147807217, -315.58922347786756, 225.94383220354212, 278.23158211969894, -289.4601356805033, 282.6244522509787, 168.73931147355142, -361.65381584554416, -323.7899546369499, 332.18901539065155, 219.0588206835655, 340.7017934868236, 300.6046846416193, 326.0464241943155, 315.40667987886627, -453.4807105153832, -438.5590433975103, 403.9822956855551, 195.5352951937457, -315.9766934066083, 225.61740774226564, 280.2596942850996, -287.1638374410395, 211.36432597306884, 246.47706151702187, -353.80615954512905, -448.64937080991876, 252.61915750423395, 218.23023230235412, 340.39867049745953, 300.00349589547477, 325.34760759729943, 314.77687090868676, -463.23343845132695, -458.08091455052636, 121.49285921328737, 195.93111678779132, -316.35193314263415, 226.22857068504527, 283.0711116555831, -359.46772862923353, 270.2698890266456, 250.62404189683184, -367.5839260033572, -315.82404338092783, 334.19126756532387, 220.03964147389524, 340.08425200376223, 299.5134710017919, 325.1487809805167, 315.33214403895545, -185.04022584420272, -138.3397639065788, 136.34223229719646, 195.635953295503, -316.240837251391, 226.36034612650474, 156.80333081994831, -355.3315912267535, 211.8852006761208, 148.2341000580218, -368.1445445620848, -324.3179987112926, 307.56714837005893, 218.85535167963644, 339.7024775478858, 299.23326844959956, 323.9012485553758, 314.7862327472303, -490.93605874694003, -127.24776608662978, 416.99103199149977, 195.8772056598209, -317.1538239509464, 227.01423144270302, 286.07155248104755, -381.97586320203663, 199.9084016284105, 257.30752863435555, -306.14069447310976, -297.0821706456318, 329.60438098535997, 218.46360872037076, 340.45398745777493, 9975.143223895953, 10000.0, 314.92164683334886, -0.18641251824010396, 0.046290719305034145, 0.0031607664823241066, 0.25128624071476224, 0.3967118761047459, 0.5358909782862611, 0.4312080362196866, 0.4432720746790437, 0.44655219889682485, 0.5164361638708215, 0.6652429516871052, 0.6394356252567377, 0.7147533696752362, 0.638825213049131, 0.6000962962934864, 0.5301125312580552, 0.4608874889636354, 0.4069235188399034, 0.45780814403042314, 0.29987998362412327, 0.3060822743245808, 0.2355598802860186, 0.262854486822107, 0.2742365129688551, 0.3169300373920123, 0.3712599147893343, 0.23626546474406496, 0.28030342647434736, 0.2381436256946472, 0.2428196499531623, 0.14255539619080082, 0.2699427225506952, 0.28810051588698116, 0.049314146028336836, 0.15209257472815807, 0.011633836915279971, 0.09626139754800533, 0.012749241654091747, 0.1815818778013636, 0.062016769785259385, -0.06749448423397553, 0.05987557528715115, -0.06250573894612899, -0.25181591050750285, -0.20696394531114493, -0.10744918015552685, 0.006402166580301127, 0.06627076865152048, 0.08365854136172857, -0.12637532956978248, -0.17037963901020703, -0.13870334489001834, -0.4181660169742827, -0.33123044083731656, -0.29547006358734507, -10000.0]




