import copy
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from importance_sampling.proposals import ArrivalGeneratorSamplePathProposal
from importance_sampling.sample_path import sample_path_from_record
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
        (self.sample_paths, self.sample_path_weights, self.sample_path_strata) = self._initialize_sample_paths(sample_path_number)
        self.sample_path_number = len(self.sample_paths)
        self.penalty_ratio = penalty_ratio
        self.generating_function = generating_function
        self.decision_model, self.state_linking_constraints, self.action_t_var = None, None, None
        self.is_trained = is_trained
        self.coefficient_model, self.coefficients = None, None
        self.workers = None
        self._policy_model_signature = None
        self.solver_name = solver_name
        self.subproblem_grb_envs = subproblem_grb_envs
        self.verbose = verbose
        self.subproblem_cold_solve_seconds = {}
        self.subproblem_initial_cuts = {}
    
    def _require_generating_function(self):
        if self.generating_function is None:
            raise ValueError("generating_function is required for penalized SAA operations.")
        return self.generating_function

    def _ensure_policy_models_current(self):
        generating_function = self._require_generating_function()
        signature = (
            generating_function,
            tuple(generating_function.coefficient_vector()),
            float(self.penalty_ratio),
            float(self.discount_factor),
            self.current_decision_var_type,
            self.future_decision_var_type,
        )
        if signature == self._policy_model_signature:
            return
        if self.decision_model is not None:
            self.decision_model.dispose()
            self.decision_model = None
        for worker in self.workers or []:
            worker.model.dispose()
        self.workers = None
        self._policy_model_signature = signature

    def _get_policy_solution(self, action_vars):
        values = tuple(np.asarray(var.X, dtype=float).copy() for var in action_vars)
        if self.current_decision_var_type == GRB.INTEGER:
            return tuple(np.rint(value).astype(int) for value in values)
        return values

    def _get_subproblem_env(self, scenario_id, default_params, parallel):
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
        paths = proposal.sample_paths(
            arrival_generator=self.arrival_generator,
            size=sample_path_number,
            target_discount_factor=self.discount_factor,
        )
        return (paths,
                proposal.path_weights(sample_path_number),
                proposal.path_strata(sample_path_number))

    def _build_importance_sampling_info(self):
        proposal = self.sample_path_proposal
        info = {
            'proposal_type': type(proposal).__name__,
            'target_discount_factor': float(self.discount_factor),
        }
        proposal_discount_factor = getattr(proposal, 'discount_factor_proposal', None)
        if proposal_discount_factor is not None:
            info['proposal_discount_factor'] = float(proposal_discount_factor)
        lengths = np.asarray([path.length for path in self.sample_paths], dtype=int)
        if lengths.size > 0:
            info['sample_path_length_stats'] = {
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'mean': float(lengths.mean()),
            }
        return info
    
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
        model.setParam("MIPGapAbs", 1e-9)
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        model.setParam("NonConvex", 2)
        generating_function = self._require_generating_function()
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        future_cost = self.penalty_ratio * generating_function.calculate_expected_continuation_value(state_var, action_var,is_var=True)
        model.setObjective(imm_cost + self.discount_factor * future_cost, GRB.MINIMIZE)
        return model, state_linking_constraints, action_var, {}
    
    def approx_Q_solve(self, state, t, action=None, verbose=False):
        self._ensure_policy_models_current()
        if self.decision_model is None:
            self.decision_model, self.state_linking_constraints, self.action_var, self.info = self.decision_model_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        original_bounds = None
        if action is not None:
            self.decision_model.update()
            original_bounds = [(var.lb.copy(), var.ub.copy()) for var in self.action_var]
        try:
            if action is not None:
                self.set_action(action_var=self.action_var, action=action)
            self.decision_model.reset()
            if not solve_and_handle_errors(self.decision_model, verbose=verbose):
                raise RuntimeError("Direct model optimal solution not found")
            solved_action = self._get_policy_solution(self.action_var)
            return float(self.decision_model.ObjVal), solved_action, dict(self.info)
        finally:
            if original_bounds is not None:
                for var, (lower, upper) in zip(self.action_var, original_bounds):
                    var.lb = lower
                    var.ub = upper
                self.decision_model.update()
    
    def pathwise_terms(self, model, generating_function, form, path, state_var, action_var,
                       include_first_cost=True):
        gamma = self.discount_factor
        W = form.period_weights(path, gamma)
        tau = path.periods
        Phi = gp.MLinExpr.zeros(generating_function.number_of_coefficients)
        cost = W[0] * self.env.cost_fn(state_var, action_var, is_var=True) if include_first_cost else 0.0
        for s in range(tau):
            expected_weight, realized_weight = form.term_weights(s, tau, W, gamma, path.terminal)
            if s == tau - 1:
                generating_function.penalty_features(
                    state_var, action_var, None, expected_weight, 0.0, is_var=True).add_to(Phi)
                break
            arrival = path.arrivals[s]
            generating_function.penalty_features(
                state_var, action_var, arrival, expected_weight, realized_weight, is_var=True).add_to(Phi)
            state_var = self.get_next_state(model=model, state=state_var, action=action_var, new_arrival=arrival)
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            cost = cost + W[s + 1] * self.env.cost_fn(state_var, action_var, is_var=True)
        return cost, Phi, state_var, action_var

    def train_master_builder_fn(self, coefficient_bound=GRB.INFINITY, regularization=None,
                                regularization_scale=None):
        master_model = gp.Model(f"SAA_train_Master", env=self.grb_env)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        master_model.setParam("Method", 1)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        theta_vars = master_model.addMVar(shape=self.sample_path_number, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e8, name="theta")
        z = theta_vars @ self.sample_path_weights
        coefficient_vars = self.generating_function.get_coefficient_var(model=master_model, coefficient_bound=coefficient_bound)
        objective = z
        if regularization is not None:
            kind = str(regularization.get('type', '')).lower()
            lam = float(regularization.get('lambda', 0.0))
            if kind not in ('l1', 'l2'):
                raise ValueError(f"regularization type must be 'l1' or 'l2'; got {regularization.get('type')!r}")
            if lam < 0:
                raise ValueError(f"regularization lambda must be >= 0; got {lam}")
            size = coefficient_vars.shape[0]
            if regularization_scale is None:
                scale = np.ones(size)
            else:
                scale = np.asarray(regularization_scale, dtype=float).reshape(-1)
            if scale.shape != (size,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("regularization scale must be a positive finite vector with one entry per coefficient")
            scaled = scale * coefficient_vars
            if kind == 'l2':
                penalty = scaled @ scaled
            else:
                abs_vars = master_model.addMVar(size, lb=0.0, name="coefficient_abs")
                master_model.addConstr(abs_vars >= scaled, name="l1_abs_pos")
                master_model.addConstr(abs_vars >= -scaled, name="l1_abs_neg")
                penalty = abs_vars.sum()
            objective = z - lam * penalty
        master_model.setObjective(objective, GRB.MAXIMIZE)
        # Gurobi finalizes objective sense on update; do this before Benders reads ModelSense.
        master_model.update()
        return master_model, coefficient_vars, theta_vars

    def train_subproblem_builder_fn(self, env, scenario_id, init_state = None):
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
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
        cost_part, feature_vec, _, _ = self.pathwise_terms(
            sub_model, generating_function, generating_function.form('training'),
            self.sample_paths[scenario_id], state_var, action_var)

        feature_var = sub_model.addMVar(
            number_of_coefficients, lb=-GRB.INFINITY, name="penalty_feature")
        sub_model.addConstr(feature_var == feature_vec, name="penalty_feature_link")
        sub_model.update()

        def objective_builder_fn(model, action_values):
            feature_var.Obj = np.asarray(action_values, dtype=float).reshape(-1)

        def cut_gradient_fn(model, action_values):
            return np.asarray(feature_var.X, dtype=float)

        sub_model.setObjective(cost_part, GRB.MINIMIZE)

        sub_model.Params.OutputFlag = 0
        cold_solve_start = time.time()
        sub_model.optimize()
        cold_solve_seconds = time.time() - cold_solve_start
        self.subproblem_cold_solve_seconds[scenario_id] = cold_solve_seconds

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
        envs = {sid: self._get_subproblem_env(sid, {"Threads": 1}, parallel)
                for sid in range(self.sample_path_number)}
        init_state_for = self._resolve_init_state_per_scenario(init_state)

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
                  f'{self.sample_paths[sid].length} in {time.time() - start} seconds')
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

    def _regularization_scale(self, mode):
        size = self._require_generating_function().number_of_coefficients
        if mode in (None, 'none'):
            return np.ones(size)
        if mode != 'feature_std':
            raise ValueError(f"unknown regularization scale {mode!r}; use 'feature_std' or 'none'")
        missing = [sid for sid in range(self.sample_path_number) if sid not in self.subproblem_initial_cuts]
        if missing:
            raise ValueError(
                "feature_std regularization scale needs a build-time initial cut for every "
                f"scenario; missing for {missing[:5]}{'...' if len(missing) > 5 else ''}")
        gradients = np.array([self.subproblem_initial_cuts[sid][1] for sid in range(self.sample_path_number)], dtype=float)
        weights = np.asarray(self.sample_path_weights, dtype=float)
        mean = weights @ gradients
        std = np.sqrt(np.maximum(weights @ (gradients - mean) ** 2, 0.0))
        positive = std > 1e-12
        fallback = float(np.median(std[positive])) if positive.any() else 1.0
        return np.where(positive, std, fallback)

    def _prepare_training_checkpoint(self, checkpoint_path, resume_checkpoint_path):
        if not checkpoint_path and not resume_checkpoint_path:
            return
        solver = self.coefficient_model
        solver.master_model.update()
        for worker in solver.workers:
            worker.model.update()
        gf = self._require_generating_function()
        identity = {
            'version': 1,
            'gurobi_version': list(gp.gurobi.version()),
            'feature_family': f'{type(gf).__module__}.{type(gf).__qualname__}',
            'master': solver.master_model.Fingerprint,
            'workers': [worker.model.Fingerprint for worker in solver.workers],
            'scenario_weights': np.asarray(self.sample_path_weights).tolist(),
            'scenario_strata': np.asarray(self.sample_path_strata).tolist(),
        }
        if resume_checkpoint_path:
            stored_paths = (resume_checkpoint_path, *solver._checkpoint_paths(resume_checkpoint_path))
            if any(Path(path).exists() for path in stored_paths):
                identity_path = Path(f'{resume_checkpoint_path}.training.json')
                if not identity_path.is_file():
                    raise ValueError('training checkpoint has no model identity; restart without resume_checkpoint_path')
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError('training checkpoint is incompatible with the current models; restart without resume_checkpoint_path')
        if checkpoint_path:
            # Never attach a new identity to another problem's old cuts.
            if not resume_checkpoint_path or Path(checkpoint_path) != Path(resume_checkpoint_path):
                solver._reset_checkpoint_store(checkpoint_path)
            identity_path = Path(f'{checkpoint_path}.training.json')
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = identity_path.with_name(identity_path.name + '.tmp')
            temporary_path.write_text(json.dumps(identity, sort_keys=True))
            os.replace(temporary_path, identity_path)

    def benders_decomposition_train(self,
                                    coefficient_bound=GRB.INFINITY,
                                    init_state = None,
                                    parallel=True,
                                    verbose=False,
                                    checkpoint_path=None,
                                    resume_checkpoint_path=None,
                                    purge_after=30,
                                    regularization=None):
        overall_start = time.time()
        master_time = None
        workers_time = None
        coefficient_vars = None

        # Each call may change the initial states, coefficient bounds or
        # regularizer. Rebuild both sides rather than reuse incompatible models.
        if self.coefficient_model is not None:
            self.coefficient_model.master_model.dispose()
            for worker in self.coefficient_model.workers:
                worker.model.dispose()
            self.coefficient_model = None
        self.subproblem_initial_cuts.clear()
        self.subproblem_cold_solve_seconds.clear()
        workers_start = time.time()
        workers = self._build_training_workers(init_state=init_state, parallel=parallel, verbose=verbose)
        workers_time = time.time() - workers_start
        if verbose:
            print(f"[TIMING] Workers building (parallel={parallel}): {workers_time:.2f}s")

        regularization_scale = None
        if regularization is not None:
            regularization_scale = self._regularization_scale(regularization.get('scale', 'feature_std'))
        self._regularization_scale_vector = regularization_scale
        master_start = time.time()
        master_model, coefficient_vars, theta_vars = self.train_master_builder_fn(
            coefficient_bound, regularization=regularization, regularization_scale=regularization_scale)
        master_time = time.time() - master_start
        if verbose:
            print(f"[TIMING] Master model building: {master_time:.2f}s")

        self.coefficient_model = BendersDecompositionSolver(
            master_model=master_model, workers=workers, imm_cost=None,
            theta_vars=theta_vars, action_vars=coefficient_vars,
            scenario_weights=self.sample_path_weights, scenario_strata=self.sample_path_strata)
        self._prepare_training_checkpoint(checkpoint_path, resume_checkpoint_path)
        init_solution = None
        solver_start = time.time()
        # A regularizer already makes the optimum bounded/unique, so the
        # min-norm tie-break is unnecessary (and its objective guard would turn
        # into a QCP under L2).
        upper_bound, info = self.coefficient_model.solve(init_solution=init_solution, is_hard_bound=True, max_iter=3000, parallel=parallel, verbose=verbose, checkpoint_path=checkpoint_path, resume_checkpoint_path=resume_checkpoint_path, min_norm_action=(regularization is None), purge_after=purge_after)
        solver_time = time.time() - solver_start
        if verbose:
            print(f"[TIMING] Solver execution (parallel={parallel}): {solver_time:.2f}s")
        
        info = dict(info or {})
        info['importance_sampling'] = self._build_importance_sampling_info()
        
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
        
        # The solver stops when the cut-model gap at its last action is within
        # tol, so the last master action is the certified solution. The
        # reported objective is that action's subproblem-evaluated (in-sample
        # SAA) value, not the master objective, which overestimates it by the
        # stopping gap.
        self.coefficients = np.asarray(coefficient_vars.X).tolist()
        self.is_trained = True
        generating_function = self._require_generating_function()
        generating_function.set_coefficients(self.coefficients)
        objective = float(info.get('evaluated_value', upper_bound))
        if regularization is not None:
            # The solver objective carries -lambda R(theta); the in-sample lower
            # bound is the unregularized weighted SAA value at theta*.
            _, saa_objective = self.coefficient_model.evaluate_action(
                np.asarray(self.coefficients, dtype=float), parallel=parallel)
            scale_vector = getattr(self, '_regularization_scale_vector', None)
            info['regularization'] = {
                'type': regularization['type'],
                'lambda': float(regularization['lambda']),
                'scale_mode': regularization.get('scale', 'feature_std'),
                'scale': None if scale_vector is None else np.asarray(scale_vector, dtype=float).tolist(),
                'regularized_objective': float(upper_bound),
                'saa_objective': float(saa_objective),
            }
            objective = float(saa_objective)
            print(f"Regularized master objective {upper_bound:.4f}; unregularized SAA value at theta*: {saa_objective:.4f}")
        return objective, self.coefficients, info

    def calculate_information_relaxation_cost(self, state, sample_path, verbose=False, period_weights=None,
                                              terminal=None, first_action=None):
        path = sample_path_from_record(sample_path, period_weights, terminal)
        generating_function = self._require_generating_function()
        theta = generating_function.coefficient_vector()
        model = gp.Model(f"IR_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        if first_action is not None:
            self.set_action(action_var=action_var, action=first_action)
        cost, Phi, _, _ = self.pathwise_terms(
            model, generating_function, generating_function.form('evaluation'), path, state_var, action_var)
        model.setObjective(cost + self.penalty_ratio * (theta * Phi).sum(), GRB.MINIMIZE)
        if not solve_and_handle_errors(model, verbose=verbose):
            raise RuntimeError("Direct model optimal solution not found")
        return model.ObjVal

    def hindsight_scenario_weights(self, form):
        if form.hindsight_scenario_weights == 'kappa':
            return np.asarray(self.sample_path_weights, dtype=float)
        return np.full(self.sample_path_number, 1.0 / self.sample_path_number)
    
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
        weights = self.hindsight_scenario_weights(self._require_generating_function().form('hindsight'))
        model.setObjective(imm_cost + theta_vars @ weights, GRB.MINIMIZE)
        model.update()
        return model, imm_cost, theta_vars, action_var, state_linking_constraints
    
    def hindsight_subproblem_builder_fn(self, env, scenario_id):
        model = gp.Model(f"Penalized_Hindsight_Subproblem_{scenario_id}", env=env)
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

        generating_function = self._require_generating_function()
        theta = generating_function.coefficient_vector()
        cost, Phi, _, _ = self.pathwise_terms(
            model, generating_function, generating_function.form('hindsight'),
            self.sample_paths[scenario_id], state_var, action_var, include_first_cost=False)
        model.setObjective(cost + self.penalty_ratio * (theta * Phi).sum(), GRB.MINIMIZE)
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
        self._ensure_policy_models_current()
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
                  f'{self.sample_paths[omega].length} in {time.time() - start} seconds')

        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)

        benders_solver = BendersDecompositionSolver(
            master_model=master_model,
            workers=self.workers,
            imm_cost=imm_cost,
            theta_vars=theta_vars,
            action_vars=flatten_action_vars,
            scenario_weights=self.hindsight_scenario_weights(self.generating_function.form('hindsight')),
            scenario_strata=self.sample_path_strata,
        )
        solver_options = dict(tol=tol, max_iter=max_iterations,
                              use_pareto_cuts=use_pareto_cuts, pareto_epsilon=pareto_epsilon,
                              max_workers=max_workers, parallel=parallel, verbose=verbose)
        if self.current_decision_var_type == GRB.CONTINUOUS:
            # LP masters have no MIPSOL callbacks; use the existing cut loop.
            obj, info = benders_solver.solve(core_alpha=core_alpha, **solver_options)
        else:
            obj, info = benders_solver.solve_with_callback(**solver_options)

        action_t = self._get_policy_solution(action_vars)
        if 'debug' in info or 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode([path.arrivals for path in self.sample_paths]),
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
            solve_action = self.approx_Q_solve
            options = {'verbose': verbose}
        elif self.solver_name == 'approx_penalized_hindsight':
            solve_action = self.hindsight_solve
            options = dict(tol=tol, max_iterations=max_iterations, use_pareto_cuts=use_pareto_cuts,
                           pareto_epsilon=pareto_epsilon, core_alpha=core_alpha, parallel=parallel,
                           max_workers=max_workers, verbose=verbose)
        else:
            raise ValueError(f"Unsupported solver_name: {self.solver_name}")
        obj, solved_action, info = solve_action(state, t, action=action, **options)
        if action is not None:
            return obj, solved_action, info
        advance_scheduling_decision, solved_overtime = solved_action
        if self.current_decision_var_type == GRB.CONTINUOUS:
            overtime_decision = np.maximum(
                np.asarray(state[0]) + self.env.convert_action_to_booking_slots(advance_scheduling_decision)
                - self.env.regular_capacity, 0.0)
        else:
            overtime_decision = self.regular_first_overtime(state, advance_scheduling_decision)
        if np.array_equal(overtime_decision, solved_overtime):
            return obj, solved_action, info
        objective, executed_action, executed_info = solve_action(
            state, t, action=(advance_scheduling_decision, overtime_decision), **options)
        executed_info = dict(executed_info or {})
        executed_info['raw_objective'] = float(obj)
        executed_info['action_repaired'] = True
        return objective, executed_action, executed_info
    

if __name__ == '__main__':
    from experiments import get_config_by_type
    from generating_function import LinearPenaltyFunction
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
    generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
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
    obj, action, info = agent.solve(test_state, t=1, action=test_action, verbose=False)
    print("Objective from Decision model:", obj)
    print('Action:', action)

    coefficients = [4.031106747948797, 3.731703603754795, 3.5110029706811474, 3.495309016916508, 3.4469374308773695, 3.620388023893611, 3.468970454934606, 3.4121752858627588, 3.420954726074342, 3.2911472532541666, 3.341244596389515, 3.153996981016462, 3.068661661893202, 2.8007719608922343, 2.6406469670509978, 2.5585743444116815, 2.714828543497788, 2.7906901430833386, 2.8381822968840424, 2.723075370537117, 2.7392035106768162, 2.8880411064365035, 2.8867249052855186, 2.8363971914968715, 2.5830434439212695, 2.5868757260868733, 2.4595544891471945, 2.616520659839807, 2.4821093774589826, 2.653224437046447, 2.182940076545492, 2.1995155948316096, 2.1037601654952596, 2.3654658480754733, 2.179703557096218, 2.5953502390566427, 2.165565702569438, 2.538463441982458, 2.8668958956350252, 2.4901683595107897, 2.6469799405986123, 2.3391017301619286, 2.483527936605242, 2.291399239364182, 2.3631739372613083, 2.3386523583067174, 2.4906574536507833, 2.339416753686237, 2.094345377136051, 2.4008308323791425, 2.718716807768942, 1.8077437358206225, 2.233975229284624, 2.525210771469574, 2.5985016563154204, -9673.529416648953, 2.5308210327457346, 2.4386721134233085, 2.5035348846759007, 2.465905006032699, 2.2962559039679036, 2.2896813510888023, 2.29080695259654, 2.220695191350387, 2.3091382091533887, 2.1536156958427455, 2.2343206542136613, 2.179743618498833, 2.1237630768155213, 1.9084598247227404, 1.8268965265760926, 1.8212737177746021, 2.110110685669497, 2.2948762379764958, 2.2916681453243655, 2.26883242119402, 2.207594017016163, 2.408146495348774, 2.393804850315064, 2.3308995020979637, 2.1867258567090175, 2.161072911658266, 2.180878039136587, 2.3192141203053325, 2.2227018308622064, 2.372184628755349, 1.9321579069455765, 1.9078986434506078, 1.8080094154065591, 2.187342514276679, 2.033164145253977, 2.472464460746778, 1.9513293452873768, 2.381388498668457, 2.7310687290973874, 2.4015202777864033, 2.5609463688579126, 2.2280847468464344, 2.5168502971664566, 2.4845325235928613, 2.5511188776763447, 2.5361634474120365, 2.733940758958852, 2.5491626283328515, 2.2550018805213767, 2.716197768395432, 3.0756269104658713, 2.179398483625846, 2.6157409906918474, 2.9031534201003524, 2.9433251026257494, 326.4308899441294, -306.80538922482265, -271.34200787771806, 366.8716962296621, 226.66343282684102, -256.36646528334677, 310.2671867390345, 215.7647398459776, -299.21558893725523, 292.9005647060494, 195.288815855778, -247.67218549613608, -344.21798125278656, 359.57431000450015, 401.8398527932586, 429.0719770921296, 346.5934142514343, 400.27383599266614, 514.9903043363447, -321.19610240285147, -276.34784729290004, 306.53825015476104, 194.19673952677476, -319.5605341423852, 224.79726636833402, 243.51460727998528, -319.8182811593342, 255.35870432894444, 200.1455209779324, -330.4558924765697, -369.519723909847, 295.445431954633, 222.9500956398424, 337.09888618023797, 296.5907140823456, 323.50446294334324, 314.9410519515495, -315.04802201804887, -281.6993863469015, 302.87866300521637, 194.41066068788496, -319.77796321651294, 225.16246610386042, 243.4663673411833, -320.0286379604695, 255.20314053410584, 200.01957353819307, -330.2293614733444, -369.53006847918004, 295.9340112723312, 224.2521708508757, 337.424256905706, 296.2411124725786, 323.02630061138007, 315.52310211153053, -313.62652754385635, -279.1328114943153, 301.94401440875845, 194.403795138669, -319.90004113772557, 225.38926890546827, 243.47361005955827, -319.9847143590632, 255.3263252766901, 199.994535362237, -329.8431731280798, -369.52644352539755, 296.0032945728999, 224.03461499156947, 337.99265968068175, 296.5647097922374, 323.26389182335515, 316.3730561977518, -313.293839431637, -279.40854062177095, 304.20183496928803, 194.49189713542364, -319.9903672654127, 225.62956586954897, 243.47361612210625, -319.96749077516324, 255.39325517434736, 200.0052266039802, -329.81174472187377, -369.52644055889687, 296.0271376404562, 224.33045232565928, 338.2909773619267, 296.920891578342, 323.3417336613584, 316.6813963227287, -312.5992670605574, -267.6591213466545, 305.12374750304843, 194.7079890811583, -320.0217092821167, 226.21855906336896, 243.57036051538307, -319.9058833485815, 255.67989138535995, 200.08981610724368, -329.7794359344207, -369.47806870610475, 296.29248874615405, 224.32954316912583, 339.11427102080415, 297.01625229783167, 323.3950273138471, 317.395570241064, -296.4742044035793, -251.51421795260285, 301.3966235484586, 194.71005551365306, -320.18115179712913, 226.1380782247943, 247.00384597050652, -318.47496133618915, 257.0892958925524, 201.3298353320679, -328.3751560088731, -373.04526024964434, 296.0316491440517, 223.9284189948903, 339.00830978644444, 297.6663251065329, 323.87314817754304, 316.84994252048637, -286.9825202348329, -323.5874077764347, 326.0737452740432, 194.64729985142003, -319.3790552675782, 225.68477402877033, 236.93739353427918, -320.50575257072524, 254.33190433829623, 199.09015429542706, -333.78975092384644, -359.29479063927283, 290.45605939650886, 221.6634928059666, 338.94583584764587, 297.9445008352086, 324.2857086475178, 316.71199901598084, -286.9760745140047, -337.8457599217927, 328.7436070755866, 194.97180798874797, -318.5770489578863, 226.43426138248105, 250.36875429684733, -321.55600483197304, 254.31932519830298, 201.72617795624683, -336.5621312063631, -363.36936440524005, 291.06341342431006, 220.7643250887013, 339.62778859725586, 298.8643347384277, 324.9218083351807, 317.4330357370545, -381.3747833066809, -214.71075970372476, 341.39101112372737, 195.40247657977307, -317.6716749417992, 227.39689191692923, 254.1991179843244, -317.13873712091663, 246.4219520056813, 205.0636013170315, -334.27806961802344, -391.0987734102637, 291.28688832782063, 220.32554616225207, 340.192070078152, 299.7124180712708, 326.22940624381226, 317.9637809752094, -263.21989921435124, -364.5592522165971, 221.75111963142263, 196.22970132951377, -316.7663692346614, 229.0876393500621, 217.17140838707564, -327.0128312789129, 244.70621275575104, 206.09965119390472, -337.1873816681, -389.7237290090816, 290.071149491514, 220.93346152421327, 341.3461336492219, 300.76514443407905, 327.1090414152786, 319.1624998651205, -260.2320600776875, -383.6721827797446, 359.3127802076815, 195.93982791944654, -316.97538512966275, 228.07363407186494, 261.91027474658404, -332.47096504574074, 260.25525944732544, 191.24437550603216, -344.0327691953935, -407.7562698064903, 303.070054119029, 219.8234398836721, 340.9802148615199, 300.38517771716397, 326.6927602781998, 318.54855783526, -241.9695610826293, -384.89181920379633, 367.7103762800107, 196.02403430671438, -316.6995142943906, 227.1132363986526, 200.15328237481663, -307.67933870334673, 261.98938845597513, 222.40827022492886, -347.6656775502779, -334.8277043240305, 281.40661339901635, 219.31275381491287, 341.0762225915205, 300.49053786450713, 326.3128154913302, 316.8009959659903, -232.8958174931886, -409.17682832794526, 375.42505230670395, 196.27488152448677, -316.3589655277174, 226.47041703675495, 269.0427701724766, -345.92387035948013, 233.62915652334596, 228.13370844021483, -353.5089173365395, -329.2852277204511, 297.89702513326665, 219.53755577327138, 341.2403286918179, 300.8828832749696, 326.8542833558331, 316.6696704965234, -222.7347069045045, -414.51269301807224, 156.71284135427777, 196.19442903777417, -315.85601334673265, 226.20195846200477, 185.4418087260783, -298.60056107239143, 255.22715798943682, 233.86673612580125, -320.38939234294594, -323.70743305860196, 274.77330563059695, 221.18097587186458, 340.84113927591534, 300.6380872915306, 326.2993145929304, 317.3159851294695, -446.610866509056, -424.40218811597697, 155.9224306880078, 196.2295893043938, -315.2170392238477, 226.11728776426753, 179.04063637673062, -353.40530485697855, 275.889496667176, 236.4045986375786, -356.0234755268375, -430.84254484893245, 325.4932924707846, 218.81161629350936, 340.4755132597875, 300.2415676907858, 326.15426899145496, 316.25261381471137, -459.1059613307116, -158.21721310038993, 143.31468210990715, 195.92866147807217, -315.58922347786756, 225.94383220354212, 278.23158211969894, -289.4601356805033, 282.6244522509787, 168.73931147355142, -361.65381584554416, -323.7899546369499, 332.18901539065155, 219.0588206835655, 340.7017934868236, 300.6046846416193, 326.0464241943155, 315.40667987886627, -453.4807105153832, -438.5590433975103, 403.9822956855551, 195.5352951937457, -315.9766934066083, 225.61740774226564, 280.2596942850996, -287.1638374410395, 211.36432597306884, 246.47706151702187, -353.80615954512905, -448.64937080991876, 252.61915750423395, 218.23023230235412, 340.39867049745953, 300.00349589547477, 325.34760759729943, 314.77687090868676, -463.23343845132695, -458.08091455052636, 121.49285921328737, 195.93111678779132, -316.35193314263415, 226.22857068504527, 283.0711116555831, -359.46772862923353, 270.2698890266456, 250.62404189683184, -367.5839260033572, -315.82404338092783, 334.19126756532387, 220.03964147389524, 340.08425200376223, 299.5134710017919, 325.1487809805167, 315.33214403895545, -185.04022584420272, -138.3397639065788, 136.34223229719646, 195.635953295503, -316.240837251391, 226.36034612650474, 156.80333081994831, -355.3315912267535, 211.8852006761208, 148.2341000580218, -368.1445445620848, -324.3179987112926, 307.56714837005893, 218.85535167963644, 339.7024775478858, 299.23326844959956, 323.9012485553758, 314.7862327472303, -490.93605874694003, -127.24776608662978, 416.99103199149977, 195.8772056598209, -317.1538239509464, 227.01423144270302, 286.07155248104755, -381.97586320203663, 199.9084016284105, 257.30752863435555, -306.14069447310976, -297.0821706456318, 329.60438098535997, 218.46360872037076, 340.45398745777493, 9975.143223895953, 10000.0, 314.92164683334886, -0.18641251824010396, 0.046290719305034145, 0.0031607664823241066, 0.25128624071476224, 0.3967118761047459, 0.5358909782862611, 0.4312080362196866, 0.4432720746790437, 0.44655219889682485, 0.5164361638708215, 0.6652429516871052, 0.6394356252567377, 0.7147533696752362, 0.638825213049131, 0.6000962962934864, 0.5301125312580552, 0.4608874889636354, 0.4069235188399034, 0.45780814403042314, 0.29987998362412327, 0.3060822743245808, 0.2355598802860186, 0.262854486822107, 0.2742365129688551, 0.3169300373920123, 0.3712599147893343, 0.23626546474406496, 0.28030342647434736, 0.2381436256946472, 0.2428196499531623, 0.14255539619080082, 0.2699427225506952, 0.28810051588698116, 0.049314146028336836, 0.15209257472815807, 0.011633836915279971, 0.09626139754800533, 0.012749241654091747, 0.1815818778013636, 0.062016769785259385, -0.06749448423397553, 0.05987557528715115, -0.06250573894612899, -0.25181591050750285, -0.20696394531114493, -0.10744918015552685, 0.006402166580301127, 0.06627076865152048, 0.08365854136172857, -0.12637532956978248, -0.17037963901020703, -0.13870334489001834, -0.4181660169742827, -0.33123044083731656, -0.29547006358734507, -10000.0]

