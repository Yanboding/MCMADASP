import argparse
import json
import pickle
import os
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

from generate_params import _save_training_result, _load_cached_training_result
import numpy as np
import pandas as pd
from gurobipy import GRB
from scipy.stats import geom

from experiments.experiment_config import get_config_by_type
from importance_sampling import build_proposal
from utils import iter_to_tuple, get_uid, safe_open, RunningStats, encode, decode, get_solution_value, acquire_grb_env, read_lines_with_pattern
from decision_maker import InfiniteSAAAgent, InfinitePenalizedSAAAgent, MyopicAgent, ALPRowGenerationAgent, ApproxQAgent
from policy_evaluator import PolicyEvaluator
from generating_function import MulticlassLinearPenaltyFunction, LinearPenaltyFunction

def jsonl_result_exists(path, uid, policy_id):
    """Return True if (uid, policy_id) already exists in JSONL output."""
    if not os.path.isfile(path):
        return False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('uid') == uid and row.get('policy_id') == policy_id:
                return True
    return False

def run_lower_bound_solver(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
    res = {}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    t = 1  # Assuming a single time step for the experiment
    config = get_config_by_type(case_type='infinite_custom',args=env_args)
    env = config.env
    config.reset_params['new_arrivals'] = sample_path
    perfect_info_lower_bound_solver = InfiniteSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.CONTINUOUS, sample_path=sample_path, is_include_discount_factor=True)
    state, info = env.reset(**config.reset_params)
    _, benchmark_value, info = perfect_info_lower_bound_solver.solve(state, t)
    res['benchmark_value'] = benchmark_value
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def run_penalized_lower_bound_solver(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
    coefficients = None
    for agent_arg in agent_args:
        if agent_arg['agent_name'] == 'col_gen_alp':
            coefficients = agent_arg["args"]['coefficients']
            break
    if coefficients is None:
        raise ValueError("No coefficients found for penalized lower bound solver.")
    res = {}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    t = 1  # Assuming a single time step for the experiment
    config = get_config_by_type(case_type='infinite_custom',args=env_args)
    env = config.env
    config.reset_params['new_arrivals'] = sample_path
    state, info = env.reset(**config.reset_params)
    penalized_lower_bound_solver = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.INTEGER, sample_path=sample_path, is_include_discount_factor=True, coeffecients=coefficients)
    _, benchmark_value, info = penalized_lower_bound_solver.solve(state, t)
    res['penalized_lower_bound'] = benchmark_value
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def experiment(experiment_name, param_value, env_args, coefficients, agent_args, sample_path, uid, job_id):
    res = {'result':[]}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    sample_path = np.array(sample_path)
    print('uid:', uid)
    print('test sample_path:', sample_path)
    t = 1  # Assuming a single time step for the experiment
    config = get_config_by_type(case_type='infinite_custom',args=env_args)
    env = config.env
    config.reset_params['new_arrivals'] = sample_path
    state, info = env.reset(**config.reset_params)
    print('init state:', state)
    perfect_info_lower_bound_solver = InfiniteSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.INTEGER, sample_path=sample_path, is_include_discount_factor=True)
    _, perfect_info_lower_bound, info = perfect_info_lower_bound_solver.solve(state, t)
    print('benchmark_value:', perfect_info_lower_bound)
    penalized_lower_bound_solver = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.INTEGER, sample_path=sample_path, is_include_discount_factor=True, coeffecients=coefficients)
    _, penalized_lower_bound, info = penalized_lower_bound_solver.solve(state, t)
    print('benchmark_value:', penalized_lower_bound)
    for agent in agent_args:
        config = get_config_by_type(case_type='infinite_custom',args=env_args)
        env = config.env
        agent_name, args = agent['agent_name'], agent['args']
        stats = {'agent_name': agent_name}
        config.reset_params['new_arrivals'] = sample_path
        state, info = env.reset(**config.reset_params)
        print('agent init state:', state)
        print('agent_name:', agent_name)
        if agent_name in {"hindsight_approx"}:
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name in {"hindsight_approx_with_penalty"}:
            agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "myopic":
            agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'col_gen_alp':
            agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'row_gen_alp':
            agent_instance = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, **args)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        states, actions, rewards = evaluator.sample_path_evaluate(state, t, sample_path)
        G = 0.0
        for tau in reversed(range(len(rewards))):
            G = env.discount_factor * G + rewards[tau]
        stats['value_function'] = G
        stats['perfect_info_lower_bound'] = perfect_info_lower_bound
        stats['penalized_lower_bound'] = penalized_lower_bound
        wait_time_by_type =[]
        waiting_time_target_violations = []
        for type_i, running_stat in env.wait_time_by_type.items():
            wait_time_by_type.append({"treatment_type":type_i,
                                      "expect":running_stat.mean,
                                      "varSum":running_stat.var_sum,
                                      "count":int(running_stat.n)})
        stats['wait_time_by_type'] = wait_time_by_type
        for type_i, running_stat in env.waiting_time_target_violations.items():
            waiting_time_target_violations.append({"treatment_type":type_i,
                                                    "expect":running_stat.mean,
                                                    "varSum":running_stat.var_sum,
                                                    "count":int(running_stat.n)})
            
        stats['waiting_time_target_violations'] = waiting_time_target_violations
        stats['overtime'] = env.overtime.tolist()
        stats['postponing_decision_number'] = env.postponing_decision_number.tolist()
        stats['waiting_number'] = env.waiting_number.tolist()
        res['result'].append(stats)
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def value_function_experiment(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
    res = {'result':[]}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    sample_path = np.array(sample_path)
    t = 1  # Assuming a single time step for the experiment
    print('agent_args:', len(agent_args))
    for agent in agent_args:
        config = get_config_by_type(case_type='infinite_custom',args=env_args)
        env = config.env
        agent_name, args = agent['agent_name'], agent['args']
        stats = {'agent_name': agent_name}
        config.reset_params['new_arrivals'] = sample_path
        state, info = env.reset(**config.reset_params)
        print(config.reset_params)
        print('init state:', state)
        if agent_name in {"hindsight_approx", "hindsight_value"}:
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "myopic":
            agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'alp':
            agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        states, actions, rewards = evaluator.sample_path_evaluate(state, t, sample_path)
        stats['value_function'] = sum(rewards)
        '''
        if agent_name == "hindsight_value":
            benchmark_solver = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
            _, benchmark_value, info = benchmark_solver.solve(state, t)
        '''
        wait_time_by_type =[]
        for type_i, running_stat in env.wait_time_by_type.items():
            wait_time_by_type.append({"treatment_type":type_i,
                                      "expect":running_stat.mean,
                                      "varSum":running_stat.var_sum,
                                      "count":int(running_stat.n)})
        stats['wait_time_by_type'] = wait_time_by_type
        stats['overtime'] = env.overtime.tolist()
        res['result'].append(stats)
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def alp_train(env_args, experiment_name, train_type="row_gen", job_id=None):
    print('Training ALP agent with args:', env_args)
    config_for_train = get_config_by_type(case_type='infinite_custom',args=env_args)
    env_for_train = config_for_train.env
    if train_type == "col_gen":
        agent = ALPEJORColumnGenerationAgent(env=env_for_train, discount_factor=env_for_train.discount_factor)
    else:
        agent = ALPRowGenerationAgent(env=env_for_train, discount_factor=env_for_train.discount_factor)
    obj_val, coefficients = agent.train(debug=False, verbose=True)
    output_file = os.path.join('experiments','results',experiment_name, f'alp_train_{train_type}_{job_id}.jsonl' if job_id else f'alp_train_{train_type}.jsonl')
    with safe_open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps({'uid':get_uid(env_args), 'result': {'agent_name': f'{train_type}_alp', 'obj_val': obj_val, 'args': {'coefficients':coefficients}}}) + '\n')

def load_pickle_if_exists(path):
    """Return file contents if the file exists, otherwise return None."""
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
    print('Simulate evaluation on uid:', uid, agent_arg['agent_name'])
    # t = 1  # Assuming a single time step for the experiment
    # 1) Build config and base env
    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    config.reset_params['new_arrivals'] = sample_path
    env = config.env
    t0 = config.reset_params.get('t', 1)

    aid = get_uid(agent_arg)
    pickle_file = os.path.join('experiments', 'results', experiment_name,
                               f'{uid}-{aid}.pickle')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    data = load_pickle_if_exists(pickle_file)
    # 2) Decide env, state trajectory, etc.
    if data is None:
        s, info = env.reset(**config.reset_params)
        states = []
        actions = []
        rewards = []
        t = t0
    else:
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        t = data['t']
        s = data['s']
        s, info = env.reset(init_state=s, t=t, new_arrivals=sample_path)
    agent_name, args = agent_arg['agent_name'], agent_arg['args']
    if agent_name in {"hindsight_approx"}:
        agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
    elif agent_name in {"hindsight_approx_with_penalty"}:
        agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, **args)
    elif agent_name == "myopic":
        agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, **args)
    elif agent_name == 'col_gen_alp':
        agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
    elif agent_name == 'row_gen_alp':
        agent_instance = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, **args)
    # evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
    # states, actions, rewards = evaluator.sample_path_evaluate(state, t, sample_path)
    for tau in range(len(sample_path[t-1:])):
        print("Current time step:", t + tau)
        states.append(s)
        start = time.time()
        a = agent_instance.policy(s, t + tau)
        end = time.time()
        print(f"Policy {t + tau} computation time: {end - start} seconds")
        actions.append(a)
        next_state, reward, done, info = env.step(a)
        rewards.append(reward)
        s = next_state
        if (t+tau) % 10 == 0:
            with open(pickle_file, 'wb') as f:
                res = {
                    's': s,
                    "t": t + tau+1,
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                }
                pickle.dump(res, f)
        
        if done:
            break
    # waiting time
    stats = {}
    scheduled_patients = []
    for (advance_scheduling_decision, overtime_decision) in actions[warm_up_periods:]:
        scheduled_patients.append(advance_scheduling_decision)
    rewards_after_warmup = rewards[warm_up_periods:]
    stats['total_cost'] = sum(rewards_after_warmup)
    discounted_cost_after_warmup = 0.0
    for tau in reversed(range(len(rewards_after_warmup))):
        discounted_cost_after_warmup = 0.99 * discounted_cost_after_warmup + rewards_after_warmup[tau]
    stats['total_scheduled_patients'] = np.sum(scheduled_patients, axis=0).tolist()
    stats['overtime'] = env.overtime.tolist()
    stats['postponing_decision_number'] = env.postponing_decision_number.tolist()
    #stats['states'] = encode(states)
    #stats['actions'] = encode(actions)
    stats['rewards'] = rewards
    res = {
        "uid": uid,
        "experiment_name": experiment_name,
        "warm_up_periods": warm_up_periods,
        "agent_name": agent_name
    }
    res['results'] = stats
    # state, sample_path, periods, remaining_new_arrivals

    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def get_solution(action_var, is_final=False):
        x_var, y_var = action_var
        if is_final:
            x = np.array([[round(var.Xn) for var in row] for row in x_var]).astype(int)
            y = np.array([round(var.Xn) for var in y_var]).astype(int)
        else:
            x = get_solution_value(x_var).astype(float)
            y = get_solution_value(y_var).astype(float)
        return (x, y)

def calculate_information_relaxation_cost(env, env_args, experiment_name, lowerbound_args, generating_function, init_state, sample_path, job_id):
    '''
        1. read hindsight_approx data
        2. use actions to determine states in each period
        3. starting at warmup_period + 1, solve lowerbound without penalty
    '''
    agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, sample_path=sample_path, generating_function=generating_function, **lowerbound_args)
    #print('warmup_state:', warmup_state, sample_path[warm_up_periods])
    my_action, benchmark_value, info = agent_instance.solve(init_state)
    costs = [cost.getValue() for cost in info['costs'][0]]
    penalties = [penalty if isinstance(penalty, int) else penalty.getValue() for penalty in info['penalties'][0]] if 'penalties' in info else []
    actions = info['actions'][0]
    scheduled_patients = []
    overtime = np.zeros(len(sample_path)+env.planning_horizon)
    for t, action_var in enumerate(actions):
        advance_scheduling_decision, overtime_decision = get_solution(action_var, is_final=False)
        scheduled_patients.append(advance_scheduling_decision.tolist())
        start = t
        end = t + len(overtime_decision)
        overtime[start:end] += overtime_decision
    params = {
        'init_state': list(item.tolist() for item in init_state),
        'sample_path': sample_path.tolist(),
        'env_args':env_args,
    }
    uid = get_uid(params)
    res = {
        "uid": uid,
        "penalized_cost": benchmark_value,
        "total_cost": sum(costs),
        "total_penalty": sum(penalties),
        "experiment_name": experiment_name,
        "agent_name": {'agent_name': "penalized_lowerbound", 'lowerbound_args': lowerbound_args},
        "costs": costs,
        "penalties": penalties,
        "scheduled_patients": scheduled_patients,
        "overtime": overtime.tolist(),
    }
    output_file = os.path.join('experiments', 'results',  experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')
    return res

def calculate_policy_costs(uid, experiment_name, policy_id, agent_name, agent_args, env_args, init_state, sample_path, warm_up_periods, generating_function, grb_env=None, grb_sub_envs=None):
    def _to_float(v):
        if hasattr(v, "getValue"):
            return float(v.getValue())
        return float(v)

    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        return value

    base_dir = os.path.join('experiments', 'results', experiment_name, 'pickles')
    os.makedirs(base_dir, exist_ok=True)
    checkpoint_file = os.path.join(base_dir, f'{uid}-{policy_id}-checkpoint.pickle')
    result_file = os.path.join(base_dir, f'{uid}-{policy_id}-result.pickle')

    cached_result = load_pickle_if_exists(result_file)
    if cached_result is not None:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        return cached_result

    sample_path = np.array(sample_path)

    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    local_generating_function = LinearPenaltyFunction(env=env, coefficients=generating_function.coefficients)
    runtime_agent_args = dict(agent_args)
    serializable_agent_args = {k: _to_jsonable(v) for k, v in runtime_agent_args.items() if k != 'grb_env'}

    if agent_name in {"approx_hindsight", "approx_penalized_hindsight"}:
        generating_function = LinearPenaltyFunction(env=env, coefficients=runtime_agent_args['penalty_coefficients'])
        runtime_agent_args.pop('penalty_coefficients')
        runtime_agent_args['generating_function'] = generating_function
        runtime_agent_args['sample_path_proposal'] = build_proposal(
            runtime_agent_args.get('sample_path_proposal', None)
        )
        agent_instance = ApproxQAgent(env, discount_factor=env.discount_factor,
                                        grb_env=grb_env,
                                        subproblem_grb_envs=grb_sub_envs,
                                        **runtime_agent_args
                                    )
    elif agent_name == "myopic":
        agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **runtime_agent_args)
    elif agent_name == 'row_gen_alp':
        agent_instance = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **runtime_agent_args)
    else:
        raise ValueError(f"Unsupported policy: {agent_name}")

    states = []
    actions = []
    costs = []
    penalties = []
    checkpoint = load_pickle_if_exists(checkpoint_file)
    if checkpoint is not None:
        states = checkpoint['states']
        actions = checkpoint['actions']
        costs = checkpoint['costs']
        penalties = checkpoint['penalties']
        t = checkpoint['t']
        s = checkpoint['s']
        solving_time_per_state = checkpoint.get('solving_time_per_state', RunningStats())
        s, _ = env.reset(init_state=s, t=t, new_arrivals=sample_path)
    else:
        t = 1
        s, _ = env.reset(init_state=init_state, t=t, new_arrivals=sample_path)
        solving_time_per_state = RunningStats()
    for tau in range(len(sample_path) - t + 2):
        current_t = t + tau
        start_time = time.time()
        _, action, _ = agent_instance.solve(s, current_t)
        solving_time_per_state += time.time() - start_time
        next_state, cost, done, _ = env.step(action)

        if current_t <= len(sample_path):
            new_arrivals = sample_path[current_t - 1]
            penalty = local_generating_function.calculate_penalty(s, action, new_arrivals)
            penalties.append(_to_float(penalty))

        states.append(s)
        actions.append(action)
        costs.append(_to_float(cost))
        s = next_state

        if current_t % 10 == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    's': s,
                    't': current_t + 1,
                    'states': states,
                    'actions': actions,
                    'costs': costs,
                    'penalties': penalties,
                    'solving_time_per_state': solving_time_per_state,
                }, f)
        if done:
            break

    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            's': s,
            't': len(costs) + 1,
            'states': states,
            'actions': actions,
            'costs': costs,
            'penalties': penalties,
            'solving_time_per_state': solving_time_per_state,
        }, f)

    scheduled_patients = []
    overtime = np.zeros(len(sample_path) + env.planning_horizon)
    postponing_decisions = []
    for idx, (state_t, action_t) in enumerate(zip(states, actions)):
        _, _, waitlist = state_t
        advance_scheduling_decision, overtime_decision = action_t
        scheduled_patients.append(advance_scheduling_decision.tolist())
        overtime[idx:idx + len(overtime_decision)] += overtime_decision
        postponing_decisions.append(waitlist - advance_scheduling_decision.sum(axis=0))

    postponing_decisions = np.array(postponing_decisions).sum(axis=0) if postponing_decisions else np.zeros(env.num_types)
    total_cost = float(sum(costs[warm_up_periods:]))
    total_penalty = float(sum(penalties[warm_up_periods:]))
    penalized_cost = total_cost + total_penalty
    warmup_state = tuple(np.array(item).tolist() for item in states[warm_up_periods])

    result = {
        'policy_id': policy_id,
        'agent_name': agent_name,
        'agent_args': serializable_agent_args,
        'penalized_cost': penalized_cost,
        'total_cost': total_cost,
        'total_penalty': total_penalty,
        "warmup_state": warmup_state,
        'costs': [float(v) for v in costs],
        'penalties': [float(v) for v in penalties],
        'scheduled_patients': scheduled_patients,
        'overtime': overtime.tolist(),
        'postponing_decisions': postponing_decisions.tolist(),
        'solving_time_per_state': solving_time_per_state.mean,
    }

    with open(result_file, 'wb') as f:
        pickle.dump(result, f)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return result

def evaluate_policy_costs_with_information_relaxation(uid, experiment_name, mutate_val, init_state, sample_path, warm_up_periods, env_args, policy_specs, penalty_coefficients, group_id, grb_env, grb_sub_envs, job_id):
    '''
    This function evaluates the costs of different policies and their gaps to the information relaxation lower bounds.
    '''
    init_state = tuple(np.array(item) for item in init_state)
    sample_path = np.array(sample_path)

    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    generating_function = LinearPenaltyFunction(env=env, coefficients=penalty_coefficients)
    # generating_function = MulticlassLinearPenaltyFunction(env=env, coefficients=penalty_coefficients)


    zero_lowerbound_args = {
        'current_decision_var_type': 'integer',
        'future_decision_var_type': 'continuous',
        'generating_function': generating_function,
        'penalty_ratio': 0,
        'grb_env': grb_env,
        'subproblem_grb_envs': grb_sub_envs,
    }
    penalized_lowerbound_args = {
        'current_decision_var_type': 'integer',
        'future_decision_var_type': 'continuous',
        'generating_function': generating_function,
        'penalty_ratio': 1,
        'grb_env': grb_env,
        'subproblem_grb_envs': grb_sub_envs,
    }

    zero_lowerbound_instance = ApproxQAgent(env, discount_factor=env.discount_factor, **zero_lowerbound_args)
    penalized_lowerbound_instance = ApproxQAgent(env, discount_factor=env.discount_factor, **penalized_lowerbound_args)

    summary_rows = []
    for policy_spec in policy_specs:
        policy_result = calculate_policy_costs(
            uid=uid,
            experiment_name=experiment_name,
            policy_id=policy_spec['policy_id'],
            agent_name=policy_spec['agent_name'],
            agent_args=policy_spec['agent_args'],
            env_args=env_args,
            init_state=tuple(np.array(item) for item in init_state),
            sample_path=sample_path,
            warm_up_periods=warm_up_periods,
            generating_function=generating_function,
            grb_env=grb_env,
            grb_sub_envs=grb_sub_envs,
        )
        warmup_sate = tuple(np.array(item) for item in policy_result.get('warmup_state', init_state))
        start = time.time()
        zero_information_relaxation_cost = zero_lowerbound_instance.calculate_information_relaxation_cost(warmup_sate, sample_path=sample_path[warm_up_periods:])
        print(f"Zero information relaxation cost computed in {time.time() - start:.1f} seconds: {zero_information_relaxation_cost}")
        start = time.time()
        penalized_information_relaxation_cost = penalized_lowerbound_instance.calculate_information_relaxation_cost(warmup_sate, sample_path=sample_path[warm_up_periods:])
        print(f"Penalized information relaxation cost computed in {time.time() - start:.1f} seconds: {penalized_information_relaxation_cost}")
        policy_result.update({
            'uid': uid,
            'group_id': group_id,
            'experiment_name': experiment_name,
            'mutate_val': mutate_val,
            'warm_up_periods': warm_up_periods,
            'zero_information_relaxation_cost': float(zero_information_relaxation_cost),
            'penalized_information_relaxation_cost': float(penalized_information_relaxation_cost),
            'gap_to_zero_information_relaxation': float(policy_result['total_cost'] - zero_information_relaxation_cost),
            'gap_to_penalized_information_relaxation': float(policy_result['penalized_cost'] - penalized_information_relaxation_cost),
        })

        if not jsonl_result_exists(output_file, uid, policy_result['policy_id']):
            with open(output_file, 'a') as f:
                f.write(json.dumps(policy_result) + '\n')
        else:
            print(f"Skip saving duplicate result: uid={uid}, policy_id={policy_result['policy_id']}")

        summary_rows.append({
            'policy_id': policy_result['policy_id'],
            'agent_name': policy_result['agent_name'],
            'penalized_cost': policy_result['penalized_cost'],
            'total_cost': policy_result['total_cost'],
            'total_penalty': policy_result['total_penalty'],
            'zero_information_relaxation_cost': float(zero_information_relaxation_cost),
            'penalized_information_relaxation_cost': float(penalized_information_relaxation_cost),
            'gap_to_zero_information_relaxation': policy_result['gap_to_zero_information_relaxation'],
            'gap_to_penalized_information_relaxation': policy_result['gap_to_penalized_information_relaxation'],
        })

    return summary_rows

def coefficient_training_test(uid, experiment_name, env_args, group_id, grb_env, job_id):
    '''
    This function tests the training of penalty coefficients using Benders decomposition and evaluates the resulting coefficients by calculating the penalized lower bound and zero-penalty lower bound with the same initial state and sample path.
    '''
    test_state = env_args['reset_params']['init_state']
    test_state = tuple(np.array(item) for item in test_state)
    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    generating_function = LinearPenaltyFunction(env=env)
    agent = InfinitePenalizedSAAAgent(env=env, discount_factor=env.discount_factor, sample_path_number=256, generating_function=generating_function, is_myopic=False, grb_env=grb_env)
    env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, init_state=test_state, parallel=False, verbose=False)
    print('Obejctive from Benders decomposition training:', obj) # Full MILP:39050.05571672409 # LP: 21524.097118570513
    print('Coefficients from Benders decomposition training:', direct_coefficients)
    env.reset_random_seeds()
    penalized_obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=direct_coefficients, ratio=1, init_state=test_state, verbose=False)
    print('Objective from sample mean penalized lower bound evaluation using original problem coefficients:', penalized_obj)
    env.reset_random_seeds()
    zero_penalized_obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=direct_coefficients, ratio=0, init_state=test_state, verbose=False)
    print('Objective from sample mean zero penalized lower bound evaluation using original problem coefficients:', zero_penalized_obj)
    print('Difference between penalized and zero-penalized objectives:', (penalized_obj - zero_penalized_obj)/zero_penalized_obj * 100)

    result = {
            'uid': uid,
            'group_id': group_id,
            'experiment_name': experiment_name,
            'penalized_lower_bound_objective': penalized_obj,
            'zero_penalized_lower_bound_objective': zero_penalized_obj,
            'gap_between_penalized_and_zero': penalized_obj - zero_penalized_obj,
            'coefficients': direct_coefficients,
        }
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

def coefficient_out_of_sample_test(uid, experiment_name, mutate_val, env_args, penalty_coefficients, init_state, sample_path, warm_up_periods, group_id, grb_env, job_id, alp_coefficients):
    '''
    This function tests the out-of-sample performance of the trained coefficients by evaluating the penalized lower bound and zero-penalty lower bound on same initial state but different sample path that were not seen during training.
    '''
    test_state = tuple(np.array(item) for item in init_state)
    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    generating_function = LinearPenaltyFunction(env=env, coefficients=penalty_coefficients)
    zero_penalized_args = {
        'current_decision_var_type': 'integer',
        'future_decision_var_type': 'continuous',
        'is_myopic': False,
        'is_include_discount_factor': False,
        'sample_path': sample_path[warm_up_periods:],
        'generating_function': generating_function,
        'penalty_ratio': 0,
        'grb_env': grb_env,
    }
    penalized_args = {
        'current_decision_var_type': 'integer',
        'future_decision_var_type': 'continuous',
        'is_myopic': False,
        'is_include_discount_factor': False,
        'sample_path': sample_path[warm_up_periods:],
        'generating_function': generating_function,
        'penalty_ratio': 1,
        'grb_env': grb_env,
    }
    zero_penalized_agent = InfinitePenalizedSAAAgent(env=env, discount_factor=env.discount_factor, **zero_penalized_args)
    penalized_agent = InfinitePenalizedSAAAgent(env=env, discount_factor=env.discount_factor, **penalized_args)
    zero_penalized_information_relaxation_cost, _, _ = zero_penalized_agent.direct_solve(test_state, t=1)
    penalized_information_relaxation_cost, _, _ = penalized_agent.direct_solve(test_state, t=1)
    result = {
            'uid': uid,
            'group_id': group_id,
            'experiment_name': experiment_name,
            'mutate_val': mutate_val,
            'penalized_lower_bound_objective': penalized_information_relaxation_cost,
            'zero_penalized_lower_bound_objective': zero_penalized_information_relaxation_cost,
            'gap_between_penalized_and_zero': penalized_information_relaxation_cost - zero_penalized_information_relaxation_cost,
            'coefficients': penalty_coefficients,
        }
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')


def train_penalty_coefficients(env_args, experiment_name, sample_path_number, mutate_val, agent_args=None, job_id=None):
    '''
    Train coefficients for the penalty function used in the hindsight
    approximation with penalty agent. ``agent_args`` may include a
    JSON-serializable ``sample_path_length_proposal`` spec dict.
    '''
    agent_args = dict(agent_args or {})
    mutate_val_str = str(mutate_val).replace('.', '_')
    out_file = f'penalty_train_discount_{mutate_val_str}.jsonl'

    cached = _load_cached_training_result(experiment_name, out_file, env_args, agent_args=agent_args)
    if cached is not None:
        cached_result = cached.get('result', {})
        cached_obj = cached_result.get('obj_val')
        cached_coefficients = cached_result.get('args', {}).get('coefficients', None)
        cached_info = cached_result.get('info', {})
        print(f"Use cached penalty coefficients for uid={cached.get('uid')}")
        return cached_obj, cached_coefficients, cached_info

    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    generating_function = LinearPenaltyFunction(env=env)
    inner = dict(agent_args.get('agent_args', {}))
    inner['generating_function'] = generating_function
    if 'sample_path_length_proposal' in inner:
        inner['sample_path_length_proposal'] = build_proposal(
            inner['sample_path_length_proposal']
        )
    pprint(inner)
    sample_path_number = agent_args['agent_args']['sample_path_number']
    # Pre-create one Gurobi token (env) per subproblem so ApproxQAgent can
    # reuse them instead of acquiring/releasing a token for every subproblem.
    subproblem_grb_envs = [
        acquire_grb_env({"Threads": 1}, verbose=False)
        for _ in range(sample_path_number)
    ]
    agent = ApproxQAgent(env=env, discount_factor=env.discount_factor,
                                      subproblem_grb_envs=subproblem_grb_envs,
                                      **inner)
    init_state = None if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
    sample_path_number = agent_args['agent_args']['sample_path_number']
    print(f"Training penalty coefficients for env_uid {get_uid(env_args)} with init_state: {init_state}, sample_path_number: {sample_path_number}, agent_args: {agent_args}")
    if init_state is not None:
        init_state = tuple(np.array(item) for item in init_state)
    required_bookings = [(env.regular_capacity + env.overtime_capacity) * env.discount_factor**(j) for j in range(env.planning_horizon)]
    required_bookings[-1] = 0
    required_bookings = np.array(required_bookings)
    E_u_alpha = np.minimum(required_bookings, env.regular_capacity)
    E_v_alpha = required_bookings - E_u_alpha
    E_w_alpha = env.arrival_generator.mean_by_type
    init_state = (E_u_alpha, E_v_alpha, E_w_alpha)
    env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    print(init_state)
    obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=1e4, init_state=init_state, parallel=True, verbose=False)
    print('Obejctive from Benders decomposition training:', obj)
    print('Coefficients from Benders decomposition training:', direct_coefficients)
    _save_training_result(
        experiment_name=experiment_name,
        file_name=out_file,
        env_args=env_args,
        agent_name='hindsight_approx_with_penalty',
        obj_val=obj,
        coefficients=direct_coefficients,
        info=info,
        agent_args=agent_args,
    )

    return obj, direct_coefficients, info
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    parser.add_argument('--params_file', help='Path to JSON file containing params payload', type=str)
    parser.add_argument('--job_id', help='Input METAJOB_ID', type=str)
    args = parser.parse_args()
    if args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    elif args.params:
        params = json.loads(args.params)
    else:
        raise ValueError('Either --params or --params_file must be provided.')

    if isinstance(params, dict):
        params = [params]
    grb_env = acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
    # Dynamically allocate a pool of Gurobi tokens (envs) for the subproblems.
    # A Gurobi env is NOT thread-safe for concurrent optimization, so the
    # minimum number of tokens needed equals the number of subproblems solved
    # concurrently, i.e. the number of available CPUs. Subproblems are then
    # partitioned across this pool and the ones sharing an env are solved
    # sequentially (see SubproblemWorker grouping in the Benders solver), so we
    # only hold `min(sample_path_number, num_cpus)` tokens instead of one per
    # subproblem.
    sample_path_number = max(
        (spec.get('agent_args', {}).get('sample_path_number', 0)
         for param in params
         for spec in param.get('policy_specs', [])),
        default=0,
    )
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    try:
        num_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    except ValueError:
        num_cpus = os.cpu_count() or 1
    num_sub_envs = min(sample_path_number, num_cpus) if sample_path_number else 0
    grb_sub_envs = [
        acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
        for _ in range(num_sub_envs)
    ]
    for param in params:
        evaluate_policy_costs_with_information_relaxation(**param, grb_env=grb_env, grb_sub_envs=grb_sub_envs, job_id=args.job_id)