import argparse
import json
import pickle
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

import numpy as np
import pandas as pd
from gurobipy import GRB

from experiments.experiment_config import get_config_by_type
from utils import iter_to_tuple, get_uid, safe_open, RunningStats, encode, decode, get_solution_value
from decision_maker import ALPEJORColumnGenerationAgent, InfiniteSAAAgent, InfinitePenalizedSAAAgent, MyopicAgent, ALPRowGenerationAgent, LinearPenaltyFunction
from policy_evaluator import PolicyEvaluator

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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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

def alp_train(env_args, experiment_name, param_value, train_type="col_gen", job_id=None):
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
        f.write(json.dumps({'uid':get_uid(env_args), 'result': {'agent_name': f'{train_type}_alp', 'obj_val': obj_val, 'param_value':param_value, 'args': {'coefficients':coefficients}}}) + '\n')

def load_pickle_if_exists(path):
    """Return file contents if the file exists, otherwise return None."""
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def simulate_evaluation(env_args, experiment_name, agent_arg,  warm_up_periods, sample_path, uid, job_id, states=None, actions=None, rewards=None):
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

def evaluate_lower_bound(env_args, experiment_name, agent_arg,  warm_up_periods, sample_path, uid, lowerbound_args, generating_function, job_id):
    '''
    I need to get the state on the warm_up_periods
    and then evaluate the lower bound solvers from there
    1) Build config and base env
    2) For each agent, run evaluation from the given state and sample_path
    3) Collect stats and save to output file
    4) Output file name: experiments/results/{experiment_name}/{job_id}.jsonl
    5) Each line in the output file is a json object with keys:
    '''
    print('Evaluate lower bound on uid:', uid, agent_arg['agent_name'])
    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    config.reset_params['new_arrivals'] = sample_path
    env = config.env
    agent_name, args = agent_arg['agent_name'], agent_arg['args']
    if "lowerbound" in agent_name:
        if agent_name == "lowerbound":
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, sample_path=sample_path, **args)
        elif agent_name == "penalized_lowerbound":
            agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, sample_path=sample_path, **args)
        state, info = env.reset(**config.reset_params)
        _, benchmark_value, info = agent_instance.solve(state, 1)
        costs = [cost.getValue() for cost in info['costs'][0]]
        penalties = [penalty if isinstance(penalty, int) else penalty.getValue() for penalty in info['penalties'][0]] if 'penalties' in info else []
        print(sum(costs))
        actions = info['actions'][0]
        scheduled_patients = []
        overtime = np.zeros(len(sample_path)+env.planning_horizon)
        for t, action_var in enumerate(actions):
            advance_scheduling_decision, overtime_decision = get_solution(action_var, is_final=False)
            scheduled_patients.append(advance_scheduling_decision.tolist())
            start = t
            end = t + len(overtime_decision)
            overtime[start:end] += overtime_decision
    else:
        if agent_name in {"hindsight_approx"}:
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "hindsight_approx_with_penalty":
            agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "myopic":
            agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'col_gen_alp':
            agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'row_gen_alp':
            agent_instance = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, **args)
        config.reset_params['new_arrivals'] = sample_path
        env = config.env
        t0 = config.reset_params.get('t', 1)
        aid_arg = {
            'agent_arg': agent_arg,
            'lowerbound_args':lowerbound_args
        }
        aid = get_uid(aid_arg)
        pickle_file = os.path.join('experiments', 'results', experiment_name,
                                'pickles', f'{uid}-{aid}.pickle')
        # Make sure the parent directories exist
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        data = load_pickle_if_exists(pickle_file)
        # 2) Decide env, state trajectory, etc.
        if data is None:
            s, info = env.reset(**config.reset_params)
            states = []
            actions = []
            costs = []
            penalties = []
            t = t0
        else:
            states = data['states']
            actions = data['actions']
            costs = data['costs']
            penalties = data['penalties']
            t = data['t']
            s = data['s']
            s, info = env.reset(init_state=s, t=t, new_arrivals=sample_path)
        print("total cost:", sum(costs), "length of costs:", len(costs), 't:', t, 'remaining sample path length:', len(sample_path)-t+1)
        if len(costs) >= len(sample_path):
            print(states[warm_up_periods:])
            print(actions[warm_up_periods:])
            print(costs[warm_up_periods:])
            print(penalties[warm_up_periods:])
            print('env.arrival_generator.mean_by_type', env.arrival_generator.mean_by_type)
            print('Evaluation already completed. Total cost:', sum(costs))
            return
        average_run_time = 0
        max_scenario = 0
        print(agent_instance)
        for tau in range(len(sample_path)-t+1):
            print("Current time step:", t + tau)
            start = time.time()
            a, upper_bound, info = agent_instance.solve(s, t + tau)
            end = time.time()
            compute_time = end - start
            average_run_time += (compute_time - average_run_time) / (tau+1)
            if 'number_of_workers' in info:
                max_scenario += (info['number_of_workers']-max_scenario) / (tau+1)
            print(f"Policy {t + tau} computation time: {compute_time} seconds, average timr: {average_run_time} seconds")
            next_state, cost, done, info = env.step(a)
            if t + tau < len(sample_path):
                new_arrivals = sample_path[t + tau]
                print('new_arrivals:', new_arrivals)
                penalty = generating_function.calculate_penalty(s, a, new_arrivals) * lowerbound_args['coefficients']
                penalties.append(penalty)
            states.append(s)
            actions.append(a)
            costs.append(cost)
            s = next_state
            
            if (t+tau) % 10 == 0:
                with open(pickle_file, 'wb') as f:
                    res = {
                        's': s,
                        "t": t + tau+1,
                        "states": states,
                        "actions": actions,
                        "costs": costs,
                        "penalties": penalties,
                    }
                    pickle.dump(res, f)
            
            if done:
                with open(pickle_file, 'wb') as f:
                    res = {
                        's': s,
                        "t": t + tau+1,
                        "states": states,
                        "actions": actions,
                        "costs": costs,
                        "penalties": penalties,
                    }
                    pickle.dump(res, f)
                break
        print("number of states:", len(states), len(actions))
        scheduled_patients = []
        overtime = np.zeros(len(sample_path)+env.planning_horizon)
        postponing_decisions = []
        for t, ((regular_bookings, overtimes, waitlist), (advance_scheduling_decision, overtime_decision)) in enumerate(zip(states, actions)):
            scheduled_patients.append(advance_scheduling_decision.tolist())
            start = t
            end = t + len(overtime_decision)
            overtime[start:end] += overtime_decision
            postponing_decisions.append(waitlist - advance_scheduling_decision.sum(axis=0))
        postponing_decisions = np.array(postponing_decisions).sum(axis=0)
        print("Total postponing decisions:", postponing_decisions)
    #agent_arg['penalty_coefficient'] = penalty_coefficient
    res = {
        "uid": uid,
        "total_cost": sum(costs),
        "total_penalty": sum(penalties),
        "experiment_name": experiment_name,
        "agent_name": {'agent_name': agent_name, 'args': args, 'lowerbound_args': lowerbound_args},
        "warm_up_periods": warm_up_periods,
        "costs": costs,
        "penalties": penalties,
        "scheduled_patients": scheduled_patients,
        "overtime": overtime.tolist(),
    }
    print("total cost:", sum(costs))
    print("total penalties:", sum(penalties))
    print("total combined:", sum(costs) + sum(penalties))
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def calcualte_penalized_lowerbound_with_same_initial_state(env_args, experiment_name, agent_arg,  lowerbound_args, generating_function, warm_up_periods, sample_path, uid, job_id):
    '''
        1. read hindsight_approx data
        2. use actions to determine states in each period
        3. starting at warmup_period + 1, solve lowerbound
    '''
    aid_arg = {
            'agent_arg': agent_arg,
            'lowerbound_args':lowerbound_args
        }
    aid = get_uid(aid_arg)
    pickle_file = os.path.join('experiments', 'results', experiment_name, 'pickles',
                            f'{uid}-{aid}.pickle')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    data = load_pickle_if_exists(pickle_file)
    # 2) Decide env, state trajectory, etc.
    warmup_state = None
    if data != None:
        states = data['states']
        actions = data['actions']
        costs = data['costs']
        penalties = data['penalties']
        config = get_config_by_type(case_type='infinite_custom', args=env_args)
        config.reset_params['new_arrivals'] = sample_path
        env = config.env
        actions = data['actions']
        costs = data['costs'][:warm_up_periods]
        penalties = data['penalties'][:warm_up_periods]
        t = 1
        warmup_state, info = env.reset(**config.reset_params)
        for tau in range(warm_up_periods):
            a = actions[tau]
            warmup_state, cost, done, info = env.step(a)
        #lowerbound_args = {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False, 'coefficients': 1}
        agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, sample_path=sample_path[warm_up_periods:], generating_function=generating_function, **lowerbound_args)
        #print('warmup_state:', warmup_state, sample_path[warm_up_periods])
        my_action, benchmark_value, info = agent_instance.solve(warmup_state)
        costs += [cost.getValue() for cost in info['costs'][0]]
        penalties += [penalty if isinstance(penalty, int) else penalty.getValue() for penalty in info['penalties'][0]] if 'penalties' in info else []
        print("total cost:", sum(costs))
        print("total penalties:", sum(penalties))
        print("total combined:", sum(costs) + sum(penalties))
        actions = info['actions'][0]
        scheduled_patients = []
        overtime = np.zeros(len(sample_path)+env.planning_horizon)
        for t, action_var in enumerate(actions):
            advance_scheduling_decision, overtime_decision = get_solution(action_var, is_final=False)
            scheduled_patients.append(advance_scheduling_decision.tolist())
            start = t
            end = t + len(overtime_decision)
            overtime[start:end] += overtime_decision
        agent_name, args = agent_arg['agent_name'], agent_arg['args']
        res = {
        "uid": uid,
        "total_cost": sum(costs),
        "total_penalty": sum(penalties),
        "experiment_name": experiment_name,
        "agent_name": {'agent_name': "penalized_lowerbound_" + agent_name, 'args': args, 'lowerbound_args': lowerbound_args},
        "warm_up_periods": warm_up_periods,
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
    else:
        print(f'{uid}-{aid}.pickle not exists')


def caclaulte_information_relexation_cost(env, env_args, experiment_name, lowerbound_args, generating_function, init_state, sample_path, job_id):
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
    

def _evaluate_one_scenario(task):
    """Top-level worker for ProcessPoolExecutor — must be module-level for pickling.
    Each process builds its own env/Gurobi env so nothing is shared across workers.
    """
    env_args, experiment_name, coefficients, init_state, sample_path, job_id = task
    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    generating_function = LinearPenaltyFunction(env, coefficients=coefficients)
    zero_args     = {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous',
                     'is_myopic': False, 'is_include_discount_factor': False, 'coefficients': 0}
    penalized_args = {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous',
                      'is_myopic': False, 'is_include_discount_factor': False, 'coefficients': 1}
    zero_res     = caclaulte_information_relexation_cost(env, env_args, experiment_name, zero_args,     generating_function, init_state, sample_path, job_id)
    penalized_res = caclaulte_information_relexation_cost(env, env_args, experiment_name, penalized_args, generating_function, init_state, sample_path, job_id)
    gap = penalized_res['penalized_cost'] - zero_res['penalized_cost']
    return gap, penalized_res['penalized_cost'], zero_res['penalized_cost']


def calculate_information_relexation_costs(env_args, experiment_name, train_sample_path_num, test_sample_path_num, job_id, num_workers=None):
    # ---------- Train penalty coefficients (sequential) ----------
    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    generating_function = LinearPenaltyFunction(env=env)
    t0 = time.time()
    agent = InfinitePenalizedSAAAgent(env=env, discount_factor=0.99, sample_path_number=train_sample_path_num,
                                      generating_function=generating_function, is_myopic=False)
    obj, coefficients, info = agent.reformulate_train(coefficient_bound=GRB.INFINITY)
    print(f"Training time: {time.time() - t0:.1f}s")

    # ---------- Pre-generate all test scenarios in the main process ----------
    # (preserves reproducibility / deterministic RNG order)
    scenarios = []
    for _ in range(test_sample_path_num):
        init_state  = env.generate_initial_state()
        sample_path = env.reset_arrivals()
        scenarios.append((env_args, experiment_name, coefficients,
                          init_state, sample_path, job_id))

    # ---------- Parallel evaluation ----------
    penalized_lowerbound_stats = RunningStats()
    zero_penalized_lowerbound_stats = RunningStats()
    gap_stats = RunningStats()
    n_workers = num_workers or min(test_sample_path_num, (os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_one_scenario, task): i
                   for i, task in enumerate(scenarios)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                gap, penalized_cost, zero_cost = future.result()
                gap_stats += gap
                penalized_lowerbound_stats += penalized_cost
                zero_penalized_lowerbound_stats += zero_cost
                print(f"Scenario {i} done | gap={gap:.2f} | penalized_cost={penalized_cost:.2f} | zero_cost={zero_cost:.2f}")
            except Exception as exc:
                print(f"Scenario {i} raised: {exc}")

    # ---------- Aggregate ----------
    relative_improvement_stats = (gap_stats / zero_penalized_lowerbound_stats.mean) / 0.01
    print('Train objective:', obj)
    print("Test penalized cost:", penalized_lowerbound_stats)
    print("Test zero penalized cost:", zero_penalized_lowerbound_stats)
    print("Gap:", gap_stats)
    print("Relative improvement stats:", relative_improvement_stats)
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    parser.add_argument('--job_id', help='Input METAJOB_ID', type=str)
    args = parser.parse_args()
    params = json.loads(args.params)
    # alp_train(**params, job_id=args.job_id)
    #experiment(**params, job_id=args.job_id)
    #value_function_experiment(**params, job_id=args.job_id)
    #run_lower_bound_solver(**params, job_id=args.job_id)
    #run_penalized_lower_bound_solver(**params, job_id=args.job_id)
    #simulate_evaluation(**params, job_id=args.job_id)
    # restore_costs(**params, job_id=args.job_id)
    # calcualte_lowerbound_with_same_initial_state(**params, job_id=args.job_id)
    
    # penalty_coefficients = [round(i,1) for i in range(0, 2)]
    # env_args = params['env_args']
    # config = get_config_by_type(case_type='infinite_custom', args=env_args)
    # env = config.env
    # coefficients = [7.5014254032033, 30.591338709685797, 97.09300537634867, 2.5151471773961083, 29.26133870970855, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, -100.0, -47.053083333327514, 1.6391031893291558e-11, 0.32999999997006774, -3.56699462365624]
    # # coefficients = [100.0, 100.0, -100.0, 100.0, -100.0, -100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0]
    
    # generating_function = LinearPenaltyFunction(env, coefficients=coefficients)
    # print(generating_function.get_coefficients(coefficients))
    # for penalty_coefficient in penalty_coefficients:
    #     lowerbound_args = {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False, 'coefficients': penalty_coefficient}
    #     evaluate_lower_bound(**params, lowerbound_args=lowerbound_args, generating_function=generating_function, job_id=args.job_id)
    #     calcualte_penalized_lowerbound_with_same_initial_state(**params, lowerbound_args=lowerbound_args, generating_function=generating_function, job_id=args.job_id)
    calculate_information_relexation_costs(**params, train_sample_path_num=350,test_sample_path_num=8000, job_id=args.job_id)