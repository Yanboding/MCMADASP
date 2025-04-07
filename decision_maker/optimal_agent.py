import os
from collections import defaultdict
import sys
import os
from pprint import pprint

from environment import MultiClassPoissonArrivalGenerator, AdvanceSchedulingEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import iter_to_tuple, convert_str_keys_to_tuple, convert_tuple_keys_to_str, keep_significant_digits
import numpy as np
import matplotlib.pyplot as plt
import json
class OptimalAgent:

    def __init__(self, env, discount_factor, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(float))
        if V is None:
            self.V = {}

    def train(self, state, t):
        def dfs(state, t):
            '''
            print('State:', t)
            print(state)
            '''
            state_tuple = iter_to_tuple(state)
            min_total_cost = float('inf')
            for action in self.env.valid_actions(state, t):
                '''
                print('Action:')
                print(action)
                '''
                action_tuple = iter_to_tuple(action)
                q = 0
                for prob, next_state, cost, done in self.env.transition_dynamic(state, action, t):
                    '''
                    print('\tNext_state:', prob)
                    print('\t',next_state)
                    print('\tCost:')
                    print('\t',cost)
                    '''
                    next_state_tuple = iter_to_tuple(next_state)
                    if (next_state_tuple, t + 1) in self.V:
                        next_state_val = self.V[(next_state_tuple, t + 1)]
                    elif done:
                        next_state_val = 0
                    else:
                        next_state_val = dfs(next_state, t + 1)
                    q += prob * (cost + self.discount_factor * next_state_val)
                self.Q[(state_tuple, t)][action_tuple] = q
                if q < min_total_cost:
                    min_total_cost = q
            self.V[(state_tuple, t)] = min_total_cost
            return self.V[(state_tuple, t)]
        dfs(state, t)

    def policy(self, state, t):
        # All actions that available in the given state
        state_tuple = iter_to_tuple(state)
        minValue = float('inf')
        bestAction = None
        for action, qValue in self.Q[(state_tuple, t)].items():
            if qValue < minValue or (qValue == minValue and action > bestAction):
                minValue = qValue
                bestAction = action
        return np.array(bestAction)

    def get_action_value(self, state, action, t):
        state_tuple = iter_to_tuple(state)
        action_tuple = iter_to_tuple(action)
        return self.Q[(state_tuple, t)][action_tuple]

    def get_action_values(self, state, t):
        state_tuple = iter_to_tuple(state)
        return self.Q[(state_tuple, t)]

    def get_state_value(self, state, t):
        state_tuple = iter_to_tuple(state)
        return self.V[(state_tuple, t)]

    def save(self, save_path):
        with open(os.path.join(save_path, 'state_value_function.json'), 'w') as json_file:
            json.dump(convert_tuple_keys_to_str(self.V), json_file)
        with open(os.path.join(save_path, 'action_value_function.json'), 'w') as json_file:
            json.dump(convert_tuple_keys_to_str(self.Q), json_file)

    def load(self, load_path):
        # Open the JSON file
        with open(os.path.join(load_path, 'state_value_function.json'), 'r') as json_file:
            # Load the JSON data into a dictionary
            self.V = convert_str_keys_to_tuple(json.load(json_file))
        with open(os.path.join(load_path, 'action_value_function.json'), 'r') as json_file:
            self.Q = defaultdict(lambda: defaultdict(int), convert_str_keys_to_tuple(json.load(json_file)))

    def action_value_plot(self, state, t):
        _, demand, _ = state
        state_tuple = iter_to_tuple(state)
        x = []
        y = []
        z = []
        for action, qValue in self.Q[(state_tuple, t)].items():
            x.append(action[0])
            y.append(action[1])
            z.append(qValue)
        # Example 1D arrays
        X = np.array(x).reshape(demand[0]+1, -1)
        Y = np.array(y).reshape(demand[0]+1, -1)
        Z = np.array(z).reshape(demand[0]+1, -1)
        # Create a 3D figure
        # Create a figure with 2 subplots
        # Create the contour plot
        fig, ax = plt.subplots(figsize=(6, 6))
        #contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')

        # Label Z values only where X and Y are integers
        #ax.clabel(contour, inline=True, fontsize=10)
        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis')
        # Set view angle to make it look more 2D
        ax.view_init(elev=90, azim=-90)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Contour Plot with Z Labels at Integer X and Y')

        # Show the plot
        plt.show()

    def action_value_3d_plot(self, state, t):
        plt.ion()

        _, demand = state
        state_tuple = iter_to_tuple(state)
        x = []
        y = []
        z = []
        for action, qValue in self.Q[(state_tuple, t)].items():
            x.append(action[0])
            y.append(action[1])
            z.append(qValue)
        # Example 1D arrays
        X = np.array(x).reshape(demand[0] + 1, -1)
        Y = np.array(y).reshape(demand[0] + 1, -1)
        Z = np.array(z).reshape(demand[0] + 1, -1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        # Set labels
        ax.set_xlabel(r'Type 1 Action($q_1$)', fontsize=12)
        ax.set_ylabel(r'Type 2 Action($q_2$)', fontsize=12)
        ax.set_zlabel('Action Value Function', fontsize=12)

        # Add labels and title
        ax.set_title('Optimal policy: ' + str(self.policy(state, t))+' Optimal Value: '+ str(keep_significant_digits(self.get_state_value(state, t), 6)), fontsize=16)
        # Increase font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='z', which='major', labelsize=11)
        # Show the plot
        plt.savefig(str((state_tuple[:-1], t)) + '.png', bbox_inches='tight')
        plt.show()

    def state_value_3d_plot(self, t):
        values = []
        x_num = -float('inf')
        min_value = -float('inf')
        for (state, period), state_value in self.V.items():
            if period == t:
                _, (w1, w2), _ = state
                x_num = max(x_num, w1)
                min_value = max(min_value, state_value)
                values.append([w1, w2, state_value])
        values = np.array(sorted(values, key=lambda x: (x[0], x[1])))
        X = values[:, 0].reshape(x_num + 1, -1)
        Y = values[:, 1].reshape(x_num + 1, -1)
        Z = values[:, 2].reshape(x_num + 1, -1)
        print(X)
        print(Y)
        print(Z)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        # Set labels
        ax.set_xlabel(r'Type 1 Waitlist($w_1$)', fontsize=12)
        ax.set_ylabel(r'Type 2 Waitlist($w_2$)', fontsize=12)
        ax.set_zlabel('State Value Function', fontsize=12)

        # Increase font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='z', which='major', labelsize=11)
        # Show the plot
        plt.savefig(str(t) + '.png')
        plt.show()

if __name__ == '__main__':
    decision_epoch = 3
    class_number = 2
    arrival_generator = MultiClassPoissonArrivalGenerator(100, 1, [1 / class_number] * class_number)
    env_params = {
        'treatment_pattern': [[1, 1]],
        'decision_epoch': decision_epoch,
        'arrival_generator': arrival_generator,
        'holding_cost': [10, 5],
        'overtime_cost': 4000,
        'duration': 1,
        'regular_capacity': 5,
        'discount_factor': 0.99,
        'problem_type': 'advance'
    }
    bookings = np.array([0])
    future_schedule = np.array([[0] * class_number for i in range(decision_epoch)])
    new_arrival = np.array([5, 6])
    init_state = (bookings, new_arrival, future_schedule)
    t = 1

    env = AdvanceSchedulingEnv(**env_params)

    print('Waiting for Optimal...')
    optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
    optimal_agent.train(init_state, t)
    optimal_value = optimal_agent.get_state_value(init_state, t)
    optimal_action = optimal_agent.policy(init_state, t)
    #print('Optimal Done:', optimal_value, 'Optimal action:', optimal_action)
    pprint(optimal_agent.get_action_values(init_state, t))
    # [[5, 0], [1, 4], [0, 2]]
    # [[5, 0], [0, 6], [0, 0]]  [[5, 0], [0, 5], [0, 1]]: 125.6823125
    # 120.78181249999999
