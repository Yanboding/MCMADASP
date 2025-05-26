import numpy as np
import copy
from scipy.stats import poisson
def get_valid_advance_actions(waitlist, number_days):
    if number_days < 1:
        raise ValueError('number_bins must be at least 1')
    def valid_actions_for_one_class(x, y):
        def backtracking(x, y, current_combination):
            if y == 0:
                if x == 0:
                    all_combinations.append(current_combination.copy())
                return

            for i in range(x + 1):
                current_combination.append(i)
                backtracking(x - i, y - 1, current_combination)
                current_combination.pop()

        all_combinations = []
        backtracking(x, y, [])
        return all_combinations

    actions = []

    def backtracking(current, waitlist):
        if len(waitlist) == 0:
            actions.append(copy.deepcopy(current))
            return
        for combination in valid_actions_for_one_class(waitlist[0], number_days):
            current.append(combination)
            backtracking(current, waitlist[1:])
            current.pop()

    backtracking([], waitlist)
    return np.array(actions).transpose((0, 2, 1))
'''

def get_valid_advance_actions(waitlist, number_days):
    if number_days < 1:
        raise ValueError('number_days must be at least 1')

    @functools.lru_cache(maxsize=None)
    def compositions(n, k):
        # Generate all tuples of k non-negative integers summing to n
        if k == 1:
            return [(n,)]
        results = []
        for i in range(n + 1):
            for tail in compositions(n - i, k - 1):
                results.append((i,) + tail)
        return results

    def get_all_class_combinations():
        return [compositions(w, number_days) for w in waitlist]

    # Cartesian product of all class-level combinations
    all_class_combinations = get_all_class_combinations()
    all_actions = list(itertools.product(*all_class_combinations))

    # Convert to ndarray: (num_actions, number_days, num_classes)
    return np.array(all_actions).transpose((0, 2, 1))
'''

def get_system_dynamic(mean_arrival, maximum_arrival, type_probs):
    total_poisson = poisson(mean_arrival)
    poisson_by_type = [poisson(mean_arrival * p) for p in type_probs]
    system_dynamic = []

    def calculate_transition_dynamic(arrival_by_type):
        prob = 1
        for num, dis in zip(arrival_by_type, poisson_by_type):
            prob *= dis.pmf(num)
        prob /= total_poisson.cdf(maximum_arrival)
        return [prob, np.array(arrival_by_type)]

    def get_transition_dynamic(total_arrival, num_types):
        def backtracking(total_arrival, num_types, current_combination):
            if num_types == 0:
                if total_arrival == 0:
                    all_combinations.append(calculate_transition_dynamic(current_combination.copy()))
                return

            for i in range(total_arrival + 1):
                current_combination.append(i)
                backtracking(total_arrival - i, num_types - 1, current_combination)
                current_combination.pop()

        all_combinations = []
        backtracking(total_arrival, num_types, [])
        return all_combinations

    for total_arrival in range(maximum_arrival + 1):
        system_dynamic += get_transition_dynamic(total_arrival, len(type_probs))

    return system_dynamic

def get_valid_allocation_actions(waitlist):
    actions = []
    def backtracking(current, current_waitlist):
        if len(current_waitlist) == 0:
            actions.append(current.copy())
            return
        for w in range(int(current_waitlist[0] + 1)):
            current.append(w)
            backtracking(current, current_waitlist[1:])
            current.pop()
    backtracking([], waitlist)
    return np.array(actions)

if __name__ =='__main__':
    #system_dynamic = get_system_dynamic_fast(10, 30, [0.4, 0.3, 0.2, 0.1])
    class_number = 8
    probability = 1 / class_number
    actions = get_valid_advance_actions([3, 3, 4], 100)
    print(actions)
    '''
    probabilities = [item[0] for item in system_dynamic]
    arrivals = [item[1] for item in system_dynamic]
    rng = np.random.default_rng(1)
    N = 100
    new_arrivals = np.array([arrivals[rng.choice(len(arrivals), p=probabilities)] for i in range(N)])
    print(new_arrivals)
    '''
