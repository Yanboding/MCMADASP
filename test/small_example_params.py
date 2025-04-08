from environment import MultiClassPoissonArrivalGenerator

decision_epoch = 3
class_number = 2
arrival_generator = MultiClassPoissonArrivalGenerator(3, 1, [1 / class_number] * class_number)
env_params = {
    'treatment_pattern': [[2, 1]],
    'decision_epoch': decision_epoch,
    'arrival_generator': arrival_generator,
    'holding_cost': [10, 5],
    'overtime_cost': 40,
    'duration': 1,
    'regular_capacity': 5,
    'discount_factor': 1,
    'problem_type':'advance'
}