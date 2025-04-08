import copy
import time
from pprint import pprint

import numpy as np
from utils import numpy_shift, iter_to_tuple
from .utility import get_valid_advance_actions, get_valid_allocation_actions
from .arrival_generator import MultiClassPoissonArrivalGenerator


class AdvanceSchedulingEnv:
    def __init__(self,
                 treatment_pattern,
                 decision_epoch,
                 arrival_generator,
                 holding_cost,
                 overtime_cost,
                 duration,
                 regular_capacity,
                 discount_factor,
                 problem_type='advance'
                 ):
        self.treatment_pattern = np.array(treatment_pattern)
        self.decision_epoch = decision_epoch
        self.arrival_generator = arrival_generator
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.discount_factor = discount_factor
        self.problem_type = problem_type
        self.valid_actions = self.valid_advance_scheduling_actions
        self.post_state = self.post_advance_scheduling_state
        self.cost_fn = self.advance_scheduling_cost_fn
        if problem_type == 'allocation':
            self.valid_actions = self.valid_allocation_actions
            self.post_state = self.post_allocation_state
            self.cost_fn = self.allocation_cost_fn
        self.num_sessions, self.num_types = self.treatment_pattern.shape

    def interpret_state(self, state):
        '''
        (number of bookings, waitlist, future first appointments)
        :param state:
        :return:
        '''
        bookings, waitlist, future_first_appts = state
        return copy.deepcopy(bookings), copy.deepcopy(waitlist), copy.deepcopy(future_first_appts)

    def valid_advance_scheduling_actions(self, state, t):
        '''
        if days to go is less or equal to 0, then valid actions is the same as waitlist
        '''
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        days_to_go = self.decision_epoch - t  # fix end booking date
        valid_actions = get_valid_advance_actions(waitlist, days_to_go + 1)
        return valid_actions

    def one_time_cost(self, bookings, outstanding_treatments, upcoming_schedule, t):
        cost = (self.holding_cost * outstanding_treatments).sum() + self.overtime_cost * np.maximum(
            (bookings[0] + (
                        upcoming_schedule * self.treatment_pattern[0]).sum()) * self.duration - self.regular_capacity,
            0)

        if t == self.decision_epoch:
            for i in range(1, len(bookings)):
                cost += self.discount_factor ** i * self.overtime_cost * np.maximum(
                    (bookings[i] + (upcoming_schedule * self.treatment_pattern[
                        i]).sum()) * self.duration - self.regular_capacity,
                    0)
        return cost
    def advance_scheduling_cost_fn(self, state, action, t):
        bookings, waitlist, future_schedule = self.interpret_state(state)
        new_future_schedule = future_schedule + action
        outstanding_treatments = new_future_schedule.sum(axis=0)
        upcoming_schedule = new_future_schedule[0]
        return self.one_time_cost(bookings, outstanding_treatments, upcoming_schedule, t)

    def post_advance_scheduling_state(self, state, action):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        if any(waitlist != action.sum(axis=0)):
            print(state)
            print(action)
            raise ValueError('Invalid Action')
        new_future_first_appts = future_first_appts + action
        next_first_appts = new_future_first_appts[0] if len(new_future_first_appts) > 0 else 0
        next_bookings = numpy_shift(bookings + (self.treatment_pattern * next_first_appts).sum(axis=1), -1)
        next_waitlist = waitlist - action.sum(axis=0)
        return (next_bookings, next_waitlist, new_future_first_appts[1:])

    def valid_allocation_actions(self, state, t):
        days_to_go = self.decision_epoch - t
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        if days_to_go <= 0:
            return np.array([waitlist])
        return get_valid_allocation_actions(waitlist)

    def allocation_cost_fn(self, state, action, t):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        outstanding_treatments = waitlist + future_first_appts.sum(axis=0)
        upcoming_schedule = future_first_appts[0] + action
        return self.one_time_cost(bookings, outstanding_treatments, upcoming_schedule, t)

    def post_allocation_state(self, state, action):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        next_first_appts = future_first_appts[0] if len(future_first_appts) > 0 else 0
        next_first_appts = next_first_appts + action
        next_bookings = numpy_shift(bookings + (self.treatment_pattern * next_first_appts).sum(axis=1), -1)
        next_waitlist = waitlist - action
        next_future_first_appts = future_first_appts[1:]
        return (next_bookings, next_waitlist, next_future_first_appts)

    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action, t)
        res = []
        for prob, delta in self.arrival_generator.get_system_dynamic():
            next_bookings, next_waitlist, new_future_first_appts = self.post_state(state, action)
            if t+1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            next_waitlist += delta
            done = t == self.decision_epoch
            if (next_waitlist.sum() < 0) or (done and next_waitlist.sum() > 0):
                print('time:', t, 'decision_epoch:', self.decision_epoch)
                print(state)
                print(action)
                raise ValueError("Invalid action")
            res.append([prob, (next_bookings, next_waitlist, new_future_first_appts), cost, done])
        return res

    def reset(self, init_state, t):
        self.t = t
        bookings, waitlist, future_first_appts = init_state
        bookings = np.array(bookings)
        waitlist = np.array(waitlist)
        future_first_appts = np.array(future_first_appts)
        self.future_first_appts_copy = copy.deepcopy(future_first_appts)
        self.state = (bookings, waitlist, future_first_appts)
        # how to handle the first arrivals
        self.new_arrivals = self.reset_arrivals(t)
        return self.interpret_state(self.state), {}

    def reset_arrivals(self, t=1):
        return self.arrival_generator.rvs(self.decision_epoch-t+1)

    def step(self, action):
        pass

if __name__ == "__main__":
    decision_epoch = 3
    class_number = 2
    arrival_generator = MultiClassPoissonArrivalGenerator(3, 3, [1 / class_number] * class_number)
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

    action = np.array([[5,0],[0,6],[0,0]])
    print(env.cost_fn(init_state, action, t))
    state = (np.array([0]), np.array([0, 0]), np.array([[0, 6], [0, 0]]))
    action = np.array([[0, 0], [0, 0]])
    t = 2
    print(env.cost_fn(state, action, t))