import copy
import time
from pprint import pprint

import numpy as np
from utils import numpy_shift, RunningStat
from .utility import get_valid_advance_actions, get_valid_allocation_actions


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

    '''
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
    '''

    def post_advance_scheduling_state(self, state, action):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        if any(waitlist != action.sum(axis=0)):
            print(state)
            print(action)
            raise ValueError('Invalid Action')
        post_action_future_appts = future_first_appts + action
        next_first_appts = post_action_future_appts[0] if len(post_action_future_appts) > 0 else 0
        post_action_bookings = bookings + (self.treatment_pattern * next_first_appts).sum(axis=1)
        post_action_waitlist = waitlist - action.sum(axis=0)
        return (post_action_bookings, post_action_waitlist, post_action_future_appts)

    def post_action_state_to_new_state(self, post_action_state, new_arrival):
        post_action_bookings, post_action_waitlist, post_action_future_appts = self.interpret_state(post_action_state)
        new_bookings = numpy_shift(post_action_bookings, -1)
        new_waitlist = post_action_waitlist + new_arrival
        new_future_schedule = post_action_future_appts[1:]
        return (new_bookings, new_waitlist, new_future_schedule)

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

    '''
    def post_allocation_state(self, state, action):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        next_first_appts = future_first_appts[0] if len(future_first_appts) > 0 else 0
        next_first_appts = next_first_appts + action
        next_bookings = numpy_shift(bookings + (self.treatment_pattern * next_first_appts).sum(axis=1), -1)
        next_waitlist = waitlist - action
        next_future_first_appts = future_first_appts[1:]
        return (next_bookings, next_waitlist, next_future_first_appts)
    '''

    # Need to verify
    def post_allocation_state(self, state, action):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        next_first_appts = future_first_appts[0] if len(future_first_appts) > 0 else 0
        current_post_first_appts = next_first_appts + action
        post_action_bookings = bookings + (self.treatment_pattern * current_post_first_appts).sum(axis=1)
        post_action_waitlist = waitlist - action
        future_first_appts[0] = current_post_first_appts
        return (post_action_bookings, post_action_waitlist, future_first_appts)

    # Need to verify
    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action, t)
        res = []
        for prob, delta in self.arrival_generator.get_system_dynamic():
            post_state = self.post_state(state, action)
            if t+1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            done = t == self.decision_epoch
            next_state = (next_bookings, next_waitlist, new_future_first_appts) = self.post_action_state_to_new_state(
                post_state, delta)
            if (next_waitlist.sum() < 0) or (done and next_waitlist.sum() > 0):
                print('time:', t, 'decision_epoch:', self.decision_epoch)
                print(state)
                print(action)
                raise ValueError("Invalid action")
            res.append([prob, next_state, cost, done])
        return res

    def reset(self, init_state=None, t=1, new_arrivals=None, percentage_occupied=0):
        if new_arrivals is not None and len(new_arrivals) != self.decision_epoch-t+1:
            raise ValueError('Invalid sample path!')
        self.t = t
        self.tau = 0
        # how to handle the first arrivals
        if new_arrivals is None:
            self.new_arrivals = self.reset_arrivals(t)
        else:
            self.new_arrivals = new_arrivals
        if init_state == None:
            init_state = self.reset_initial_state(t, percentage_occupied)
        bookings, waitlist, future_first_appts = init_state
        bookings = np.array(bookings)
        waitlist = np.array(waitlist)
        future_first_appts = np.array(future_first_appts)
        self.future_first_appts_copy = copy.deepcopy(future_first_appts)
        self.state = (bookings, waitlist, future_first_appts)
        # measure of performance
        self.wait_time_by_type = {j: RunningStat((1,)) for j in range(self.num_types)}
        self.overtime = np.array([0]*(self.decision_epoch+self.num_sessions - t))
        return self.interpret_state(self.state), {'wait_time_by_type':self.wait_time_by_type, 'overtime':self.overtime}

    def reset_arrivals(self, t=1):
        return self.arrival_generator.rvs(self.decision_epoch-t+1)

    def reset_initial_state(self, t, percentage_occupied):
        # find out the average appointment slot required in first period
        average_pattern = self.arrival_generator.type_probs @ self.treatment_pattern.T
        min_capacity_occupied = self.regular_capacity * percentage_occupied
        # initialize the current booking slots with all zeros
        booking_horizon = self.decision_epoch-t+1
        booked_slots = np.zeros(booking_horizon+self.num_sessions-1)
        init_bookings = np.array([int(min_capacity_occupied)]*self.num_sessions)
        init_future_schedule = np.zeros((booking_horizon, self.num_types))
        booked_slots[:self.num_sessions] += init_bookings
        # initialize the bookings z to be regular_capacity * min_percentage_occupied
        # for each period j:
        #    if the percentage of the occupancy is less than the min_percentage_occupied
        #        do the calculation of number of treatment I should allocate in period j
        #        the remaining number of slots/ average appointment slot
        #        sample new arrivals from each type
        #        update the current booking slots
        for j in range(booking_horizon):
            if booked_slots[j] < min_capacity_occupied:
                remaining_slot_num = max(min_capacity_occupied - booked_slots[j], 0)
                new_arrival = np.round(remaining_slot_num/average_pattern[0])
                init_future_schedule[j] = self.arrival_generator.arrival_type_rvs(new_arrival)[0]
                booked_slots[j:j+self.num_sessions] += init_future_schedule[j] @ self.treatment_pattern.T
        init_waitlist = self.new_arrivals[0]
        return (init_bookings, init_waitlist, init_future_schedule)

    def step(self, action):
        # t+tau
        cost = self.cost_fn(self.state, action, self.t+self.tau)
        post_state = (post_action_bookings, post_action_waitlist, post_action_future_appts) = self.post_state(self.state, action)
        done = self.t + self.tau == self.decision_epoch
        if (post_action_waitlist.sum() < 0) or (done and post_action_waitlist.sum() > 0):
            print('time:', self.t, 'decision_epoch:', self.decision_epoch)
            print(self.state)
            print(action)
            raise ValueError("Invalid action")
        # record performance metric
        # implement info: include the type-dependent waiting times and overtime use
        wait_times = np.arange(action.shape[0])
        for i in range(action.shape[1]):
            self.wait_time_by_type[i].record_batch(wait_times, action[:, i])
        self.overtime[self.tau] = np.maximum(post_action_bookings[0]-self.regular_capacity, 0)
        if done:
            self.overtime[self.tau:] = np.maximum(post_action_bookings-self.regular_capacity, 0)
        # update state
        self.tau += 1
        if self.t + self.tau > self.decision_epoch:
            delta = np.zeros(self.num_types, dtype=int)
        else:
            delta = self.new_arrivals[self.tau]
        self.state = self.post_action_state_to_new_state(post_state, delta)
        return self.state, cost, done, {'wait_time_by_type':self.wait_time_by_type, 'overtime':self.overtime}

if __name__ == "__main__":
    from waiting_cost import wait_time_penalty
    from experiments import Config
    import matplotlib.pyplot as plt
    config = Config.from_multiappt_default_case()
    env_params = config.env_params
    env = AdvanceSchedulingEnv(**env_params)
    bookings = np.array([2]*(env.decision_epoch+env.num_sessions-1))
    action = np.array([[2,1] for i in range(env.decision_epoch)])
    print(bookings)
    print(action)
    print(env.advance_scheduling_one_time_cost_fn(bookings, action, 1))