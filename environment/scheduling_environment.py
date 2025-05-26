import copy
import numpy as np
from scipy.stats import truncnorm

from utils import numpy_shift, RunningStat
from .utility import get_valid_advance_actions

class SchedulingEnv:

    def __init__(self,
                 treatment_pattern,
                 decision_epoch,
                 arrival_generator,
                 holding_cost,
                 overtime_cost,
                 duration,
                 regular_capacity,
                 discount_factor
                 ):
        self.treatment_pattern = np.array(treatment_pattern)
        self.decision_epoch = decision_epoch
        self.arrival_generator = arrival_generator
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.discount_factor = discount_factor
        self.num_sessions, self.num_types = self.treatment_pattern.shape

    def valid_actions(self, state, t):
        '''
        if days to go is less or equal to 0, then valid actions is the same as waitlist
        '''
        bookings, waitlist = copy.deepcopy(state)
        days_to_go = self.decision_epoch - t  # fix end booking date
        valid_actions = get_valid_advance_actions(waitlist, days_to_go + 1)
        return valid_actions

    def convert_action_to_booking_slots(self, action):
        appointment_slots = action @ self.treatment_pattern.T
        N, P = appointment_slots.shape
        total_len = len(action) + self.num_sessions - 1
        booked_slots = np.zeros(total_len, dtype=appointment_slots.dtype)

        # 2.  Vectorised diagonal add:
        #     element (i,j) in `appointment_slots` goes to position i+j in `booked_slots`.
        idx = np.arange(P) + np.arange(N)[:, None]  # shape (N,P)
        np.add.at(booked_slots, idx.ravel(), appointment_slots.ravel())
        return booked_slots

    def cost_fn(self, state, action, t):
        '''
        bookings = [1,2,3,4,5]
        action = [[1,2,3],[4,5,6]]
        '''
        bookings, _ = state
        waiting_cost = sum(sum(self.discount_factor ** k * self.holding_cost(k, i) for k in range(j + 1)) * action[j, i]
                           for j in range(len(action))
                           for i in range(len(action[0])))
        new_bookings = bookings + self.convert_action_to_booking_slots(action)
        overtime_cost = self.overtime_cost * np.maximum(new_bookings[0] * self.duration - self.regular_capacity, 0)
        if t == self.decision_epoch:
            # Only consider the tail overtime if we're at the last decision epoch
            for k in range(1, len(new_bookings)):
                overtime_cost += self.discount_factor ** k * self.overtime_cost * np.maximum(new_bookings[k] * self.duration - self.regular_capacity, 0)
        return waiting_cost + overtime_cost

    def post_action_state(self, state, action):
        bookings, waitlist = copy.deepcopy(state)
        if any(waitlist != action.sum(axis=0)):
            print(state)
            print(action)
            raise ValueError('Invalid Action')
        new_bookings = bookings + self.convert_action_to_booking_slots(action)
        new_waitlist = waitlist - action.sum(axis=0)
        return (new_bookings, new_waitlist)

    def post_action_state_to_new_state(self, post_action_state, new_arrival):
        post_action_bookings, post_action_waitlist = copy.deepcopy(post_action_state)
        new_bookings = post_action_bookings[1:]
        return (new_bookings, new_arrival)

    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action, t)
        res = []
        for prob, delta in self.arrival_generator.get_system_dynamic():
            post_action_state = self.post_action_state(state, action)
            if t + 1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            done = t == self.decision_epoch
            next_state = self.post_action_state_to_new_state(post_action_state, delta)
            res.append([prob, next_state, cost, done])
        return res

    def reset(self, init_state=None, t=1, new_arrivals=None, percentage_occupied=0, seed=None):
        if new_arrivals is not None and len(new_arrivals) != self.decision_epoch - t + 1:
            print("length of new arrivals:", len(new_arrivals), "length of decision epoch:",self.decision_epoch - t + 1)
            raise ValueError('Invalid sample path!')
        self.t = t
        self.tau = 0
        # how to handle the first arrivals
        if new_arrivals is None:
            self.new_arrivals = self.reset_arrivals(t)
        else:
            self.new_arrivals = new_arrivals
        if init_state == None:
            init_state = self.reset_initial_state(t, percentage_occupied, seed)
        bookings, waitlist = init_state
        bookings = np.array(bookings)
        waitlist = np.array(waitlist)
        self.state = (bookings, waitlist)
        # measure of performance
        self.wait_time_by_type = {j: RunningStat((1,)) for j in range(self.num_types)}
        self.overtime = np.array([0] * (self.decision_epoch + self.num_sessions - t))
        return copy.deepcopy(self.state), {'wait_time_by_type': self.wait_time_by_type,
                                                  'overtime': self.overtime}

    def reset_arrivals(self, t=1):
        return self.arrival_generator.rvs(self.decision_epoch-t+1)

    def reset_initial_state(self, t, percentage_occupied, seed=None):
        # find out the average appointment slot required in first period
        capacity_occupied = self.regular_capacity * percentage_occupied
        # initialize the current booking slots with all zeros
        booking_horizon = self.decision_epoch-t + self.num_sessions
        if seed is not None:
            np.random.seed(seed)
        # Step 1: Generate from truncated normal distribution
        mean = 1.0
        std_dev = 0.3
        lower, upper = 0, 2
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=booking_horizon)
        # Step 2: Scale so that the average is exactly 100 * p
        scaled = samples / samples.mean() * capacity_occupied
        return (scaled, self.new_arrivals[0])

    def step(self, action):
        # t+tau
        cost = self.cost_fn(self.state, action, self.t + self.tau)
        post_action_state =(post_action_bookings, _)= self.post_action_state(self.state, action)
        done = self.t + self.tau == self.decision_epoch
        # record performance metric
        # implement info: include the type-dependent waiting times and overtime use
        wait_times = np.arange(action.shape[0])
        for i in range(action.shape[1]):
            self.wait_time_by_type[i].record_batch(wait_times, action[:, i])
        self.overtime[self.tau] = np.maximum(post_action_bookings[0] - self.regular_capacity, 0)
        if done:
            self.overtime[self.tau:] = np.maximum(post_action_bookings - self.regular_capacity, 0)
        # update state
        self.tau += 1
        if self.t + self.tau > self.decision_epoch:
            delta = np.zeros(self.num_types, dtype=int)
        else:
            delta = self.new_arrivals[self.tau]
        self.state = self.post_action_state_to_new_state(post_action_state, delta)
        return self.state, cost, done, {'wait_time_by_type': self.wait_time_by_type, 'overtime': self.overtime}


if __name__ == '__main__':
    from experiments import ExperimentConfig
    config = ExperimentConfig.from_multiappt_default_case()
    env = config.env
    init_state = config.init_state
    action = np.array([[3,3], [3,3], [0,0]])
    print(env.transition_dynamic(init_state, action, 1))