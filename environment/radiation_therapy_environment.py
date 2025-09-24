import copy
import functools
import itertools
import time

import numpy as np
from scipy.stats import truncnorm

from utils import numpy_shift, RunningStats, integer_partitions_fixed_bins, generate_advance_actions, \
    bounded_compositions


class RTEnv:
    def __init__(self,
                 treatment_pattern,
                 decision_epoch, # Note: not useful in the EJOR paper
                 booking_window_size,
                 arrival_generator,
                 holding_cost,
                 overtime_cost,
                 postponing_cost,
                 duration,
                 regular_capacity,
                 overtime_capacity,
                 discount_factor,
                 random_seed
                 ):
        self.treatment_pattern = np.array(treatment_pattern)
        self.decision_epoch = decision_epoch
        self.booking_window_size = booking_window_size # if the problem is finite, bokking_window_size == decision_epoch
        self.arrival_generator = arrival_generator
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.postponing_cost = postponing_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.overtime_capacity = overtime_capacity
        self.discount_factor = discount_factor
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.num_sessions, self.num_types = self.treatment_pattern.shape
        self.planning_horizon = self.booking_window_size + self.num_sessions - 1

    def get_state(self, state, is_var=False):
        if not is_var:
            return copy.deepcopy(state)
        return state

    def convert_action_to_booking_slots(self, advance_scheduling_decision):
        appointment_slots = advance_scheduling_decision @ self.treatment_pattern.T
        N, P = appointment_slots.shape
        total_len = len(advance_scheduling_decision) + self.num_sessions - 1
        booked_slots = np.zeros(total_len, dtype=appointment_slots.dtype)

        # 2.  Vectorised diagonal add:
        #     element (i,j) in `appointment_slots` goes to position i+j in `booked_slots`.
        idx = np.arange(P) + np.arange(N)[:, None]  # shape (N,P)
        np.add.at(booked_slots, idx.ravel(), appointment_slots.ravel())
        return booked_slots

    def generate_states(self):
        maximum_number_of_waitlist = self.arrival_generator.maximum_arrival
        for bookings_tuple in itertools.product(range(self.regular_capacity + 1), repeat=self.planning_horizon-1):
            for overtimes_tuple in itertools.product(range(self.overtime_capacity + 1), repeat=self.planning_horizon-1):
                for waitlist in bounded_compositions(maximum_number_of_waitlist, self.num_types):
                    yield np.array(bookings_tuple+(0,)), np.array(overtimes_tuple+(0,)), waitlist

    def valid_actions(self, state):
        regular_bookings, overtimes, waitlist = state
        per_type_generators = [bounded_compositions(w_i, self.booking_window_size) for w_i in waitlist]
        for advance_scheduling_decision in itertools.product(*per_type_generators):
            advance_scheduling_decision = np.array(advance_scheduling_decision).T
            new_booking_slots = self.convert_action_to_booking_slots(advance_scheduling_decision)
            overtime_decision = np.maximum(regular_bookings + new_booking_slots - self.regular_capacity, 0)
            if any(overtime_decision > self.overtime_capacity):
                continue
            yield (advance_scheduling_decision, overtime_decision)

    def generate_state_action_pairs(self):
        for state in self.generate_states():
            for action in self.valid_actions(state):
                yield (state, action)

    def cost_fn(self, state, action):
        regular_bookings, overtimes, waitlist = state
        advance_scheduling_decision, overtime_decision = action
        waiting_cost = sum(sum(self.discount_factor ** k * self.holding_cost(k, i) for k in range(j + 1)) * advance_scheduling_decision[j, i]
                           for j in range(len(advance_scheduling_decision))
                           for i in range(len(advance_scheduling_decision[0])))

        overtime_cost = sum(self.discount_factor ** j * self.overtime_cost(j) * overtime_decision[j] for j in range(len(overtime_decision)))
        remaining_treatments = waitlist - advance_scheduling_decision.sum(axis=0)
        postponing_cost = sum(self.postponing_cost(i) * remaining_treatments[i] for i in range(self.num_types))
        return waiting_cost + overtime_cost + postponing_cost

    def post_action_state(self, state, action, is_var=False):
        regular_bookings, overtimes, waitlist = self.get_state(state, is_var)
        advance_scheduling_decision, overtime_decision = action
        post_action_regular_bookings = regular_bookings + self.convert_action_to_booking_slots(
            advance_scheduling_decision) - overtime_decision
        post_action_overtimes = overtimes + overtime_decision
        post_action_waitlist = waitlist - advance_scheduling_decision.sum(axis=0)
        return (post_action_regular_bookings, post_action_overtimes, post_action_waitlist)

    def post_action_state_to_new_state(self, post_action_state, new_arrival, is_var=True):
        post_action_regular_bookings, post_action_overtimes, post_action_waitlist = self.get_state(post_action_state, is_var)
        new_regular_bookings = numpy_shift(post_action_regular_bookings, num_places=-1)
        new_overtimes = numpy_shift(post_action_overtimes, num_places=-1)
        new_waitlist  = post_action_waitlist + new_arrival
        return (new_regular_bookings, new_overtimes, new_waitlist)

    def get_next_state(self, state, action, new_arrival, is_var=False):
        post_action_state = self.post_action_state(state, action, is_var)
        next_state = self.post_action_state_to_new_state(post_action_state, new_arrival, is_var)
        return next_state

    def get_next_bookings(self, bookings, action):
        advance_scheduling_decision, overtime_decision = action
        post_action_bookings = bookings + self.convert_action_to_booking_slots(advance_scheduling_decision) - overtime_decision
        new_bookings = numpy_shift(post_action_bookings, num_places=-1)
        return new_bookings

    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action)
        res = []
        for prob, delta in self.arrival_generator.get_system_dynamic():
            if t + 1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            done = t == self.decision_epoch
            next_state = self.get_next_state(state, action, delta)
            res.append([prob, next_state, cost, done])
        return res

    # simulation
    def reset(self, init_state=None, t=1, new_arrivals=None, percentage_occupied=0):
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
            init_state = self.reset_initial_state(percentage_occupied, self.new_arrivals[0])
        bookings, overtimes, waitlist = init_state
        bookings = np.array(bookings)
        overtimes = np.array(overtimes)
        waitlist = np.array(waitlist)
        self.state = (bookings, overtimes, waitlist)
        # measure of performance
        self.wait_time_by_type = {j: RunningStats() for j in range(self.num_types)}
        self.overtime = np.array([0] * (self.planning_horizon - t + 1))
        return copy.deepcopy(self.state), {'wait_time_by_type': self.wait_time_by_type,
                                           'overtime': self.overtime}

    def reset_arrivals(self, t=1):
        return self.arrival_generator.rvs(self.decision_epoch - t + 1)

    def reset_initial_state(self, percentage_occupied, new_arrivals):
        # find out the average appointment slot required in first period
        capacity_occupied = (self.regular_capacity + self.overtime_capacity) * percentage_occupied
        # initialize the current booking slots with all zeros
        booking_horizon = self.planning_horizon + self.num_sessions - 1
        # Step 1: Generate from truncated normal distribution
        mean = 1.0
        std_dev = 0.3
        lower, upper = 0, 2
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=booking_horizon, random_state=self.rng)
        # Step 2: Scale so that the average is exactly 100 * p
        total_bookings = samples / samples.mean() * capacity_occupied
        overtimes = np.maximum(total_bookings - self.regular_capacity, 0)
        regular_bookings = total_bookings - overtimes
        return (regular_bookings, overtimes, new_arrivals)

    def step(self, action):
        advance_scheduling_decision, overtime_decision = action
        # t+tau
        cost = self.cost_fn(self.state, action)
        post_action_state = self.post_action_state(self.state, action)
        post_action_regular_bookings, post_action_overtimes, post_action_waitlist = post_action_state
        done = self.t + self.tau == self.decision_epoch
        # record performance metric
        # implement info: include the type-dependent waiting times and overtime use
        wait_times = np.arange(advance_scheduling_decision.shape[0])
        for i in range(advance_scheduling_decision.shape[1]):
            self.wait_time_by_type[i].record_batch(wait_times, advance_scheduling_decision[:, i])
        self.overtime[self.tau] = post_action_overtimes[0]
        if done:
            self.overtime[self.tau:] = post_action_overtimes
        # update state
        self.tau += 1
        if self.t + self.tau > self.decision_epoch:
            delta = np.zeros(self.num_types, dtype=int)
        else:
            delta = self.new_arrivals[self.tau]
        self.state = self.post_action_state_to_new_state(post_action_state, delta)
        return self.state, cost, done, {'wait_time_by_type': self.wait_time_by_type, 'overtime': self.overtime}


if __name__ == '__main__':
    from experiments import get_config_by_type
    config = get_config_by_type('rt_default', 0)
    env = config.env
    for (state, action) in env.generate_state_action_pairs():
        print(state)
        print(action)
