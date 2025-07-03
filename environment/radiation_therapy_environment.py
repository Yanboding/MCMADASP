import copy
import functools
import itertools
import time

import numpy as np
from scipy.stats import truncnorm

from utils import numpy_shift, RunningStat


class RTEnv:
    def __init__(self,
                 treatment_pattern,
                 decision_epoch,
                 planning_horizon,
                 arrival_generator,
                 holding_cost,
                 overtime_cost,
                 postponing_cost,
                 duration,
                 regular_capacity,
                 overtime_capacity,
                 discount_factor
                 ):
        self.treatment_pattern = np.array(treatment_pattern)
        self.decision_epoch = decision_epoch
        self.planning_horizon = planning_horizon
        self.arrival_generator = arrival_generator
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.postponing_cost = postponing_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.overtime_capacity = overtime_capacity
        self.discount_factor = discount_factor
        self.num_sessions, self.num_types = self.treatment_pattern.shape

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

    def validation(self, state, action):
        bookings, overtimes, waitlist = state
        advance_scheduling_decision, overtime_decision = action
        # limits the number of bookings for each treatment type to be less than or equal to the number of treatments waiting to be booked
        if not all(advance_scheduling_decision.sum(axis=0) <= waitlist):
            raise ValueError('The number of bookings exceeds the number of treatments waiting to be booked')
        new_bookings = self.convert_action_to_booking_slots(advance_scheduling_decision)
        # restricts the total number of appointment slots booked today for day m to be less than or equal to the available treatment capacity that day
        if not all(bookings + new_bookings <= self.regular_capacity + overtime_decision):
            raise ValueError('The number of overtime slots decision cannot cover new bookings')
        # limits the total overtime utilization on day m to be less than the overtime capacity
        if not all(overtimes + overtime_decision <= self.overtime_capacity):
            raise ValueError('The number of overtime slots used exceeds the the overtime capacity')
        if not np.all(advance_scheduling_decision == advance_scheduling_decision.astype(int)):
            raise ValueError("Advance scheduling decision is not all integer")
        if not np.all(overtime_decision == overtime_decision.astype(int)):
            raise ValueError("Overtime decision is not all integer")

    def cost_fn(self, state, action):
        '''
        bookings = [1,2,3,4,5]
        action = ([[1,2,3],[4,5,6]], [1,2,3,4])
        '''
        bookings, overtimes, waitlist = state
        advance_scheduling_decision, overtime_decision = action
        waiting_cost = sum(sum(self.discount_factor ** k * self.holding_cost(k, i) for k in range(j + 1)) * advance_scheduling_decision[j, i]
                           for j in range(len(advance_scheduling_decision))
                           for i in range(len(advance_scheduling_decision[0])))

        overtime_cost = sum(self.discount_factor ** j * self.overtime_cost(j) * overtime_decision[j] for j in range(len(overtime_decision)))
        remaining_treatments = waitlist - advance_scheduling_decision.sum(axis=0)
        postponing_cost = sum(self.postponing_cost(i) * remaining_treatments[i] for i in range(self.num_types))
        return waiting_cost + overtime_cost + postponing_cost

    def post_action_state(self, state, action):
        # check validation
        self.validation(state, action)
        bookings, overtimes, waitlist = copy.deepcopy(state)
        advance_scheduling_decision, overtime_decision = action
        new_bookings = bookings + self.convert_action_to_booking_slots(advance_scheduling_decision) - overtime_decision
        new_overtimes = overtimes + overtime_decision
        new_waitlist = waitlist - advance_scheduling_decision.sum(axis=0)
        return (new_bookings, new_overtimes, new_waitlist)

    def post_action_state_to_new_state(self, post_action_state, new_arrival):
        post_action_bookings, post_action_overtimes, post_action_waitlist = copy.deepcopy(post_action_state)
        new_bookings = numpy_shift(post_action_bookings, num_places=-1)
        new_overtimes = numpy_shift(post_action_overtimes, num_places=-1)
        new_waitlist  = post_action_waitlist + new_arrival
        return (new_bookings, new_overtimes, new_waitlist)

    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action)
        res = []
        for prob, delta in self.arrival_generator.get_system_dynamic():
            post_action_state = self.post_action_state(state, action)
            if t + 1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            done = t == self.decision_epoch
            next_state = self.post_action_state_to_new_state(post_action_state, delta)
            res.append([prob, next_state, cost, done])
        return res

    def valid_actions(self, state):
        bookings, overtimes, waitlist = copy.deepcopy(state)
        number_days = self.planning_horizon
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
            res = []
            for w in waitlist:
                comp = []
                for num in range(w+1):
                    comp += compositions(num, number_days)
                res.append(comp)
            return res

        # Cartesian product of all class-level combinations
        all_class_combinations = get_all_class_combinations()
        for advance_scheduling_combination in itertools.product(*all_class_combinations):
            advance_scheduling_decision = np.array(advance_scheduling_combination).T
            new_bookings = self.convert_action_to_booking_slots(advance_scheduling_decision)
            overtime_decision = np.maximum(new_bookings + bookings - self.regular_capacity, 0)
            action = (advance_scheduling_decision, overtime_decision)
            try:
                self.validation(state, action)
                yield action
            except ValueError:
                continue

    # simulation
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
            init_state = self.reset_initial_state(percentage_occupied, self.new_arrivals[0], seed)
        bookings, overtimes, waitlist = init_state
        bookings = np.array(bookings)
        overtimes = np.array(overtimes)
        waitlist = np.array(waitlist)
        self.state = (bookings, overtimes, waitlist)
        # measure of performance
        self.wait_time_by_type = {j: RunningStat((1,)) for j in range(self.num_types)}
        self.overtime = np.array([0] * (self.decision_epoch + self.num_sessions - t))
        return copy.deepcopy(self.state), {'wait_time_by_type': self.wait_time_by_type,
                                           'overtime': self.overtime}

    def reset_arrivals(self, t=1):
        return self.arrival_generator.rvs(self.decision_epoch - t + 1)

    def reset_initial_state(self, percentage_occupied, new_arrivals, seed=None):
        # find out the average appointment slot required in first period
        capacity_occupied = (self.regular_capacity + self.overtime_capacity) * percentage_occupied
        # initialize the current booking slots with all zeros
        booking_horizon = self.planning_horizon + self.num_sessions - 1
        if seed is not None:
            np.random.seed(seed)
        # Step 1: Generate from truncated normal distribution
        mean = 1.0
        std_dev = 0.3
        lower, upper = 0, 2
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=booking_horizon)
        # Step 2: Scale so that the average is exactly 100 * p
        total_bookings = samples / samples.mean() * capacity_occupied
        overtimes = np.maximum(total_bookings - self.regular_capacity, 0)
        bookings = total_bookings - overtimes
        return (bookings, overtimes, new_arrivals)

    def step(self, action):
        advance_scheduling_decision, overtime_decision = action
        # t+tau
        cost = self.cost_fn(self.state, action)
        post_action_state = (post_action_bookings, post_action_overtimes, post_action_waitlist) = self.post_action_state(self.state, action)
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
    print(env.reset_initial_state(0.5, np.array([6,6])))
