import copy
import itertools

import numpy as np
from scipy.stats import truncnorm

from utils import numpy_shift, RunningStats, integer_partitions_fixed_bins, bounded_compositions


class AdvSchedulingEnv:

    def __init__(self,
                 treatment_pattern,
                 decision_epoch,
                 arrival_generator,
                 holding_cost,
                 overtime_cost,
                 duration,
                 regular_capacity,
                 discount_factor,
                 random_seed=None
                 ):
        self.treatment_pattern = np.array(treatment_pattern)
        self.decision_epoch = decision_epoch
        self.arrival_generator = arrival_generator
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.discount_factor = discount_factor
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.num_sessions, self.num_types = self.treatment_pattern.shape
        self.planning_horizon = decision_epoch + self.num_sessions - 1

    def set_decision_epoch(self, decision_epoch):
        self.decision_epoch = decision_epoch
        self.planning_horizon = decision_epoch + self.num_sessions - 1

    def get_next_regular_bookings(self, state, action, is_var):
        regular_bookings, waitlist = self.get_state(state, is_var)
        advance_scheduling_decision, overtime_decision = action
        new_regular_bookings = regular_bookings + self.convert_action_to_booking_slots(
            advance_scheduling_decision) - overtime_decision
        return new_regular_bookings[1:]

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

    def generate_advance_actions(self, waitlist, booking_window_size):
        per_type_generators = [integer_partitions_fixed_bins(w_i, booking_window_size) for w_i in waitlist]

        # Cartesian product across types builds a full schedule column-by-column
        for columns in itertools.product(*per_type_generators):
            # columns is a tuple of I length-N tuples; convert to N rows
            yield np.array(columns).T

    def valid_actions(self, state, t, is_var=False):
        regular_bookings, waitlist = self.get_state(state, is_var)
        for advance_scheduling_decision in self.generate_advance_actions(waitlist, self.decision_epoch-t+1):
            new_booking_slots = self.convert_action_to_booking_slots(advance_scheduling_decision)
            overtime_decision = np.maximum(regular_bookings + new_booking_slots - self.regular_capacity, 0)
            yield (advance_scheduling_decision, overtime_decision)

    def generate_regular_hour_bookings(self, t):
        for p in itertools.product(range(self.regular_capacity + 1), repeat=self.planning_horizon - t + 1):
            yield np.array(p)

    def generate_arrivals(self):
        for N in range(self.arrival_generator.maximum_arrival + 1):
            for arrivals in integer_partitions_fixed_bins(total=N, bins=self.num_types):
                yield np.array(arrivals)
    def generate_states(self):
        for t in range(1, self.decision_epoch+1):
            for regular_hour_bookings in self.generate_regular_hour_bookings(t):
                for arrivals in self.generate_arrivals():
                    yield (regular_hour_bookings, arrivals), t

    def generate_state_action_pairs(self):
        for state, t in self.generate_states():
            for action in self.valid_actions(state, t):
                yield (state, action, t)

    def cost_fn(self, state, action, t):
        advance_scheduling_decision, overtime_decision = action
        waiting_cost = sum(sum(self.discount_factor ** k * self.holding_cost(k, i) for k in range(j + 1)) *
                           advance_scheduling_decision[j, i]
                           for j in range(len(advance_scheduling_decision))
                           for i in range(len(advance_scheduling_decision[0])))
        overtime_cost = sum(self.discount_factor ** j * self.overtime_cost(j) * overtime_decision[j] for j in
                            range(len(overtime_decision)))
        return waiting_cost + overtime_cost

    def get_state(self, state, is_var=False):
        if not is_var:
            return copy.deepcopy(state)
        return state

    def post_action_state(self, state, action, is_var=False):
        regular_bookings, waitlist = self.get_state(state, is_var)
        advance_scheduling_decision, overtime_decision = action
        new_regular_bookings = regular_bookings + self.convert_action_to_booking_slots(advance_scheduling_decision) - overtime_decision
        new_waitlist = waitlist - advance_scheduling_decision.sum(axis=0)
        return (new_regular_bookings, new_waitlist)

    def post_action_state_to_new_state(self, post_action_state, new_arrival, is_var=False):
        post_action_regular_bookings, post_action_waitlist = self.get_state(post_action_state, is_var)
        new_regular_bookings = post_action_regular_bookings[1:]
        return (new_regular_bookings, new_arrival)

    def get_next_state(self, state, action, new_arrival, is_var=False):
        return self.post_action_state_to_new_state(self.post_action_state(state, action, is_var), new_arrival, is_var)

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

    def reset(self, init_state=None, t=1, new_arrivals=None, percentage_occupied=0, init_new_arrivals=None, seed=None):
        if new_arrivals is not None and len(new_arrivals) != self.decision_epoch - t + 1:
            print("length of new arrivals:", len(new_arrivals), "length of decision epoch:",
                  self.decision_epoch - t + 1)
            raise ValueError('Invalid sample path!')
        self.t = t
        self.tau = 0
        # how to handle the first arrivals
        if new_arrivals is None:
            self.new_arrivals = self.reset_arrivals(t)
        else:
            self.new_arrivals = new_arrivals
        if init_new_arrivals is not None:
            self.new_arrivals[0] = init_new_arrivals
        if init_state == None:
            init_state = self.reset_initial_state(t, percentage_occupied, seed)
        regular_bookings, waitlist = init_state
        regular_bookings = np.array(regular_bookings)
        waitlist = np.array(waitlist)
        self.state = (regular_bookings, waitlist)
        # measure of performance
        self.wait_time_by_type = {j: RunningStats() for j in range(self.num_types)}
        self.overtime = np.array([0] * (self.decision_epoch + self.num_sessions - t))
        return copy.deepcopy(self.state), {'wait_time_by_type': self.wait_time_by_type,
                                           'overtime': self.overtime}

    def reset_arrivals(self, t=1):
        return self.arrival_generator.rvs(self.decision_epoch - t + 1)

    def reset_initial_state(self, t, percentage_occupied, seed=None):
        # find out the average appointment slot required in first period
        capacity_occupied = self.regular_capacity * percentage_occupied
        # initialize the current booking slots with all zeros
        booking_horizon = self.decision_epoch - t + self.num_sessions
        # Step 1: Generate from truncated normal distribution
        mean = 1.0
        std_dev = 0.3
        lower, upper = 0, 2
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=booking_horizon, random_state=self.rng)
        # Step 2: Scale so that the average is exactly 100 * p
        scaled = samples / samples.mean() * capacity_occupied
        return (scaled, self.new_arrivals[0])

    def step(self, action):
        # t+tau
        advance_scheduling_decision, overtime_decision = action
        cost = self.cost_fn(self.state, action, self.t + self.tau)
        post_action_state = (post_action_bookings, _) = self.post_action_state(self.state, action)
        done = self.t + self.tau == self.decision_epoch
        # record performance metric
        # implement info: include the type-dependent waiting times and overtime use
        wait_times = np.arange(advance_scheduling_decision.shape[0])
        for i in range(advance_scheduling_decision.shape[1]):
            self.wait_time_by_type[i].record_batch(wait_times, advance_scheduling_decision[:, i])
        self.overtime[self.tau:] = self.overtime[self.tau:]+ overtime_decision
        # update state
        self.tau += 1
        if self.t + self.tau > self.decision_epoch:
            delta = np.zeros(self.num_types, dtype=int)
        else:
            delta = self.new_arrivals[self.tau]
        self.state = self.post_action_state_to_new_state(post_action_state, delta)
        return self.state, cost, done, {'wait_time_by_type': self.wait_time_by_type, 'overtime': self.overtime}

if __name__ =='__main__':
    from experiments import get_config_by_type
    config = get_config_by_type('base_case')
    env = config.env

    def get_utilization(total_rate):
        """Modifier function for the demand rate experiment."""
        config = get_config_by_type('base_case')
        env = config.env
        base_case_rate = sum(config.env.arrival_generator.mean_by_type)
        new_arrival_rates = env.arrival_generator.mean_by_type * (total_rate/base_case_rate)
        return np.dot(new_arrival_rates,env.treatment_pattern.sum(axis=0))/env.regular_capacity
    print(get_utilization(12))
    print(get_utilization(16))
    print(get_utilization(20))