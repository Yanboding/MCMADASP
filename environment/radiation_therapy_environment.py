import copy
import itertools

import numpy as np
from scipy.stats import truncnorm, geom, qmc

from utils import numpy_shift, RunningStats, bounded_compositions
import gurobipy as gp

class RTEnv:

    def __init__(self,
                 treatment_pattern,
                 booking_window_size,
                 arrival_generator,
                 holding_cost,
                 overtime_cost,
                 postponing_cost,
                 duration,
                 regular_capacity,
                 overtime_capacity,
                 discount_factor,
                 init_state_random_seed,
                 stop_time_random_seed=42,
                 ):
        self.treatment_pattern = np.array(treatment_pattern)
        self.booking_window_size = booking_window_size
        self.arrival_generator = arrival_generator
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.postponing_cost = postponing_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.overtime_capacity = overtime_capacity
        self.discount_factor = discount_factor
        self.init_state_random_seed = init_state_random_seed
        self.stop_time_random_seed = stop_time_random_seed
        self.init_state_rng = np.random.default_rng(init_state_random_seed)
        self.stop_time_rng = np.random.default_rng(stop_time_random_seed)
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

    def cost_fn(self, state, action, is_var=False):
        regular_bookings, overtimes, waitlist = state
        advance_scheduling_decision, overtime_decision = action
        waiting_cost = gp.quicksum(
            gp.quicksum(self.discount_factor ** k * self.holding_cost(k, i) for k in range(j)) * advance_scheduling_decision[j, i]
            for j in range(len(advance_scheduling_decision))
            for i in range(len(advance_scheduling_decision[0]))
        )
        overtime_cost = gp.quicksum(self.discount_factor ** j * self.overtime_cost(j) * overtime_decision[j] for j in range(len(overtime_decision)))
        remaining_treatments = waitlist - advance_scheduling_decision.sum(axis=0)
        postponing_cost = gp.quicksum(self.postponing_cost(i) * remaining_treatments[i] for i in range(self.num_types))
        cost = waiting_cost + overtime_cost + postponing_cost
        if not is_var:
            cost = cost.getValue()
        return cost

    def post_action_state(self, state, action, is_var=False):
        regular_bookings, overtimes, waitlist = self.get_state(state, is_var)
        advance_scheduling_decision, overtime_decision = action
        new_booking_slots = self.convert_action_to_booking_slots(advance_scheduling_decision)
        post_action_regular_bookings = regular_bookings + new_booking_slots - overtime_decision
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

    def transition_dynamic(self, state, action):
        cost = self.cost_fn(state, action)
        res = []
        for prob, delta in self.arrival_generator.get_system_dynamic():
            next_state = self.get_next_state(state, action, delta)
            # In infinite horizon, 'done' is False unless you have an absorbing state
            done = False
            res.append([prob, next_state, cost, done])
        return res

    # simulation
    def reset(self, init_state=None, t=1, new_arrivals=None, percentage_occupied=0.99):
        self.t = t
        self.tau = 0
        # how to handle the first arrivals
        if new_arrivals is None:
            self.new_arrivals = self.reset_arrivals()
        else:
            self.new_arrivals = new_arrivals
        self.decision_epoch = len(self.new_arrivals)
        if init_state == None:
            init_state = self.reset_initial_state(percentage_occupied, self.new_arrivals[0])
        bookings, overtimes, waitlist = init_state
        bookings = np.array(bookings)
        overtimes = np.array(overtimes)
        waitlist = np.array(waitlist)
        self.state = (bookings, overtimes, waitlist)
        # measure of performance
        self.wait_time_by_type = {j: RunningStats() for j in range(self.num_types)}
        total_periods = self.decision_epoch + self.planning_horizon - 1
        self.overtime = np.array([0] * (total_periods - t + 1))
        self.waiting_time_target_violations = {j: RunningStats() for j in range(self.num_types)}
        self.postponing_decision_number = np.array([[0] * self.num_types for _ in range(self.decision_epoch - t + 1)])
        self.waiting_number = np.array([[0] * self.num_types for _ in range(self.decision_epoch - t + 1)])
        return copy.deepcopy(self.state), {'wait_time_by_type': self.wait_time_by_type,
                                           'overtime': self.overtime,
                                           'target_violations': self.waiting_time_target_violations,
                                           'postponing_decision_number': self.postponing_decision_number}

    def reset_arrivals(self, stop_time=None):
        # Generate a single random number from the geometric distribution
        if stop_time is None:
            stop_time = geom.rvs((1- self.discount_factor), random_state=self.stop_time_rng)
        return self.arrival_generator.rvs(stop_time)
    
    def quasi_reset_arrivals(self, stop_time=None):
        if stop_time is None:
            quantile = self.qmc_rng.random(n=1)[0][0]
            print(quantile)
            stop_time = geom.ppf(0.99, (1- self.discount_factor)).astype(int)
            print(stop_time)
        return self.arrival_generator.quasi_rvs(stop_time)

    def reset_initial_state(self, decay_factor, new_arrivals):
        # find out the average appointment slot required in first period
        required_bookings = []
        for j in range(self.planning_horizon):
            mean = (self.regular_capacity + self.overtime_capacity) * decay_factor**(j+1)
            std_dev = 1
            required_booking = truncnorm.rvs(0, float('inf'), loc=mean, scale=std_dev, random_state=self.init_state_rng)
            # randomized rounding to preserve mean
            k = int(np.floor(required_booking))
            p = required_booking - k
            if self.init_state_rng.random() < p:
                required_booking = k + 1
            else:
                required_booking = k

            required_bookings.append(required_booking)
        required_bookings[-1] = 0
        required_bookings = np.array(required_bookings)
        regular_bookings = np.minimum(required_bookings, self.regular_capacity)
        overtimes = np.minimum(np.maximum(required_bookings - self.regular_capacity, 0), self.overtime_capacity)
        return (regular_bookings, overtimes, new_arrivals)

    def step(self, action):
        regular_bookings, overtimes, waitlist = self.state
        advance_scheduling_decision, overtime_decision = action
        cost = self.cost_fn(self.state, action)
        post_action_state = self.post_action_state(self.state, action)
        post_action_regular_bookings, post_action_overtimes, post_action_waitlist = post_action_state
        done = self.t + self.tau == self.decision_epoch
        # record performance metric
        # implement info: include the type-dependent waiting times and overtime use
        wait_times = np.arange(advance_scheduling_decision.shape[0])
        for i in range(advance_scheduling_decision.shape[1]):
            self.wait_time_by_type[i].record_batch(wait_times, advance_scheduling_decision[:, i])
            is_violate_waiting_time_target = (wait_times - self.holding_cost.get_waiting_target(i) >=0).astype(int)
            self.waiting_time_target_violations[i].record_batch(is_violate_waiting_time_target, advance_scheduling_decision[:, i])
        self.overtime[self.tau] = post_action_overtimes[0]
        if done:
            self.overtime[self.tau:] = post_action_overtimes
        self.postponing_decision_number[self.tau] = post_action_waitlist
        self.waiting_number[self.tau] = waitlist
        # update state
        self.tau += 1
        if self.t + self.tau > self.decision_epoch:
            delta = np.zeros(self.num_types, dtype=int)
        else:
            delta = self.new_arrivals[self.t + self.tau - 1]
        self.state = self.post_action_state_to_new_state(post_action_state, delta)
        return self.state, cost, done, {'wait_time_by_type': self.wait_time_by_type, 
                                        'overtime': self.overtime, 
                                        'target_violations': self.waiting_time_target_violations,
                                        'postponing_decision_number': self.postponing_decision_number,
                                        'waiting_number': self.waiting_number}


if __name__ == '__main__':
    from experiments import get_config_by_type
    config = get_config_by_type('ejor_default')
    env = config.env
    for i in range(1):
        print(env.quasi_reset_arrivals())
