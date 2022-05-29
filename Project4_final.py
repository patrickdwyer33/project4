import random as rand
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rand.seed()

class Distribution:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def draw_uniform(self):
        x = rand.uniform(self.start, self.end)
        return x

    def draw_squared(self):
        z_squared = rand.uniform(self.start, self.end)
        x = np.sqrt(z_squared)
        return x

R0_1 = Distribution(0, 1)
R0_3 = Distribution(0, 3)
R2_3 = Distribution(2, 3)
#fn == 0 --> uniform, fn == 1 --> squared
def Generate_Bids_1(N, dist, fn, m):
    bid_data = []
    for i in range(0, N):
        bids = []
        if fn == 0:
            for j in range(0, m):
                b = dist.draw_uniform()
                bids.append(b)
        if fn == 1:
            for j in range(0, m):
                b = dist.draw_squared()
                bids.append(b)
        bids.sort()
        bids_tup = tuple(bids)
        bid_data.append(bids_tup)
    return bid_data

def Generate_Bids_2(N, dist, fn):
    bids = []
    for i in range(0, N):
        if fn == 0:
            b = dist.draw_uniform()
        if fn == 1:
            b = dist.draw_squared()
        bids.append(b)
    return bids
#T10 & T15 for 1.7-2.0
T10 = []
T11 = []
T12 = []
T13 = []
T14 = []
T15 = []
T16 = []
for i in range(0,100):
    B10 = Generate_Bids_1(150, R0_1, 0, 2)
    B11 = Generate_Bids_1(150, R0_1, 1, 2)
    B12 = Generate_Bids_1(150, R0_3, 0, 2)
    B13 = Generate_Bids_1(150, R2_3, 0, 2)
    B14 = Generate_Bids_1(150, R0_1, 0, 5)
    B15 = Generate_Bids_1(150, R0_1, 0, 10)
    B16 = Generate_Bids_1(150, R0_1, 0, 25)
    T10.append(B10)
    T11.append(B11)
    T12.append(B12)
    T13.append(B13)
    T14.append(B14)
    T15.append(B15)
    T16.append(B16)




df = pd.DataFrame()


def SPA_Payoff(bids, reserve):
    lst = list(bids)
    lst.sort()
    second_highest_bid = bids[-2]
    highest_bid = bids[-1]
    assert(highest_bid >= second_highest_bid)
    if reserve > second_highest_bid and reserve > highest_bid:
        return 0
    elif reserve <= highest_bid and reserve <= second_highest_bid:
        return second_highest_bid
    else:
        return reserve

def Choose_Action(probs):
    x = rand.uniform(0,1)
    cur = 0
    next = probs[0]
    for i in range(0,len(probs)):
        if i == (len(probs) - 1):
            return i
        elif x >= cur and x <= next:
            return i
        else:
            cur = cur + probs[i]
            next = next + probs[i+1]
    return -1

class EW:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.e = np.sqrt(np.log(k)/n)
        self.payoffs = {}

    def initialize(self, first_bids):
        lst = list(first_bids)
        lst.sort()
        start = lst[0]
        end = lst[-1]

        diff = 0
        if start == end:
            diff = 0.1
        else:
            diff = (end - start)
        self.h = end + diff
        k_max = end
        k_min = start
        actions = [10]
        k = self.k - 1
        if k_min < 0:
            k_min = 0
        if k_min > 0:
            actions.append(0)
            k = k - 1
        k_step = (k_max-k_min) / (k-1)

        for i in range(0,k):
            action = k_min + (i*k_step)
            actions.append(action)
        self.actions = actions
        probs = []
        for i in range(0,self.k):
            prob = 1/self.k
            probs.append(prob)
        self.probs = probs

    def run_step(self, bids, action):
        prob_sum = 0
        probs = []
        round_payoffs = []
        actual_payoff = 0
        for i in range(0, self.k):
            a = self.actions[i]
            a_payoff = SPA_Payoff(bids, a)
            if i == action:
                actual_payoff = a_payoff
            round_payoffs.append(a_payoff)
            a_totalpayoff = self.payoffs.get(a, 0) + a_payoff
            self.payoffs[a] = a_totalpayoff
            exp = a_totalpayoff / self.h
            prob = (1+self.e)**exp
            probs.append(prob)
            prob_sum = prob + prob_sum
        for i in range(0,self.k):
            p = probs[i]
            prob = p / prob_sum
            probs[i] = prob
        self.probs = probs

        return actual_payoff

first_trials = {}
first_trials["T10"] = [T10, "Expected Revenue: 1.0", "Reserve Chosen: 1.0"]
first_trials["T11"] = [T11, "Expected Revenue: 1.1", "Reserve Chosen: 1.1"]
first_trials["T12"] = [T12, "Expected Revenue: 1.2", "Reserve Chosen: 1.2"]
first_trials["T13"] = [T13, "Expected Revenue: 1.3", "Reserve Chosen: 1.3"]
first_trials["T14"] = [T14, "Expected Revenue: 1.4", "Reserve Chosen: 1.4"]
first_trials["T15"] = [T15, "Expected Revenue: 1.5", "Reserve Chosen: 1.5"]
first_trials["T16"] = [T16, "Expected Revenue: 1.6", "Reserve Chosen: 1.6"]

for trial in first_trials:
    info = first_trials[trial]
    all_bids = info[0]
    AVGP_name = info[1]
    R_name = info[2]
    AVGPS = []
    RS = []
    for j in range(0, 100):
        bids = all_bids[j]
        Ew = EW(30, 150)
        first_bids = bids[0]
        Ew.initialize(first_bids)
        A = [0]
        pay = Ew.run_step(first_bids, 1)
        P = [pay]
        AVGP = [pay]
        for i in range(1, 150):
            round_bids = bids[i]
            round_action_idx = Choose_Action(Ew.probs)
            round_action = Ew.actions[round_action_idx]
            A.append(round_action)
            pay = Ew.run_step(round_bids, round_action_idx)
            #payoff = pay + P[(i-1)]
            P.append(pay)
            avg_payoff = pay# / (i+1)
            AVGP.append(avg_payoff)

        AVGPS.append(AVGP)
        RS.append(A)
    AVGP = []
    R = []
    for i in range(0, 150):
        avg_sum = 0
        avg_res_sum = 0
        for j in range(0, 100):
            temp = AVGPS[j]
            temp_res = RS[j]
            val = temp[i]
            val_res = temp_res[i]
            avg_sum = avg_sum + val
            avg_res_sum = avg_res_sum + val_res
        avg_sum = avg_sum / 100
        avg_res_sum = avg_res_sum / 100
        AVGP.append(avg_sum)
        R.append(avg_res_sum)
    df[AVGP_name] = AVGP
    df[R_name] = R

second_trials = {}
second_trials["T17"] = [T10, 2, "Expected Revenue: 1.7", "Reserve Chosen: 1.7"]
second_trials["T18"] = [T10, 4, "Expected Revenue: 1.8", "Reserve Chosen: 1.8"]
second_trials["T19"] = [T15, 2, "Expected Revenue: 1.9", "Reserve Chosen: 1.9"]
second_trials["T20"] = [T15, 4, "Expected Revenue: 2.0", "Reserve Chosen: 2.0"]

for trial in second_trials:
    info = second_trials[trial]
    all_bids = info[0]
    l = info[1]
    AVGP_name = info[2]
    AVGPS = []
    for j in range(0, 100):
        bids = all_bids[j]
        Ew = EW(30, 150)
        first_bids = []
        P = []
        AVGP = []
        A = []
        for i in range(0, l):
            round_bids = bids[i]
            for b in round_bids:
                first_bids.append(b)
            round_pay = SPA_Payoff(round_bids, 1)
            A.append(0)
            P.append(round_pay)
            avg_payoff = round_pay
            AVGP.append(avg_payoff)
        first_bids.sort()
        Ew.initialize(first_bids)

        for i in range(l, 150):
            round_bids = bids[i]
            round_action_idx = Choose_Action(Ew.probs)
            round_action = Ew.actions[round_action_idx]
            A.append(round_action)
            pay = Ew.run_step(round_bids, round_action_idx)
            P.append(pay)
            avg_payoff = pay
            AVGP.append(avg_payoff)
        AVGPS.append(AVGP)
        RS.append(A)
    AVGP = []
    R = []
    for i in range(0, 150):
        avg_sum = 0
        avg_res_sum = 0
        for j in range(0, 100):
            temp = AVGPS[j]
            temp_res = RS[j]
            val = temp[i]
            val_res = temp_res[i]
            avg_sum = avg_sum + val
            avg_res_sum = avg_res_sum + val_res
        avg_sum = avg_sum / 100
        avg_res_sum = avg_res_sum / 100
        AVGP.append(avg_sum)
        R.append(avg_res_sum)
    df[AVGP_name] = AVGP
    df[R_name] = R

#code we used to plot. Simply change the y column to explore other variations
#df["Round"] = df.index
#df.plot(x = 'Round', y='Reserve Chosen: 1.0', kind = 'scatter')
#plt.show()

T21 = []
for i in range(0,1000):
    B10 = Generate_Bids_2(100, R0_1, 0)
    B11 = Generate_Bids_2(100, R0_1, 1)
    trial = [B10, B11]
    T21.append(trial)

class Auction:
    def __init__(self, v, c):
        self.metric = (v-c)*v
        self.v = v
        self.c = c

class EWA:
    action_payoffs = []
    observed_auctions = []
    probs = []
    k = 0
    e = 0
    n = 0
    h = 0
    total_payoff = 0

    #input: round #
    #updates: cls.action_payoffs
    @classmethod
    def construct_actions(cls, round):
        cls.action_payoffs = []
        cls.k = round + 1
        cls.h = 1
        sorted_auctions = cls.observed_auctions.copy()
        sorted_auctions.sort(key=lambda x: x.metric)
        for i in range(0, cls.k):
            action_payoff = 0
            for j in range(0, round):
                auction = cls.observed_auctions[j]
                v = auction.v
                c = auction.c
                diff = v - c
                if diff > cls.h:
                    cls.h = diff
                if i == 0:
                    payoff = 0
                    if diff >= 0:
                        payoff = diff
                    action_payoff = action_payoff + payoff
                else:
                    cutoff = sorted_auctions[(i-1)]
                    payoff = 0
                    if diff >= 0 and auction.metric >= cutoff.metric:
                        payoff = diff
                    action_payoff = action_payoff + payoff
            cls.action_payoffs.append(action_payoff)

    #input: round #
    #updates: cls.probs
    @classmethod
    def update_probs(cls, round):
        cls.probs = []
        cls.construct_actions(round)
        cls.e = np.sqrt(np.log(cls.k)/cls.n)
        prob_sum = 0
        for i in range(0, cls.k):
            action_payoff = cls.action_payoffs[i]
            exp = action_payoff / cls.h
            weight = (1+cls.e)**exp
            cls.probs.append(weight)
            prob_sum = prob_sum + weight
        for i in range(0, cls.k):
            prob = cls.probs[i]
            p = prob / prob_sum
            cls.probs[i] = p

    #input: num_trials
    #updates: all class attributes
    @classmethod
    def initialize(cls, n):
        cls.action_payoffs = []
        cls.observed_auctions = []
        cls.probs = []
        cls.k = 0
        cls.e = 0
        cls.n = n
        cls.h = 1

    #input: num_trials
    #returns: payoff_list, opt_payoff_list
    @classmethod
    def run(cls, n, trial):
        cls.initialize(n)
        payoff_list = []
        optimal_payoff_list = []
        values = trial[0]
        costs = trial[1]
        first_val = values[0]
        first_cost = costs[0]
        first_dif = first_val - first_cost
        optimal_first_payoff = 0
        total_payoff = 0
        total_optimal_payoff = 0
        actions_taken = [0]
        if first_dif >= 0:
            payoff_list.append(first_dif)
            total_payoff = first_dif
            if first_dif >= 0 and ((2*first_val)-1) >= 0 and ((2*first_val)-(1.5*first_cost)) >= 1:
                optimal_first_payoff = first_dif
                total_optimal_payoff = optimal_first_payoff
            else:
                optimal_first_payoff = 0
            optimal_payoff_list.append(optimal_first_payoff)
        else:
            payoff_list.append(0)
            total_payoff = 0
            optimal_payoff_list.append(0)
        first_auction = Auction(first_val, first_cost)
        cls.observed_auctions.append(first_auction)
        for i in range(1, n):
            cls.update_probs(i)
            probs = cls.probs
            a = Choose_Action(probs)
            action_taken = a / (i+1)
            actions_taken.append(action_taken)
            v = values[i]
            c = costs[i]
            auction = Auction(v, c)

            diff = v - c
            sorted_auctions = cls.observed_auctions.copy()
            sorted_auctions.sort(key=lambda x: x.metric)
            round_payoff = 0

            if a == 0:
                if diff >= 0:
                    round_payoff = diff
            else:
                cutoff = sorted_auctions[(a-1)]
                if diff >= 0 and auction.metric >= cutoff.metric:
                    round_payoff = diff

            optimal_payoff = 0
            if ((2*v)-1) >= 0 and ((2*v)-(1.5*c)) >= 1 and diff >= 0:
                optimal_payoff = diff

            total_optimal_payoff = total_optimal_payoff + optimal_payoff
            avg_optimal_payoff = total_optimal_payoff / (i+1)

            total_payoff = total_payoff + round_payoff
            avg_total_payoff = total_payoff / (i+1)

            payoff_list.append(round_payoff)
            optimal_payoff_list.append(optimal_payoff)
            cls.observed_auctions.append(auction)

        return payoff_list, optimal_payoff_list, actions_taken


big_payoff_list = []
big_opt_payoff_list = []
big_action_list = []
for i in range(0, 1000):
    trial = T21[i]
    payoff_list, optimal_payoff_list, action_list = EWA.run(100, trial)
    big_payoff_list.append(payoff_list)
    big_opt_payoff_list.append(optimal_payoff_list)
    big_action_list.append(action_list)

avg_round_payoffs = []
avg_opt_round_payoffs = []
avg_actions = []
diffs = []
for i in range(0, 100):
    sum_round_payoff = 0
    sum_opt_round_payoff = 0
    sum_action = 0
    for j in range(0, 1000):
        payoff_list = big_payoff_list[j]
        optimal_list = big_opt_payoff_list[j]
        action_list = big_action_list[j]
        round_payoff = payoff_list[i]
        opt_payoff = optimal_list[i]
        action = action_list[i]
        sum_round_payoff = sum_round_payoff + round_payoff
        sum_opt_round_payoff = sum_opt_round_payoff + opt_payoff
        sum_action = sum_action + action
    avg_round_payoff = sum_round_payoff / 1000
    avg_opt_round_payoff = sum_opt_round_payoff / 1000
    avg_action = sum_action / 1000
    diff = avg_round_payoff - avg_opt_round_payoff
    diffs.append(diff)
    avg_round_payoffs.append(avg_round_payoff)
    avg_opt_round_payoffs.append(avg_opt_round_payoff)
    avg_actions.append(avg_action)

#code we used for plotting the second part
#you can plot either 'avg_round_payoffs', 'avg_opt_round_payoffs', or 'avg_actions'
#df2 = pd.DataFrame()
#df2["Reserve Chosen"] = avg_actions
#df2["Round"] = df2.index
#df2.plot(x = 'Round', y='Reserve Chosen', kind = 'scatter')
#plt.show()

#this code is from our first attempt at this problem. We believe it can achieve
#the true optimal allocation strategy for any possible distributions,
#but the code as it is is intractable
#we included it because we think it's interesting and you might too
#class EW_Auction:
#    action_list = []
#    observed_values = []
#    k = 0
#    e = 0
#    n = 0
#    h = 0
#    total_payoff = 0
#
#    # takes in # of observed rounds and updates action_list
#    @classmethod
#    def construct_actions(cls, cur_round):
#        action_size = cur_round + 1
#        buffer = action_size + 1
#        cls.action_list = []
#        for i in range(0, buffer):
#            lst = []
#            cls.construct_lists(lst, i, buffer, action_size)
#        cls.k = len(cls.action_list)
#        cls.e = np.sqrt(np.log(cls.k)/cls.n)
#
#
#    @classmethod
#    def construct_lists(cls, lst, val, max, action_size):
#        lst.append(val)
#        if len(lst) == action_size:
#            cpy = lst.copy()
#            cls.action_list.append(cpy)
#            return
#        if len(lst) > action_size:
#            return
#        for i in range(val, max):
#            cpy = lst.copy()
#            cls.construct_lists(cpy, i, max, action_size)
#        return
#
#    @classmethod
#    def calculate_action_payoff(cls, i):
#        action = cls.action_list[i]
#        values_sorted = cls.observed_values
#        values_sorted.sort()
#        costs_sorted = cls.observed_costs
#        costs_sorted.sort()
#        length = len(values_sorted)
#        total_payoff = 0
#        for j in range(0, length):
#            payoff = 0
#            v = cls.observed_values[j]
#            c = cls.observed_costs[j]
#            if v >= c:
#                v_index = 0
#                for k in range(0, length):
#                    cur_v = values_sorted[k]
#                    if v <= cur_v:
#                        break
#                    else:
#                        v_index = v_index + 1
#                max_c = action[v_index]
#                c_index = 0
#                for k in range(0, length):
#                    cur_c = costs_sorted[k]
#                    if c <= cur_c:
#                        break
#                    else:
#                        c_index = c_index + 1
#                if c_index < max_c:
#                    payoff = v - c
#            total_payoff = total_payoff + payoff
#        return total_payoff
#
#    @classmethod
#    def calculate_round_payoff(cls, i, v, c):
#        action = cls.action_list[i]
#        values_sorted = cls.observed_values
#        values_sorted.sort()
#        costs_sorted = cls.observed_costs
#        costs_sorted.sort()
#        length = len(values_sorted)
#        payoff = 0
#        if v >= c:
#            v_index = 0
#            for k in range(0, length):
#                cur_v = values_sorted[k]
#                if v <= cur_v:
#                    break
#                else:
#                    v_index = v_index + 1
#            max_c = action[v_index]
#            c_index = 0
#            for k in range(0, length):
#                cur_c = costs_sorted[k]
#                if c <= cur_c:
#                    break
#                else:
#                    c_index = c_index + 1
#            if c_index < max_c:
#                payoff = v - c
#        optimal_payoff = 0
#        if ((2*v)-1) >= 0 and ((2*v)-((3/2)*c)) >= 1:
#            optimal_payoff = v - c
#        return payoff, optimal_payoff
#
#    @classmethod
#    def update_probs(cls):
#        differences = []
#        for i in range(0, len(cls.observed_values)):
#            v = cls.observed_values[i]
#            c = cls.observed_costs[i]
#            diff = v - c
#            differences.append(diff)
#        cls.h = max(differences)
#        if cls.h <= 0:
#            cls.h = 1
#        cls.probs = []
#        prob_sum = 0
#        for i in range(0, cls.k):
#            weight = cls.calculate_action_payoff(i)
#            exponent = weight / cls.h
#            prob = (1+cls.e)**exponent
#            prob_sum = prob + prob_sum
#            cls.probs.append(prob)
#        for i in range(0, cls.k):
#            prob = cls.probs[i]
#            rel_prob = prob / prob_sum
#            cls.probs[i] = rel_prob
#
#    @classmethod
#    def initialize(cls, n):
#        cls.action_list = []
#        cls.observed_values = []
#        cls.observed_costs = []
#        cls.probs = []
#        cls.k = 0
#        cls.e = 0
#        cls.n = n
#        cls.h = 0
#        cls.total_payoff = 0
#
#    @classmethod
#    def run(cls, n, trial):
#        cls.initialize(n)
#        payoff_list = []
#        optimal_payoff_list = []
#        values = trial[0]
#        costs = trial[1]
#        first_val = values[0]
#        first_cost = costs[0]
#        first_dif = first_val - first_cost
#        cls.total_payoff = 0
#        optimal_first_payoff = 0
#        if first_dif >= 0:
#            cls.total_payoff = first_dif
#            payoff_list.append(first_dif)
#            if ((2*first_val)-1) >= 0 and ((2*first_val)-((3/2)*first_cost)) >= 1:
#                    optimal_first_payoff = first_dif
#            else:
#                optimal_first_payoff = 0
#            optimal_payoff_list.append(optimal_first_payoff)
#        else:
#            payoff_list.append(0)
#            optimal_payoff_list.append(0)
#        cls.observed_values.append(first_val)
#        cls.observed_costs.append(first_cost)
#        for i in range(1, n):
#            print(i)
#            cls.construct_actions(i)
#            cls.update_probs()
#            probs = cls.probs
#            a = Choose_Action(probs)
#            v = values[i]
#            c = costs[i]
#            round_payoff, optimal_payoff = cls.calculate_round_payoff(i, v, c)
#            payoff_list.append(round_payoff)
#            optimal_payoff_list.append(optimal_payoff)
#            payoff_so_far = cls.total_payoff
#            total_payoff = payoff_so_far + round_payoff
#            cls.total_payoff = total_payoff
#            cls.observed_values.append(v)
#            cls.observed_costs.append(c)
#        return payoff_list, optimal_payoff_list
#
