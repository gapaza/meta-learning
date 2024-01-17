import numpy as np
import os

import config

from problem.multi_objective.AbstractMultiKnapsack import AbstractMultiKnapsack


class FirstOrderSynergyMultiKnapsack(AbstractMultiKnapsack):

    def __init__(self, n=12, s=5, new_problem=False):
        super().__init__(n, new_problem)
        self.f_name = self.__class__.__name__ + '.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n
        self.s = s

        # --> Synergy Conditions
        self.weight_range = [0, 100]
        self.val_range = [0, 100]
        self.fo_syn_prob = 0.10
        self.fo_syn_low_range = [0, 0.0]
        self.fo_syn_high_range = [50, 100]

        # --> Constraints
        self.max_weights = [120, 150, 200, 100, 20]

        # --> Items
        if new_problem is False:
            self.items = self.load_items()
        else:
            values = np.random.randint(1, 10, self.n)
            low_synergy_fo = np.random.uniform(self.fo_syn_low_range[0], self.fo_syn_low_range[1], (self.n * self.s, self.n * self.s))
            high_synergy_fo = np.random.uniform(self.fo_syn_high_range[0], self.fo_syn_high_range[1], (self.n * self.s, self.n * self.s))
            np.fill_diagonal(low_synergy_fo, 0)
            np.fill_diagonal(high_synergy_fo, 0)
            synergy_fo = np.where(np.random.rand(self.n, self.n) < self.fo_syn_prob, high_synergy_fo, low_synergy_fo)
            synergy_fo = np.triu(synergy_fo)  # Take the upper triangle
            synergy_fo += np.triu(synergy_fo, 1).T  # Reflect to the lower triangle
            weight = np.random.randint(self.weight_range[0], self.weight_range[1], self.n)
            self.items = [[values[i], weight[i], synergy_fo[i]] for i in range(self.n)]


    def evaluate(self, solution):
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # --> Solution is a list of n binary values

        # ---------------------------------------
        # Objectives
        # ---------------------------------------
        synergy_counter = []
        weight = 0
        value = 0
        idx = 0
        for i in range(self.s):
            kp_idx = i
            kp_weight = 0
            kp_value = 0
            for j in range(self.n):
                item_idx = j
                if solution[idx] == 1:
                    kp_weight += self.items[item_idx][1]
                    val = 0
                    val += self.items[item_idx][0]



                idx += 1
            if kp_weight < self.max_weights[kp_idx]:
                value += kp_value
            weight += kp_weight



        return value, weight




















