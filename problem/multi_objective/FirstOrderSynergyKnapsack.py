import numpy as np
import os

import config

from problem.multi_objective.AbstractKnapsack import AbstractKnapsack


class FirstOrderSynergyKnapsack(AbstractKnapsack):

    def __init__(self, n=10, new_problem=False, id=''):
        super().__init__(n, new_problem, id)
        self.f_name = self.__class__.__name__ + '_'+str(n)+'_'+id+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)

        # --> Synergy Conditions
        self.weight_range = [0, 100]
        self.val_range = [0, 100]
        self.fo_syn_prob = 0.10
        self.fo_syn_low_range = [0, 0.0]
        self.fo_syn_high_range = [50, 100]

        # --> Normalize
        self.value_norm = 5000
        self.weight_norm = 2000

        # --> Items
        if new_problem is False:
            self.items = self.load_items()
        else:
            values = np.random.randint(1, 10, self.n)
            low_synergy_fo = np.random.uniform(self.fo_syn_low_range[0], self.fo_syn_low_range[1], (self.n, self.n))
            high_synergy_fo = np.random.uniform(self.fo_syn_high_range[0], self.fo_syn_high_range[1], (self.n, self.n))
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

        # 1. Calculate value (item values + first-order synergy values)
        value = 0
        for i in range(self.n):
            if solution[i] == 1:
                value += self.items[i][0]
        for i in range(self.n):
            if solution[i] == 1:
                for j in range(self.n):
                    if solution[j] == 1:
                        value += self.items[i][2][j]

        # 2. Calculate weight
        weight = 0
        for i in range(self.n):
            if solution[i] == 1:
                weight += self.items[i][1]

        # 3. Normalize
        # print(value, weight)
        weight = weight / self.weight_norm
        value = value / self.value_norm

        # ---------------------------------------
        # Constraints
        # ---------------------------------------

        if weight > self.max_weight:
            value = 0.0
        if sum(solution) == 0:
            value = 0.0
            weight = 1.0

        return value, weight




















