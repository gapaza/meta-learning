import random
import numpy as np
import os
import pickle

import config


class O1Knapsack2S:

    def __init__(self, n=10, new_problem=False):
        # get name of class
        self.f_name = self.__class__.__name__ + '.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n

        # --> Synergy Conditions
        self.val_range = [0, 100]

        self.fo_syn_prob = 0.10
        self.fo_syn_low_range = [0, 0.0]
        self.fo_syn_high_range = [50, 100]

        self.so_syn_prob = 0.10
        self.so_syn_low_range = [0, 0.0]
        self.so_syn_high_range = [50, 100]

        # --> Normalizing Constants
        self.value_norm = 295369.8736952084
        self.weight_norm = 3399.0

        # --> Items
        if new_problem is True:
            self.items = self.generate_items(n)
        else:
            self.items = self.load_items()

    def save(self):
        # Save self.items to a pickle file
        with open(self.path, 'wb') as f:
            pickle.dump(self.items, f)

    def load_items(self):
        # Load self.items from a pickle file
        with open(self.path, 'rb') as f:
            items = pickle.load(f)
        return items

    def generate_items(self, n):
        # --- Objective Functions ---
        # 1. Maximize value
        # value = sum of item values + sum of first order synergy values + sum of second order synergy values
        # 2. Minimize weight
        # weight = sum of item weights

        # ---------------------------------------
        # Value Function
        # ---------------------------------------

        # --> 1. Item values
        values = np.random.randint(self.val_range[0], self.val_range[1], n)

        # --> 2. First-order synergy values (symmetric)

        low_synergy_fo = np.random.uniform(self.fo_syn_low_range[0], self.fo_syn_low_range[1], (n, n))
        high_synergy_fo = np.random.uniform(self.fo_syn_high_range[0], self.fo_syn_high_range[1], (n, n))

        synergy_fo = np.where(np.random.rand(n, n) < self.fo_syn_prob, high_synergy_fo, low_synergy_fo)

        # Make the synergy matrix symmetric
        synergy_fo = np.triu(synergy_fo)  # Take the upper triangle
        synergy_fo += np.triu(synergy_fo, 1).T  # Reflect to the lower triangle

        # --> 3. Second-order synergy values (symmetric)
        low_synergy = np.random.uniform(self.so_syn_low_range[0], self.so_syn_low_range[1], (n, n, n))
        high_synergy = np.random.uniform(self.so_syn_high_range[0], self.so_syn_high_range[1], (n, n, n))

        # Decide where to put high synergies based on a probability (e.g., 0.1)
        mask = np.random.rand(n, n, n) < self.so_syn_prob
        second_order_synergy = np.where(mask, high_synergy, low_synergy)

        # Make the synergy tensor symmetric along the last two dimensions
        for i in range(n):
            second_order_synergy[i] = np.maximum.reduce([
                np.triu(second_order_synergy[i]),
                np.triu(second_order_synergy[i], 1).T]
            )

        # ---------------------------------------
        # Weight Function
        # ---------------------------------------

        weight = np.random.randint(1, 100, n)

        # ---------------------------------------
        # Combine into items
        # ---------------------------------------

        items = [[values[i], synergy_fo[i], weight[i], second_order_synergy[i]] for i in range(n)]
        return items


    def evaluate(self, solution):
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]
        # Solution is a list of n binary values
        # 1. Calculate value
        value = 0
        for i in range(self.n):
            if solution[i] == 1:
                value += self.items[i][0]
        # Add synergy values
        for i in range(self.n):
            if solution[i] == 1:
                for j in range(self.n):
                    if solution[j] == 1:
                        value += self.items[i][1][j]

        # Add second-order synergy values
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for k in range(j + 1, self.n):
                    if solution[i] == 1 and solution[j] == 1 and solution[k] == 1:
                        value += self.items[i][3][j][k]

        # 2. Calculate weight
        weight = 0
        for i in range(self.n):
            if solution[i] == 1:
                weight += self.items[i][2]

        # 3. Normalize
        # print(weight, value)
        weight = weight / self.weight_norm
        value = value / self.value_norm

        return value, weight




if __name__ == '__main__':
    n = 60

    kp = O1Knapsack2S(n=n, new_problem=False)


    solution = ''.join([str(1) for _ in range(n)])
    # solution = ''.join([str(random.choice([0, 1])) for _ in range(n)])




    value, weight = kp.evaluate(solution)


    print('Solution:', solution)
    print('Value:', value)
    print('Weight:', weight)


    kp.save()






