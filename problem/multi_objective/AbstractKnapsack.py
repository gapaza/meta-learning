import random
import numpy as np
import os
import pickle

import config


class AbstractKnapsack:

    def __init__(self, n=10, new_problem=False, id=''):
        # get name of class
        self.f_name = self.__class__.__name__ + '_'+str(n)+'_'+id+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n

        # --> Loss Landscape
        self.val_range = [0, 100]
        self.weight_range = [0, 100]

        # --> Constraints
        self.max_weight = 120

        # --> Normalize
        self.value_norm = 1000
        self.weight_norm = 1000

        # --> Items
        if new_problem is False:
            self.items = self.load_items()
        else:
            values = np.random.randint(self.val_range[0], self.val_range[1], self.n)
            weight = np.random.randint(self.weight_range[0], self.weight_range[1], self.n)
            self.items = [[values[i], weight[i]] for i in range(self.n)]

    def evaluate(self, solution):
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # --> Solution is a list of n binary values

        # ---------------------------------------
        # Objectives
        # ---------------------------------------

        # 1. Calculate value
        value = 0
        for i in range(self.n):
            if solution[i] == 1:
                value += self.items[i][0]

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

    # ---------------------------------------
    # Random Solution
    # ---------------------------------------

    def random_solution(self):
        return ''.join([str(random.choice([0, 0, 1])) for _ in range(self.n)])

    # ---------------------------------------
    # Save and Load
    # ---------------------------------------

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.items, f)
    def load_items(self):
        # Load self.items from a pickle file
        with open(self.path, 'rb') as f:
            items = pickle.load(f)
        return items


if __name__ == '__main__':
    n = 60

    kp = AbstractKnapsack(n=n, new_problem=False)

    sol = ''.join([str(1) for _ in range(n)])
    # sol = kp.random_solution()

    value, weight = kp.evaluate(sol)

    print('Solution:', sol)
    print('Value:', value)
    print('Weight:', weight)

    kp.save()






