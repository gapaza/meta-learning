import random
import numpy as np
import os
import pickle

import config


class AbstractMultiKnapsack:

    def __init__(self, n=12, s=5, new_problem=False, id=''):
        # get name of class
        self.f_name = self.__class__.__name__ + '_'+str(n)+'_'+str(s)+'_'+id+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n
        self.s = s

        # --> Loss Landscape
        self.val_range = [0, 100]
        self.weight_range = [0, 100]

        # --> Constraints
        self.max_weights = [120, 150, 200, 100, 20]

        # --> Normalize
        self.value_norm = 1000
        self.weight_norm = 2000

        # --> Items
        if new_problem is False:
            self.items = self.load_items()
        else:
            values = np.random.randint(self.val_range[0], self.val_range[1], self.n)
            weight = np.random.randint(self.weight_range[0], self.weight_range[1], self.n)
            self.items = [[values[i], weight[i]] for i in range(self.n)]

    def evaluate(self, solution):
        if len(solution) != (self.n * self.s):
            raise Exception('Solution must be of length n * s:', self.n, self.s)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # ---------------------------------------
        # Objectives
        # ---------------------------------------
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
                    kp_value += self.items[item_idx][0]
                    kp_weight += self.items[item_idx][1]
                idx += 1
            if kp_weight < self.max_weights[kp_idx]:
                value += kp_value
            weight += kp_weight

        # 3. Normalize
        # print(value, weight)
        weight = weight / self.weight_norm
        value = value / self.value_norm

        return value, weight

    # ---------------------------------------
    # Random Solution
    # ---------------------------------------

    def random_solution(self):
        return ''.join([str(random.choice([0, 0, 1])) for _ in range(self.n * self.s)])

    # ---------------------------------------
    # Save and Load
    # ---------------------------------------

    def save(self):
        # Save self.items to a pickle file
        with open(self.path, 'wb') as f:
            pickle.dump(self.items, f)

    def load_items(self):
        # Load self.items from a pickle file
        with open(self.path, 'rb') as f:
            items = pickle.load(f)
        return items


if __name__ == '__main__':
    n = 12
    s = 5

    kp = AbstractMultiKnapsack(new_problem=False)

    sol = ''.join([str(1) for _ in range(n * s)])
    # sol = kp.random_solution()

    value, weight = kp.evaluate(sol)

    print('Solution:', sol)
    print('Value:', value)
    print('Weight:', weight)

    kp.save()






