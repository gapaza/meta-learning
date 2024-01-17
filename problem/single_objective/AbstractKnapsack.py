import random
import numpy as np
import os
import pickle
import config


class AbstractKnapsack:

    def __init__(self, n=60, s=1, new_problem=False, id=''):
        # get name of class
        self.f_name = self.__class__.__name__ + '_'+str(n)+'_'+str(s)+'_'+id+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n
        self.s = s
        self.seq_len = s * n

        # --> Loss Landscape
        self.val_range = [10, 100]
        self.weight_range = [10, 100]

        # --> Items / Max Weights / Obj Norm
        if new_problem is False:
            self.items, self.max_weights, self.obj_norm = self.load_items()
        else:

            # --> Items
            values = np.random.randint(self.val_range[0], self.val_range[1], self.n)
            weight = np.random.randint(self.weight_range[0], self.weight_range[1], self.n)
            self.items = [[values[i], weight[i]] for i in range(self.n)]

            # --> Max Weights
            if s == 1:
                total_weight = np.sum(weight)
                self.max_weights = [total_weight / 3.0]
            else:
                total_weight = np.sum(weight)
                min_weight = np.min(weight)
                max_weight = total_weight - min_weight
                min_weight = min_weight * 2.0
                self.max_weights = np.random.randint(min_weight, max_weight, self.s).tolist()

            # --> Obj Norm
            self.obj_norm = np.sum(values)

    # ---------------------------------------
    # Evaluate
    # ---------------------------------------

    def evaluate(self, solution):
        if len(solution) != (self.n * self.s):
            raise Exception('Solution must be of length n * s:', self.n, self.s, len(solution))
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # Calculate total value
        total_value = 0
        idx = 0
        for i in range(self.s):
            kp_value = 0
            kp_weight = 0
            for j in range(self.n):
                kp_value += solution[idx] * self.items[j][0]
                kp_weight += solution[idx] * self.items[j][1]
                idx += 1
            if kp_weight <= self.max_weights[i]:
                total_value += kp_value

        # Normalize
        total_value = total_value / self.obj_norm
        return total_value



    # ---------------------------------------
    # Random Solution
    # ---------------------------------------

    def random_solution(self):
        return ''.join([str(random.choice([0, 0, 1])) for _ in range(self.n * self.s)])


    # ---------------------------------------
    # Save and Load
    # ---------------------------------------

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump([self.items, self.max_weights, self.obj_norm], f)

    def load_items(self):
        # Load self.items from a pickle file
        with open(self.path, 'rb') as f:
            obj = pickle.load(f)
            items, max_weights, obj_norm = obj
        return items, max_weights, obj_norm
