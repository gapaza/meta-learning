import random
import numpy as np
import os
import pickle

import config


class O1Knapsack:

    def __init__(self, n=10, new_problem=False):
        # get name of class
        self.f_name = self.__class__.__name__ + '.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)
        self.n = n

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
        items = []

        # ---------------------------------------
        # Value Function
        # ---------------------------------------

        values = np.random.randint(1, 100, n)

        # ---------------------------------------
        # Weight Function
        # ---------------------------------------

        weight = np.random.randint(1, 100, n)

        # ---------------------------------------
        # Items
        # ---------------------------------------

        items = [[values[i], weight[i]] for i in range(n)]

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

        # 2. Calculate weight
        weight = 0
        for i in range(self.n):
            if solution[i] == 1:
                weight += self.items[i][2]

        # 3. Normalize
        weight = weight / 2477.0
        value = value / 11563.653237215553

        return value, weight


if __name__ == '__main__':
    n = 50

    kp = O1Knapsack(n=n, new_problem=True)

    solution = ''.join([str(1) for _ in range(n)])

    value, weight = kp.evaluate(solution)

    print('Solution:', solution)
    print('Value:', value)
    print('Weight:', weight)

    kp.save()






