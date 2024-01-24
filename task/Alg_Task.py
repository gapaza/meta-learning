import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import json
import config
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import get_universal_crossover, get_fast_universal_crossover
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.selection.tournament import compare
from task.AbstractTask import AbstractTask

from buffer.CrossoverBuffer import CrossoverBuffer
from design.BitVector import BitVector as Design
import utils

from problem.multi_objective.ConstantTruss import ConstantTruss




class Alg_Task(AbstractTask):

    def __init__(self, run_num=0, barrier=None, problem=None, limit=50, actor_load_path=None, critic_load_path=None, debug=False, c_type='uniform', max_nfe=5000):
        super(Alg_Task, self).__init__(run_num, barrier, problem, limit, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type

        # HV
        self.ref_point = np.array([0, 2])  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = set()
        self.unique_designs_vals = []

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = 30  # 32 FU_NSGA2, 30 U_NSGA2
        self.max_nfe = max_nfe
        self.nfe = 0
        self.limit = limit
        self.steps_per_design = 30  # 30 | 60

        # Population
        self.population = []
        self.hv = []  # hv progress over time
        self.nfes = []  # nfe progress over time


    def run(self):
        results_file = os.path.join(self.run_dir, self.c_type + '.png')
        if os.path.exists(results_file):
            print('ALG TASK ALREADY COMPLETED:', self.run_num)
            return []

        print('--> RUNNING ALG TASK:', self.run_num)

        self.init_population()
        self.eval_population()

        terminated = False
        counter = 0
        while terminated is False and self.nfe < self.max_nfe:
            # print('EPOCH', counter, '/', self.limit)

            # 1. Create offspring
            self.create_offspring()

            # 2. Evaluate offspring
            self.eval_population()

            # 3. Prune population
            self.prune_population()

            # 4. Log iteration
            self.record(None)
            self.activate_barrier()
            counter += 1

            if counter >= self.limit:
                terminated = True

        performance = [[nfe, hv] for hv, nfe in zip(self.hv, self.nfes)]
        return performance

    # -------------------------------------
    # Population Functions
    # -------------------------------------

    def calc_pop_hv(self):
        objectives = self.eval_population()
        F = np.array(objectives)
        hv = self.hv_client.do(F)
        return hv

    def init_population(self):
        self.population = []
        for x in range(self.pop_size):
            design = Design(evaluator=self.problem, num_bits=self.steps_per_design)
            self.population.append(design)

    def eval_population(self):
        evals = []
        for design in self.population:
            if design.evaluated is False:
                self.nfe += 1
            vals = design.evaluate()
            evals.append(vals)
            design_str = design.get_vector_str()
            if design_str not in self.unique_designs:
                self.unique_designs.add(design_str)
                self.unique_designs_vals.append(vals)
        return evals

    def prune_population(self):
        objectives = self.eval_population()
        F = np.array(objectives)

        # 1. Determine survivors
        fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
        survivors = []
        for k, front in enumerate(fronts, start=1):
            for idx in front:
                survivors.append(idx)
        perished = [idx for idx in range(len(self.population)) if idx not in survivors]

        # 2. Update population (sometimes the first front is larger than pop_size)
        new_population = []
        for idx in survivors:
            new_population.append(self.population[idx])
        self.population = new_population

    # -------------------------------------
    # Tournament Functions
    # -------------------------------------

    def binary_tournament(self, solutions=None):
        if solutions is None:
            solutions = self.population

        p1 = random.randrange(len(solutions))
        p2 = random.randrange(len(solutions))
        while p1 == p2:
            p2 = random.randrange(len(solutions))

        player1 = solutions[p1]
        player2 = solutions[p2]

        winner_idx = compare(
            p1, player1.rank,
            p2, player2.rank,
            'smaller_is_better',
            return_random_if_equal=False
        )
        if winner_idx is None:
            winner_idx = compare(
                p1, player1.crowding_dist,
                p2, player2.crowding_dist,
                'larger_is_better',
                return_random_if_equal=True
            )
        return winner_idx

    def head_to_head(self, player1, player2):
        winner = compare(
            player1, player1.rank,
            player2, player2.rank,
            'smaller_is_better',
            return_random_if_equal=False
        )
        if winner is None:
            winner = compare(
                player1, player1.crowding_dist,
                player2, player2.crowding_dist,
                'larger_is_better',
                return_random_if_equal=True
            )
        return winner

    def create_offspring(self):
        objectives = self.eval_population()
        F = np.array(objectives)

        # Set pareto rank and crowding distance
        fronts = self.nds.do(F)
        ff_solns = []
        for k, front in enumerate(fronts, start=1):
            crowding_of_front = utils.calc_crowding_distance(F[front, :])
            for i, idx in enumerate(front):
                self.population[idx].crowding_dist = crowding_of_front[i]
                self.population[idx].rank = k
                if k == 1:
                    ff_solns.append(self.population[idx])
        if len(ff_solns) > self.offspring_size:
            ff_solns = ff_solns[:self.offspring_size]

        # Get parent pairs
        pairs = []
        while len(pairs) < self.offspring_size:
            parent1_idx = self.binary_tournament()
            parent2_idx = self.binary_tournament()
            while parent2_idx == parent1_idx:
                parent2_idx = self.binary_tournament()
            pairs.append([
                self.population[parent1_idx],
                self.population[parent2_idx]
            ])

        # Create offspring
        offspring = self.crossover_parents(pairs)
        self.population.extend(offspring)

    def crossover_parents(self, parent_pairs):
        offspring = []
        for pair in parent_pairs:
            parent1 = pair[0]
            parent2 = pair[1]
            child = Design(evaluator=self.problem, c_type=self.c_type)
            child.crossover(parent1, parent2)
            child.mutate()
            offspring.append(child)
        return offspring

    def record(self, epoch_info):
        self.hv.append(self.calc_pop_hv())
        self.nfes.append(self.nfe)


    def plot(self):

        # 1. Plot HV
        plt.figure(figsize=(8, 8))
        plt.plot(self.nfes, self.hv)
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('HV Progress')
        plt.savefig(os.path.join(self.run_dir, self.c_type + '.png'))
        plt.close('all')

        # 2. Plot designs
        plt.figure(figsize=(8, 8))
        for obj_vals in self.unique_designs_vals:
            plt.scatter(obj_vals[0] * -1.0, obj_vals[1], color='blue')
        plt.xlabel('Vertical Stiffness')
        plt.ylabel('Volume Fraction')
        plt.title('Designs')
        plt.savefig(os.path.join(self.run_dir, self.c_type + '_designs.png'))


if __name__ == '__main__':
    problem = ConstantTruss(n=30)
    problem.init_problem(0, run_val=True)

    task_runner = Alg_Task(
        run_num=1000,
        barrier=None,
        problem=problem,
        limit=1000,
        actor_load_path=None,
        critic_load_path=None,
        c_type='uniform',
        max_nfe=3000,
    )
    task_runner.run()
    task_runner.plot()





