import numpy as np
import os
import config
import time
from tqdm import tqdm
import random
import pickle

from problem.multi_objective.AbstractKnapsack import AbstractKnapsack
from py4j.java_gateway import JavaGateway

class ConstantTruss(AbstractKnapsack):

    def __init__(self, n=10, new_problem=True, id=''):
        super().__init__(n, new_problem, id)
        self.f_name = self.__class__.__name__ + '_'+str(n)+'_'+id+'.pkl'
        self.path = os.path.join(config.root_dir, 'problem', 'store', self.f_name)

        self.gateway = JavaGateway()
        self.my_java_object = self.gateway.entry_point

        # Problem Config
        self.config_vals = []

        # Normalization
        self.stiff_norm = None
        self.vol_norm = None

        # --- Validation parameters ---
        # 1. member radius (250e-6 in m)
        # 2. side length (10e-3 in m)
        # 3. Young's modulus (1.8162e6 in Pa) ~ a bit stiffer than soft tissue which is < 1e6 Pa
        self.val_problem = [250e-6, 10e-3, 1.8162e6]

        # --- Training parameters ---
        # 1. Acrylic 3.2GPa --> 3.2e9 Pa
        # 2. Cartilage 20MPa --> 20e6 Pa
        # 3. Rigid Polymers 200MPa --> 200e6 Pa
        self.member_radii = [200e-6, 150e-6, 100e-6, 300e-6]
        self.side_lengths = [5e-3, 15e-3, 30e-3, 45e-3]
        self.youngs_modulus = [1e5, 5e6, 20e6, 200e6]
        self.train_problems = []
        for r in self.member_radii:
            for s in self.side_lengths:
                for y in self.youngs_modulus:
                    self.train_problems.append([r, s, y])
        # random.shuffle(self.train_problems)


    # ---------------------------------------
    # Problem Init
    # ---------------------------------------

    def init_problem(self, p_num, run_val=False):
        print('Initializing problem:', p_num)
        if run_val is True:
            self.init_val_problem()
        else:
            self.init_train_problem(p_num)
        self.calc_norm_values()

    def init_train_problem(self, config_num):
        self.config_vals = self.train_problems[config_num]
        self.my_java_object.initProblem2(self.config_vals[0], self.config_vals[1], self.config_vals[2])

    def init_val_problem(self):
        self.config_vals = self.val_problem
        self.my_java_object.initProblem2(self.config_vals[0], self.config_vals[1], self.config_vals[2])

    # ---------------------------------------
    # Evaluation
    # ---------------------------------------

    def evaluate(self, solution, normalize=True):
        # print('Evaluating:', solution)
        if len(solution) != self.n:
            raise Exception('Solution must be of length n:', self.n)
        if type(solution) == str:
            solution = [int(i) for i in solution]

        # --> Solution is a list of n binary values
        if sum(solution) == 0:
            return 0.0, self.vol_norm

        # ---------------------------------------
        # Objectives
        # ---------------------------------------

        java_list_design = self.convert_design(solution)
        objectives = self.my_java_object.evaluateDesign(java_list_design)
        objectives = list(objectives)
        vertical_stiffness = objectives[0]
        volume_fraction = objectives[1]

        if vertical_stiffness == 0.0:  # infeasible design
            volume_fraction = self.vol_norm
        # if np.isnan(vertical_stiffness):  # infeasible design
        #     vertical_stiffness = 0.0
        #     volume_fraction = 1.0
        # if volume_fraction == 1.0:
        #     vertical_stiffness = 0.0

        vertical_stiffness *= -1.0  # make value positive (it is currently negative)

        # Normalize
        if normalize is True:
            vertical_stiffness /= self.stiff_norm
            volume_fraction /= self.vol_norm

        return vertical_stiffness, volume_fraction

    def convert_design(self, design):
        design_array = np.array(design, dtype=np.float64)
        java_list_design = self.gateway.jvm.java.util.ArrayList()
        for i in range(len(design_array)):
            java_list_design.append(design_array[i])
        return java_list_design

    # ---------------------------------------
    # Normalization
    # ---------------------------------------

    def get_norm_values(self):
        return self.stiff_norm, self.vol_norm

    def calc_norm_values(self):

        # -- Check if norm values have already been calculated --
        s_fname = 'norm_values_' + '_'.join([str(i) for i in self.config_vals]) + '.pkl'
        print('Checking for saved norm values:', s_fname)
        s_path = os.path.join(config.root_dir, 'problem', 'store', s_fname)
        if os.path.exists(s_path):
            print('--> USING SAVED NORM VALUES')
            vals = pickle.load(open(s_path, 'rb'))
            self.stiff_norm = vals[0]
            self.vol_norm = vals[1]
            print('Stiffness norm:', self.stiff_norm)
            print('Volume norm:', self.vol_norm)
            return

        # -- Create bitstring population --
        sample_size = 2000

        # 1. Uniformly draw how many 1s each design sample will have
        num_ones = np.random.randint(1, self.n, sample_size)

        # 2. Create bitstrings, randomly assigning 1s to the positions for each design
        bitstrings = []
        for i in range(sample_size):
            design = np.zeros(self.n)
            design[:num_ones[i]] = 1
            np.random.shuffle(design)
            bitstrings.append(design)

        # 3. Evaluate bitstrings
        v_stiffness = []
        vol_frac = []
        for i in tqdm(range(sample_size), desc='Calculating normalization values'):
            v, vol = self.evaluate(bitstrings[i], normalize=False)
            v_stiffness.append(v)
            vol_frac.append(vol)

        # 4. Calculate normalization values
        margin = 0.1
        self.stiff_norm = np.max(v_stiffness) + (np.max(v_stiffness) * margin)
        self.vol_norm = np.max(vol_frac) # Max volume fraction is 1.0 (or at least should be)
        print('Stiffness norm:', self.stiff_norm)
        print('Volume norm:', self.vol_norm)

        # 5. Save norm values
        vals = [self.stiff_norm, self.vol_norm]
        with open(s_path, 'wb') as f:
            pickle.dump(vals, f)


if __name__ == '__main__':
    # Config 0: 0.124, 0.3680
    test_design = '000000101110011100001110110011'
    test_design_input = [int(i) for i in test_design]

    truss = ConstantTruss(n=30)

    truss.init_problem(0, run_val=False)
    v_stiff, mass_frac = truss.evaluate(test_design_input)
    print(v_stiff, mass_frac)

    #
    # truss.init_problem(1)
    # v_stiff, mass_frac = truss.evaluate(test_design_input)
    # print(v_stiff, mass_frac)




