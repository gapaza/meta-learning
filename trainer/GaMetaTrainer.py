import random
import config
import os
import pickle
import tensorflow as tf
import uuid
import numpy as np


# ------------------------------------
# Problems
# ------------------------------------

# from problem.multi_objective.AbstractKnapsack import AbstractKnapsack
from problem.multi_objective.SecondOrderSynergyKnapsack import SecondOrderSynergyKnapsack
from problem.multi_objective.FirstOrderSynergyKnapsack import FirstOrderSynergyKnapsack as AbstractKnapsack
from problem.multi_objective.AbstractMultiKnapsack import AbstractMultiKnapsack
from problem.multi_objective.ConstantTruss import ConstantTruss

# ------------------------------------
# Models
# ------------------------------------

# from task.GA_Task import GA_Task
# from task.LGA_Task import LGA_Task as GA_Task
# from task.FGA_Task import FGA_Task as GA_Task
from task.PPOMO_Task import PPOMO_Task as GA_Task


# from model import get_universal_crossover
# from model import get_large_universal_crossover as get_universal_crossover
from model import get_universal_solver_mo as get_universal_crossover



class GaMetaTrainer:

    def __init__(
            self,
            num_task_variations=3,
            new_tasks=False,
            checkpoint_path_actor=None,
            checkpoint_path_critic=None,
            task_sample_size=4,
            task_epochs=50,
    ):

        # 1. Initialize models
        self.checkpoint_path_actor = checkpoint_path_actor
        self.checkpoint_path_critic = checkpoint_path_critic
        self.actor, self.critic = get_universal_crossover(
            checkpoint_path_actor=self.checkpoint_path_actor,
            checkpoint_path_critic=self.checkpoint_path_critic,
        )
        self.save_path_actor = os.path.join(config.models_dir, 'universal_crossover_actor')
        self.save_path_critic = os.path.join(config.models_dir, 'universal_crossover_critic')

        # 2. Initialize optimizers
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

        # 3. Initialize Task Parameters
        self.num_task_variations = num_task_variations
        self.new_tasks = new_tasks
        self.task_parameters_path = os.path.join(config.results_save_dir, 'universal_trainer_task_parameters.pkl')
        self.run_parameters = {}
        if new_tasks is True:
            self.train_task_parameters = []
            self.train_task_ids = []
            for _ in range(self.num_task_variations):
                self.train_task_parameters.append([60])
                self.train_task_ids.append([str(uuid.uuid1())])
            self.val_task_parameters = [60]
            self.val_task_ids = [str(uuid.uuid1())]
            self.run_parameters['task_parameters'] = self.train_task_parameters
            self.run_parameters['task_ids'] = self.train_task_ids
            self.run_parameters['val_task_parameters'] = self.val_task_parameters
            self.run_parameters['val_task_ids'] = self.val_task_ids
            self.save()
        else:
            self.load()

        # 4. Initialize tasks
        self.train_tasks, self.val_tasks = self.init_tasks()
        self.truss_task = ConstantTruss(n=30)

        # Run info
        self.eval_freq = 15
        self.max_epochs = 15
        self.epoch = 0
        self.task_sample_size = task_sample_size
        self.task_epochs = task_epochs

    def init_tasks(self):

        # 1. Create tasks
        train_tasks = []
        for idx, params in enumerate(self.run_parameters['task_parameters']):
            task_ids = self.run_parameters['task_ids'][idx]
            train_tasks.append(AbstractKnapsack(n=params[0], new_problem=self.new_tasks, id=task_ids[0]))

        val_tasks = []
        val_task_ids = self.run_parameters['val_task_ids']
        val_tasks.append(
            AbstractKnapsack(
                n=self.run_parameters['val_task_parameters'][0],
                new_problem=self.new_tasks,
                id=val_task_ids[0]
            )
        )

        # 2. Save tasks
        for task in train_tasks:
            task.save()
        for task in val_tasks:
            task.save()

        return train_tasks, val_tasks

    def save(self):
        with open(self.task_parameters_path, 'wb') as f:
            pickle.dump(self.run_parameters, f)

    def load(self):
        with open(self.task_parameters_path, 'rb') as f:
            self.run_parameters = pickle.load(f)

    def save_models(self):
        actor_save_path = self.save_path_actor + '_{}'.format(self.epoch)
        critic_save_path = self.save_path_critic + '_{}'.format(self.epoch)
        self.actor.save_weights(actor_save_path)
        self.critic.save_weights(critic_save_path)
        return actor_save_path, critic_save_path

    # -----------------------------------
    # Run
    # -----------------------------------

    def train(self):
        terminated = False
        while not terminated and self.epoch < self.max_epochs:
            self.run_epoch()
            self.epoch += 1


    def run_truss_task(self, run_num=0, start_nfe=0, config_num=0, run_val=False):
        self.truss_task.init_problem(config_num, run_val=run_val)
        actor_save_path, critic_save_path = self.save_models()
        alg = GA_Task(
            run_num=run_num,
            problem=self.truss_task,
            limit=self.task_epochs,
            actor_load_path=actor_save_path,
            critic_load_path=critic_save_path,
            debug=True,
            c_type='uniform',
            start_nfe=start_nfe,
            config_num=config_num,
        )
        alg.run()
        return alg

    def train_truss_task(self):

        # 1. Eval initial model on val task (config 0)
        alg = self.run_truss_task(run_num=1000, config_num=0, run_val=True)

        # 2. Run tasks, collect parameters
        val_runs = 1
        terminated = False
        while not terminated and self.epoch < self.max_epochs:
            all_actor_params = []
            all_critic_params = []

            for x in range(self.task_sample_size):

                # For now just run a single truss task
                run_num = self.epoch + 1  # this is also the task index (0 index is the val task)
                if run_num == 20:  # task 20 is bugged and has local optima
                    self.epoch += 1
                    continue
                alg = self.run_truss_task(run_num=run_num, config_num=run_num, run_val=False)
                temp_actor, temp_critic = get_universal_crossover(alg.actor_save_path, alg.critic_save_path)
                all_actor_params.append(temp_actor.get_weights())
                all_critic_params.append(temp_critic.get_weights())
                self.epoch += 1

            # Reptile update
            print('Updating meta-model')
            # 1. Average the parameters + computer pseudo-gradients
            mean_actor_params = []
            mean_critic_params = []
            for layer_weights in zip(*all_actor_params):
                layer_mean = np.mean(layer_weights, axis=0)
                mean_actor_params.append(layer_mean)
            for layer_weights in zip(*all_critic_params):
                layer_mean = np.mean(layer_weights, axis=0)
                mean_critic_params.append(layer_mean)

            curr_actor_params = self.actor.get_weights()
            curr_critic_params = self.critic.get_weights()

            actor_pseudo_gradients = [original - new for original, new in zip(curr_actor_params, mean_actor_params)]
            critic_pseudo_gradients = [original - new for original, new in zip(curr_critic_params, mean_critic_params)]

            # 2. Apply the pseudo-gradient updates to the models
            actor_grads_and_vars = zip(actor_pseudo_gradients, self.actor.trainable_variables)
            critic_grads_and_vars = zip(critic_pseudo_gradients, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(actor_grads_and_vars)
            self.critic_optimizer.apply_gradients(critic_grads_and_vars)

            # 3. Evaluate against val task again (config 0)
            alg = self.run_truss_task(run_num=1000+val_runs, config_num=0, run_val=True)
            val_runs += 1




    def run_tasks(self):
        actor_save_path, critic_save_path = self.save_models()

        # 1. Sample tasks (all tasks)
        task_indices = [x for x in range(len(self.train_tasks))]
        train_tasks = [self.train_tasks[x] for x in task_indices]

        # 2. Run tasks, collect parameters
        possible_nfes = [x for x in range(0, 5001, 500)]
        for task, task_idx in zip(train_tasks, task_indices):
            start_nfe = 0  # random.choice(possible_nfes)
            alg = GA_Task(
                run_num=task_idx,
                problem=task,
                limit=self.task_epochs,
                actor_load_path=actor_save_path,
                critic_load_path=critic_save_path,
                debug=True,
                c_type='uniform',
                start_nfe=start_nfe,
            )
            alg.run()




    def run_epoch(self):
        actor_save_path, critic_save_path = self.save_models()

        # --> Evaluate initial model
        if self.epoch+1 % self.eval_freq == 0:
            self.evaluate(self.val_tasks[0])

        # -----------------------------------
        # Run Tasks
        # -----------------------------------

        # 1. Sample tasks
        task_indices = [x for x in range(len(self.train_tasks))]
        task_indices = random.sample(task_indices, self.task_sample_size)
        train_tasks = [self.train_tasks[x] for x in task_indices]

        # 2. Run tasks, collect parameters
        all_algs = []
        all_actor_params = []
        all_critic_params = []
        possible_nfes = [x for x in range(0, 5001, 500)]
        for task, task_idx in zip(train_tasks, task_indices):
            start_nfe = 0  # random.choice(possible_nfes)
            debug = len(all_algs) == 0
            alg = GA_Task(
                run_num=task_idx,
                problem=task,
                limit=self.task_epochs,
                actor_load_path=actor_save_path,
                critic_load_path=critic_save_path,
                debug=debug,
                c_type='uniform',
                start_nfe=start_nfe,
            )
            all_algs.append(alg)
        for alg in all_algs:
            alg.start()
        for alg_idx, alg in enumerate(all_algs):
            alg.join()
            temp_actor, temp_critic = get_universal_crossover(alg.actor_save_path, alg.critic_save_path)
            all_actor_params.append(temp_actor.get_weights())
            all_critic_params.append(temp_critic.get_weights())

        # -----------------------------------
        # Reptile update
        # -----------------------------------

        # 1. Average the parameters + computer pseudo-gradients
        mean_actor_params = []
        mean_critic_params = []
        for layer_weights in zip(*all_actor_params):
            layer_mean = np.mean(layer_weights, axis=0)
            mean_actor_params.append(layer_mean)
        for layer_weights in zip(*all_critic_params):
            layer_mean = np.mean(layer_weights, axis=0)
            mean_critic_params.append(layer_mean)

        curr_actor_params = self.actor.get_weights()
        curr_critic_params = self.critic.get_weights()

        actor_pseudo_gradients = [original - new for original, new in zip(curr_actor_params, mean_actor_params)]
        critic_pseudo_gradients = [original - new for original, new in zip(curr_critic_params, mean_critic_params)]

        # 2. Apply the pseudo-gradient updates to the models
        actor_grads_and_vars = zip(actor_pseudo_gradients, self.actor.trainable_variables)
        critic_grads_and_vars = zip(critic_pseudo_gradients, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(actor_grads_and_vars)
        self.critic_optimizer.apply_gradients(critic_grads_and_vars)
        print('--> UPDATED META-MODEL')



    # -----------------------------------
    # Evaluate
    # -----------------------------------

    def evaluate(self, task):
        actor_save_path, critic_save_path = self.save_models()

        actor_eval, critic_eval = get_universal_crossover()
        actor_eval.load_target_weights(self.actor, trainable=True)
        critic_eval.load_target_weights(self.critic, trainable=True)

        eval_task_run_num = 1000 + self.epoch
        eval_task = GA_Task(
            run_num=eval_task_run_num,
            problem=task,
            limit=100,
            actor_load_path=actor_save_path,
            critic_load_path=critic_save_path
        )
        eval_task.run()




