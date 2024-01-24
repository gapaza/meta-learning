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
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.selection.tournament import compare
from task.AbstractTask import AbstractTask
from design.BitVector import BitVector as Design
import utils
import scipy.signal
from task.Alg_Task import Alg_Task
from model import get_universal_solver_mo
import tensorflow_addons as tfa
import pickle


global_mini_batch_size = 12

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class PPOMO_Task(AbstractTask):

    def __init__(
            self,
            run_num=0,
            barrier=None,
            problem=None,
            limit=50,
            actor_load_path=None,
            critic_load_path=None,
            debug=False,
            c_type='uniform',
            start_nfe=0,
            config_num=0,
            run_val=False,
            val_itr=0
    ):
        super(PPOMO_Task, self).__init__(run_num, barrier, problem, limit, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type
        self.start_nfe = start_nfe
        self.config_num = config_num
        self.run_val = run_val
        self.val_itr = val_itr

        # HV
        self.ref_point = np.array([0, 2])  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = set()
        self.unique_designs_vals = []

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = global_mini_batch_size  # 32 FU_NSGA2, 30 U_NSGA2
        self.mini_batch_size = global_mini_batch_size
        self.num_cross_obs_designs = 10
        self.max_nfe = 6000
        self.nfe = 0
        self.limit = limit
        self.steps_per_design = 30  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.01  # usually 0.01
        self.entropy_coef = 0.02  # was 0.1 for large exploration
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 2

        # Population
        self.population = []
        self.hv = []  # hv progress over time
        self.nfes = []  # nfe progress over time

        # Results
        self.uniform_ga_file = os.path.join(self.run_dir, 'uniform_ga.json')
        self.uniform_ga = []
        self.random_search_file = os.path.join(self.run_dir, 'random_search.json')
        self.random_search = []
        self.plot_freq = 50

        # MO Weights
        self.epoch_token_weights = [
            [
                0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4,
                # 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                # 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                # 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                # 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
             ],
        ]

        num_keys = 9  # 9 is best so far
        values = np.linspace(0.1, 0.9, num_keys)
        self.token_weight_map = {key: value for key, value in enumerate(values)}
        # self.token_weight_map = {
        #     0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4,
        #     4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8,
        #     8: 0.9,
        # }

        # TODO: switch
        self.init_comparison_data()

    def init_comparison_data(self):
        # Uniform GA
        if os.path.exists(self.uniform_ga_file):
            with open(self.uniform_ga_file, 'r') as f:
                self.uniform_ga = json.load(f)
        else:
            print('--> RUNNING UNIFORM GA')
            task_runner = Alg_Task(
                run_num=self.run_num,
                barrier=None,
                problem=self.problem,
                limit=1000,
                actor_load_path=None,
                critic_load_path=None,
                c_type='uniform',
                max_nfe=3000,
            )
            self.uniform_ga = task_runner.run()
            if len(self.uniform_ga) > 0:
                task_runner.plot()
            with open(self.uniform_ga_file, 'w') as f:
                json.dump(self.uniform_ga, f)

        # # Random Search
        # if os.path.exists(self.random_search_file):
        #     with open(self.random_search_file, 'r') as f:
        #         self.random_search = json.load(f)
        # else:
        #     print('--> RUNNING RANDOM SEARCH')
        #     task_runner = Alg_Task(
        #         run_num=self.run_num,
        #         barrier=None,
        #         problem=self.problem,
        #         limit=1000,
        #         actor_load_path=None,
        #         critic_load_path=None,
        #         c_type='random',
        #         max_nfe=3000,
        #     )
        #     self.random_search = task_runner.run()
        #     if len(self.random_search) > 0:
        #         task_runner.plot()
        #     with open(self.random_search_file, 'w') as f:
        #         json.dump(self.random_search, f)

    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.0001
        self.train_actor_iterations = 250
        self.train_critic_iterations = 40
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)


        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_universal_solver_mo(self.actor_load_path, self.critic_load_path)

    def build_optimizers(self):
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)

        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

    def run(self):
        self.build()
        self.init_population()


        for x in range(self.limit):
            print('Epoch', x)
            children, epoch_info = self.fast_mini_batch()
            self.population.extend(children)
            self.prune_population()
            curr_time = time.time()
            self.record(epoch_info)
            print('--> RECORD TIME:', round(time.time() - curr_time, 3))

        # Save the parameters of the current actor and critic
        self.c_actor.save_weights(self.actor_save_path)
        self.c_critic.save_weights(self.critic_save_path)

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

    def eval_population(self):
        evals = []
        for design in self.population:
            evals.append(design.evaluate())
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
    # PPO Functions
    # -------------------------------------

    def get_cross_obs(self):
        # Want weight tensor of shape: (mini_batch_size, 1)

        # --- Hard coded token weights ---
        # weights_idx = self.epoch_token_weights[0]
        # weights = [self.token_weight_map[w] for w in weights_idx]

        # --- Random same token weights ---
        # weight_idx = random.choice(list(self.token_weight_map.keys()))
        # weights_idx = [weight_idx for _ in range(self.mini_batch_size)]
        # weights = [self.token_weight_map[weight_idx] for _ in range(self.mini_batch_size)]

        # --- Random different token weights ---

        # sample 3 unique (without replacement) keys from the token weight map
        weight_sample_size = 3
        weight_idx = random.sample(list(self.token_weight_map.keys()), weight_sample_size)  # three total weights
        weights_idx = weight_idx * int(self.mini_batch_size / weight_sample_size)  # repeat the three weights to fill the mini batch
        weights = [self.token_weight_map[w] for w in weights_idx]  # convert the weight idx to the actual weight

        weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
        weights_tensor = tf.expand_dims(weights_tensor, axis=-1)  # shape: (mini_batch_size, 1)
        idx_tensor = tf.convert_to_tensor(weights_idx, dtype=tf.float32)
        idx_tensor = tf.expand_dims(idx_tensor, axis=-1)
        return weights, weights_tensor, idx_tensor

    def fast_mini_batch(self):
        children = []

        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_rewards_stiffness = [[] for _ in range(self.mini_batch_size)]   # vertical stiffness (maximize)
        all_rewards_volume = [[] for _ in range(self.mini_batch_size)]   # volume (minimize)
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Preprocess cross attention observation input
        weights, cross_obs_tensor, cross_obs_idx_tensor = self.get_cross_obs()  # (mini_batch_size, 1)

        # -------------------------------------
        # Sample Actor
        # -------------------------------------

        start_gen = time.time()
        for t in range(self.steps_per_design):
            action_log_prob, action = self.sample_actor(observation, cross_obs_tensor)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                m_action = int(deepcopy(act))
                designs[idx].append(m_action)
                observation_new[idx].append(m_action + 2)

            # Determine reward for each batch element
            if len(designs[0]) == self.steps_per_design:
                done = True
                for idx, design in enumerate(designs):

                    # Record design
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    reward, design_obj, stiffness, volume_frac = self.calc_reward(design_bitstr, weights[idx])
                    all_rewards[idx].append(reward)
                    all_rewards_stiffness[idx].append(stiffness)
                    all_rewards_volume[idx].append(volume_frac)
                    children.append(design_obj)
                    all_total_rewards.append(reward)
            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(designs):
                    all_rewards[idx].append(reward)
                    all_rewards_stiffness[idx].append(reward)
                    all_rewards_volume[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new
        print('--> ACTOR TIME:', round(time.time() - start_gen, 3))
        # print(observation[0])
        # print(observation[-1])

        # -------------------------------------
        # Sample Critic
        # -------------------------------------
        start_critic = time.time()
        value_stiff_t, value_vol_t = self.sample_critic(critic_observation_buffer, cross_obs_tensor)
        value_t = value_stiff_t + value_vol_t

        value_stiff_t = value_stiff_t.numpy().tolist()  # (30, 31)
        value_vol_t = value_vol_t.numpy().tolist()  # (30, 31)
        for idx, value_stiff, value_vol in zip(range(self.mini_batch_size), value_stiff_t, value_vol_t):
            last_stiff = value_stiff[-1]  # the predicted objective is already weighted
            last_vol = value_vol[-1]  # the predicted objective is already weighted
            reward = last_stiff + last_vol
            all_rewards[idx].append(reward)
            all_rewards_stiffness[idx].append(last_stiff)
            all_rewards_volume[idx].append(last_vol)
        print('--> CRITIC TIME:', round(time.time() - start_critic, 3))

        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------
        proc_time = time.time()
        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        all_returns_mo = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):
            rewards = np.array(all_rewards[idx])
            values = np.array(value_t[idx])
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor
            rewards_volume = np.array(all_rewards_volume[idx])
            ret_tensor_volume = discounted_cumulative_sums(
                rewards_volume, self.gamma
            )
            ret_tensor_volume = np.array(ret_tensor_volume, dtype=np.float32)
            rewards_stiffness = np.array(all_rewards_stiffness[idx])
            ret_tensor_stiffness = discounted_cumulative_sums(
                rewards_stiffness, self.gamma
            )
            ret_tensor_stiffness = np.array(ret_tensor_stiffness, dtype=np.float32)
            all_returns_mo[idx] = np.column_stack((ret_tensor_stiffness, ret_tensor_volume))

            ret_tensor = discounted_cumulative_sums(
                rewards, self.gamma
            )  # [:-1]
            ret_tensor = np.array(ret_tensor, dtype=np.float32)
            all_returns[idx] = ret_tensor



        advantage_mean, advantage_std = (
            np.mean(all_advantages),
            np.std(all_advantages),
        )
        all_advantages = (all_advantages - advantage_mean) / advantage_std

        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(all_actions, dtype=tf.int32)
        logprob_tensor = tf.convert_to_tensor(all_logprobs, dtype=tf.float32)
        advantage_tensor = tf.convert_to_tensor(all_advantages, dtype=tf.float32)

        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        return_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)
        return_tensor_mo = tf.convert_to_tensor(all_returns_mo, dtype=tf.float32)
        print('--> ADV/RET TIME:', round(time.time() - proc_time, 3))

        # -------------------------------------
        # Train Actor
        # -------------------------------------
        curr_time = time.time()
        policy_update_itr = 0
        for i in range(self.train_actor_iterations):
            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                observation_tensor,
                action_tensor,
                logprob_tensor,
                advantage_tensor,
                cross_obs_tensor,
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break
        print('--> ACTOR TRAIN TIME:', round(time.time() - curr_time, 3))
        # print('--> KL:', kl.numpy())
        # print('--> ENTROPY:', entr.numpy())
        # print('--> POLICY LOSS:', policy_loss.numpy())
        # print('--> ACTOR LOSS:', actor_loss.numpy())

        # -------------------------------------
        # Train Critic
        # -------------------------------------
        curr_time = time.time()
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                return_tensor_mo,
                cross_obs_tensor,
            )
        print('--> CRITIC TRAIN TIME:', round(time.time() - curr_time, 3))
        # print('--> VALUE LOSS:', value_loss.numpy())

        # Print designs
        for weight, design, reward_t in zip(weights, children, all_total_rewards):
            print(design.get_vector_str(), round(weight, 3), round(reward_t, 3))


        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_total_rewards),
            'c_loss': value_loss.numpy(),
            'p_loss': policy_loss.numpy(),
            'p_iter': policy_update_itr,
            'entropy': entr.numpy(),
            'kl': kl.numpy()
        }
        return children, epoch_info

    def calc_reward(self, bitstr, obj_weight):



        stiffness, volume_frac = self.problem.evaluate(bitstr)
        stiffness = float(stiffness)
        volume_frac = float(volume_frac)
        pop_design = [stiffness * -1.0, volume_frac]

        # Calculate reward
        w1 = obj_weight
        w2 = 1.0 - obj_weight
        stiff_term = w1 * stiffness
        vol_term = w2 * (1.0 - volume_frac)
        reward = stiff_term + vol_term

        # Create design object
        design = Design(design_vector=[int(i) for i in bitstr], evaluator=self.problem, num_bits=self.steps_per_design, c_type=self.c_type)
        design.set_objectives(pop_design[0], pop_design[1])
        if bitstr not in self.unique_designs:
            self.unique_designs.add(bitstr)
            self.unique_designs_vals.append(deepcopy([stiffness, volume_frac]))
            self.nfe += 1

        return reward, design, stiff_term, vol_term

    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation, cross_obs):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, cross_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, cross_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs = self.c_actor([observation_input, cross_input])

        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions

    def sample_critic(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_obs, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, None), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx):
        t_value = self.c_critic([observation_input, parent_input])  # (batch, seq_len, 2)
        t_value_stiff = t_value[:, :, 0]  # (batch, 1)
        t_value_vol = t_value[:, :, 1]  # (batch, 1)
        return t_value_stiff, t_value_vol

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.int32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 30), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 1), dtype=tf.float32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            parent_buffer,
    ):
        # print('-- TRAIN ACTOR --')
        # print('observation buffer:', observation_buffer.shape)
        # print('action buffer:', action_buffer.shape)
        # print('logprob buffer:', logprobability_buffer.shape)
        # print('advantage buffer:', advantage_buffer.shape)
        # print('parent buffer:', parent_buffer.shape)
        # print('pop vector:', pop_vector.shape)

        with tf.GradientTape() as tape:
            pred_probs = self.c_actor([observation_buffer, parent_buffer])
            pred_log_probs = tf.math.log(pred_probs)
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)

            # Total loss
            loss = 0

            # PPO Surrogate Loss
            ratio = tf.exp(
                logprobability - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
            loss += policy_loss

            # Entropy Term
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration
            loss = loss - (self.entropy_coef * entr)

        policy_grads = tape.gradient(loss, self.c_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.c_actor.trainable_variables))

        #  KL Divergence
        pred_probs = self.c_actor([observation_buffer, parent_buffer])
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)
        return kl, entr, policy_loss, loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(global_mini_batch_size, 31), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 31, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(global_mini_batch_size, 1), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values = self.c_critic(
                [observation_buffer, parent_buffer])  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss

    # ---------------------------------
    # Helper Functions
    # ---------------------------------

    def print_crossover(self, decisions, parents, design):
        p1_items = sum(parents[0].vector)
        p2_items = sum(parents[1].vector)

        d_str = ''.join([str(bit) for bit in design])
        d_copy = ''.join([str(bit) for bit in deepcopy(decisions)])
        print('--> (' + str(self.run_num) + ')', p1_items, '+', p2_items, 'gives', sum(design), d_str)

    # ---------------------------------
    # Plotting
    # ---------------------------------

    def record(self, epoch_info):
        if epoch_info is None:
            return

        # Record new epoch / print
        if self.debug is True:
            print(f"Proc GA_Task {self.run_num}", end=' ')
            for key, value in epoch_info.items():
                print(f"{key}: {value}", end=' | ')
            print('')

        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])
        self.hv.append(self.calc_pop_hv())
        self.nfes.append(self.nfe)
        
        if len(self.hv) % self.plot_freq == 0:
            print('--> PLOTTING')
        else:
            return

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(figsize=(12, 8))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.returns)
        plt.xlabel('Epoch')
        plt.ylabel('Mini-batch Return')
        plt.title('PPO Return Plot')

        # Critic loss plot
        plt.subplot(gs[0, 1])
        plt.plot(epochs, self.c_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Critic loss')
        plt.title('Critic Loss Plot')

        # Policy entropy plot
        plt.subplot(gs[0, 2])
        plt.plot(epochs, self.entropy)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy Plot')

        # KL divergence plot
        plt.subplot(gs[1, 0])
        plt.plot(epochs, self.kl)
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.title('KL Divergence Plot')

        # HV Plot
        plt.subplot(gs[1, 1])
        plt.plot(self.nfes, self.hv, label='PPO HV')
        plt.plot([r[0] for r in self.uniform_ga], [r[1] for r in self.uniform_ga], label='Uniform GA HV')
        # plt.plot([r[0] for r in self.random_search], [r[1] for r in self.random_search], label='Random Search HV')
        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.title('Hypervolume Plot')
        plt.legend()

        # Design Space
        plt.subplot(gs[1, 2])
        for point in self.unique_designs_vals:
            plt.scatter(point[0], point[1], marker='.', color='b')
        plt.xlabel('Value')
        plt.ylabel('Weight')
        plt.title('Design Plot ' + str(len(self.unique_designs_vals)) + ' designs')
        plt.tight_layout()

        # Save and close
        save_path = os.path.join(self.run_dir, 'plots.png')
        if self.run_val is True:
            save_path = os.path.join(self.run_dir, 'plots_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')

        # HV file
        hv_prog_file_path = os.path.join(self.run_dir, 'hv.json')
        if self.run_val is True:
            hv_prog_file_path = os.path.join(self.run_dir, 'hv_val_' + str(self.val_itr) + '.json')
        hv_progress = [(n, h) for n, h in zip(self.nfes, self.hv)]
        with open(hv_prog_file_path, 'w', encoding='utf-8') as file:
            json.dump(hv_progress, file, ensure_ascii=False, indent=4)






