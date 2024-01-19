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
import scipy.signal


from task.Alg_Task import Alg_Task


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class FGA_Task(AbstractTask):

    def __init__(self, run_num=0, barrier=None, problem=None, limit=50, actor_load_path=None, critic_load_path=None, debug=False, c_type='uniform', start_nfe=0):
        super(FGA_Task, self).__init__(run_num, barrier, problem, limit, actor_load_path, critic_load_path)
        self.debug = debug
        self.c_type = c_type
        self.start_nfe = start_nfe

        # HV
        self.ref_point = np.array([0, 2])  # value, weight
        self.hv_client = HV(self.ref_point)
        self.nds = NonDominatedSorting()
        self.unique_designs = set()
        self.unique_designs_vals = []

        # Algorithm parameters
        self.pop_size = 30  # 32 FU_NSGA2, 10 U_NSGA2
        self.offspring_size = 30  # 32 FU_NSGA2, 30 U_NSGA2
        self.mini_batch_size = 30
        self.num_cross_obs_designs = 10
        self.max_nfe = 6000
        self.nfe = 0
        self.limit = limit
        self.steps_per_design = 60  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.entropy_coef = 0.0  # was 0.1 for large exploration
        self.counter = 0
        self.decision_start_token_id = 1
        self.num_actions = 2

        # Population
        self.population = []
        self.hv = []  # hv progress over time
        self.nfes = []  # nfe progress over time

        # Results
        self.uniform_ga = []
        self.random_search = []
        self.init_comparison_data()

    def init_comparison_data(self):
        task_runner = Alg_Task(
            run_num=0,
            barrier=None,
            problem=self.problem,
            limit=1000,
            actor_load_path=None,
            critic_load_path=None,
            c_type='uniform',
        )
        self.uniform_ga = task_runner.run()
        task_runner = Alg_Task(
            run_num=0,
            barrier=None,
            problem=self.problem,
            limit=1000,
            actor_load_path=None,
            critic_load_path=None,
            c_type='random',
        )
        self.random_search = task_runner.run()


    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.00001
        self.critic_learning_rate = 0.00001
        self.train_actor_iterations = 120
        self.train_critic_iterations = 25
        if self.actor_optimizer is None:
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_universal_crossover(self.actor_load_path, self.critic_load_path)

    def run(self):
        print('--> RUNNING GA TASK:', self.run_num)
        self.build()

        self.init_population()
        self.eval_population()

        training_started = self.nfe >= self.start_nfe
        terminated = False
        counter = 0
        while terminated is False and self.nfe < self.max_nfe:
            # print('EPOCH', counter, '/', self.limit)
            if training_started is False and self.nfe >= self.start_nfe:
                training_started = True
                print('--> Training Started:', self.nfe, '/', self.start_nfe)

            # 1. Create offspring
            epoch_info = self.create_offspring()

            # 2. Evaluate offspring
            self.eval_population()

            # 3. Prune population
            self.prune_population()

            # 4. Log iteration
            self.record(epoch_info)
            self.activate_barrier()

            if self.nfe >= self.start_nfe:
                counter += 1
                if counter >= self.limit:
                    terminated = True

        # Save the parameters of the current actor and critic
        self.c_actor.save_weights(self.actor_save_path)
        self.c_critic.save_weights(self.critic_save_path)

        return

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
            design = Design(evaluator=self.problem, num_bits=self.steps_per_design, c_type=self.c_type)
            self.population.append(design)

    def eval_population(self):
        evals = []
        for design in self.population:
            if self.nfe < self.start_nfe and design.evaluated is False:
                self.nfe += 1
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
    # Tournament Functions
    # -------------------------------------

    def binary_tournament(self):
        p1 = random.randrange(len(self.population))
        p2 = random.randrange(len(self.population))
        while p1 == p2:
            p2 = random.randrange(len(self.population))

        player1 = self.population[p1]
        player2 = self.population[p2]

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
        for k, front in enumerate(fronts, start=1):
            crowding_of_front = utils.calc_crowding_distance(F[front, :])
            for i, idx in enumerate(front):
                self.population[idx].crowding_dist = crowding_of_front[i]
                self.population[idx].rank = k

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
        if self.nfe < self.start_nfe:
            offspring, epoch_info = self.crossover_parents_init(pairs)
        else:
            offspring, epoch_info = self.crossover_parents(pairs)
        self.population.extend(offspring)
        return epoch_info

    def crossover_parents(self, parent_pairs):
        mini_batch = parent_pairs
        # children, epoch_info = self.run_mini_batch(mini_batch)
        children, epoch_info = self.fast_mini_batch(mini_batch)
        return children, epoch_info

    def crossover_parents_init(self, parent_pairs):
        offspring = []
        for pair in parent_pairs:
            parent1 = pair[0]
            parent2 = pair[1]
            child = Design(evaluator=self.problem, c_type=self.c_type)
            child.crossover(parent1, parent2)
            child.mutate()
            offspring.append(child)
        return offspring, None


    # -------------------------------------
    # PPO Functions
    # -------------------------------------

    def new_buffer(self):
        buffer = CrossoverBuffer(self.steps_per_design, self.mini_batch_size)
        return buffer

    def get_cross_obs(self, p1, p2):
        p1_str = p1.get_vector_str()
        p2_str = p2.get_vector_str()
        pop_strs = [design.get_vector_str() for design in self.population]
        p1_idx = pop_strs.index(p1_str)
        p2_idx = pop_strs.index(p2_str)

        cross_obs_indices = [x for x in range(self.num_cross_obs_designs)]
        if p1_idx not in cross_obs_indices and p2_idx not in cross_obs_indices:
            cross_obs_indices[-2] = p1_idx
            cross_obs_indices[-1] = p2_idx
        elif p1_idx not in cross_obs_indices:
            for idx in range(len(cross_obs_indices)-1, -1, -1):
                if cross_obs_indices[idx] != p2_idx:
                    cross_obs_indices[idx] = p1_idx
                    break
        elif p2_idx not in cross_obs_indices:
            for idx in range(len(cross_obs_indices)-1, -1, -1):
                if cross_obs_indices[idx] != p1_idx:
                    cross_obs_indices[idx] = p2_idx
                    break

        design_id_tokens = []
        for idx in cross_obs_indices:
            if idx == p1_idx:
                design_id_tokens.append(3)
            elif idx == p2_idx:
                design_id_tokens.append(4)
            else:
                design_id_tokens.append(2)
        cross_obs_designs = [self.population[idx] for idx in cross_obs_indices]

        #  0 and 1 are decisions, 2 is begin non-parent token, 3 is parent 1 token, 4 is parent 2 token
        pop_vector = []
        pop_pos_vector = []
        for idx, design in enumerate(cross_obs_designs, start=0):
            pop_vector.append(design_id_tokens[idx])
            pop_vector.extend(deepcopy(design.vector))
            pop_pos_vector.extend([idx+1 for _ in range(len(design.vector) + 1)])

        return pop_vector, pop_pos_vector

    def fast_mini_batch(self, mini_batch):
        children = []

        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]

        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Preprocess cross attention observation input
        parent_obs = []
        for pair in mini_batch:
            pop_vector, pop_pos_vector = self.get_cross_obs(pair[0], pair[1])
            parent_obs.append(pop_vector)

        # -------------------------------------
        # Sample Actor
        # -------------------------------------

        start_gen = time.time()
        for t in range(self.steps_per_design):
            action_log_prob, action, attn_scores = self.sample_actor(observation, parent_obs)  # returns shape: (batch,) and (batch,)
            action_log_prob = action_log_prob.numpy().tolist()
            if t == 0:
                # We are looking at the first design in the batch
                self.plot_attention_scores(attn_scores, mini_batch[0], parent_obs)

            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):

                # Get action (either inherit from parent a or b)
                all_actions[idx].append(deepcopy(act))
                all_logprobs[idx].append(action_log_prob[idx])
                ppair = mini_batch[idx]
                m_action = int(deepcopy(act))

                # Get parent bit
                p1_bit = ppair[0].vector[t]
                p2_bit = ppair[1].vector[t]
                if m_action == 0:
                    m_bit = p1_bit
                elif m_action == 1:
                    m_bit = p2_bit
                else:
                    raise ValueError('--> INVALID ACTION VALUE:', act)
                designs[idx].append(m_bit)
                observation_new[idx].append(m_action + 2)

            # Determine reward for each batch element
            if len(designs[0]) == self.steps_per_design:
                done = True
                for idx, design in enumerate(designs):

                    # TODO: potentially move mutation operator somewhere else
                    bit_idx = random.randint(0, self.steps_per_design - 1)  # ClimateCentric
                    if design[bit_idx] == 0:
                        design[bit_idx] = 1
                    else:
                        design[bit_idx] = 0

                    # Record design
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    reward, design_obj = self.calc_reward(design_bitstr)
                    all_rewards[idx].append(reward)
                    children.append(design_obj)
                    all_total_rewards.append(reward)
            else:
                done = False
                reward = 0.0
                for idx, _ in enumerate(designs):
                    all_rewards[idx].append(reward)

            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new
        print('--> ACTOR TIME:', round(time.time() - start_gen, 3))
        print(observation[0])
        print(observation[-1])

        # -------------------------------------
        # Sample Critic
        # -------------------------------------
        start_critic = time.time()
        value_t = self.sample_critic(critic_observation_buffer, parent_obs).numpy()
        value_t = value_t.tolist()  # (100, 61)
        for idx, value in enumerate(value_t):
            all_rewards[idx].append(value[-1])
        print('--> CRITIC TIME:', round(time.time() - start_critic, 3))

        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------
        proc_time = time.time()
        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):
            rewards = np.array(all_rewards[idx])
            values = np.array(value_t[idx])
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor
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
        cross_obs = tf.convert_to_tensor(parent_obs, dtype=tf.float32)

        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        return_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)
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
                cross_obs,
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
                critic_observation_buffer,
                return_tensor,
                parent_obs,
            )
        print('--> CRITIC TRAIN TIME:', round(time.time() - curr_time, 3))
        # print('--> VALUE LOSS:', value_loss.numpy())

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

    def run_mini_batch(self, mini_batch):
        children = []
        buffer = self.new_buffer()

        all_last_values = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        designs = [[] for x in range(self.mini_batch_size)]
        epoch_designs = []
        observation = [[self.decision_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]

        # Preprocess cross attention observation input
        parent_obs = []  # (batch, design_len * pop_size + pop_size)
        parent_obs_pos = []
        for pair in mini_batch:
            p1 = pair[0]
            p2 = pair[1]
            pop_vector, pop_pos_vector = self.get_cross_obs(p1, p2)
            parent_obs.append(pop_vector)
            parent_obs_pos.append(pop_pos_vector)

        start_gen = time.time()
        for t in range(self.steps_per_design):

            # 1. Sample actor
            action_log_prob, action, attn_scores = self.sample_actor(observation, parent_obs, parent_obs_pos)  # returns shape: (batch,) and (batch,)
            if t == 0:
                # We are looking at the first design in the batch
                self.plot_attention_scores(attn_scores, mini_batch[0], parent_obs)


            # 2. Update observation for each batch element
            observation_new = deepcopy(observation)
            for idx, act in enumerate(action.numpy()):

                # Get action (either inherit from parent a or b)
                all_actions[idx].append(deepcopy(act))
                ppair = mini_batch[idx]
                m_action = float(deepcopy(act))

                # Get parent bit
                p1_bit = ppair[0].vector[t]
                p2_bit = ppair[1].vector[t]
                if m_action == 0:
                    m_bit = p1_bit
                elif m_action == 1:
                    m_bit = p2_bit
                else:
                    raise ValueError('--> INVALID ACTION VALUE:', act)
                designs[idx].append(m_bit)
                observation_new[idx].append(m_bit + 2)

            # 3. Determine reward for each batch element
            rewards = []
            if len(designs[0]) == self.steps_per_design:
                done = True
                for idx, design in enumerate(designs):

                    # TODO: potentially move mutation operator somewhere else
                    bit_idx = random.randint(0, self.steps_per_design-1)  # ClimateCentric
                    if design[bit_idx] == 0:
                        design[bit_idx] = 1
                    else:
                        design[bit_idx] = 0

                    # Record design
                    design_bitstr = ''.join([str(bit) for bit in design])
                    epoch_designs.append(design_bitstr)

                    # Evaluate design
                    reward, design_obj = self.calc_reward(design_bitstr)
                    rewards.append(reward)
                    children.append(design_obj)
            else:
                reward = 0.0
                for _ in designs:
                    rewards.append(reward)
                done = False

            t_value = self.sample_critic(observation, parent_obs, parent_obs_pos)
            logprobability_t = action_log_prob

            last_obs = []
            for agent_obs in observation:
                last_obs.append(agent_obs[-1])

            buffer.store_batch(deepcopy(last_obs), action, rewards, t_value, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if done
            if done is True:
                t_value = self.sample_critic(observation, parent_obs, parent_obs_pos)  # returns shape: (batch,) and (batch,)
                critic_observation_buffer = deepcopy(observation)
                all_last_values = t_value
                buffer.next_trajectory(parent_obs)
                break

        print('--> GEN TIME:', round(time.time() - start_gen, 3))
        new_rewards = buffer.finish_trajectory(all_last_values)

        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
            parent_obs_buffer,
        ) = buffer.get()

        # Actor Tensors
        observation_buffer = tf.convert_to_tensor(observation_buffer, dtype=tf.float32)
        action_buffer = tf.convert_to_tensor(action_buffer, dtype=tf.int32)
        logprobability_buffer = tf.convert_to_tensor(logprobability_buffer, dtype=tf.float32)
        advantage_buffer = tf.convert_to_tensor(advantage_buffer, dtype=tf.float32)
        parent_obs = tf.convert_to_tensor(parent_obs, dtype=tf.float32)  # (batch, 2, 60)
        parent_obs_pos = tf.convert_to_tensor(parent_obs_pos, dtype=tf.float32)  # (batch, pop_size, 60)

        # Critic Tensors
        critic_observation_buffer = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        return_buffer = tf.convert_to_tensor(return_buffer, dtype=tf.float32)  # (batch, seq_len, num_objectives)




        curr_time = time.time()
        policy_update_itr = 0
        for i in range(self.train_actor_iterations):
            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                observation_buffer,
                action_buffer,
                logprobability_buffer,
                advantage_buffer,
                parent_obs,
                parent_obs_pos,
            )
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # print('--> ACTOR TRAIN TIME:', round(time.time() - curr_time, 3))

        # Train critic
        curr_time = time.time()
        # value_loss = self.train_critic_loop(
        #     critic_observation_buffer,
        #     return_buffer,
        #     parent_obs,
        #     parent_obs_pos
        # )
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_buffer,
                return_buffer,
                parent_obs,
                parent_obs_pos
            )
        # print('--> CRITIC TRAIN TIME:', round(time.time() - curr_time, 3))



        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(new_rewards),
            'c_loss': value_loss.numpy(),
            'p_loss': policy_loss.numpy(),
            'p_iter': policy_update_itr,
            'entropy': entr.numpy(),
            'kl': kl.numpy()
        }
        return children, epoch_info

    def calc_reward(self, bitstr):
        value, weight = self.problem.evaluate(bitstr)
        value = float(value)
        weight = float(weight)
        pop_design = [value * -1.0, weight]

        extended_pop = self.eval_population()
        if pop_design not in extended_pop:
            extended_pop.append(pop_design)
        pop_design_idx = extended_pop.index(pop_design)
        F = np.array(extended_pop)
        design_reward, crowding_reward = 0, 0  # assume design not in pareto pop
        fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
        for k, front in enumerate(fronts, start=1):
            crowding_of_front = utils.calc_crowding_distance(F[front, :])
            advantage = 1.0 / k
            for i, idx in enumerate(front):
                if idx == pop_design_idx:
                    design_reward = advantage
                    crowding_reward = crowding_of_front[i]
        design = Design(design_vector=[int(i) for i in bitstr], evaluator=self.problem, num_bits=self.steps_per_design, c_type=self.c_type)
        design.set_objectives(pop_design[0], pop_design[1])
        if bitstr not in self.unique_designs:
            self.unique_designs.add(bitstr)
            self.unique_designs_vals.append([value, weight])
            self.nfe += 1
        return design_reward, design

    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        parent_input = tf.convert_to_tensor(parent_obs, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, parent_input, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(30, None), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 610), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, parent_input, inf_idx):
        # print('sampling actor', inf_idx)
        # pred_probs, attn_scores = self.c_actor([observation_input, parent_input, parent_pos_input])
        pred_probs, attn_scores = self.c_actor.inference(observation_input, parent_input, inf_idx)

        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)

        return actions_log_prob, actions, attn_scores




    def sample_critic(self, observation, parent_obs):
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        parent_input = tf.convert_to_tensor(parent_obs, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input, parent_input, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(30, None), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 610), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, observation_input, parent_input, inf_idx):
        t_value, attn_scores = self.c_critic.inference(observation_input, parent_input, inf_idx)
        t_value = tf.squeeze(t_value, axis=-1)
        # t_value = t_value[:, inf_idx]
        return t_value








    @tf.function(input_signature=[
        tf.TensorSpec(shape=(30, 60), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 60), dtype=tf.int32),
        tf.TensorSpec(shape=(30, 60), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 60), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 610), dtype=tf.float32),
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
            pred_probs, attn_scores = self.c_actor([observation_buffer, parent_buffer])
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
        pred_probs, attn_scores = self.c_actor([observation_buffer, parent_buffer])
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
        tf.TensorSpec(shape=(30, 61), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 61), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 610), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
            parent_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            pred_values, attn_scores = self.c_critic(
                [observation_buffer, parent_buffer])  # (batch, seq_len, 2)
            pred_values = tf.squeeze(pred_values, axis=-1)  # (batch, seq_len)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(30, 61), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 61), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 610), dtype=tf.float32),
        tf.TensorSpec(shape=(30, 610), dtype=tf.float32)
    ])  # add num_iterations to the signature
    def train_critic_loop(self, observation_buffer, return_buffer, parent_buffer, pop_vector):
        total_value_loss = 0.0

        for i in tf.range(self.train_critic_iterations):
            with tf.GradientTape() as tape:
                pred_values, attn_scores = self.c_critic(
                    [observation_buffer, parent_buffer, pop_vector])
                pred_values = tf.squeeze(pred_values, axis=-1)

                # Value Loss (mse)
                value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

            total_value_loss += value_loss

            critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return total_value_loss / tf.cast(self.train_critic_iterations, tf.float32)


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
        plt.plot([r[0] for r in self.random_search], [r[1] for r in self.random_search], label='Random Search HV')
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
        plt.savefig(os.path.join(self.run_dir, 'plots.png'))
        plt.close('all')

        # HV file
        hv_prog_file_path = os.path.join(self.run_dir, 'hv.json')
        hv_progress = [(n, h) for n, h in zip(self.nfes, self.hv)]
        with open(hv_prog_file_path, 'w', encoding='utf-8') as file:
            json.dump(hv_progress, file, ensure_ascii=False, indent=4)




    def plot_attention_scores(self, attn_scores, parent_pair, parent_obs):
        # print(attn_scores.shape)  # (30, 64, 60, 610)
        # - 30 is batch size
        # - 64 is number of heads
        # - 60 is number of seq elements (queries)
        # - 610 is number of key elements (keys)
        batch_index = 0
        head_index = 0
        query_index = 0

        attention_mean_across_heads = tf.reduce_mean(attn_scores, axis=1)  # average across heads
        attention_vector_mean_heads = attention_mean_across_heads[batch_index, query_index, :]
        attention_vector_mean_heads /= tf.reduce_sum(attention_vector_mean_heads)  # normalize


        p1 = parent_pair[0]
        p2 = parent_pair[1]
        pop_designs = [design.get_vector_str() for design in self.population]
        p1_idx_pop = pop_designs.index(p1.get_vector_str())
        p2_idx_pop = pop_designs.index(p2.get_vector_str())

        p_obs = parent_obs[batch_index]
        p1_idx = -1
        if 3 in p_obs:
            p1_idx = p_obs.index(3)

        p2_idx = -1
        if 4 in p_obs:
            p2_idx = p_obs.index(4)

        plt.figure(figsize=(15, 4))  # Wider figure
        attention_vector = attn_scores[batch_index, head_index, query_index]
        attention_vector = attention_vector / tf.reduce_sum(attention_vector, axis=-1, keepdims=True)
        ax = sns.heatmap(attention_vector_mean_heads.numpy()[None, :], cmap='viridis', cbar=True)
        plt.title(f'Attention Map (Averaged over heads, Query {query_index}) \n Parent 1: {p1_idx} - {p1_idx_pop} | Parent 2: {p2_idx} - {p2_idx_pop}')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')

        # Set x-axis ticks every 30 units
        tick_positions = np.arange(0, attention_vector.shape[-1], 61)  # Start at 0 and end at the length of the axis
        tick_labels = tick_positions  # This simply uses the positions as labels, update this if you want custom labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        save_path = os.path.join(self.run_dir, 'attention.png')
        plt.savefig(save_path)
        plt.close('all')







