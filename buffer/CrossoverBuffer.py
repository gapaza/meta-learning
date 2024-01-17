import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import scipy.signal
import time
import config
from copy import deepcopy

# Clarify
# - advantage_buffer
# - return_buffer (used as target values for critic training)


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class CrossoverBuffer:
    # Buffer for storing trajectories
    def __init__(self, trajectory_len, mini_b_size, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.action_pointer, self.trajectory_pointer = 0, 0

        # Buffer initialization
        self.observation_buffer = [[0.0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]
        self.action_buffer = [[0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]

        self.value_buffer = [[0.0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]

        self.reward_buffer = [[0.0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]
        self.logprobability_buffer = [[0.0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]

        self.cross_obs_buffer = [[0.0] for _ in range(mini_b_size)]

        self.advantage_buffer = [[0.0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]
        self.return_buffer = [[0.0 for _ in range(trajectory_len)] for _ in range(mini_b_size)]
        self.return_buffer_mo = [[ [0.0, 0.0] for _ in range(trajectory_len)] for _ in range(mini_b_size)]  # Use for back-propagation of value model


    def store_batch(self, observations, action, reward, value, logprobability):
        for traj_idx in range(len(observations)):
            self.observation_buffer[traj_idx][self.action_pointer] = observations[traj_idx]
            self.action_buffer[traj_idx][self.action_pointer] = action[traj_idx]
            self.reward_buffer[traj_idx][self.action_pointer] = reward[traj_idx]
            self.value_buffer[traj_idx][self.action_pointer] = value[traj_idx]
            self.logprobability_buffer[traj_idx][self.action_pointer] = logprobability[traj_idx]
        self.action_pointer += 1


    def next_trajectory(self, parent_obs_vector):
        for idx in range(len(parent_obs_vector)):
            self.cross_obs_buffer[idx] = deepcopy(parent_obs_vector[idx])
        self.action_pointer = 0

    def finish_trajectory(self, last_values):
        new_rewards = []
        for traj_idx, reward_traj in enumerate(self.reward_buffer):
            new_rewards.append(sum(self.reward_buffer[traj_idx]))

        for i in range(len(last_values)):
            last_value_total = last_values[i]

            rewards = np.append(self.reward_buffer[i], last_value_total)
            values = np.append(self.value_buffer[i], last_value_total)
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            adv_tensor = np.array(adv_tensor, dtype=np.float32)
            self.advantage_buffer[i] = adv_tensor


            ret_tensor = discounted_cumulative_sums(
                rewards, self.gamma
            )  # [:-1]
            ret_tensor = np.array(ret_tensor, dtype=np.float32)
            self.return_buffer[i] = ret_tensor

        return new_rewards



    # --------------------------
    # Get Buffer Data
    # --------------------------

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.action_pointer, self.trajectory_start_index, self.trajectory_pointer = 0, 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )

        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std

        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
            self.cross_obs_buffer,
        )










