from model import get_universal_crossover

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


# from official.nlp.modeling.ops import sampling_module
from model import sampling_module






class Sampling:

    def __init__(self):
        self.actor, self.critic = get_universal_crossover()
        self.start_token_id = 1
        self.batch_size = 100
        self.decoding_steps = 60
        self.cross_attn_input = 610
        self.embed_dim = 64

    def generate2(self):
        params = {}
        params['num_heads'] = 64
        params['num_layers'] = 1
        params['batch_size'] = 100
        params['n_dims'] = 64
        params['max_decode_length'] = 60

        cache = {
            f'{i}': {
                'key': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'],
                               int(params['n_dims'] / params['num_heads'])], dtype=tf.float32),
                'value': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'],
                               int(params['n_dims'] / params['num_heads'])], dtype=tf.float32)
            } for i in range(params['num_layers'])
        }
        cache['pop_inputs'] = tf.zeros((100, 610), dtype=tf.float32)



        def _symbols_to_logits_fn():
            """Calculates logits of the next tokens."""

            def symbols_to_logits_fn(ids, i, temp_cache):
                pop_inputs = temp_cache.get('pop_inputs')  # shape: (100, 610)
                logits, attn_scores = self.actor.inference(ids, pop_inputs, i, temp_cache)
                logits = tf.cast(logits, tf.float32)  # shape: (100, curr_seq_len, 2)
                logits = logits[:, i, :]
                return logits, temp_cache

            return symbols_to_logits_fn


        top_k_obj = sampling_module.SamplingModule(
            length_normalization_fn=None,
            dtype=tf.float32,
            symbols_to_logits_fn=_symbols_to_logits_fn(),
            vocab_size=3,
            max_decode_length=params['max_decode_length'],
            eos_id=10,
            sample_temperature=tf.constant(1.0),
            top_k=tf.constant(3),
            padded_decode=False,
            enable_greedy=False
        )

        initial_ids = tf.ones((params['batch_size'],), dtype=tf.float32)
        # initial_ids = tf.expand_dims(initial_ids, axis=-1)
        # initial_ids = tf.constant([9, 1])

        ids, _ = top_k_obj.generate(initial_ids=initial_ids, initial_cache=cache)
        print("top-k sampled Ids:", ids)

    def generate(self):
        self.actor.new_cache()

        observation = [[self.start_token_id] for x in range(self.batch_size)]
        designs = [[] for x in range(self.batch_size)]


        cross_obs = tf.zeros((self.batch_size, self.cross_attn_input))
        cross_obs_encoded = tf.zeros((self.batch_size, self.cross_attn_input, self.embed_dim), dtype=tf.float32)

        for t in range(self.decoding_steps):
            curr_time = time.time()
            action_log_prob, action, attn_scores = self.sample_actor(observation, cross_obs, cross_obs_encoded)
            for idx, act in enumerate(action.numpy()):
                # Get action (either inherit from parent a or b)
                m_action = int(deepcopy(act))
                observation[idx].append(m_action + 2)
                designs[idx].append(m_action)
            print('--> STEP', t, '-', time.time() - curr_time)


    def sample_actor(self, observation, parent_obs, cross_obs_encoded):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.float32)
        parent_input = tf.convert_to_tensor(parent_obs, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, parent_input, inf_idx, cross_obs_encoded)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(100, None), dtype=tf.float32),
        tf.TensorSpec(shape=(100, 610), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(100, 610, 64), dtype=tf.float32),
    ])
    def _sample_actor(self, observation_input, parent_input, inf_idx, cross_obs_encoded):


        # pred_probs, attn_scores = self.actor([observation_input, parent_input, parent_pos_input])
        pred_probs, attn_scores = self.actor.inference(observation_input, parent_input, inf_idx, cross_obs_encoded)


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





if __name__ == '__main__':
    client = Sampling()

    start_time = time.time()
    client.generate()
    # client.generate2()
    print('--> FINAL TIME:', time.time() - start_time)











