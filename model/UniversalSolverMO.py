import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="UniversalSolverMO", name="UniversalSolverMO")
class UniversalSolverMO(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.use_cont_weights = config.bd_actor_continuous_weights
        self.learn_weights = config.bd_actor_learn_weights
        self.num_weight_vectors = config.bd_num_weight_vecs

        self.num_objectives = 2
        self.num_weights = self.num_objectives - 1
        self.vocab_size = 4
        self.vocab_output_size = 2
        self.gen_design_seq_length = 30
        self.embed_dim = config.bd_embed_dim
        self.num_heads = config.bd_actor_heads
        self.dense_dim = config.bd_actor_dense

        # Weight Embedding
        if self.use_cont_weights is True:
            self.weight_embedding_layer = layers.Dense(self.embed_dim)
        else:
            self.weight_embedding_layer = layers.Embedding(input_dim=self.num_weight_vectors, output_dim=self.embed_dim, mask_zero=False)

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1')
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4')
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5')


        # Design Prediction Head
        self.design_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs

        # 0. Weights: (batch, num_weights)
        if self.learn_weights is True:
            weight_seq = self.weight_embedding_layer(weights)
        else:
            weight_seq = tf.expand_dims(weights, axis=-1)
            weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

        # 1. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 2. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)


        # 3. Design Prediction Head
        design_prediction_logits = self.design_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction  # For training


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="UniversalSolverCriticMO", name="UniversalSolverCriticMO")
class UniversalSolverCriticMO(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.learn_weights = config.bd_critic_learn_weights
        self.num_objectives = 2
        self.num_weights = self.num_objectives - 1
        self.vocab_size = 4
        self.gen_design_seq_length = 31
        self.embed_dim = config.bd_critic_embed_dim
        self.num_heads = config.bd_critic_heads
        self.dense_dim = config.bd_critic_dense
        self.num_weight_vectors = config.bd_num_weight_vecs

        # Weight Embedding
        # self.weight_embedding_layer = layers.Dense(self.embed_dim)
        self.weight_embedding_layer = layers.Embedding(input_dim=self.num_weight_vectors, output_dim=self.embed_dim, mask_zero=False)

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1')
        # self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')




    def call(self, inputs, training=False, mask=None):
        design_sequences, weights = inputs
        # weights: (batch, 1)

        # 0. Weights: (batch, num_weights)
        if self.learn_weights is True:
            weight_seq = self.weight_embedding_layer(weights)
        else:
            weight_seq = tf.expand_dims(weights, axis=-1)
            weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])

        # 1. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 2. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)
        # decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)

        # 3. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)















