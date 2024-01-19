from tensorflow import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
import numpy as np
from keras_nlp.layers import TransformerEncoder


# from keras_nlp.layers import TransformerDecoder
from model.TransformerDecoder import TransformerDecoder

from keras_nlp.layers import TokenAndPositionEmbedding

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit

# @keras.saving.register_keras_serializable(package="UniversalSolver", name="UniversalSolver")
class UniversalSolver(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.gen_design_seq_length = 60  # this is the max length of a design
        self.embed_dim = 32
        self.num_heads = 32
        self.dense_dim = 512

        # Solution Embedding
        self.decision_embedding_layer = TokenAndPositionEmbedding(
            4,  # (0) mask, (1) start_token, (2) 1 bit, (3) 2 bit
            self.gen_design_seq_length,  # the first token is the objective weighting
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1')
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')
        self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4')
        self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5')

        # Inheritance Prediction Head
        self.design_prediction_head = layers.Dense(
            2,  # 0 or 1
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        decisions = inputs  # decisions: (batch, 60)

        # 1. Embed decisions
        decisions_embedded = self.decision_embedding_layer(decisions, training=training)  # shape (batch, 59, embed_dim)

        # 2. Decode solution
        decoded_design = decisions_embedded
        decoded_design, attn_scores = self.decoder_1(decoded_design, use_causal_mask=True)
        decoded_design, attn_scores = self.decoder_2(decoded_design, use_causal_mask=True)
        decoded_design, attn_scores = self.decoder_3(decoded_design, use_causal_mask=True)
        decoded_design, attn_scores = self.decoder_4(decoded_design, use_causal_mask=True)
        decoded_design, attn_scores = self.decoder_5(decoded_design, use_causal_mask=True)

        # 5. Design Prediction Head
        design_prediction_logits = self.design_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction, attn_scores

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def load_target_weights(self, model_instance, trainable=False):
        """
        Updates the weights of the current instance with the weights of the provided model instance.
        """
        for target_layer, source_layer in zip(self.layers, model_instance.layers):
            target_layer.set_weights(source_layer.get_weights())



        self.trainable = trainable


# @keras.saving.register_keras_serializable(package="UniversalSolverCritic", name="UniversalSolverCritic")
class UniversalSolverCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.gen_design_seq_length = 61  # this is the max length of a design
        self.embed_dim = 32
        self.num_heads = 32
        self.dense_dim = 512

        # Solution Embedding
        self.decision_embedding_layer = TokenAndPositionEmbedding(
            4,  # (0) mask, (1) start_token, (2) 1 bit, (3) 2 bit
            self.gen_design_seq_length,  # the first token is the objective weighting
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first,
                                            name='decoder_1')
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')

        # Design Prediction Head
        self.design_prediction_head = layers.Dense(
            1,  # value
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        decisions = inputs  # decisions: (batch, 60)

        # 1. Embed decisions
        decisions_embedded = self.decision_embedding_layer(decisions, training=training)  # shape (batch, 59, embed_dim)

        # 2. Decode solution
        decoded_design = decisions_embedded
        decoded_design, attn_scores = self.decoder_1(decoded_design, use_causal_mask=True)
        decoded_design, attn_scores = self.decoder_2(decoded_design, use_causal_mask=True)

        # 5. Design Prediction Head
        design_prediction_logits = self.design_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction, attn_scores

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def load_target_weights(self, model_instance, trainable=False):
        """
        Updates the weights of the current instance with the weights of the provided model instance.
        """
        for target_layer, source_layer in zip(self.layers, model_instance.layers):
            target_layer.set_weights(source_layer.get_weights())

        self.trainable = trainable








