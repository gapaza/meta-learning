from tensorflow import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
import numpy as np
from keras_nlp.layers import TransformerEncoder
from keras_nlp.layers import TokenAndPositionEmbedding

# import tensorflow_models as tfm



# from keras_nlp.layers import TransformerDecoder
from model.TransformerDecoder import TransformerDecoder
# from tensorflow_models.nlp.models import TransformerDecoder



# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit


# @keras.saving.register_keras_serializable(package="UniversalCrossoverCritic", name="UniversalCrossoverCritic")
class UniversalCrossoverCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.pop_seq_len = 610  # 310 | 610
        self.gen_design_seq_length = 61  # 31 | 61
        self.embed_dim = 64
        self.num_heads = 64
        self.dense_dim = 512
        self.pop_size = 10

        # Decision Embedding
        self.decision_embedding_layer = TokenAndPositionEmbedding(
            4,  # (0) mask, (1) start_token, (2) parent_1 bit, (3) parent_2 bit
            self.gen_design_seq_length,  # the first token is the objective weighting
            self.embed_dim,
            mask_zero=True
        )

        # Population Embeddings
        self.pop_embedding_layer = TokenAndPositionEmbedding(
            5,  # 0 and 1 are decisions, 2 is begin non-parent token, 3 is parent 1 token, 4 is parent 2 token
            self.pop_seq_len,  # self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=False
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1')
        # self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4')
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5')

        # self.decoder_1 = TransformerDecoder(
        #     num_layers=1,
        #     num_attention_heads=self.num_heads,
        #     intermediate_size=self.dense_dim
        # )


        # Output Prediction Head
        self.output_modeling_head = layers.Dense(1, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        decisions, pop, pop_pos = inputs
        # decisions: (batch, 60)
        # pop:     (batch, 60 * pop_size + pop_size)
        # pop_pos: (batch, 60 * pop_size + pop_size)

        # 1. Embed population
        pop_embedded = self.pop_embedding_layer(pop, training=training)  # shape (batch, 60 * pop_size + pop_size, embed_dim)

        # 2. Embed decisions
        decisions_embedded = self.decision_embedding_layer(decisions, training=training)  # shape (batch, 59, embed_dim)

        # 3. Decode design
        decoded_design = decisions_embedded
        decoded_design, attn_scores = self.decoder_1(decoded_design, encoder_sequence=pop_embedded, use_causal_mask=True)
        # decoded_design, attn_scores = self.decoder_2(decoded_design, encoder_sequence=pop_embedded, use_causal_mask=True)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=pop_embedded, use_causal_mask=True)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=pop_embedded, use_causal_mask=True)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=pop_embedded, use_causal_mask=True)

        # decoded_design, attn_scores = self.decoder_1(decoded_design, pop_embedded,
        #                                              use_causal_mask=True)

        # 4. Design Prediction Head
        design_prediction_logits = self.output_modeling_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction, attn_scores

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # ---------------------------------------
    # Positional Encoding (from vaswani et al.)
    # ---------------------------------------

    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    # -------------------------------------------
    # Load weights
    # -------------------------------------------

    def load_target_weights(self, model_instance, trainable=False):
        """
        Updates the weights of the current instance with the weights of the provided model instance.
        Args:
            model_instance (SatelliteMLP): Instance of SatelliteMLP whose weights will be used.
        """
        for target_layer, source_layer in zip(self.layers, model_instance.layers):
            target_layer.set_weights(source_layer.get_weights())
        self.trainable = trainable




# @keras.saving.register_keras_serializable(package="FastUniversalCrossoverCritic", name="FastUniversalCrossoverCritic")
class FastUniversalCrossoverCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.pop_seq_len = 61
        self.gen_design_seq_length = 61  # this is the max length of a design
        self.embed_dim = 32
        self.num_heads = 32
        self.dense_dim = 512
        self.pop_size = 32

        # Decision Embedding
        self.decision_embedding_layer = TokenAndPositionEmbedding(
            4,  # (0) mask, (1) start_token, (2) parent_1 bit, (3) parent_2 bit
            self.gen_design_seq_length,  # the first token is the objective weighting
            self.embed_dim,
            mask_zero=True
        )

        # Population encoders
        self.encoder_1 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=False, name='encoder_1')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first,
                                            name='decoder_1')
        # self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4')
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(1, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        decisions, pop, pop_pos = inputs
        # decisions: (batch, 60)
        # pop:     (batch, 60 * pop_size + pop_size)
        # pop_pos: (batch, 60 * pop_size + pop_size)

        # reshape pop to (batch, 61, pop_size) then repeat each bit 8 times to get (batch, 61, 256)
        pop = tf.reshape(pop, [-1, self.pop_seq_len, self.pop_size])
        # pop = tf.repeat(pop, repeats=4, axis=2)
        pop = tf.cast(pop, tf.float32)

        # 3. Embed decisions
        decisions_embedded = self.decision_embedding_layer(decisions, training=training)  # shape (batch, 59, embed_dim)

        # 4. Decode design
        decoded_design = decisions_embedded
        decoded_design, attn_scores = self.decoder_1(decoded_design, encoder_sequence=pop, use_causal_mask=True)
        # decoded_design = self.decoder_2(decoded_design, encoder_sequence=pop, use_causal_mask=True)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=pop, use_causal_mask=True)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=pop, use_causal_mask=True)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=pop, use_causal_mask=True)

        # 5. Design Prediction Head
        design_prediction_logits = self.output_modeling_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction, attn_scores

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # ---------------------------------------
    # Positional Encoding (from vaswani et al.)
    # ---------------------------------------

    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    # -------------------------------------------
    # Load weights
    # -------------------------------------------

    def load_target_weights(self, model_instance, trainable=False):
        """
        Updates the weights of the current instance with the weights of the provided model instance.
        Args:
            model_instance (SatelliteMLP): Instance of SatelliteMLP whose weights will be used.
        """
        for target_layer, source_layer in zip(self.layers, model_instance.layers):
            target_layer.set_weights(source_layer.get_weights())
        self.trainable = trainable




