import os
import pickle
from datetime import datetime
import platform
import json

# CPU vs GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Set to use legacy keras
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
# os.environ["KERAS_BACKEND"] = "tensorflow"




#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'meta-learning')
results_dir = os.path.join(root_dir, 'results')
analysis_dir = os.path.join(root_dir, 'analysis')
models_dir = os.path.join(root_dir, 'model', 'store')
results_save_dir = os.path.join(results_dir, 'RUN')





#######################
# --> Bit Decoder <-- #
#######################
bd_num_weight_vecs = 4
bd_embed_dim = 32

bd_actor_continuous_weights = True
bd_actor_learn_weights = False
bd_actor_dense = 512
bd_actor_heads = 32

bd_critic_learn_weights = False
bd_critic_dense = 512
bd_critic_heads = 16
bd_critic_decoders = 1






