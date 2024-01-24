import tensorflow as tf
import config
import os



# ---------------------------------------
# Universal Crossover
# ---------------------------------------


from model.UniversalCrossover import UniversalCrossover
from model.UniversalCrossoverCritic import UniversalCrossoverCritic

def get_universal_crossover(checkpoint_path_actor=None, checkpoint_path_critic=None):
    pop_size = 10
    design_len = 30
    pop_input_len = design_len * pop_size + pop_size

    actor_model = UniversalCrossover()
    decisions = tf.zeros((1, design_len))
    parents = tf.zeros((1, pop_input_len))
    actor_model([decisions, parents])

    critic_model = UniversalCrossoverCritic()
    decisions = tf.zeros((1, design_len + 1))
    parents = tf.zeros((1, pop_input_len))
    critic_model([decisions, parents])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model


from model.UniversalCrossover import FastUniversalCrossover
from model.UniversalCrossoverCritic import FastUniversalCrossoverCritic


# ---------------------------------------
# Fast Universal Crossover
# ---------------------------------------

def get_fast_universal_crossover(checkpoint_path_actor=None, checkpoint_path_critic=None):
    pop_size = 32
    design_len = 60
    pop_input_len = design_len * pop_size + pop_size

    actor_model = FastUniversalCrossover()
    decisions = tf.zeros((1, design_len))
    parents = tf.zeros((1, pop_input_len))
    pop = tf.zeros((1, pop_input_len))
    actor_model([decisions, parents, pop])

    critic_model = FastUniversalCrossoverCritic()
    decisions = tf.zeros((1, design_len + 1))
    parents = tf.zeros((1, pop_input_len))
    pop = tf.zeros((1, pop_input_len))
    critic_model([decisions, parents, pop])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    # actor_model.summary()

    return actor_model, critic_model

# ---------------------------------------
# Large Universal Crossover
# ---------------------------------------


from model.LargeUniversalCrossover import LargeUniversalCrossover, LargeUniversalCrossoverCritic

def get_large_universal_crossover(checkpoint_path_actor=None, checkpoint_path_critic=None):
    pop_size = 20
    design_len = 60
    pop_input_len = design_len * pop_size + pop_size

    actor_model = LargeUniversalCrossover()
    decisions = tf.zeros((1, design_len))
    parents = tf.zeros((1, pop_input_len))
    pop = tf.zeros((1, pop_input_len))
    actor_model([decisions, parents, pop])

    critic_model = LargeUniversalCrossoverCritic()
    decisions = tf.zeros((1, design_len + 1))
    parents = tf.zeros((1, pop_input_len))
    pop = tf.zeros((1, pop_input_len))
    critic_model([decisions, parents, pop])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    # actor_model.summary()

    return actor_model, critic_model


# ---------------------------------------
# Universal Solver
# ---------------------------------------

from model.UniversalSolver import UniversalSolver, UniversalSolverCritic

def get_universal_solver(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = 60

    actor_model = UniversalSolver()
    decisions = tf.zeros((1, design_len))
    actor_model(decisions)

    critic_model = UniversalSolverCritic()
    decisions = tf.zeros((1, design_len + 1))
    critic_model(decisions)

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model





# ---------------------------------------
# Universal Solver MO
# ---------------------------------------

from model.UniversalSolverMO import UniversalSolverMO, UniversalSolverCriticMO

def get_universal_solver_mo(checkpoint_path_actor=None, checkpoint_path_critic=None):
    design_len = 30

    actor_model = UniversalSolverMO()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, 1))
    actor_model([decisions, weights])

    critic_model = UniversalSolverCriticMO()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, 1))
    critic_model([decisions, weights])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model






