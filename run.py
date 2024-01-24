import multiprocessing as mp
import os
import config


from task.Alg_Task import Alg_Task

from trainer.GaMetaTrainer import GaMetaTrainer
from trainer.TrussMetaTrainer import TrussMetaTrainer


def ga_trainer():
    num_task_variations = 100
    new_tasks = False
    checkpoint_path_actor = None  # os.path.join(config.models_dir, 'universal_crossover_actor_0')  # None
    checkpoint_path_critic = None  # os.path.join(config.models_dir, 'universal_crossover_critic_0')  # None
    task_sample_size = 8
    task_epochs = 400
    trainer = GaMetaTrainer(
        num_task_variations=num_task_variations,
        new_tasks=new_tasks,
        checkpoint_path_actor=checkpoint_path_actor,
        checkpoint_path_critic=checkpoint_path_critic,
        task_sample_size=task_sample_size,
        task_epochs=task_epochs
    )
    trainer.train()


def truss_trainer():
    checkpoint_path_actor = None  # os.path.join(config.models_dir, 'universal_crossover_actor_0')  # None
    checkpoint_path_critic = None  # os.path.join(config.models_dir, 'universal_crossover_critic_0')  # None
    task_sample_size = 12
    task_epochs = 400
    trainer = TrussMetaTrainer(
        checkpoint_path_actor=checkpoint_path_actor,
        checkpoint_path_critic=checkpoint_path_critic,
        task_sample_size=task_sample_size,
        task_epochs=task_epochs,
        max_tasks=24
    )
    trainer.train()







if __name__ == '__main__':
    # set multiprocessing start method to spawn
    mp.set_start_method('spawn')
    truss_trainer()





