import multiprocessing as mp
import os
import config


from trainer.GaMetaTrainer import GaMetaTrainer



if __name__ == '__main__':
    # set multiprocessing start method to
    mp.set_start_method('spawn')

    num_task_variations = 100
    new_tasks = False
    checkpoint_path_actor = None  # os.path.join(config.models_dir, 'universal_crossover_actor_20')  # None
    checkpoint_path_critic = None  # os.path.join(config.models_dir, 'universal_crossover_critic_20')  # None
    task_sample_size = 4
    task_epochs = 100
    trainer = GaMetaTrainer(
        num_task_variations=num_task_variations,
        new_tasks=new_tasks,
        checkpoint_path_actor=checkpoint_path_actor,
        checkpoint_path_critic=checkpoint_path_critic,
        task_sample_size=task_sample_size,
        task_epochs=task_epochs
    )
    trainer.train()


