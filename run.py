import multiprocessing as mp
import os
import config


from task.Alg_Task import Alg_Task

from trainer.GaMetaTrainer import GaMetaTrainer



if __name__ == '__main__':
    # set multiprocessing start method to
    mp.set_start_method('spawn')


    checkpoint_path_actor = None
    checkpoint_path_critic = None
    model_num = None
    if model_num is not None:
        checkpoint_path_actor = os.path.join(config.models_dir, 'universal_crossover_actor_' + str(model_num))
        checkpoint_path_critic = os.path.join(config.models_dir, 'universal_crossover_critic_' + str(model_num))


    num_task_variations = 100
    new_tasks = False
    task_sample_size = 1
    task_epochs = 10
    trainer = GaMetaTrainer(
        num_task_variations=num_task_variations,
        new_tasks=new_tasks,
        checkpoint_path_actor=checkpoint_path_actor,
        checkpoint_path_critic=checkpoint_path_critic,
        task_sample_size=task_sample_size,
        task_epochs=task_epochs
    )
    trainer.train()






    # test_task = trainer.val_tasks[0]
    # task_runner = Alg_Task(
    #     run_num=0,
    #     barrier=None,
    #     problem=test_task,
    #     limit=1000,
    #     actor_load_path=None,
    #     critic_load_path=None,
    #     c_type='uniform',
    # )
    # results = task_runner.run()
    # print(results)





