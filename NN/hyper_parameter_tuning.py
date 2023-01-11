import os
from train import *

config_dir ='/root/Desktop/data_and_log/NN_grid_config_files'
if not os.path.exists(config_dir):
    os.mkdir(config_dir)
run_id = 1
l_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
regularizations = ['L2', 'L1', 'na']
reg_lambdas = [0.1, 0.05, 0.01, 0.005, 0.001]

trainer = TrainerDubins3D()

for l in l_rates:
    for reg in regularizations:
        if reg=='na':
            file_name = f'hyper_params_{run_id}.py'
            config_path = os.path.join(config_dir, file_name)
            with open(config_path, 'w') as conf:
                conf.write(f"learning_rate = {l}")
                conf.write("\n")
                conf.write(f"regularization = {reg}")
                conf.write("\n")
                conf.write(f"regularization_lambda = {reg_l}")
            trainer.train_with_true_local_map(run_id=run_id,
                    learning_rate=l, reg=reg, reg_lambda=0)
            run_id += 1
        else:
            for reg_l in reg_lambdas:
                file_name = f'hyper_params_{run_id}.py'
                config_path = os.path.join(config_dir, file_name)
                with open(config_path, 'w') as conf:
                    conf.write(f"learning_rate = {l}")
                    conf.write("\n")
                    conf.write(f"regularization = {reg}")
                    conf.write("\n")
                    conf.write(f"regularization_lambda = {reg_l}")
                trainer.train_with_true_local_map(run_id=run_id,
                        learning_rate=l, reg=reg, reg_lambda=reg_l)
                run_id += 1

