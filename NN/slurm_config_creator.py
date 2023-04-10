import os
import sys
sys.path.append('/root/Desktop/git_repo/TTR_based_IRL/')

import params.dubins3D_params as db3
from train import *

config_dir ='/root/Desktop/data_and_log/slurm_configs'
if not os.path.exists(config_dir):
    os.mkdir(config_dir)
run_id = 1
transform = [True, False]
l_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
regularizations = ['L2', 'L1', 'na']
reg_lambdas = [0.1, 0.05, 0.01, 0.005, 0.001]

trainer = TrainerDubins3D()

for trans in transform:
    for l in l_rates:
        for reg in regularizations:
            if reg=='na':
                file_name = f'config_{run_id}.sh'
                config_path = os.path.join(config_dir, file_name)
                with open(config_path, 'w') as c:
                    c.write(f"#!/bin/bash\n")
                    c.write(f"#SBATCH --job-name=config_{run_id}\n")
                    c.write("#SBATCH --output=/home/aliarab/scratch/ttr4rl/logs/%x.log\n")
                    c.write("#SBATCH --time=3:00:00\n#SBATCH --gres=gpu:1\n")
                    c.write("#SBATCH --cpus-per-task=2\n#SBATCH --mem=12G\n")
                    c.write("#SBATCH --mail-user=asahraee@sfu.ca\n")
                    c.write("#SBATCH --mail-type=NONE\n#SBATCH --account=def-ester")

                    c.write("conda activate opt_dp\ncd /home/aliarab/src/ttr4rl/NN\n")
                    c.write(f"python train.py --use_default_params False --learning_rate {l} --transform {trans} --reg_type {reg} --reg_lambda 0 --config_id {run_id}")
                run_id += 1
            else:
                for reg_l in reg_lambdas:
                    file_name = f'config_{run_id}.sh'
                    config_path = os.path.join(config_dir, file_name)
                    with open(config_path, 'w') as c:
                        c.write(f"#!/bin/bash\n")
                        c.write(f"#SBATCH --job-name=config_{run_id}\n")
                        c.write("#SBATCH --output=/home/aliarab/scratch/ttr4rl/logs/%x.log\n")
                        c.write("#SBATCH --time=3:00:00\n#SBATCH --gres=gpu:1\n")
                        c.write("#SBATCH --cpus-per-task=2\n#SBATCH --mem=12G\n")
                        c.write("#SBATCH --mail-user=asahraee@sfu.ca\n")
                        c.write("#SBATCH --mail-type=NONE\n")
                        c.write("#SBATCH --account=def-ester\n")
                        c.write("conda activate opt_dp\n")
                        c.write("cd /home/aliarab/src/ttr4rl/NN\n")
                        c.write(f"python train.py --use_default_params False --learning_rate {l} --transform {trans} --reg_type {reg} --reg_lambda {reg_l} --config_id {run_id}")
                    run_id += 1
