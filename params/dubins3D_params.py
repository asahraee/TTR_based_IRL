import math
import numpy as np

# All parameters must be passed as numpy array and not as a list
# Grid Parameters
min_bounds = np.array([-10, -10, -math.pi])
max_bounds = np.array([10, 10, math.pi])
dims = 3
pts_each_dim = np.array([100, 100, 60])
periodic_dims = np.array([2])

# Obstacle Map Parameters
obstcle_no = 10
obstcle_edge_no = 4
obstcle_size = 2
inflation = 0.1

# Dynamics Parameters
w_range = np.array([-1, 1]) 
v = 1.5
dstb_max = np.array([0.5, 0.5, 0.5])
dyn_bounds = dict(w_range=w_range, v=v, dstb_max=dstb_max)

# TTR Parameters
computation_error = 0.001     # Convergence error
goal_radi = 0.1            # Goal radius
samples_no = 200         # Number of generated trainig data for each map-goal pair

# Lidar Parameters

# Ground truth local map paramters
size_pix = np.array([200, 200])   # Size of the local maps in pixels
loc2glob = 0.5          # The ratio between local map and global map size in meters
                        # assuming it's the same for x and y coordinates
# General Parameters
map_no = 10000               # Number of random maps


# Parameters used for training
run_id = 0
data_log_dir = '/root/Desktop/data_and_log/data_7'
trainer_log_dir = '/root/Desktop/data_and_log/train_log'
saved_model = '/root/Desktop/data_and_log/saved_model'
learning_rate = 0.0001
batch_size = 512
num_epochs = 200
validation_ratio = 0.01
test_ratio = 0
regularization = 'L2' # L1, L2, na
regularization_lambda = 0.0005
data_trans = True # whether to have data transform or not
