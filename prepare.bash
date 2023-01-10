#!/usr/env/bin bash
# Need to run this script if creating container from dockerfile or sevd image which hasn't buil robot packages

# Making and sourcing all the robot packages used
cd /root/Desktop/poject/robots_ws
catkin_make -DPYTHON_EXECUTABLE=~/miniconda/envs/opt_dp/bin/python3.8
source /root/Desktop/project/robts_ws/devel/setup.bash

# Making and sourcing the turtlebot lidar generation package
cd /root/Desktop/project/dataset_generation/lidar_gen_ws
catkin_make -DPYTHON_EXECUTABLE=~/miniconda/envs/opt_dp/bin/python3.8
source /root/Desktop/project/dataset_generation/lidar_gen_ws/devel/setup.bash
