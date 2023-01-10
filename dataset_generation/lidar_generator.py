#! /usr/bin/env python3.8
import numpy as np
import math
import sys
import os
import roslaunch
import rospy
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetModelState

# Local map
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from tf.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation

# For debugging
import pickle
from map_generator import *
from ttr_generator import *
sys.path.append('/root/Desktop/project')
sys.path.append('/root/Desktop/project/optimized_dp')


#################################################################################
class TurtleLidarGen:
    def __init__(self, **kwargs):
        '''
        to be done
        '''
        rospy.init_node('turtle_lidar_reader', anonymous=True)
        self._package_dir = '/root/Desktop/project/dataset_generation/lidar_gen_ws/src/turtle_lidar_reading/'
        if not os.path.exists(self._package_dir): os.mkdir(self._package_dir)
        self._launch_file_path = self._package_dir + 'launch/turtle_lidar_reading.launch'
        #if not os.path.exists(self._launch_file_path): os.mkdir(self._launch_file_path)
        self._model_dir = self._package_dir + 'models/'
        if not os.path.exists(self._model_dir): 
            os.makedirs(self._model_dir+'random_boxes/', True, True)
            self._model_config()
        self._map_sdf_path = self._model_dir + 'random_boxes/random_boxes.sdf'
        print('model_dir = ', self._model_dir)
        print('map_sdf_path = ', self._map_sdf_path)
        self._turtle_model = kwargs['turtle_model'] if 'turtle_model' in kwargs else 'burger'
        
        self._laser_scan = rospy.Subscriber('/scan', LaserScan, self._scan_cb, queue_size=1)
        # The value that tells us what to replace infinity with in laser scan range post processing
        self._laser_replace_inf = kwargs['inf'] if 'inf' in kwargs else 0
        self._set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._rate = rospy.Rate(0.5)
        self._lidar_rng_i = []
        self._lidar_data = []

    def _get_arc(self, p1_y, p1_x, p0_y, p0_x):
        if p1_x - p0_x == 0:
            if p1_y - p0_y == 0:
                return -1
            else:
                return np.pi/2
        tan = (p1_y - p0_y)/(p1_x - p0_x)
        arc = np.arctan(tan)
        return arc

    def _model_config(self):
        
        with open(self._model_dir+'random_boxes/model.config', 'w') as out:
            out.write("<?xml version='1'?>\n")
            out.write("<model>\n")
            out.write(" "*2 + "<name>Random Boxes</name>\n")
            out.write(" "*2 + "<version>1.0</version>\n")
            out.write(" "*2 + "<sdf version='1.5'>random_boxes.sdf</sdf>\n")
            out.write(" "*2 + "<author>\n")
            out.write(" "*4 + "<name>A Sahraee</name>\n")
            out.write(" "*4 + "<email>asahraee@sfu.ca</email>\n")
            out.write(" "*2 + "</author>\n")
            out.write(" "*2 + "<description>\n")
            out.write(" "*4 + "randomly scattered boxes\n")
            out.write(" "*2 + "</description>\n")
            out.write("</model>")

        
    def _map_sdf(self, walls):
        '''
        to be done
        '''
        obs_no = 1
        edge_no = 1
        with open(self._map_sdf_path, 'w') as out:
            out.write('<?xml version=\'1.0\'?>\n<sdf version=\'1.6\'>\n')
            out.write(' '*2 + '<model name=\'walls\'>\n')
            out.write(' '*4 + '<pose frame=\'\'>0 0 0 0 -0 0</pose>\n')
            out.write(' '*4 + '<static>1</static>\n')

            for obs_vertx in walls:
                out.write(' '*4 + f'<link name=\'obstcle_{obs_no}\'>\n')

                for [x1, y1, x2, y2] in obs_vertx:
                    len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    theta = self._get_arc(y2, x2, y1, x1)

                    len_str = '%.2f' % len
                    cx_str = '%.2f' % cx
                    cy_str = '%.2f' % cy
                    theta_str = '%.2f' % theta
                    
                    out.write(' '*6 + f'<collision name=\'edge_{edge_no}_Collision\'>\n')
                    out.write(' '*8 + '<geometry>\n' + ' '*10 + '<box>\n')
                    out.write(' '*12 + f'<size>{len_str} 0.1 2</size>\n')
                    out.write(' '*10 + '</box>\n' + ' '*8 + '</geometry>\n')
                    out.write(' '*8 + f'<pose frame=\'\'>{cx_str} {cy_str} 1 0 -0 {theta_str}</pose>\n')
                    out.write(' '*6 + '</collision>\n')
                    
                    out.write(' '*6 + f'<visual name=\'edge_{edge_no}_Visual\'>\n')
                    out.write(' '*8 + f'<pose frame=\'\'>{cx_str} {cy_str} 1 0 -0 {theta_str}</pose>\n')
                    out.write(' '*8 + '<geometry>\n' + ' '*10 + '<box>\n')
                    out.write(' '*12 + f'<size>{len_str} 0.1 2</size>\n')
                    out.write(' '*10 + '</box>\n' + ' '*8 + '</geometry>\n')
                    out.write(' '*8 + '<material>\n' + ' '*10 + '<script>\n')
                    out.write(' '*12 + '<uri>file://media/materials/scripts/gazebo.material</uri>\n')
                    out.write(' '*12 + '<name>Gazebo/Green</name>\n')
                    out.write(' '*10 + '</script>\n')
                    out.write(' '*10 + '<ambient>1 1 1 1</ambient>\n' + ' '*8 + '</material>\n')
                    out.write(' '*8 + '<meta>\n' + ' '*10 + '<layer>0</layer>\n' + ' '*8 + '</meta>\n')
                    out.write(' '*6 + '</visual>\n')
                    
                    edge_no = edge_no + 1
                    
                    #out.write(' '*6 + f'<pose frame=\'\'>{cx_str} {cy_str} 0 0 -0 {theta_str}</pose>\n')
                out.write(' '*4 + '</link>\n')
                
                obs_no = obs_no + 1

            out.write(' '*2 + '</model>\n' + '</sdf>\n')
            out.close()

    def _launch_file(self, state_0):
        
        state_x_str = '%.2f' % state_0[0]
        state_y_str = '%.2f' % state_0[1]
        state_yaw_str = '%.2f' % state_0[2]
        
        # Create Launch File
        with open(self._launch_file_path, 'w') as out:
            out.write('<?xml version=\"1.0\"?>\n<launch>\n')
            out.write(' '*2 + '<include file=\"$(find turtlebot3_gazebo)/launch/turtlebot3_empty_world.launch\">\n')
            out.write(' '*4 + f'<arg name=\"world\" value=\"{self._package_dir}worlds/random_python.world\"/>\n')
            out.write(' '*4 + f'<arg name=\"x_pos\" value=\"{state_x_str}\"/>\n')
            out.write(' '*4 + f'<arg name=\"y_pos\" value=\"{state_y_str}\"/>\n')
            out.write(' '*4 + f'<arg name=\"yaw\" value=\"{state_yaw_str}\"/>\n')
            out.write(' '*2 + '</include>\n</launch>\n')
            out.close()
    
    def _scan_cb(self, msg):
        self._lidar_rng_i_raw = msg.ranges
        self._lidar_rng_i = self.process(self._replace_inf)
        self._lidar_info = {'angle_min':msg.angle_min, 'angle_max':msg.angle_max,\
                'angle_increment':msg.angle_increment,'rng_min':msg.range_min,\
                'rng_max':msg.range_max}

    def _process(self, val):
        # This function replaces the infinity values in the lidar range
        # with the passed value as input
        return [rng if not math.isinf(rng) else val\
                for rng in self._lidar_rng_i_raw]


    def get_raw_data(self, lidar_indx=None):
        '''
        to be done ...
        '''
        if lidar_indx==None:
            return self._lidar_data , self._lidar_info
        else:
            return self._lidar_data[lidar_indx] , self._lidar_info

    def local_map(self, scale, path,format='png',lidar_indx=None):
        '''
        to be done
        '''
        # scale shows how many grid points are used in 1 meter of range -> grid number/meter
        max_rng = scale*self._lidar_info['rng_max']
        thetas = np.arange(self._lidar_info['angle_min'],self._lidar_info['angle_max'],\
                self._lidar_info['angle_increment'])
        if lidar_indx==None:
            i = 0
            for lidar_rng_i in self._lidar_data:
                rngs = scale * np.array(lidar_rng_i)
                fig = plt.figure()
                ax = fig.add_subplot(projection='polar')
                c = ax.scatter(thetas, rngs, s=1, c='k', cmap='gray', alpha=1)
                plt.show()
                if format==array:
                    TypeError('Not written yet')
                elif format=='png':
                    fig.savefig(f'{path}/local_map_{i}.png')
                else:
                    TypeError('format not supported, choose array or png')
                i = i + 1
                plt.close(fig)
                        
        else:
            rngs = scale * np.array(self._lidar_rng_i[lidar_indx])
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')
            c = ax.scatter(thetas, rngs, s=1, c='k', cmap='gray', alpha=1)
            plt.show()
            fig.savefig(f'{path}/local_map_{lidar_indx}.png')
            i = i + 1
            plt.close(fig)

    def _launch_start(self):
       # # method 1: using subprocess
       # """
       #     Does work as well from service/topic callbacks using launch files
       # """
       # package = 'YOUR_PACKAGE'
       # launch_file = 'YOUR_LAUNCHFILE.launch'
       # command = "roslaunch  {0} {1}".format(package, launch_file)
       # p = subprocess.Popen(command, shell=True)
       # state = p.poll()
       # if state is None:
       #     rospy.loginfo("process is running fine")
       # elif state < 0:
       #     rospy.loginfo("Process terminated with error")
       # elif state > 0:
       # rospy.loginfo("Process terminated without error")
        
        # method 2: using roslaunch.parent (they say it's unstable but jun used it)
        #rospy.init_node('turtle_lidar_reader', anonymous=True)
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [self._launch_file_path])
        self.launch.start()

    def generate_lidar_data(self, walls, states, pos_ind=[0, 1, 2]):

        # turn map infrmation to gazebo sdf model (for each map only once)
        self._map_sdf(walls)
        
        # let gazebo know where the obstacle model file is
        os.environ['GAZEBO_MODEL_PATH'] = self._model_dir
        print('gazebo model path: ', os.environ['GAZEBO_MODEL_PATH'])
        os.environ['TURTLEBOT3_MODEL'] = self._turtle_model
        print('turtle model: ', os.environ['TURTLEBOT3_MODEL'])

        # generate launch file for launching turtlebot in the specified world
        # the world file is already in place for reading the sdf model
        self._launch_file([states[0][pos_ind[0]], states[0][pos_ind[1]], states[0][pos_ind[2]]])
        
        # Launch the launch file which was created (for each map only once)
        self._launch_start()

        # wait for the first laser scan message
        rospy.wait_for_message('/scan', LaserScan)
        iter_1 = True
        # call the service for spawning the robot for all the states beginning with first second state
        rospy.wait_for_service('/gazebo/set_model_state')
        state_msg = ModelState()
        state_msg.model_name= 'turtlebot3_' + self._turtle_model
        for state in states:  # need to change returned dataset in ttr code to match with this one better??????????
            if iter_1:
                iter_1 = False
            else:
                #q = [0, 0, 0, 0]
                #q = quaternion_from_euler(0, 0, state[pos_ind[2]])
                rot = Rotation.from_euler('xyz', [0, 0, state[pos_ind[2]]], degrees=False)
                q = rot.as_quat()
                state_msg.pose.position.x = state[pos_ind[0]]
                state_msg.pose.position.y = state[pos_ind[1]]
                state_msg.pose.position.z = 0
                state_msg.pose.orientation.x = q[0]
                state_msg.pose.orientation.y = q[1]
                state_msg.pose.orientation.z = q[2]
                state_msg.pose.orientation.w = q[3]
            
                # call set_model_state service
                try:
                    print('trying to set state')
                    print(os.environ["TURTLEBOT3_MODEL"])
                    rsp = self._set_state(state_msg)
                    print(f"state = {state} set")
                except rospy.ServiceException as e:
                    print("Set Model State service call failed: %s" % e)

            # wait for laser scan message
            rospy.wait_for_message('/scan', LaserScan)
            self._rate.sleep()
            print('sleeping')
            
            # process scan message and replace infinity with value
            self._process(val=0)

            # save this iterations reading
            self._lidar_data.append(self._lidar_rng_i)

        self.launch.shutdown()
            

def main():
    '''to be done'''
    g = Grid(np.array([-10, -10, -(math.pi)]), np.array([10, 10, math.pi]), 3, np.array([100, 100, 60]), np.array([2]))
    map_generator = MapGen2D(obstacle_no=10, obstacle_edge_no=4, grid=g)
    ttr_generator = TTRGen(grid=g)

    ### generating map and ttr
    map_obj = map_generator.generate_map(0.1)
    ttr_obj = ttr_generator.generate_ttr_data(map_obj, 3)
    #########
    ### saving map and ttr objects
    with open('map_file', 'wb') as map_file_handl:
        pickle.dump(map_obj, map_file_handl)
    print('Map Saved')

    with open('ttr_file', 'wb') as ttr_file_handl:
        pickle.dump(ttr_obj, ttr_file_handl)
    print('TTR Saved')
    ##########
    #### Loading map and ttrt objects
    #with open('map_file', 'rb') as map_file_handl:
    #    map_obj = pickle.load(map_file_handl)
    #print('Map Loaded')
    #with open('ttr_file', 'rb') as ttr_file_handl:
    #    ttr_obj = pickle.load(ttr_file_handl)
    #print('TTR Loaded')
    ##########
    pos_ttr = ttr_obj.get_pos_and_ttr()
    states = pos_ttr[0]
    walls = map_obj.get_obstcles()
    lidar_generator = TurtleLidarGen()
    lidar_generator.generate_lidar_data(walls, states)
    print(lidar_generator.get_raw_data())
if __name__=='__main__': main()

