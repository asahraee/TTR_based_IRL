import sys
import pandas as pd
import pickle

from map_generator import *
from ttr_generator import *
from lidar_generator import *
from map_crop import *
from obstacle_distance_generator import *
import os

sys.path.append('/root/Desktop/git_repo/TTR_based_IRL')
sys.path.append('/root/Desktop/git_repo/TTR_based_IRL/optimized_dp')

import params.dubins3D_params as db3

class DataGen:
    def __init__(self, **kwargs):
        '''
        To be done
        '''
        # kwargs = {log_dir: directory that all the data files will be saved, 
        #           read_lidar: if True uses lidar data,
        #           use_lidar_map: if True uses local image generated from lidar,
        #           use_true_map: if True crops and saves ground truth map}
        
        self._log_dir = kwargs['log_dir'] if 'log_dir' in kwargs else\
                db3.data_log_dir
        self._read_lidar = kwargs['read_lidar'] if 'read_lidar' in\
                kwargs else False
        self._save_lidar_map = kwargs['use_lidar_map'] if 'use_lidar_map' in\
                kwargs else False
        self._use_true_map = kwargs['use_true_map'] if 'use_true_map' in\
                kwargs else True
        self._save_obstcl_walls = kwargs['save_obstacle_walls'] if 'save_obstacle_walls' in\
                kwargs else False
        self._save_obstcl_distance = kwargs['save_obstacle_distance'] \
                if 'save_obstacle_distance' in kwargs else False

        self._save_map_np = kwargs['save_map_array'] if 'save_map_array' in\
                kwargs else False

        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

    def dubins3D_data(self, debug=False):
        
        g = Grid(np.array(db3.min_bounds), np.array(db3.max_bounds),
                db3.dims, np.array(db3.pts_each_dim), 
                np.array(db3.periodic_dims))
        map_generator = MapGen2D(obstcle_no=db3.obstcle_no,
                obstcle_edge_no=db3.obstcle_edge_no,
                obstcle_size=db3.obstcle_size, grid=g)
        ttr_generator = TTRGen(grid=g,
                computation_error=db3.computation_error,
                goal_radi=db3.goal_radi, dyn='DubinsCar3D',
                dyn_bounds=db3.dyn_bounds)
        map_cropper = MapCrop(dir=os.path.join(self._log_dir,'images/')
                ,rotate=True,size=db3.size_pix, loc2glob=db3.loc2glob,
                map_min=np.array(db3.min_bounds[0:2]),
                map_max=np.array(db3.max_bounds[0:2]))
        if self._read_lidar:
            lidar_generator = TurtleLidarGen(turtle_model='burger')
        if self._save_obstcl_distance:
            obst_dist_generator = ObstclDistGen()

        pos_content = []
        if self._read_lidar:
            lidar_content = []
        if self._save_obstcl_walls:
            wall_content = []
        if self._save_obstcl_distance:
            rel_dist_content = []
            close_point_content = []
        labels = []
        lidar_rng = []
        header_pos = ['x_s', 'y_s', 'theta_s'
                , 'x_g', 'y_g', 'theta_g'
                , 'x_r', 'y_r', 'obstacle_flag', 'ttr']
        #if self._read_lidar:
        #    header_lidar = ['start_position', 'lidar_range']

        csv_path = os.path.join(self._log_dir,'csv/')
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)

        if self._save_map_np:
            np_path = os.path.join(self._log_dir, 'np/')
            if not os.path.exists(np_path):
                os.mkdir(np_path)



        for i in range(db3.map_no):
            # Generating position and ttr data
            map_is_valid = False
            # creating map and ttr
            while not map_is_valid:
                print('trying a new map')
                map_i = map_generator.generate_map(db3.inflation)
                ttr_i = ttr_generator.generate_ttr_data(map_i,
                        db3.samples_no)
                map_is_valid = False if ttr_i is None else True
            ## saving for debug
            #with open(f'map{i}.obj', 'wb') as h1:
            #    pickle.dump(map_i, h1)
            #with open(f'ttr{i}.obj', 'wb') as h2:
            #    pickle.dump(ttr_i, h2)
            
            ## lodaing for debug
            #with open(f'map{i}.obj', 'rb') as h1:
            #    map_i = pickle.load(h1)
            #with open(f'ttr{i}.obj', 'rb') as h2:
            #    ttr_i = pickle.load(h2)

            
            pos_ttr_data = ttr_i.get_pos_and_ttr()
            # The output is a list of the form 
            #[[list of start pts][list of goal pts][list of TTRs]]
            #print('data: ', pos_ttr_data)
            walls = map_i.get_obstcles()
            # Saving obstacle walls
            #print('walls: ', walls)
            if self._save_obstcl_walls:
                for obs in walls:
                    wall_i = [i]
                    for x1, y1, x2, y2 in obs:
                        wall_i.extend([x1, y1, x2, y2])
                    #print('wall_i size: ', len(wall_i))
                    wall_content.append(wall_i.copy())
                #print('wall-content:\n', np.array(wall_content, dtype=float))
                pd.DataFrame(wall_content).to_csv(os.path.join(csv_path,
                'obstacle_walls.csv')) #, header=['global_ap_no', 'x1', 'y1', 'x2', 'y2'])
            # saving map numpy array
            if self._save_map_np:
                np.save(np_path+f'g{i}.npy', map_i.get_inflated_map())
            # saving relative distance from each start point to every obstacle
            if self._save_obstcl_distance:
                temp_out = obst_dist_generator.generate_obs_dist(walls,
                        pos_ttr_data[0])
                rel_dist_i = []
                close_point_i = []
                for start2obs in temp_out:
                    rel_dist_ij = []
                    close_point_ij = []
                    for x, y, d in start2obs:
                        rel_dist_ij.append(d)
                        close_point_ij.extend([x, y])
                    rel_dist_i.append(rel_dist_ij.copy())
                    close_point_i.append(close_point_ij.copy())
                rel_dist_content.extend(rel_dist_i)
                close_point_content.extend(close_point_i)
                pd.DataFrame(np.array(rel_dist_content)).to_csv(
                        os.path.join(csv_path, 'relative_distance.csv'))                        #, header=['d'])
                pd.DataFrame(np.array(close_point_content)).to_csv(
                        os.path.join(csv_path, 'obstacle_close_points.csv'))
                        #, header=['x1', 'y1', 'x2', 'y2', '...'])


            
            # Generating lidar data
            if self._read_lidar:
                # call lidar module to get raw lidar data
                # lidar needs to be generated, for every state.
                lidar_generator.generate_lidar_data(walls, pos_ttr_data[0])
                lidar_rng_i, lidar_info_i = lidar_generator.get_raw_data()
                # lidar_rng_i is a list of lists of the form:
                # [[ranges for start position 1][ranges for start positio 2][...]]
                lidar_rng.extend(lidar_rng_i.copy()) # not used right now, but maybe later


            if self._save_lidar_map:
                '''to be done ...'''

            if self._use_true_map:
                # Call the map crop module to get the real local map
                l = map_cropper.crop_local(map_i.get_inflated_map(),
                        i, pos_ttr_data[0])
                labels.extend(l)


            
            starts = np.array(pos_ttr_data[0])
            goals = np.array(pos_ttr_data[1])
            relative = goals - starts
            
            x_s = starts[:, 0]
            y_s = starts[:, 1]
            theta_s = starts[:, 2]
            
            x_g = goals[:, 0]
            y_g = goals[:, 1]
            theta_g = goals[:, 2]

            x_r = relative[:, 0]
            y_r = relative[:, 1]
            
            # obs_flag=true: path goes through obstacles, so no valid path is found
            obs_flag = np.array([t>100 for t in pos_ttr_data[2]]).squeeze()

            ttr = np.array(pos_ttr_data[2]).squeeze()
            
            # Saving position and ttr content in csv format
            stack = np.stack((x_s, y_s, theta_s
                            , x_g, y_g, theta_g
                            , x_r, y_r, obs_flag, ttr), axis=1)
            pos_content.extend(stack)
            pd.DataFrame(pos_content).to_csv(os.path.join(csv_path,
                'pos_ttr.csv'), header=header_pos)

            # Saving lidar ranges and lidar info in csv format
            if self._read_lidar:
                lidar_content.extend(np.array(lidar_rng_i))
                pd.DataFrame(lidar_content).to_csv(os.path.join(csv_path,
                'lidar_range.csv'))
                if i == 0:
                    with open (os.path.join(self._log_dir,'lidar_info.pickle'),
                            'wb') as h:
                        pickle.dump(lidar_info_i, h, pickle.HIGHEST_PROTOCOL)
 
            # Saving the image labels in a csv file
            pd.DataFrame(labels).to_csv(os.path.join(csv_path,
                'image_labels.csv'), header=None)



    def dubind4D_data(self):
        '''
        To be done
        '''


def main():
    data_gen = DataGen(read_lidar=True, use_true_map=True,
            save_obstacle_walls=True, save_obstacle_distance=True,
            save_map_array=True)
    data_gen.dubins3D_data()

if __name__ == '__main__': main()

