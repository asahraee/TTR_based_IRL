import sys
import pandas as pd

from map_generator import *
from ttr_generator import *
from lidar_generator import *
from map_crop import *
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
        self._use_lidar_map = kwargs['use_lidar_map'] if 'use_lidar_map' in\
                kwargs else False
        self._use_true_map = kwargs['use_true_map'] if 'use_true_map' in\
                kwargs else True
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

        file_content = []
        labels = []
        lidar_rng = []
        header = ['x_r', 'y_r', 'theta_s', 'ttr']
        if self._read_lidar:
            header.append('lidar_rng')
        csv_path = os.path.join(self._log_dir,'csv/')
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)

        for i in range(db3.map_no+1):
            map_is_valid = False
            while not map_is_valid:
                print('trying a new map')
                map_i = map_generator.generate_map(db3.inflation)
                ttr_i = ttr_generator.generate_ttr_data(map_i,
                        db3.samples_no)
                map_is_valid = False if ttr_i is None else True

            pos_ttr_data = ttr_i.get_pos_and_ttr()
            walls = map_i.get_obstcles()
            # The output is a list of the form 
            #[[list of start pts][list of goal pts][list of TTRs]]
            #print('data: ', pos_ttr_data)
            if self._read_lidar:
                # call lidar module to get raw lidar data
                # lidar needs to be generated, for every state.
                lidar_generator.generate_lidar_data(walls, pos_ttr_data[0])
                lidar_rng_i, lidar_info_i = lidar_generator.get_raw_data()
                # lidar_rng_i is a list of lists of the form:
                # [[ranges for start position 1][ranges for start positio 2][...]]
                lidar_rng.extend(lidar_rng_i) # not used right now, but maybe later


            if self._use_lidar_map:
                '''to be done ...'''

            if self._use_true_map:
                # Call the map crop module to get the real local map
                l = map_cropper.crop_local(map_i.get_inflated_map(),
                        i, pos_ttr_data[0])
                labels.extend(l)
            starts = np.array(pos_ttr_data[0])
            goals = np.array(pos_ttr_data[1])
            relative = goals - starts
            x_r = relative[:, 0]
            y_r = relative[:, 1]
            theta_s = starts[:, 2]
            ttr = np.array(pos_ttr_data[2]).squeeze()
            print('xShape: ', x_r.shape, ' yShape: ', y_r.shape, ' thetaShape:', theta_s.shape, ' ttr_shape: ', ttr.shape) 
            stack = np.stack((x_r, y_r, theta_s, ttr), axis=1)
            print('stackShape: ', stack.shape)
            if self._read_lidar:
                file_content.extend(np.concatenate((stack, lidar_rng_i), axis=1))
            else:
                file_content.extend(stack)

            pd.DataFrame(file_content).to_csv(os.path.join(csv_path,
                'pos_ttr.csv'), header=header)
            # Saving the image labels in a csv file
            pd.DataFrame(labels).to_csv(os.path.join(csv_path,
                'image_labels.csv'), header=None)



    def dubind4D_data(self):
        '''
        To be done
        '''


def main():
    data_gen = DataGen(read_lidar=False, use_true_map=True)
    data_gen.dubins3D_data()

if __name__ == '__main__': main()

