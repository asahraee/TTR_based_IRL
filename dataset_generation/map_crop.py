import sys
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/root/Desktop/git_repo/TTR_based_IRL')

import params.dubins3D_params as db3


class MapCrop:
    def __init__(self, **kwargs):
        # kwargs = {rotate: true or false, size: size of final local map image in pixels, dir: the directory that local and global images will be saved, loc2glob: the ratio of the local map size to the global, map_min: the lowest bound of the map in [x,y] form, map_max: the upperbound in [x, y] form and not in pixels}
        self._log_dir = kwargs['dir'] if 'dir' in kwargs else\
                os.path.join(db3.data_log_dir, '/images')
        
        self._rotate = kwargs['rotate'] if 'rotate' in kwargs else True
        self._size = kwargs['size'] if 'size' in kwargs else [28, 28]
        self._loc2glob = kwargs['loc2glob'] if 'loc2glob' in kwargs else 0.5
        self._map_min = kwargs['map_min'] if 'map_min' in kwargs else None
        self._map_max = kwargs['map_max'] if 'map_max' in kwargs else None

        # if map's bounds are not given, raise error
        if self._map_min.any()==None or self._map_max.any() == None:
            raise TypeError('Map bounds are not passed in the form of map_min: [x_min, y_min] and map_max: [x_max, y_max]')
        
        # if the log directory doesn't exist create it and create the global and local folders in it
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)
            os.mkdir(os.path.join(self._log_dir, 'global'))
            os.mkdir(os.path.join(self._log_dir, 'local'))

    def _save_glob_map(self, obstcl_map, map_no):
        g_image_path = os.path.join(self._log_dir, f'global/g_{map_no}.png')
        # if the global map doesn't exist in the dir/global/
            #create the figure, save it with the proper calculated pixel size
        if not os.path.exists(g_image_path):
            Image.fromarray(obstcl_map.astype(np.uint8)*np.uint(255)).resize(
                (self._size/self._loc2glob).astype(np.int32)).save(
                            g_image_path)
        return g_image_path


    def _rotate_glob_map(self, g_image_path, angle, center_pix):
        # Angle is passed in radian and converted to degrees
        angle = math.degrees(angle)
        # rotate the global map and rewrite the temp_rotated every time called
        temp_rotated = Image.open(g_image_path).rotate(angle, center=center_pix)
        return temp_rotated
    
    def _find_pixel(self, x):
        # check
        # needs to return the center pixel as tuple
        min = np.asarray(self._map_min)
        max = np.asarray(self._map_max)
        z = (x - min)*np.asarray([self._size[1], self._size[0]])/(max - min)
        pix = (round(z[1]), round(z[0]))
        return pix
        

    def _get_corners(self, center_pix):
        # center_pix is the pixel closest to the center state
        # calculate left top right and bottom for image crop
        right_pixs, low_pixs = (self._size - 1)//2
        left_pixs, up_pixs = self._size - [right_pixs, low_pixs] - 1
        #print('left_no: ', left_pixs, 'right_no: ', right_pixs, 'up_no: ', up_pixs, 'low_no: ', low_pixs)
        left = center_pix[1] - left_pixs
        upper = center_pix[0] - up_pixs
        right = center_pix[1] + right_pixs
        lower = center_pix[0] + low_pixs
        return left, upper, right, lower

    def crop_local(self, obstcl_map, map_no, states, pos_indx=[0, 1, 2]):
        # states is a list of numpy arrays
        # map_no is used as part of the name for saving the global map as image
        centers = [states[i][pos_indx[0:2]] for i in range(len(states))]
        labels = []
        #print('states dimension: ', np.shape(states))
        #print('centers dimension: ', np.shape(centers))
        # angles show the amount of rotattion needed for the global map, not the heading anymore
        angles = [states[i][pos_indx[2]] - (math.pi/2) for i in range(
            len(states))]
        #print('angles dimension: ', np.shape(angles))
        total_no = len(centers)
        # global map image path
        g_image_path = self._save_glob_map(obstcl_map, map_no)
        print('Global map {map_no} generated')
        for i, (center, angle) in enumerate(zip(centers, angles)):
            # print center to check, it must be numpy array of size (2,)
            #print('center: ', center)
            # the pixel closest to the current state which is used as the center of local map
            center_pix = self._find_pixel(center)
            # find the corners of the local map based on size
            corners = self._get_corners(center_pix)
            # roate the global map based on heading if needed
            if self._rotate == True:
                temp_rotated = self._rotate_glob_map(g_image_path, angle, center_pix)
            else:
                temp_rotated = Image.open(g_image_path)
            # crop the global image and save it as a new local
            label = f'g{map_no}_loc{i}.png'
            temp_log_dir = os.path.join(self._log_dir, 'local', label)
            fin_im = temp_rotated.crop(corners)
            fin_im.save(temp_log_dir)
            labels.append(label)
            print(f'local map {i}/{total_no-1} generated')
        return labels

def test():
    array = 256*np.random.randint(2, size=(256,256)).astype(np.uint8)
    im = Image.fromarray(array).convert('L').save(os.path.join(os.getcwd(),'testfig.png'), format='PNG')

    im2 = Image.open('./testfig.png').rotate(45).save('./test2.png')


if __name__=='__main__': test()
