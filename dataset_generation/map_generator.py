import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

sys.path.append('/root/Desktop/project/')

from utils.graham_scan import graham_scan
from optimized_dp.Grid.GridProcessing import *

# This class is used to generate one random map at a time with sufficient information (including goal and obstacle position)
class MapGen2D:
    def __init__(self, **kwargs):
        # Number of obstacles in the map
        self._obstcle_no = kwargs['obstcle_no'] if 'obstcle_no' in kwargs else 5
        # Number of edges of each obstacle
        self._edge_no = kwargs['obstcle_edge_no'] if 'obstcle_edge_no' in kwargs else 4
        # 2 * distance form centre of the obstacles in each coordinate [x,y]
        self._size = kwargs['obstcle_size'] if 'obstcle_size' in kwargs else 2
        # Lower bound for the center of obstacle of the map's grid for all dimensions
        self._min_obs_cntr = np.array((kwargs['grid'].min[0:2] + self._size/2) if 'grid' in kwargs else [-8.0, -8.0])
        # Upper bound for the center of obstacle of the map's grid for all dimensions
        self._max_obs_cntr = np.array((kwargs['grid'].max[0:2] - self._size/2) if 'grid' in kwargs else [8.0, 8.0])
        # Number of grid points in each dimension
        self._grid = kwargs['grid'] if 'grid' in kwargs else None
        if self._grid == None: raise TypeError('grid not passed to class')
        
    def _add_polygon(self, x, y):
        # This fnction receives 2 np.arrays of random coordinates and returns 2 np.arrays
        # walls represent the edges of the obstacles and are given by [x1 y1 x2 y2] of the corner point
        # data is a 2D map (only x and y) has a value of 100 for every point outside the obstacle,-100 otherwise
        data = -np.ones(self._grid.pts_each_dim[0:2]) * 1000
        walls = []
        
        # Sorting the corners in a clockwise order
        order = graham_scan(x, y, self._edge_no)
        print('order= ', order)
        # Creating an array of the edges & indicating every grid point inside an obstacle
        for i in range(len(order)):
            nexti = 0 if i == len(order)-1 else i+1
            ax = tuple(range(2,self._grid.dims))
            print('ax= ', ax)
            data = np.maximum(data,  (np.squeeze(self._grid.vs[1], axis = ax)-y[order[i]])*(np.squeeze(self._grid.vs[0], axis = ax)-x[order[nexti]]) \
                    -(np.squeeze(self._grid.vs[1], axis = ax)-y[order[nexti]])*(np.squeeze(self._grid.vs[0], axis = ax)-x[order[i]]) )
            walls.append([x[order[i]], y[order[i]], x[order[nexti]], y[order[nexti]]])
        
        # The value of the cross product is a representaive of the distance from obstacle
        # It is not imporatnt for TTR calculation or obstacle indicator
        new = np.where(data<0, 1, 0)
        data = new
        print('data shape = ', new.shape, '1s in data    ', np.count_nonzero(new))
        return [data, walls]
    
    def _extend_obstcle_geo(self, inflation):
        # extend obstacles approximately using geometric methods
        # grid prcision in each dimension of the map (distance between grid points
        grid_precisn = np.divide((self._grid.max[0:2] - self._grid.min[0:2]), self._grid.pts_each_dim[0:2])
        # Number of grid points which are inflated in each dimension : neigborhood
        nhood = [int(inflation/ grid_precisn[0]) +1, int(inflation/ grid_precisn[1])+1]
        # Index of grid points which are obstacles
        obs_indx = np.asarray(self._obstcle_indicator==1).nonzero()
        inf_map = self._obstcle_indicator + 0
        for i, j in zip(obs_indx[0], obs_indx[1]):
            inf_map[i-nhood[0]:i+nhood[0],j-nhood[1]:j+nhood[1]] =\
                inf_map[i-nhood[0]:i+nhood[0],j-nhood[1]:j+nhood[1]]+ 1
                #np.ones(inf_map[i-nhood[0]:i+nhood[0],j-nhood[1]:j+nhood[1]].shape())
        map = np.where(inf_map>=1,1,0)
        self._inflated_map = map
        return self

    def _extend_obstcle_BRS(self, inflation):
        # extend the obstacles precisely using reachability
        # This might prove necesary when feeding the map to TTR functions
        x=inflation
    
    def _choose_goal(self):
        # Method 1: randomly select one grid point which is not an obstacle
        goal_indx = tuple(np.random.randint(self._grid.pts_each_dim))
        while self._inflated_map[goal_indx[0:2]]==1:
            goal_indx = tuple(np.random.randint(self._grid.pts_each_dim))
        goal = np.zeros(self._grid.dims)
        for i in range(self._grid.dims):
            goal[i] = np.squeeze(np.array(self._grid.vs[i]))[goal_indx[i]]
        self._goal = goal
        self._goal_indx = goal_indx
        
        # Method 2: randomly select x and y value.
        return self
    
    def get_goal(self):
        return self._goal
    
    def get_obstcle_map(self):
        return self._obstcle_indicator
    
    def get_inflated_map(self):
        return self._inflated_map
    
    def is_obstcle_pnt(self, x , y):
        # Checks if a given coordinate is obstacle
        n = np.zeros(self._grid.dim).tolist()[2:]
        pnt = tuple(np.array([x, y]).append(n))
        indx = self._grid.get_index(pnt)
        if self._inflated_map[index[0:2]] !=0:
            return 1
        else:
            return 0
    def is_obstcle_ind(self, i, j):
        #checks to see if a given grid index is obstacle
        if self._inflated_map[i,j] != 0:
            return 1
        else:
            return 0

    def get_obstcles(self):
        return self._walls
    
    def generate_map(self, inflation = 2):
        walls = []
        for i in range(self._obstcle_no):
            corners = (np.random.rand(self._edge_no, 2) - 0.5) * self._size
            corners[:, 0] = corners[:, 0] + (np.random.rand(1, 1)[0][0] - 0.5) * (self._max_obs_cntr - self._min_obs_cntr)[0]
            corners[:, 1] = corners[:, 1] + (np.random.rand(1, 1)[0][0] - 0.5) * (self._max_obs_cntr - self._min_obs_cntr)[1]
            if i == 0:
                [obstcle_indicator, edges] = self._add_polygon(corners[:, 0], corners[:, 1])
                walls.append(edges)
            else:
                [tmp, edges] = self._add_polygon(corners[:, 0], corners[:, 1])
                obstcle_indicator = np.maximum(tmp, obstcle_indicator)
                walls.append(edges)
        self._obstcle_indicator = obstcle_indicator
        self._walls = walls
        self._extend_obstcle_geo(inflation)._choose_goal()
        return self


def main():
    g = Grid(np.array([-10.0, -10.0, -1.0, -math.pi]), np.array([10.0, 10.0, 2.0, math.pi]), 4, np.array([100, 100, 30, 36]))
    kwargs = {'obstcle_no': 5, 'obstcle_edge_no': 4, 'grid': g}
    map_generator = MapGen2D(**kwargs)
    map1 = map_generator.generate_map(inflation=1)
    image1 = map1.get_obstcle_map()
    image2 = map1.get_inflated_map()
    print('test>0', np.count_nonzero(np.asarray(image1>0)),' , test<1', np.count_nonzero(np.asarray(image1<1)))
    im1 = plt.imshow(image1, cmap=plt.cm.gray)
    plt.show(block=True)
    im2 = plt.imshow(image2, cmap=plt.cm.gray)
    plt.show(block=True)
    print('isObs', map1.is_obstcle_ind(2,10), image1[2,10])
if __name__=='__main__': main()
