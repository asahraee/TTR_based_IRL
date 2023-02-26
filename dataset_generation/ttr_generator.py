import numpy as np
import math
import sys
sys.path.append('/root/Desktop/git_repo/TTR_based_IRL')
sys.path.append('/root/Desktop/git_repo/TTR_based_IRL/optimized_dp')

# Utility functions to initialize the problem
from optimized_dp.Grid.GridProcessing import Grid
from optimized_dp.Shapes.ShapesFunctions import *
# Specify the  file that includes dynamic systems
from optimized_dp.dynamics.DubinsCar import *
# Plot options
from optimized_dp.plot_options import *
# Solver core
from optimized_dp.solver import *
# Solver requirements
from optimized_dp.Plots.plotting_utilities import *
# Path generation requirements
from utils.ttrgen_dynamics import *
from utils.utils import *

# Test function requirements
from map_generator import MapGen2D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

class TTRGen:
    def __init__(self, **kwargs):
        self._grid = kwargs['grid'] if 'grid' in kwargs else None
        self._eps = kwargs['computation_error']\
                if 'computation_error' in kwargs else 0.001
        self._goal_r = kwargs['goal_radi']\
                if 'goal_radi' in kwargs else 2
        #self._goal = kwargs['goal'] if 'goal' in kwargs else None
        self._dyn = kwargs['dyn'] if 'dyn' in kwargs else 'DubinsCar3D'
        # This is just the name of function, DubinsCar,humanoid6D,etc
        # Larger n means more paths genrated
        self._n = 10 
        self._dyn_bounds = kwargs['dyn_bounds']\
                if 'dyn_bounds' in kwargs\
                else dict(w_range=np.array([-1,1]),
                        v=1, dstb_max=np.array([0,0,0]))
                # This would be a dict and is different for each 'dyn'
        # An explanation must be provided in dataset_generator.py
        if self._grid == None:
            raise TypeError('grid must be passed to TTRGen class')
        
        if self._dyn == 'DubinsCar3D':
            self._hcl_dyn = DubinsCar
            self._ttr_dyn = DubinsCar3D

        # Must add other dynamics??????????!!!!!!!!!!!!#



    def _select_start_pos(self, obstcle_map):
        # Selects random start positions to find an optimal path from
        
        st_indx = tuple(np.random.randint(self._grid.pts_each_dim))
        while obstcle_map.is_obstcle_ind(st_indx[0], st_indx[1])==1:
            st_indx = tuple(np.random.randint(self._grid.pts_each_dim))
        start = np.zeros(self._grid.dims)
        for i in range(self._grid.dims):
            start[i] = np.squeeze(np.array(self._grid.vs[i]))[st_indx[i]]
        return start

    def _find_path(self, start, dt):
        # Finds the optimal path from each given start position
        # to the given goal position
        '''to be done
            ...
        '''
        '''
        while goal not reached:
            control = choose control
            disturbance = choose disturbance
            state[i] = step(state[i-1], control, disturbnce)
            i = i+1

        '''
        
    def _choose_state(self, path):
        # Selects random start and goal positions from every given path
        '''to be done
                ...

        '''
        path_len = path[0].size
        st_indx = np.random.randint(np.ones(path.shape[0])*path_len)
        gl_indx = np.random.randint(np.ones(path.shape[0])*path_len)
        while gl_indx.all() == st_indx.all():
            gl_indx = np.random.randint(np.ones(path.shape[0])*path_len)
        strt = np.zeros(self._grid.dims)
        goal = np.zeros(self._grid.dims)
        for i in range(self._grid.dims):
            strt[i] = path[i][st_indx[i]]
            goal[i] = path[i][gl_indx[i]]
        return strt, goal

    def _calculate_ttr(self, st, gl):
        # Calculates TTR between 2 given points in the optimal path
        '''to be done
                ...

        '''
        st_ttr = get_intpolat_value(self._grid, self._phi, st)
        gl_ttr = get_intpolat_value(self._grid, self._phi, gl)
        return st_ttr - gl_ttr

    def get_pos_and_ttr(self, indx=None):
        # Reads TTR and position data for external use
        # If indx given, returns one data bundle at indx position
        # If indx is no given, returns the whole dataset
        '''to be done
                ...

        '''
        if indx==None:
            return self._train_dataset
        else:
            #?????????????????? need to fix this to recieve the dim, this is not just for 3d
            return [self._train_dataset[0][indx],\
                    self._train_dataset[1][indx],\
                    self.train_dataset[2][indx]]

    def generate_ttr_data(self, obstcle_map, samples_no):
        # Wraps up all the required functions
        # Call to this function initializes all computation
        # and prepares the training data
        
        # Chosing dynamics
        kwargs = self._dyn_bounds
        hcl_dyn_sys = self._hcl_dyn(wMax=kwargs['w_range'][1],
                speed=kwargs['v'],
                dMax=kwargs['dstb_max'], uMode='min')
        # Dimensions to ingnore in calculation of distance from target
        dim2ignore = list(range(2,self._grid.dims))
        
        # Choosing a goal for the map
        self._goal = np.array(obstcle_map.get_goal())
        #print('goal = ', self._goal)
        
        # Specifying the target set and plot options
        init_set = CylinderShape(self._grid, dim2ignore,
                np.array(self._goal), self._goal_r)
        po = PlotOptions( "3d_plot", plotDims=[0,1,2], slicesCut=[],
                min_isosurface=1.5, max_isosurface=1.5)
        
        # obstacle map as a list to be read by hcl
        map_list = (obstcle_map.get_inflated_map()).tolist()
        print('started TTR calculation')
        V_0 = TTRSolver(hcl_dyn_sys, self._grid, init_set,
                map_list, self._eps, po)
        print('Solving for optimal paths')
        self._phi = V_0

        
        pos_no = 0 # number of training data pool
        kwargs['grid'] = self._grid
        kwargs['TTR'] = self._phi
        ttr_dyn_sys = self._ttr_dyn(**kwargs) # path generation dynamics

        train_dataset_start=[]
        train_dataset_goal=[]
        train_dataset_ttr=[]
        self.paths = []
        while pos_no < samples_no:
            path_is_valid = False
            failed_trials = 0
            # Generatea valid path from a random start point
            while not path_is_valid:
                print('trying a new path')
                start = self._select_start_pos(obstcle_map)
                path, valid = ttr_dyn_sys.generate_path(start,
                        self._goal, t_span=150,
                        goal_proximity=self._goal_r)
                path_is_valid = valid
                failed_trials += 1
                if failed_trials > 3:
                    print("No valid path found after 3 trials")
                    return None
                    #break


            pnts_no = path[0].size// self._n if path[0].size>=self._n\
                    else 1
            if (samples_no - pos_no) < pnts_no:
                #pnts_no = (samples_no*(self._n - 1))// self._n
                pnts_no = samples_no - pos_no
            count = 0
            # for drawing the path and debugging
            self.paths.append(path)
            # Choose random points as start and goal from generated path
            while count<pnts_no:
                st_pnt, gl_pnt = self._choose_state(path)
                ttr = self._calculate_ttr(st_pnt, gl_pnt)
                # add position and ttr values to dataset
                if ttr !=0:
                    count = count + 1
                    if ttr>0:
                        train_dataset_start.append(st_pnt)
                        train_dataset_goal.append(gl_pnt)
                        train_dataset_ttr.append(ttr)
                    else:
                        train_dataset_start.append(gl_pnt)
                        train_dataset_goal.append(st_pnt)
                        train_dataset_ttr.append(-ttr)
            # Repeat untill we have enough samples i.e. equal or greater than samples_no
            pos_no = pos_no + pnts_no

        self._train_dataset = [train_dataset_start, train_dataset_goal, train_dataset_ttr]
        
        return self


#////////////////////////////DEBUG

#This class is used to genearte some data to debug the network. The TTR generator in this class calls for geneartion of a new goal every time it is generating a new sample, and then calulates a start position with constant relative distance from the goal instead of randomly generating one

class TTRGen_debug:
    def __init__(self, **kwargs):
        self._grid = kwargs['grid'] if 'grid' in kwargs else None
        self._eps = kwargs['computation_error']\
                if 'computation_error' in kwargs else 0.001
        self._goal_r = kwargs['goal_radi']\
                if 'goal_radi' in kwargs else 0.1
        #self._goal = kwargs['goal'] if 'goal' in kwargs else None
        self._dyn = kwargs['dyn'] if 'dyn' in kwargs else 'DubinsCar3D'
        # This is just the name of function, DubinsCar,humanoid6D,etc
        # Larger n means more paths genrated
        self._n = 50
        self._dyn_bounds = kwargs['dyn_bounds']\
                if 'dyn_bounds' in kwargs\
                else dict(w_range=np.array([-1,1]),
                        v=1, dstb_max=np.array([0,0,0]))
                # This would be a dict and is different for each 'dyn'
        # An explanation must be provided in dataset_generator.py
        if self._grid == None:
            raise TypeError('grid must be passed to TTRGen class')

        if self._dyn == 'DubinsCar3D':
            self._hcl_dyn = DubinsCar
            self._ttr_dyn = DubinsCar3D


    def _select_start_pos(self, obstcle_map, rel_dist):
        # Selects a start position which has the passed relative distance to the goal and a theta equal to the third element of rel_dist array
        start = self._goal - rel_dist
        start[2] = rel_dist[2] + 0
        return start



    def get_pos_and_ttr(self, indx=None):
        # Reads TTR and position data for external use
        # If indx given, returns one data bundle at indx position
        # If indx is no given, returns the whole dataset
        if indx==None:
            return self._train_dataset
        else:
            return [self._train_dataset[0][indx],\
                    self._train_dataset[1][indx],\
                    self.train_dataset[2][indx]]

    def generate_ttr_data(self, obstcle_map, samples_no, rel_dist):
        # Wraps up all the required functions
        # Call to this function initializes all computation
        # and prepares the training data

        # Chosing dynamics
        kwargs = self._dyn_bounds
        hcl_dyn_sys = self._hcl_dyn(wMax=kwargs['w_range'][1],
                speed=kwargs['v'],
                dMax=kwargs['dstb_max'], uMode='min')
        # Dimensions to ingnore in calculation of distance from target
        dim2ignore = list(range(2,self._grid.dims))

        # Specifying plot options
        po = PlotOptions( "3d_plot", plotDims=[0,1,2], slicesCut=[],
                min_isosurface=1.5, max_isosurface=1.5)

        # obstacle map as a list to be read by hcl
        map_list = (obstcle_map.get_inflated_map()).tolist()

        pos_no = 0 # number of training data pool
        #kwargs['grid'] = self._grid
        #kwargs['TTR'] = self._phi

        train_dataset_start=[]
        train_dataset_goal=[]
        train_dataset_ttr=[]
        self.paths = []
        while pos_no < samples_no:
            # Choosing a goal for the map
            self._goal = np.array(obstcle_map._choose_goal().get_goal())
            # Specifying the target set
            init_set = CylinderShape(self._grid, dim2ignore,
                np.array(self._goal), self._goal_r)
            print('started TTR calculation')
            V_0 = TTRSolver(hcl_dyn_sys, self._grid, init_set,
                    map_list, self._eps, po)
            self._phi = V_0
            start = self._select_start_pos(obstcle_map, rel_dist)
            st_ttr = get_intpolat_value(self._grid, self._phi, start)
            train_dataset_start.append(start)
            train_dataset_goal.append(self._goal)
            train_dataset_ttr.append(st_ttr)
            pos_no += 1

        self._train_dataset = [train_dataset_start, train_dataset_goal, train_dataset_ttr]

        return self



def test():
    import pickle

    g = Grid(minBounds=np.array([-10.0, -10.0, -math.pi]), maxBounds=np.array([10.0, 10.0, math.pi]),
                             dims=3, pts_each_dim=np.array([100, 100, 60]), periodicDims=[2])
    # Generate a new map: only on first run
    map = MapGen2D(obstcle_no=10,grid=g)
    map.generate_map(inflation=0.1)
    print('Map generated')
    
    # Save map
    with open('map_file','wb') as map_file_handl:
        pickle.dump(map,map_file_handl)
    print('Map saved')
    
    # Load map: from second run onward, to load the previous map
    #with open('map_file','rb') as map_file_handl:
    #    map = pickle.load(map_file_handl)
    #print('Map loaded')

    ttr_generator = TTRGen(grid=g, goal_radi=0.5)

    dataset = ttr_generator.generate_ttr_data(map,500)

    if dataset is not None:
        data = dataset.get_pos_and_ttr()
        #print('data: ', data)

        fig1 = plt.figure()
        plt.imshow(ttr_generator._phi[:,:,59], cmap=plt.cm.gray)
        #plt.show()
        #fig2 = plt.figure()
        #plt.imshow(map.get_obstcle_map(), cmap=plt.cm.gray)
        #plt.show()
    
        ######## plotting the trajectory on the map #######
        #y_obs, x_obs = np.nonzero(map.get_inflated_map())
        #x_obs = x_obs*((g.max[0] - g.min[0])/g.pts_each_dim[0]) + g.min[0]
        #y_obs = -y_obs*((g.max[1] - g.min[1])/g.pts_each_dim[1]) + g.max[1]
        
        x_obs, y_obs = np.nonzero(map.get_inflated_map())
        x_obs = x_obs*((g.max[0] - g.min[0])/g.pts_each_dim[0]) + g.min[0]
        y_obs = y_obs*((g.max[1] - g.min[1])/g.pts_each_dim[1]) + g.min[1]

        fig3 = plt.figure()
        ax = fig3.add_subplot()
        ax.set_xlim([-10,10])
        ax.set_ylim([-10, 10])
        plt.scatter(x=x_obs, y=y_obs, c='r')
        #print('paths: ', dataset.paths)
        for i, path in enumerate(dataset.paths):
            plt.scatter(x=path[0], y=path[1], c=np.ones(
                len(path[0]))*i, marker=".")
        print('number of paths = ', i+1)
        plt.scatter(dataset._goal[0], dataset._goal[1],
                    marker="x")
        plt.show()

        # saving map, ttr and path states for plotting in matlab
        ii = 5 # counter
        dir = '/root/Desktop/data_and_log/test4ttr'
        def join(x): return os.path.join(dir, x)
        grid_name = join(f'grid_{ii}')
        map_name = join(f'map_{ii}')
        ttr_name = join(f'ttr_{ii}')
        
        with open(grid_name, 'wb') as grid_handl:
            pickle.dump(g, grid_handl)
        with open(map_name,'wb') as map_handl:
            pickle.dump(map, map_handl)
        with open(ttr_name, 'wb') as ttr_handl:
            pickle.dump(ttr_generator, ttr_handl)



    

if __name__ == '__main__': test()
