# This code is used to define the dynamics that are used in ttr path finding function.
# The difference between this one and the one used in optimized_dp is the format.
# The optimized_dp one is written to be compatible with HCL.
import numpy as np
import math
from scipy.interpolate import griddata as intpolate
from scipy.integrate import solve_ivp
from utils.utils import *
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class dyn:
    # This is the base class with the common functions
    def __init__(self, ttr, grid):
        self._ttr = ttr
        #print('ttr_shape', ttr.shape)
        fig11 = plt.figure('passed ttr')
        plt.imshow(ttr[:,:,27], cmap=plt.cm.gray)
        #print('ttr max = ', np.amax(ttr))
        self._grid = grid
        self._grid_prec = min([i / j for i, j in\
                zip((self._grid.max - self._grid.min),
                    self._grid.pts_each_dim)])
    def step(dt):
        ''' to be done
        '''

    def _get_value(self, state):
        # Calculates the TTR for every given state by linear approximation
        out_of_bound = False
        for i in range(len(state)):
            if i in self._grid.pDim: continue
            elif state[i] >= self._grid.max[i] or state[i] < self._grid.min[i]:
                out_of_bound = True
                break
        T = get_intpolat_value(self._grid, self._ttr, state) if not out_of_bound else math.nan
        #print('value: ', T)
        return T


    def local_deriv(self, state, eps):
        # Calculates local grad at a given state, assumes the state is within bound
        # state must be numpy.array
        deriv = np.zeros(state.size)
        reset = np.asarray(state)
        for i in range(state.size):
            eps_vec = np.zeros(reset.shape)
            eps_vec[i] = eps
            c = 0
            if reset[i] - eps >= self._grid.min[i]:
                neigb1 = reset - eps_vec
                c = c + 1
            else:
                neigb1 = reset - np.zeros(reset.shape)
            if reset[i] + eps < self._grid.max[i]:
                neigb2 = reset + eps_vec
                c = c + 1
            else:
                neigb2 = reset + np.zeros(reset.shape)
            L = self._get_value(neigb1)
            if math.isnan(L):
                #print('reset: ', reset, 'state: ', state)
                #print('neigb1: ', neigb1)
                #print('left is nan')
                time.sleep(5)
                return np.zeros(state.size)
            R = self._get_value(neigb2)
            deriv[i] = (R - L)/(c*eps)
        
        return deriv


class DubinsCar3D(dyn):
    # To Use: The class is chosen by my_car = DubinsCar3D(**kwargs)
    #         Then the path is genrated by calling mycar.generate_path
    def __init__(self, **kwargs):
        '''
        w_range = input's bound --> [w_min w_max] np.array
        v = constant speed  --> scalar value
        dstb_max = maximum disturbance for each dimension [d1_max d2_max d3_max] np.array
        '''
        self._w_rng = kwargs['w_range'] if 'w_range' in kwargs else None
        self._v = kwargs['v'] if 'v' in kwargs else None
        self._dstb_max = kwargs['dstb_max'] if 'dstb_max' in kwargs else None
        
        if self._w_rng.any() == None or self._v == None or self._dstb_max.any() == None:
            raise TypeError('Not enough input arguments')
        
        if 'TTR' in kwargs and 'grid' in kwargs:
            super().__init__(kwargs['TTR'], kwargs['grid'])
        else:
            raise TypeError('Grid or TTR function is not passed')



    def _opt_ctrl(self, mode, deriv):
        # deriv = gradient of ttr with respect to every state. [d_phi_x d_phi_y d_phi_theta]
        # This function returns a scalar
        if mode=='min':
            w = self._w_rng[0] if deriv[2] >=0 else self._w_rng[1]
        elif mode=='max':
            w = self._w_rng[1] if deriv[2] >=0 else self._w_rng[0]
        else:
            raise TypeError('Mode not properly specified')
        return w

    def _opt_dstb(self, mode, deriv):

        d = np.zeros(self._dstb_max.shape) # np.array of disturbance = output of the function
        
        if mode == 'min':
            for i in range(self._dstb_max.size):
                d[i] = -self._dstb_max[i] if deriv[i] >=0 else self._dstb_max[i]
        elif mode == 'max':
            for i in range(self._dstb_max.size):
                d[i] = self._dstb_max[i] if deriv[i] >=0 else -self._dstb_max[i]
        else:
            raise TypeError('Mode not properly specified')
        return d

    def _dyns(self, x):
        '''
        d_x = v*cos(theta) + d1
        d_y = v*sin(theta) + d2
        d_theta = w + d3

        x = [x y theta(in rad)] np.array --> input
        dx = [d_x d_y d_theta] np.array --> output
        '''

        dx = np.zeros(x.shape)
        # correcting for periodic theta
        while (x[2] >= self._grid.min[2] + 2*math.pi) or (x[2] < self._grid.min[2]):
            if x[2] >= self._grid.min[2] + 2*math.pi:
                x[2] = x[2] - (2 * math.pi)
            elif x[2] < self._grid.min[2]:
                x[2] = x[2] + (2 * math.pi)
        #print('out of the loop')

        ttr_deriv = self.local_deriv(x, 0.01)
        if any(np.isnan(ttr_deriv)): time.sleep(5)

        d = self._opt_dstb('max', ttr_deriv) # [d1 d2 d3] np.array
        w = self._opt_ctrl('min', ttr_deriv)
        
        ####### for debugging
        val = self._get_value(x)
        nanflag = True if (math.isnan(val)) else False
        obstacle = True if val >= 30 else False
        string = f'state = {x} \n ttr = {val} \n ttr_deriv = {ttr_deriv} \n d = {d} \n w = {w} \n nanflag = {nanflag} \n obstacle = {obstacle} \n ------- \n'
        #with open('/root/Desktop/project/dataset_generation/test4ttr/path_params.txt', 'a') as t:
        #   t.writelines(string)


        #print('state = ', x)
        #print('ttr = ', val)
        #print('disturbance(d) = ', d)
        #print('control(w) = ', w)
        #print('------')
        #############
        dx[0] = self._v * math.cos(x[2]) + d[0]
        dx[1] = self._v * math.sin(x[2]) + d[1]
        dx[2] = w + d[2]
        
        return dx

    def generate_path(self, start, goal, t_span = 150, goal_proximity = 2):
        # generates a path from a given start location to a given goal location
        # as long as the time doesn't exceed t_span
        # the ttr value, the grid and the bounds must be previously passed to the class
        # parameters to solve_ivp can be twiked later for better performance
        # start and goal must be np.arrays????????????
        
        def f_wrapper(t, y): return self._dyns(y)
        def goal_reached(t, y):
            return ((y[0]-goal[0])**2 + (y[1]-goal[1])**2\
                    - goal_proximity**2)
        goal_reached.terminal = True
        
        # Terminal events for going out of grid bound
        def x_min_reached(t, y): return y[0] - self._grid.min[0]
        def y_min_reached(t, y): return y[1] - self._grid.min[1]
        def x_max_reached(t, y): return y[0] - self._grid.max[0]
        def y_max_reached(t, y): return y[1] - self._grid.max[1]

        x_min_reached.terminal = True
        y_min_reached.terminal = True
        x_max_reached.terminal = True
        y_max_reached.terminal = True
        #x_min_reached.direction = -1
        #y_min_reached.derection = -1
        #x_max_reached.direction = 1
        #y_max_reached.direction = 1

        sol = solve_ivp(f_wrapper, (0, t_span), start, method='RK45', max_step=0.01, dense_output=True,
                events=[goal_reached, x_min_reached, y_min_reached,\
                        x_max_reached, y_max_reached])

        print('t_event: ', sol.t_events)
        valid = True if all(event.size==0 for event in\
                sol.t_events[1:]) and (not sol.t_events[0].size==0)\
                else False
        #print('valid = ', valid)
        #print('path =', sol.y.shape)
        return sol.y, valid

def test():
    x = np.array([[[1],[2],[3]]])
    y = np.array([[[4, 5, 6]]])
    v = np.broadcast(x, y)
    z = np.array([[[7]], [[8]], [[9]]])
    print('shp x', x.shape, 'shp y', y.shape, 'zshp', z.shape)
    v2 = np.broadcast(x,y,z)
    v = np.broadcast(v, z)

    for i in v2:
        print(i)



if __name__ =='__main__': test()
