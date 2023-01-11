import numpy as np
from scipy.interpolate import griddata as inpolate
from scipy.interpolate import interpn
import time
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

step = 0

def get_intpolat_value_wo_xtrapol(grid, value, state):
    global step
    step =+ 1
    priodic_inds = grid.pDim
    # priodic_inds is a list
    near_indx = np.array(list(grid.get_index(state)))
    #print('near_index: ', near_indx)
    #print('near_index_value', value[tuple(near_indx)])
    for i in range(near_indx.size):
        indlist = np.array(list(range(near_indx[i]-1,near_indx[i]+2)))
        shape = np.ones(near_indx.shape, dtype='int32')
        shape[i] = indlist.size
        new = np.reshape(indlist, tuple(shape))
        #new = np.broadcast_to(indlist, tuple(shape))
        indx = np.broadcast(indx, new) if i != 0 else new
    indx = list(indx)
    pts = []
    vals = []
    for i in indx:
        i = list(i)
        pnt = np.zeros(grid.dims)
        out_of_bound = False
        for j in range(grid.dims):
            if j in priodic_inds:
                while i[j] >= grid.pts_each_dim[j]:
                    i[j] = i[j] - grid.pts_each_dim[j]
                while i[j] < 0:
                    i[j] = i[j] + grid.pts_each_dim[j]
            else:
                if i[j] >= grid.pts_each_dim[j] or i[j] < 0:
                    out_of_bound = True
            if out_of_bound: break
            pnt[j] = np.squeeze(grid.vs[j])[i[j]]
        #print('indx', i)
        #print('ot_of_bound', out_of_bound)
        if out_of_bound: continue
        val = value[tuple(i)]
        #print('indx_value: ', val)
        pts.append(pnt)
        vals.append(val)
    #print('pts: ', np.array(pts).shape)
    #print('vals: ', np.array(vals).shape)
    T = inpolate(np.array(pts), np.array(vals), np.array(state), fill_value=10000)
    #print('T', T)
    return T

def get_intpolat_value(grid, value0, state):
    value = value0 + 0
    def value_func_nd(*ind):
        ret = value[tuple(ind)]
        return ret
    near_indx = np.array(list(grid.get_index(state)))
    indx = [np.array([]) for _ in range(grid.dims)]
    pnts = [np.array([]) for _ in range(grid.dims)]
    g = [np.squeeze(grid.vs[j]) for j in range(grid.dims)]
    #print('state = ', state, 'near_indx = ', near_indx)
    for dim in range(grid.dims):
        if dim in grid.pDim:
            grid_prec = (grid.max[dim] - grid.min[dim])/grid.pts_each_dim[dim]
            if near_indx[dim] + 1 >= grid.pts_each_dim[dim]:
                indx[dim] = [near_indx[dim]-1, near_indx[dim], 0]
                #pnts[dim] = [g[dim][near_indx[dim]-1] + i*grid_prec for i in range(3)]
                pnts[dim] = [g[dim][near_indx[dim]-1], g[dim][near_indx[dim]], g[dim][0] + 2*math.pi]
                #print('indx[dim]', indx[dim], 'pnts[dim]', pnts[dim])
            elif near_indx[dim] - 1 < 0:
                indx[dim] = [grid.pts_each_dim[dim] - 1, near_indx[dim], near_indx[dim] + 1]
                #pnts[dim] = [g[dim][near_indx[dim]+1] - i*grid_prec for i in range(2,-1,-1)]
                pnts[dim] = [g[dim][i]+0 for i in indx[dim]]
                pnts[dim][0] -= 2*math.pi
            else:
                indx[dim] = list(range(near_indx[dim]-1, near_indx[dim]+2))
                pnts[dim] = [g[dim][j] for j in indx[dim]]
        else:
            if near_indx[dim] + 1 >= grid.pts_each_dim[dim]:
                indx[dim] = list(range(near_indx[dim] - 2, grid.pts_each_dim[dim]))
            elif near_indx[dim] - 1 < 0:
                indx[dim] = list(range(0, near_indx[dim] + 3))
            else:
                indx[dim] = list(range(near_indx[dim] - 1, near_indx[dim] + 2))
            pnts[dim] = [g[dim][j] for j in indx[dim]]
        #print('dim: ', dim, 'indx: ', indx[dim])
    pnts = tuple(pnts)
    indx = tuple(indx)
    values = value_func_nd(*np.meshgrid(*indx, indexing='ij'))
    near_value = value[tuple(near_indx)] 
    result = interpn(pnts, values, state, bounds_error=False)
    # for debugging
    if step % 20 == 0:
        if near_value >= 1000:
            flag = True
        else: flag = False
        #print(flag)
        #if flag: time.sleep(2)
        string = f'flag = {flag} \n near_indx = {near_indx} \n near_value = {near_value} \n values = {values} \n state = {state} \n interpn_result = {result} \n ---- \n'

    return interpn(pnts, values, state, bounds_error=False)

def get_intpolat_value_test(grid, value, state):
    near_indx = list(grid.get_index(state))
    val = value[tuple(near_indx)]
    flag = True if val >= 10000 else False
    string = f'flag = {flag} \n near_indx = {near_indx} \n near_value = {val} \n state = {state} \n  ---- \n'
    return val
