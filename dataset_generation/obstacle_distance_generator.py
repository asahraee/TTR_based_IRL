#TODO
#Can be done later when the walls and start points are given

import numpy as np

class ObstclDistGen:
    #def __init__(self, ???)

    def _separate_obs(self, walls):
        '''
        This function receives all the walls in the map and devides them into
        seperate groups corresponding to diffrent obstacles.
        
        parameters  walls is a list of all the edges of all the obstacles 
                    [[x1 y1 x2 y2][...] for all the edges]
        returns     list(list(list())):
                    [[[x1, y1, x2, y2] for all the edges] for all the obstacles]
        '''
        obstcl = []
        obstcls = []
        for i, edge in enumerate(walls):
            if not obstcl:
                obstcl.append(edge)
                continue
            if edge[0]==walls[i-1][2] and edge[1]==walls[i-1][3]:
                obstcl.append(edge)
            else:
                obstcls.append(obstcl.copy())
                obstcl = []
        return obstcls


    def _preprocess(self, obstcl):
        '''
        This function calculates and saves the multipliers
        for evey wall of the obstacle
        parameters  obstcl is a list of all the edges of the obstcle of the form 
                    [x1, y1, x2, y2]
        returns     list of lists: [[m1i, m2i]...] for all the walls of one obstacle

        '''
        ms = [] 
        for [x1, y1, x2, y2] in obstcl:
            denom = (x2 - x1)**2 + (y2-y1)**2
            m1 = (x2 - x1)/denom
            m2 = (y2 - y1)/denom
            ms.append([m1, m2])
        return ms


    def _find_closest(self, start, obs_walls, obs_ms):
        '''
        This function goes through all the walls for a given point and an obstacle
        and returns the closest point and shortest distance to that obstacle
        parameters  start is [x, y, theta] of the robot's position
                    obs_walls is a list of all the dges of the obstcale of the form
                        [[x1, y1, x2, y2][x2, y2, x3, y3][...]]
                    obs_ms is a list of parameters for every obstcle edge to 
                        calculate the closest point
        returns     a list [xp yp d] where xp and yp are the coordinates of the
                        closest point and d is the distance between point p 
                        and start state
        '''
        best_dist = np.inf
        
        for edge, m in zip(obs_walls, obs_ms):
            t = (start[0] - edge[0])*m[0] +(start[1] - edge[1])*m[1]

            if t <= 0:
                point = [edge[0], edge[1]]
            elif t >= 1:
                point = [edge[2], edge[3]]
            elif t > 0 and t < 1:
                point = [(1 - t)*edge[0] + t*edge[2], (1 - t)*edge[1] + t*edge[3]]

            d = ((start[0] - point[0])**2 + (start[1] - point[1])**2)**0.5

            if d < best_dist:
                best_dist = d
                closest_point = point.copy()
        ans = []
        ans.extend(closest_point)
        ans.append(best_dist)
        
        return ans


    def generate_obs_dist(self, walls, starts):
        '''
        This function uses all the above functions in order to prepare the closest
        point and the realtive distance to each obstacle on the map. The order of
        the given distance and closest point are the same as the order of the
        walls received as input.
        parameters  walls: a list of lists of al the edges on the map of the form
                        [[x1 y1 x2 y2][...]]
                    starts: a list of list of all the start positions on this map
                        [[x1 y1 theta1][x2 y2 theta2]...]
        return      [[[xp1, yp1, dp1][...] for al the obstacles]
                      [[]...] for all the start posiitions]
                      list(list(list))
        '''
        #obstcls = self._separate_obs(walls)
        obstcls = walls
        obstcls_ms = []
        for obstcl in obstcls:
            ms = self._preprocess(obstcl)
            obstcls_ms.append(ms)
        
        all2obs = []

        for start in starts:
            start2obs = []
            for obs, obs_ms in zip(obstcls, obstcls_ms):
                ans = self._find_closest(start, obs, obs_ms)
                start2obs.append(ans)
            all2obs.append(start2obs.copy())

        return all2obs
