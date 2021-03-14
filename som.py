#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: André Pacheco
Email: pacheco.comp@gmail.com

This class implements the Self Organized Maps

If you find any bug, please email me =)
"""

import numpy as np
from time import sleep, time


class SOM:
    # It's gonna be an array m x n x dim, where m and n are the nodes' grid and
    # dim is the weight's dimension.
    w_nodes = None
    
    alpha0 = None # It's the initial learning rate
    sigma0 = None # It's the initial radius
    data_in = None # The input data
    grid = None   # The lattice of the grid
    
    def __init__ (self, data_in, grid=(10, 10), alpha=0.1, sigma=None):
        dim = data_in.shape[1]
        self.w_nodes = np.random.uniform(-1, 1, [grid[0], grid[1], dim])
        #self.wNodes = np.random.randn (grid[0], grid[1], dim)    
        
        self.alpha0 = alpha
        if (sigma is None):
            self.sigma0 = max(grid) / 2.0
        else:
            self.sigma0 = sigma
        
        self.data_in = np.asarray(data_in)
        self.grid = grid
        
    def train (self, max_it=100, verbose=True, analysis=False, time_sleep = 0.5):
        n_samples = self.data_in.shape[0]
        m = self.w_nodes.shape[0]
        n = self.w_nodes.shape[1]

        # The time constant needs to be computed just one time, so we so it before the loop starts        
        time_cte = (max_it / np.log(self.sigma0))
        if analysis:
            print (f"- time_cte = {time_cte}")
            
        time_init = 0
        time_end = 0
        for epc in range(max_it):
            # Computing the constants
            alpha = self.alpha0*np.exp(-epc/time_cte)
            sigma = self.sigma0 * np.exp(-epc/time_cte)
            if verbose:
                print (f"- Epoch: {epc}\n- Expected time: {(time_end-time_init) * (max_it - epc)} sec")
            time_init = time()

            for k in range(n_samples):
                
                # Getting the winner node
                mat_dist = self.distance (self.data_in[k, :], self.w_nodes)
                pos_win = self.get_win_node_pos(mat_dist)
                deltaW  = 0                
                h = 0
                for i in range(m):
                    for j in range(n):
                        # Computing the distance between two nodes
                        dNode = self.get_distance_nodes([i, j], pos_win)
                        
                        
                        #if dNode <= sigma: 
                            
                        # Computing the winner node's influence
                        h = np.exp ((-dNode**2)/(2*sigma**2))
                        
                        # Updating the weights
                        deltaW = (alpha * h * (self.data_in[k, :] - self.w_nodes[i, j, :]))
                        self.w_nodes[i, j, :] += deltaW
                            
                        if analysis:  
                            print('Epoch =', epc)
                            print('Sample =', k)
                            print('-' * 50)
                            print('alpha =', alpha)
                            print('sigma =', sigma)
                            print('h =',  h)
                            print('-' * 50)
                            print(f'Winner Node = [{pos_win[0]},{pos_win[1]}]')
                            print(f'Current Node = [{i},{j}]')
                            print('dist. nodes =', dNode)
                            print('deltaW =', deltaW)
                            print('wNode before =', self.w_nodes[i, j, :])
                            print('wNode after =', self.w_nodes[i, j, :] + deltaW)
                            print('\n')
                            sleep(time_sleep)

            time_end = time()
        

    # This code uses the Euclidean distance. You may change this distance, if you want to.
    # This method computes the distance between the inputs and weights throughout the 3D matrix
    def distance (self,a,b):
        return np.sqrt(np.sum((a-b)**2,2,keepdims=True))        

    # Method to get the distance between two nodes in the grid
    def get_distance_nodes (self, n1, n2):
        n1 = np.asarray(n1)
        n2 = np.asarray(n2)
        return np.sqrt(np.sum((n1-n2)**2))
        
    # This method gets the position of the winner node     
    def get_win_node_pos (self, dists):
        arg = dists.argmin()
        m = dists.shape[0]
        return arg//m, arg%m
        
    # Method to get the centroid of a input data
    def getCentroid (self, data):
        data = np.asarray(data)        
        N = data.shape[0]
        centroids = list()
        
        for k in range(N):
            matDist = self.distance (data[k,:], self.w_nodes)
            centroids.append (self.get_win_node_pos(matDist))
            
        return centroids
        
    # Methods to save and load trained nodes
    def save_trained_som (self, fileName='trainedSOM.csv'):
        np.savetxt(fileName, self.w_nodes)

    def set_trained_som (self, fileName):
        self.w_nodes = np.loadtxt(fileName)