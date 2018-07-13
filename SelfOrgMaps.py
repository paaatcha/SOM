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
import matplotlib.pyplot as plt

class SOM:
    # It's gonna be an array m x n x dim, where m and n are the nodes' grid and
    # dim is the weight's dimension.
    wNodes = None 
    
    alpha0 = None # It's the initial learning rate
    sigma0 = None # It's the initial radius
    dataIn = None # The input data
    grid = None   # The lattice of the grid
    
    def __init__ (self, dataIn, grid=[10,10], alpha=0.1, sigma=None):
        dim = dataIn.shape[1]
        self.wNodes = np.random.uniform(-1,1,[grid[0], grid[1], dim])
        #self.wNodes = np.random.randn (grid[0], grid[1], dim)    
        
        self.alpha0 = alpha
        if (sigma is None):
            self.sigma0 = max(grid) / 2.0
        else:
            self.sigma0 = sigma
        
        self.dataIn = np.asarray(dataIn)
        self.grid = grid
        
        
    def train (self, maxIt=100, verbose=True, analysis=False, timeSleep = 0.5):
        nSamples = self.dataIn.shape[0]
        m = self.wNodes.shape[0]        
        n = self.wNodes.shape[1]        
    
    
        # The time constant needs to be computed just one time, so we so it before the loop starts        
        timeCte = (maxIt/np.log(self.sigma0))        
        if analysis:
            print 'timeCte = ', timeCte
            
        timeInit = 0        
        timeEnd = 0
        for epc in xrange(maxIt):
            # Computing the constants
            alpha = self.alpha0*np.exp(-epc/timeCte)
            sigma = self.sigma0 * np.exp(-epc/timeCte)
            
            if verbose:
                print 'Epoch: ', epc, ' - Expected time: ', (timeEnd-timeInit)*(maxIt-epc), ' sec'
                
            timeInit = time()

            for k in xrange(nSamples):    
                
                # Getting the winner node
                matDist = self.distance (self.dataIn[k,:], self.wNodes)
                posWin = self.getWinNodePos(matDist)                              
                
                deltaW  = 0                
                h = 0    
                          
                
                for i in xrange(m):
                    for j in xrange(n):      
                        # Computing the distance between two nodes
                        dNode = self.getDistanceNodes([i,j],posWin)                       
                        
                        
                        #if dNode <= sigma: 
                            
                        # Computing the winner node's influence
                        h = np.exp ((-dNode**2)/(2*sigma**2))
                        
                        # Updating the weights
                        deltaW = (alpha*h*(self.dataIn[k,:] - self.wNodes[i,j,:]))                       
                        self.wNodes[i,j,:] += deltaW
                            
                        if analysis:  
                            print 'Epoch = ', epc
                            print 'Sample = ', k
                            print '-------------------------------'
                            print 'alpha = ', alpha
                            print 'sigma = ', sigma                            
                            print 'h = ',  h
                            print '-------------------------------'
                            print 'Winner Node = [', posWin[0],', ',posWin[1],']'
                            print 'Current Node = [',i,', ',j,']' 
                            print 'dist. Nodes = ', dNode
                            print 'deltaW = ', deltaW                        
                            print 'wNode before = ', self.wNodes[i,j,:]
                            print 'wNode after = ', self.wNodes[i,j,:] + deltaW
                            print '\n'                        
                            sleep(timeSleep) 
                            
            timeEnd = time()                       
        

    # This code uses the Euclidean distance. You may change this distance, if you want to.
    # This method computes the distance between the inputs and weights throught the 3D matrix
    def distance (self,a,b):
        return np.sqrt(np.sum((a-b)**2,2,keepdims=True))        

    # Method to get the distance between two nodes in the grid
    def getDistanceNodes (self,n1,n2):
        n1 = np.asarray(n1)
        n2 = np.asarray(n2)
        return np.sqrt(np.sum((n1-n2)**2))
        
    # This method gets the position of the winner node     
    def getWinNodePos (self,dists):
        arg = dists.argmin()
        m = dists.shape[0]
        return arg//m, arg%m
        
    # Method to get the centroid of a input data
    def getCentroid (self, data):
        data = np.asarray(data)        
        N = data.shape[0]
        centroids = list()
        
        for k in xrange(N):
            matDist = self.distance (data[k,:], self.wNodes)
            centroids.append (self.getWinNodePos(matDist))
            
        return centroids
        
    # Methods to save and load trained nodes
    def saveTrainedSOM (self, fileName='trainedSOM.csv'):
        np.savetxt(fileName, self.wNodes)

    def setTrainedSOM (self, fileName):
        self.wNodes = np.loadtxt(fileName)



#Training inputs for RGBcolors
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
      
colors2 = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],     
      [1., 1., 0.],
      [1., 1., 1.],     
      [1., 0., 0.]])      
      
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

s = SOM(colors,[20,30], alpha=0.3)
plt.imshow(s.wNodes)

s.train(maxIt=30)

plt.imshow(s.wNodes)
plt.show()
