import numpy as np
import random

class KMeans:
    k = 0
    x_train = np.array([])
    mean_vec = np.array([])
    z = np.array([])
    
    
    def __init__ (self, k, x_train):
        
        self.k = k
        self.x_train = x_train
        self.mean_vec = np.zeros(shape = (k, 24))
        self.z = np.zeros(np.size(self.x_train))
    
    
    def DistMeasure(self, x1, x2):
        
        return np.dot(x1 - x2, x1 - x2)
    
    
    def MinDistCluster(self, x_vec, mean_vec):
        
        min_dist = self.DistMeasure(x_vec, mean_vec[0])
        cluster_num = 0
        for i in range(0, np.size(mean_vec)):
            value = self.DistMeasure(x_vec, mean_vec[i])
            if (value < min_dist):
                min_dist = value
                cluster_num = i
                
        return cluster_num
    
    
    def fit(self, iterations, precision):

        for i in range(0, self.k):
            kth_mean = random.sample(range(0, 500), 24)
            self.mean_vec[i] = kth_mean
        
        iter_num = 0
        while iter_num < iterations:
            N_z = np.zeros(self.k)
            new_mean_vec = np.zeros(shape = (self.k, 24))
            
            for i in range(0, np.size(self.x_train)):
                cluster_num = self.MinDistCluster(self.x_train[i], self.mean_vec)
                new_mean_vec[cluster_num] = new_mean_vec[cluster_num] + self.xi_train[i]
                self.z[i] = cluster_num
                N_z[cluster_num] = N_z[cluster_num] + 1
                
            for i in range(0, self.k):
                new_mean_vec = (1 / N_z[i]) * new_mean_vec
            
            self.mean_vec = new_mean_vec
            print(iter_num)
            iter_num = iter_num + 1
            
            
    def ClusterPredict(self, X):
        pred_arr = np.zeros(np.size(X))
        for i in range(0, np.size(X)):
            pred_arr[i] = self.MinDistCluster(X[i], self.mean_vec)
        
        return pred_arr