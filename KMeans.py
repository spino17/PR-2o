import numpy as np
import random
import math

class KMeans:
    k = 0
    x_train = np.array([])
    mean_vec = np.array([])
    z = np.array([])
    vec_dim = 0
    total_size = 0
    
    def __init__ (self, k, x_train):
        
        self.k = k
        self.x_train = x_train
        self.mean_vec = np.zeros(shape = (k, np.size(x_train[0])))
        self.z = np.zeros(np.size(self.x_train))
        self.vec_dim = np.size(x_train[0])
        self.total_size = int(np.size(x_train) / self.vec_dim)
    
    
    def DistMeasure(self, x1, x2):
        
        return math.sqrt(np.dot(x1 - x2, x1 - x2))
    
    
    def MinDistCluster(self, x_vec, mean_vec):
        
        min_dist = self.DistMeasure(x_vec, mean_vec[0])
        cluster_num = 0
        for i in range(0, self.k):
            value = self.DistMeasure(x_vec, mean_vec[i])
            if (value <= min_dist):
                min_dist = value
                cluster_num = i
                
        return cluster_num
    
    
    def fit(self, iterations, precision):

        for i in range(0, self.k):
            kth_mean = random.sample(range(0, 500), self.vec_dim)
            self.mean_vec[i] = kth_mean
        
        iter_num = 0
        while iter_num < iterations:
            N_z = np.zeros(self.k)
            new_mean_vec = np.zeros(shape = (self.k, self.vec_dim))
            
            for i in range(0, self.total_size):
                cluster_num = self.MinDistCluster(self.x_train[i], self.mean_vec)
                new_mean_vec[cluster_num] = new_mean_vec[cluster_num] + self.x_train[i]
                self.z[i] = cluster_num
                N_z[cluster_num] = N_z[cluster_num] + 1
                
            for i in range(0, self.k):
                if (N_z[i] == 0):
                    new_mean_vec[i] = new_mean_vec[i]
                else:
                    new_mean_vec[i] = (1 / N_z[i]) * new_mean_vec[i]
            
            self.mean_vec = new_mean_vec
            #print(iter_num)
            iter_num = iter_num + 1
            
        print(N_z[0], N_z[1], N_z[2])
        print(self.mean_vec)
            
            
    def ClusterPredict(self, X):
        pred_arr = np.array([])
        for x_vec in X:
            pred_arr = np.append(pred_arr, self.MinDistCluster(x_vec, self.mean_vec))

        return pred_arr
