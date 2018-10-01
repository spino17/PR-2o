import numpy as np
import random
import math
import matplotlib.pyplot as plt

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
            iter_num = iter_num + 1
            print("iternation no: ", iter_num)
            
        #print(N_z[0], N_z[1], N_z[2])
        print(N_z)
        print(self.mean_vec)
            
            
    def ClusterPredict(self, X):
        pred_arr = np.array([])
        for x_vec in X:
            pred_arr = np.append(pred_arr, self.MinDistCluster(x_vec, self.mean_vec))

        return pred_arr
    
    
    def BoVW(self, mean_vec, img_vec):
        bag = np.zeros(self.k)
        for vec in img_vec:
            cluster_num = self.MinDistCluster(vec, mean_vec)
            bag[cluster_num] = bag[cluster_num] + 1
            
        return bag
        
    
    def PlotCluster(self): # for 2-D feature vectors
        y_pred = self.ClusterPredict(self.x_train)
        plt.scatter(self.x_train[y_pred == 0, 0], self.x_train[y_pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
        plt.scatter(self.x_train[y_pred == 1, 0], self.x_train[y_pred == 1, 1], s = 20, c = 'green', label = 'Cluster 2')
        plt.scatter(self.x_train[y_pred == 2, 0], self.x_train[y_pred == 2, 1], s = 20, c = 'blue', label = 'Cluster 3')
        plt.scatter(self.mean_vec[:, 0], self.mean_vec[:, 1], s = 50, c = 'yellow', label = 'Centroids')
        plt.show()
