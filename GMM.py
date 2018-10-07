import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

class GMM:
    k = 0 # number of clusters
    x_train = np.array([]) # training dataset
    mean_vec = np.array([]) # k means
    cov_mat = np.array([]) # k covariance matrix
    mixture_coeff = np.array([]) # mixture coefficients of k clusters
    N_eff = np.array([]) # effective number of data points in k clusters
    vec_dim = 0 # dimension of data points
    total_size = 0 # total size of dataset
    
    
    # initialization of parameter is taken from k-means clustering
    def __init__ (self, k, x_train, mean_vec, cov_mat, mixture_coeff):
        
        self.k = k
        self.x_train = x_train
        self.mean_vec = mean_vec
        self.cov_mat = cov_mat
        self.mixture_coeff = mixture_coeff
        self.N_eff = np.zeros(self.k)
        self.vec_dim = np.size(x_train[0])
        self.total_size = int(np.size(x_train) / self.vec_dim)
    
    # gaussian with parameters as arguments of the function
    def N(self, x_vec, mean_vec, cov_mat):
        denominator = math.sqrt(2*3.1417*abs(np.linalg.det(cov_mat)))
        exponent_term = -np.dot(x_vec - mean_vec, inv(cov_mat).dot(x_vec - mean_vec)) / 2
        return math.exp(exponent_term) / denominator
    
    # function to fit the dataset on the k gaussian clusters to get maximum value of log likelihood
    def fit(self, precision):
        
        iter_num = 0
        cost_func_f = 1000000
        cost_func_i = 0
        while cost_func_f - cost_func_i > precision:
            cost_func_i = cost_func_f
            cost_func_f = 0
            new_mean_vec = np.zeros(shape = (self.k, self.vec_dim))
            new_cov_mat = np.zeros((self.k, self.vec_dim, self.vec_dim))
            new_mixture_coeff = np.zeros(self.k)
                
            for i in range(0, self.k):
                
                cov_term1 = np.zeros(shape = (self.vec_dim, self.vec_dim))
                sum_x = np.zeros(self.vec_dim)
                
                for j in range(0, self.total_size):
                    total_prob = 0
                    x_vec = self.x_train[j]
                    for k in range(0, self.k):
                        total_prob = total_prob + self.mixture_coeff[k] * self.N(x_vec, self.mean_vec[k], self.cov_mat[k])
                    
                    gamma = (self.mixture_coeff[i] * self.N(x_vec, self.mean_vec[i], self.cov_mat[i])) / total_prob
                    new_mean_vec[i] = new_mean_vec[i] + gamma * x_vec
                    cov_term1 = cov_term1 + gamma * np.outer(x_vec, x_vec)
                    self.N_eff[i] = self.N_eff[i] + gamma
                    sum_x = sum_x + gamma * x_vec
                    
                    cost_func_f = cost_func_f + gamma * math.log(self.mixture_coeff[i] * self.N(x_vec, self.mean_vec[i], self.cov_mat[i]))

                
                new_mean_vec[i] = new_mean_vec[i] / int(self.N_eff[i])
                new_cov_mat[i] = cov_term1 - np.outer(sum_x, new_mean_vec[i]) - np.outer(new_mean_vec[i], sum_x) + self.N_eff[i] * np.outer(new_mean_vec[i], new_mean_vec[i])
                new_cov_mat[i] = new_cov_mat[i] / int(self.N_eff[i])
                new_mixture_coeff[i] = self.N_eff[i]
            
            total_N_eff = np.sum(self.N_eff)
            for i in range(0, self.k):
                new_mixture_coeff[i] = new_mixture_coeff[i] / total_N_eff
                
                
            self.mean_vec = new_mean_vec
            self.cov_mat = new_cov_mat
            self.mixute_coeff = new_mixture_coeff
            
            print("iteration no: %d diff: %f" % (iter_num, cost_func_f - cost_func_i))
            print(self.mean_vec)
            iter_num = iter_num + 1
    