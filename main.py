import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ImageHandler import ImageHandler
from KMeans import KMeans
from GMM import GMM

"""
img_dir = "" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
x_train = []
i = 1
"""
"""
for f1 in files:
    img = cv2.imread(f1)
    print(f1)
    img_obj = ImageHandler(img)
    patch = img_obj.ToPatches()
    x_train = x_train + patch
    patch = np.array(patch)
    np.savetxt(f1[:-4] + '.txt', patch)
    i = i + 1

x_train = np.array(x_train)
np.savetxt('dataset.txt', x_train)
"""
x_train = np.loadtxt('dataset.txt')

"""
kmeans = KMeans(32, x_train)

kmeans.fit(50, 0.002)

np.savetxt('means_new.txt', kmeans.mean_vec)
new_means = np.loadtxt('means_new.txt')
new_means_int = new_means.astype(int)

from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters = 32, init = 'k-means++', max_iter = 200, n_init = 10,random_state = 0)
y_Kmeans = Kmeans.fit_predict(x_train)
#print(Kmeans.cluster_centers_[:])

np.savetxt('means_1.txt', Kmeans.cluster_centers_)
means_1 = np.loadtxt('means_1.txt')
means_1_int = means_1.astype(int)

"""
"""
img_dir = "" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
image = np.zeros(32).reshape((1, 32))
for f1 in files:
    img_vec = np.loadtxt(f1[:-4] + '.txt')
    #image = kmeans.BoVW(means, img_vec)
    image_row = kmeans.BoVW(means, img_vec).reshape((1, 32))
    image = np.concatenate((image, image_row), axis = 0)
    print(f1)
    
np.savetxt('image_data.txt', image)
image = np.loadtxt('image_data.txt')
"""
img = cv2.imread('31.png')
print(img)
img_obj = ImageHandler(img)
x = img_obj.ToShiftedPatches()
x = np.array(x)
np.savetxt('cell_1.txt', x)

kmeans_obj = KMeans(3, x)
kmeans_obj.fit(3, 0.002)

means = kmeans_obj.mean_vec
cov_mat_list = kmeans_obj.CovMatrix()
mixture_coeff = kmeans_obj.MixtureCoeff()

print(cov_mat_list)

"""from sklearn.cluster import KMeans
obj = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)
y_Kmeans = obj.fit_predict(x)
print(obj.cluster_centers_[:])"""

GMM_obj = GMM(3, x, means, cov_mat_list, mixture_coeff)
GMM_obj.fit(0.0002)

print(GMM_obj.mean_vec)
print(GMM_obj.cov_mat)
print(GMM_obj.mixture_coeff)

y_pred = GMM_obj.ClusterPredict(x)
plt.scatter(GMM_obj.x_train[y_pred == 0, 0], GMM_obj.x_train[y_pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(GMM_obj.x_train[y_pred == 1, 0], GMM_obj.x_train[y_pred == 1, 1], s = 20, c = 'green', label = 'Cluster 2')
plt.scatter(GMM_obj.x_train[y_pred == 2, 0], GMM_obj.x_train[y_pred == 2, 1], s = 20, c = 'blue', label = 'Cluster 3')
plt.scatter(GMM_obj.mean_vec[:, 0], GMM_obj.mean_vec[:, 1], s = 50, c = 'yellow', label = 'Centroids')
plt.show()

plt.scatter(GMM_obj.x_train[:, 0], GMM_obj.x_train[:, 1])
plt.show()


from sklearn.mixture import GaussianMixture 
obj = GaussianMixture(3, tol = 0.0002, covariance_type = 'full').fit(x)
print(obj.means_)
print(obj.covariances_)
print(obj.weights_)

y_prediction = obj.predict(x)
plt.scatter(x[y_prediction == 1, 0], x[y_prediction == 1, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_prediction == 2, 0], x[y_prediction == 2, 1], s = 20, c = 'green', label = 'Cluster 2')
plt.scatter(x[y_prediction == 3, 0], x[y_prediction == 3, 1], s = 20, c = 'blue', label = 'Cluster 3')
plt.scatter(obj.means_[:, 0], obj.means_[:, 1], s = 50, c = 'yellow', label = 'Centroids')
plt.show()
