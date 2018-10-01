import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ImageHandler import ImageHandler
from KMeans import KMeans
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
#print(x_train[0])

kmeans = KMeans(32, x_train)
"""
kmeans.fit(50, 0.0002)

np.savetxt('means.txt', kmeans.mean_vec)"""
means = np.loadtxt('means.txt')
means_int = means.astype(int)

"""from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters = 32, init = 'k-means++', max_iter = 50, n_init = 10,random_state = 0)
y_Kmeans = Kmeans.fit_predict(x_train)
print(Kmeans.cluster_centers_[:])

np.savetxt('means_1.txt', Kmeans.cluster_centers_)"""
means_1 = np.loadtxt('means_1.txt')
means_1_int = means_1.astype(int)

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



