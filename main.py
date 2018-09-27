import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ImageHandler import ImageHandler
from KMeans import KMeans
import random
from demo import demo

img_dir = "" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
i = 1
for f1 in files:
    img = cv2.imread(f1)
    img_obj = ImageHandler(img)
    patch = img_obj.ToPatches()
    
    index = str(i)
    with open('1' + index + '.txt', 'w') as f:
        for item in patch:
            f.write("%s\n" % item)    
    i = i + 1
