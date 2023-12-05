# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:02:09 2023

@author: Utilizador
"""

import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d
from skimage.io import imsave,imread
img = imread("vegetables.png")
w,h,d = img.shape
cols = np.reshape(img/255.0, (w * h, d))

# for each n_cluster we have 1 centroid that represents 1 color of the final image
kmeans = KMeans(n_clusters=4).fit(cols)
centroids = kmeans.cluster_centers_
labels = kmeans.predict(cols)

c_cols = np.zeros(cols.shape)
for ix in range(cols.shape[0]):
    c_cols[ix,:]=centroids[labels[ix]]

final_img = np.reshape(c_cols,(w,h,3))
imsave('image.png',final_img)
