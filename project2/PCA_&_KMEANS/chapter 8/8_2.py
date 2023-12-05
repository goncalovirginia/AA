# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:39:56 2023

@author: rmgma
"""

import pandas as pd

# Specify the path to your CSV file
data = pd.read_csv('./data.csv')

# Read the CSV file into a pandas

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
print(pca.components_)
t_data = pca.transform(data)

from matplotlib import pyplot as plt

plt.scatter(t_data[:, 0], t_data[:, 1])