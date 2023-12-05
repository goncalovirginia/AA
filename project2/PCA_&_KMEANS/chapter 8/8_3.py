# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:27:03 2023

@author: Utilizador
"""
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
X_pca = pca.transform(X)


from matplotlib import pyplot as plt
def plot_iris(X,y,file_name):
    plt.figure(figsize=(7,7))
    plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='blue', alpha=0.5)
    plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='red', alpha=0.5)
    plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='green', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_name,dpi=200,bbox_inches='tight')
    plt.close()

plot_iris(X_pca,y,"8_3_plot.png")