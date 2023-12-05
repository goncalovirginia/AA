# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:03:07 2023

@author: rmgma
"""
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.feature_selection import f_classif
from matplotlib import pyplot as plt


def plot_iris(X,y,file_name):
    plt.figure(figsize=(7,7))
    plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='blue', alpha=0.5)
    plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='red', alpha=0.5)
    plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='green', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_name,dpi=200,bbox_inches='tight')
    plt.close()
    
X = np.delete(X, 0, 1)
X = np.delete(X, 0, 1)
plot_iris(X,y,"8_1_plot.png")

f_statistic, p_values = f_classif(X, y)

