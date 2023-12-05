# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:27:53 2023

@author: rmgma
"""

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)