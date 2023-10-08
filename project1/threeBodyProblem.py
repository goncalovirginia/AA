import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

trainRows = np.loadtxt("project1/X_train.csv", skiprows=1, usecols=range(0, 13), delimiter=",")

X = trainRows[0:255]
y = trainRows[1:256]

pipeline = make_pipeline(PolynomialFeatures(6), LinearRegression())
pipeline.fit(X, y)

print("Prediction: {}".format(pipeline.predict(np.array(X[0]).reshape(1, -1))))
print("Actual: {}".format(y[0]))
