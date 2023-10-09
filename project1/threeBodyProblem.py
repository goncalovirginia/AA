import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

def bodyCoords(body, rows) :
    column = 4 * (body - 1 ) + 1
    coords = []
    for row in rows :
        coords.append((row[column], row[column+1]))
    return coords

trainRows = np.loadtxt("project1/X_train.csv", skiprows=1, usecols=range(0, 13), delimiter=",")[0:256]

X = trainRows[0:255]
y = trainRows[1:256]

for polynomialDegree in range(1, 10) :
    pipeline = make_pipeline(PolynomialFeatures(polynomialDegree), LinearRegression())
    pipeline.fit(X, y)

    #print("Prediction: {}".format(pipeline.predict(np.array(X[0]).reshape(1, -1))))
    #print("Actual: {}".format(y[0]))

    body1Coords = bodyCoords(1, y)
    body2Coords = bodyCoords(2, y)
    body3Coords = bodyCoords(3, y)

    predictions = pipeline.predict(X)
    body1CoordsPred = bodyCoords(1, predictions)
    body2CoordsPred = bodyCoords(2, predictions)
    body3CoordsPred = bodyCoords(3, predictions)

    body1RMSE = math.sqrt(mean_squared_error(body1Coords, body1CoordsPred))
    body2RMSE = math.sqrt(mean_squared_error(body2Coords, body2CoordsPred))
    body3RMSE = math.sqrt(mean_squared_error(body3Coords, body3CoordsPred))

    print(polynomialDegree)
    print(body1RMSE)
    print(body2RMSE)
    print(body3RMSE)