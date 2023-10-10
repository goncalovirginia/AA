import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

LAST_ROW = 1285000
SAMPLE_STEP = 257

dataFrame = pd.read_csv("project1/X_train.csv").drop(columns='Id')

def bodyCoords(body, rows) :
    column = 4 * (body - 1 ) + 1
    coords = []
    for row in rows :
        coords.append([row[column], row[column+1]])
    return coords

def plotSample(start) :
    coords = dataFrame[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']][start : start + SAMPLE_STEP];
    print(coords)

    plt.figure("Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coords['x_1'], coords['y_1'], label='1')
    plt.plot(coords['x_2'], coords['y_2'], label='2')
    plt.plot(coords['x_3'], coords['y_3'], label='3')
    plt.show()

def plotAllSamples() :
    for start in range(0, LAST_ROW, SAMPLE_STEP) :
        plotSample(start)

def createXy() :
    X = []
    y = []

    for start in range(0, SAMPLE_STEP-1, SAMPLE_STEP) :
        #sample = dataFrame.iloc[start:start+SAMPLE_STEP].values
        sample = dataFrame.iloc[start:start+SAMPLE_STEP]
        startRow = sample.iloc[0].drop(columns='t').values
        times = sample[['t']].values

        for row in times :
            row.extend

        print(times)
        X.extend(sample[0:SAMPLE_STEP-1])
        y.extend(sample[1:SAMPLE_STEP])

    return [X, y]

X, y = createXy()

def trainModel(polynomialDegree) :
    pipeline = make_pipeline(PolynomialFeatures(polynomialDegree), LinearRegression())
    pipeline.fit(X, y)

    body1Coords = bodyCoords(1, y)
    body2Coords = bodyCoords(2, y)
    body3Coords = bodyCoords(3, y)

    predictions = []
    predictions.extend(pipeline.predict(np.array(X[0]).reshape(1, -1)))
    
    for row in range(1, len(X)) :
        print(predictions[-1])
        predictions.extend(pipeline.predict(np.array(predictions[-1]).reshape(1, -1)))

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



