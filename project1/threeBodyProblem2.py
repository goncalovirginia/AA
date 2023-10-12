""" 
This model is trained on the idea of predicting line n+1 with line n of each sample.
It then tries to predict the new positions and velocities by recursively using its own output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

LAST_ROW = 1285000
NUM_SAMPLES = 5000
SAMPLE_LENGTH = 257

dataFrame = pd.read_csv("project1/X_train.csv").drop(columns=['t', 'Id'])

def bodyCoords(body, rows) :
    column = 4 * (body - 1 ) + 1
    coords = []
    for row in rows :
        coords.append([row[column], row[column+1]])
    return coords

def plotSample(start) :
    coords = dataFrame[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']][start : start + SAMPLE_LENGTH];
    print(coords)

    plt.figure("Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coords['x_1'], coords['y_1'], label='1')
    plt.plot(coords['x_2'], coords['y_2'], label='2')
    plt.plot(coords['x_3'], coords['y_3'], label='3')
    plt.show()

def plotAllSamples() :
    for start in range(0, LAST_ROW, SAMPLE_LENGTH) :
        plotSample(start)

def createXy(numSamples) :
    X = []
    y = []

    for sampleStart in range(0, numSamples*SAMPLE_LENGTH, SAMPLE_LENGTH) :
        sample = dataFrame.iloc[sampleStart:sampleStart+SAMPLE_LENGTH].to_numpy()
        X.extend(sample[0:SAMPLE_LENGTH-1])
        y.extend(sample[1:SAMPLE_LENGTH])

    return [X, y]

def predict(pipeline, startRow) :
    predictions = []
    predictions.extend(pipeline.predict(np.array(startRow).reshape(1, -1)))
    
    for sample in range(0, NUM_SAMPLES) :
        for sampleRow in range(0, SAMPLE_LENGTH) :
            print(predictions[-1])
            predictions.extend(pipeline.predict(np.array(predictions[-1]).reshape(1, -1)))

    return predictions

def calculateRMSE(realValues, predictedValues) :
    body1Coords = bodyCoords(1, realValues)
    body2Coords = bodyCoords(2, realValues)
    body3Coords = bodyCoords(3, realValues)

    body1CoordsPred = bodyCoords(1, predictedValues)
    body2CoordsPred = bodyCoords(2, predictedValues)
    body3CoordsPred = bodyCoords(3, predictedValues)

    body1RMSE = math.sqrt(mean_squared_error(body1Coords, body1CoordsPred))
    body2RMSE = math.sqrt(mean_squared_error(body2Coords, body2CoordsPred))
    body3RMSE = math.sqrt(mean_squared_error(body3Coords, body3CoordsPred))

    return [body1RMSE, body2RMSE, body3RMSE]

X, y = createXy(NUM_SAMPLES)

for polynomialDegree in range(4, 10) :
    pipeline = make_pipeline(PolynomialFeatures(polynomialDegree), LinearRegression())
    pipeline.fit(X, y)
    predictions = predict(pipeline, X[0])
    body1RMSE, body2RMSE, body3RMSE = calculateRMSE(y, predictions)

    print(polynomialDegree)
    print(body1RMSE)
    print(body2RMSE)
    print(body3RMSE)

