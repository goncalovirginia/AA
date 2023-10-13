""" 
This model is trained on the idea of predicting line n+1 with line n of each sample.
It then tries to predict the new positions and velocities by recursively using its own output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MaxAbsScaler
import math

LAST_ROW = 1285000
NUM_SAMPLES = 5000
SAMPLE_LENGTH = 257

dataFrame = pd.read_csv("project1/X_train3.csv").drop(columns=['t'])

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

    return [pd.DataFrame(X, columns=dataFrame.columns), pd.DataFrame(y, columns=dataFrame.columns)]

def predictRecursively(pipeline, startRow) :
    predictions = []
    predictions.extend(pipeline.predict(np.array(startRow)))
    
    for sample in range(0, NUM_SAMPLES) :
        for sampleRow in range(0, SAMPLE_LENGTH) :
            print(predictions[-1])
            predictions.extend(pipeline.predict(np.array(predictions[-1])))

    return predictions

X, y = createXy(NUM_SAMPLES)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Linear Regression:")
pipeline = make_pipeline(PolynomialFeatures(3), LinearRegression())
pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)
y_predicted = pd.DataFrame(y_predicted, columns=dataFrame.columns)
y_predicted = y_predicted[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
print("RMSE: {}".format(math.sqrt(mean_squared_error(y_test[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']], y_predicted))))

print("Acceleration MaxAbsScaler Linear Regression:")
pipeline = make_pipeline(make_column_transformer((MaxAbsScaler(), ['a_x_1', 'a_y_1', 'a_x_2', 'a_y_2', 'a_x_3', 'a_y_3']), remainder='passthrough'), PolynomialFeatures(3), LinearRegression())
pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)
y_predicted = pd.DataFrame(y_predicted, columns=dataFrame.columns)
y_predicted = y_predicted[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
print("RMSE: {}".format(math.sqrt(mean_squared_error(y_test[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']], y_predicted))))