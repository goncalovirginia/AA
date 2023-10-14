""" 
This model is trained on the idea of predicting line n+1 with line n of each sample.
It then tries to predict the new positions and velocities by recursively using its own output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MaxAbsScaler
import math

np.seterr(divide='ignore', invalid='ignore')

LAST_ROW = 1285000
NUM_SAMPLES = 5000
SAMPLE_LENGTH = 257

dataFrame = pd.read_csv("project1/X_train_accelerations_pairdistances.csv").drop(columns=['t', 'Id'])

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

def accelerations(p1, p2, p3):
	a_1 = - (p1 - p2)/(math.dist(p1, p2)**3) - (p1 - p3)/(math.dist(p1, p3)**3)
	a_2 = - (p2 - p1)/(math.dist(p2, p1)**3) - (p2 - p3)/(math.dist(p2, p3)**3)
	a_3 = - (p3 - p1)/(math.dist(p3, p1)**3) - (p3 - p2)/(math.dist(p3, p2)**3)
	return [a_1[0], a_1[1], a_2[0], a_2[1], a_3[0], a_3[1]]

def rowAccelerations(row) :
    return accelerations(np.array([row['x_1'], row['y_1']]), np.array([row['x_2'], row['y_2']]), np.array([row['x_3'], row['y_3']]))

def addAccelerationsFeatures(X) :
    X[['a_x_1', 'a_y_1', 'a_x_2', 'a_y_2', 'a_x_3', 'a_y_3']] = X.apply(rowAccelerations, axis=1, result_type='expand').fillna(0.0)
    return X

def rowPairDistances(row) :
    coords1 = [row['x_1'], row['y_1']]
    coords2 = [row['x_2'], row['y_2']]
    coords3 = [row['x_3'], row['y_3']]
    return [math.dist(coords1, coords2), math.dist(coords1, coords3), math.dist(coords2, coords3)]

def addPairDistancesFeatures(X) :
    X[['d_1_2', 'd_1_3', 'd_2_3']] = X.apply(rowPairDistances, axis=1, result_type='expand')
    return X

X, y = createXy(NUM_SAMPLES)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

print("Linear Regression:")
pipeline = make_pipeline(#FunctionTransformer(addAccelerationsFeatures), FunctionTransformer(addPairDistancesFeatures), 
    PolynomialFeatures(3), LinearRegression())
pipeline.fit(X_train, y_train)
y_predicted = pd.DataFrame(pipeline.predict(X_test), columns=dataFrame.columns)
print("RMSE: {}".format(math.sqrt(mean_squared_error(y_test[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']], y_predicted[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]))))

#predictRecursively(pipeline, X.iloc[0])

print("Acceleration MaxAbsScaler Linear Regression:")
pipeline = make_pipeline(#FunctionTransformer(addAccelerationsFeatures), FunctionTransformer(addPairDistancesFeatures), 
    make_column_transformer((MaxAbsScaler(), ['a_x_1', 'a_y_1', 'a_x_2', 'a_y_2', 'a_x_3', 'a_y_3']), remainder='passthrough'), 
    PolynomialFeatures(3), LinearRegression())
pipeline.fit(X_train, y_train)
y_predicted = pd.DataFrame(pipeline.predict(X_test), columns=dataFrame.columns)
print("RMSE: {}".format(math.sqrt(mean_squared_error(y_test[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']], y_predicted[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]))))