import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

LAST_ROW = 1285000
NUM_SAMPLES = 5000
SAMPLE_LENGTH = 257

X_train_dataframe = pd.read_csv("project1/X_train3.csv")

def plotSample(startRow) :
    coords = X_train_dataframe[['x_1', 'y_1', 'x_2' , 'y_2', 'x_3', 'y_3']][startRow : startRow + SAMPLE_LENGTH];

    plt.figure("Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coords['x_1'], coords['y_1'], label='1')
    plt.plot(coords['x_2'], coords['y_2'], label='2')
    plt.plot(coords['x_3'], coords['y_3'], label='3')
    plt.show()

def plotAllSamples() :
    for startRow in range(0, LAST_ROW, SAMPLE_LENGTH) :
        plotSample(startRow)

def create_X_train_formatted(X, y) :
    #data_x2 = data_x.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
    columns = y.drop('t', axis=1).columns
    X['t'] = y['t']
    X.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
    X.to_csv("data_x.csv")
    for j in range(1,257):
        X.iloc[j] = y.iloc[0]

# Predict with X_test.csv and generate the submission csv file
def predictAndGenerateSubmissionCsv(filename) :
    X_test_dataframe = pd.read_csv("project1/X_test.csv").drop(columns=['Id'])
    X_test_dataframe = X_test_dataframe.rename(columns={'t': 't', 'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3'})

    for i in range(1,4):
        X_test_dataframe['v_x_'+ str(i)] = 0.0
        X_test_dataframe['v_y_'+ str(i)] = 0.0
        X_test_dataframe['a_x_'+ str(i)] = 0.0
        X_test_dataframe['a_y_'+ str(i)] = 0.0
    
    X_test_dataframe = X_test_dataframe[X.columns]

    y_predicted = pipeline.predict(X_test_dataframe)

    # Format output file
    y_predicted_dataframe = pd.DataFrame(y_predicted, columns = X.columns)
    y_predicted_dataframe = pd.DataFrame(y_predicted_dataframe, columns = ['Id', 'x_1', 'y_1', 'x_2', 'y_2','x_3', 'y_3'])
    y_predicted_dataframe['Id'] = X_test_dataframe.index

    y_predicted_dataframe.to_csv('project1/{}.csv'.format(filename), index=False)

def createAccelerationColumns() :
    timeAndVelocities = X_train_dataframe.drop(columns=['x_1', 'y_1', 'x_2', 'y_2','x_3', 'y_3', 'Id']).to_numpy()
    accelerationsRows = []
    
    for sample in range(0, NUM_SAMPLES) :
        for row in range(0, SAMPLE_LENGTH-1) :
            currRow = timeAndVelocities[sample*SAMPLE_LENGTH + row]
            nextRow = timeAndVelocities[sample*SAMPLE_LENGTH + row + 1]
            accelerationsRow = []
            deltaTime = nextRow[0] - currRow[0]
            if deltaTime == 0 :
                deltaTime = 1
            
            for column in range(1, 7) :
                accelerationsRow.append((nextRow[column] - currRow[column])/deltaTime)

            accelerationsRows.append(accelerationsRow)
        
        accelerationsRows.append(accelerationsRows[-1])
        
    return pd.DataFrame(accelerationsRows, columns=['a_x_1', 'a_y_1', 'a_x_2', 'a_y_2', 'a_x_3', 'a_y_3'])
    
def createRelativeDistanceColumns(X) :
    coords = X[['x_1', 'y_1', 'x_2', 'y_2','x_3', 'y_3']].to_numpy()
    relativeDistanceRows = []
    
    for row in coords :
        d_1_2 = math.dist([row[0], row[1]], [row[2], row[3]])
        d_1_3 = math.dist([row[0], row[1]], [row[4], row[5]])
        d_2_3 = math.dist([row[2], row[3]], [row[4], row[5]])
        relativeDistanceRows.append([d_1_2, d_1_3, d_2_3])

    return pd.DataFrame(relativeDistanceRows, columns=['d_1_2', 'd_1_3', 'd_2_3'])

def format_X_with_accelerations(X):
    for i in range(1,4):
        X[f'a_x_{i}'] = 0.0
        X[f'a_y_{i}'] = 0.0

    formatted_columns = ['t', 'x_1', 'y_1', 'v_x_1', 'v_y_1', 'a_x_1', 'a_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'a_x_2', 'a_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3', 'a_x_3', 'a_y_3']
    return X[formatted_columns]

# Create train and test splits
X = pd.read_csv("project1/X_train_formatted3.csv")
y = pd.read_csv("project1/X_train3.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)

# Linear Regression

print("Linear Regression:")
pipeline = make_pipeline(PolynomialFeatures(4), LinearRegression())            
pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)
y_df_predicted = pd.DataFrame(y_predicted, columns=['t', 'x_1', 'y_1', 'v_x_1', 'v_y_1', 'a_x_1', 'a_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'a_x_2', 'a_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3', 'a_x_3', 'a_y_3'])
y_predicted = y_df_predicted.drop(columns=['v_x_1', 'v_y_1', 'a_x_1', 'a_y_1', 'v_x_2', 'v_y_2', 'a_x_2', 'a_y_2', 'v_x_3', 'v_y_3', 'a_x_3', 'a_y_3'])
y_test = y_test.drop(columns=['v_x_1', 'v_y_1', 'a_x_1', 'a_y_1', 'v_x_2', 'v_y_2', 'a_x_2', 'a_y_2', 'v_x_3', 'v_y_3', 'a_x_3', 'a_y_3'])
print("RMSE: {}".format(math.sqrt(mean_squared_error(y_test, y_predicted))))

#predictAndGenerateSubmissionCsv("AccelerationsTrainSize25LinearRegressionDegree4")
