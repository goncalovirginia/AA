import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import onnxruntime as rt
from skl2onnx import to_onnx as sklearn_to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

LAST_ROW = 1285000
NUM_SAMPLES = 5000
SAMPLE_LENGTH = 257

def plotSample(startRow) :
    coords = X[['x_1', 'y_1', 'x_2' , 'y_2', 'x_3', 'y_3']][startRow : startRow + SAMPLE_LENGTH];

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

def format_X_Train_for_input(X_train):
    X1 = X_train.copy()
    for i in range(0, NUM_SAMPLES):
        for j in range(0, SAMPLE_LENGTH):
            X1.iloc[[i*SAMPLE_LENGTH + j]] = X_train.iloc[[i*SAMPLE_LENGTH]]
    X1['t'] = X_train['t']
    return X1

def addFeatureEngineering(df) :
    df = addAccelerationsFeatures(df)
    df = addPairDistancesFeatures(df)
    df = addPairPotentialsFeatures(df)
    return df

# Predict with X_test.csv and generate the submission csv file
def predictAndGenerateSubmissionCsv(filename) :
    X_test_dataframe = pd.read_csv("project1/X_test.csv").drop(columns=['Id'])
    X_test_dataframe = X_test_dataframe.rename(columns={'t': 't', 'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3'})

    for i in range(1,4):
        X_test_dataframe['v_x_'+ str(i)] = 0.0
        X_test_dataframe['v_y_'+ str(i)] = 0.0

    X_test_dataframe = addAccelerationsFeatures(X_test_dataframe)
    X_test_dataframe = addPairDistancesFeatures(X_test_dataframe)
    X_test_dataframe = addPairPotentialsFeatures(X_test_dataframe)
    X_test_dataframe = X_test_dataframe[X.columns.tolist()]

    y_predicted = pipeline.predict(X_test_dataframe)

    # Format output file
    y_predicted_dataframe = pd.DataFrame(y_predicted, columns = X.columns)
    y_predicted_dataframe = pd.DataFrame(y_predicted_dataframe, columns = ['Id', 'x_1', 'y_1', 'x_2', 'y_2','x_3', 'y_3'])
    y_predicted_dataframe['Id'] = X_test_dataframe.index

    y_predicted_dataframe.to_csv('project1/{}.csv'.format(filename), index=False)

def safeDiv(n, d) :
    return n / d if d else np.array([-0.0, -0.0])

def accelerations(p1, p2, p3):
	a_1 = - safeDiv(p1 - p2, math.dist(p1, p2)**3) - safeDiv(p1 - p3, math.dist(p1, p3)**3)
	a_2 = - safeDiv(p2 - p1, math.dist(p2, p1)**3) - safeDiv(p2 - p3, math.dist(p2, p3)**3)
	a_3 = - safeDiv(p3 - p1, math.dist(p3, p1)**3) - safeDiv(p3 - p2, math.dist(p3, p2)**3)
	return [a_1[0], a_1[1], a_2[0], a_2[1], a_3[0], a_3[1]]

def rowAccelerations(row) :
    return accelerations(np.array([row['x_1'], row['y_1']]), np.array([row['x_2'], row['y_2']]), np.array([row['x_3'], row['y_3']]))

def addAccelerationsFeatures(X) :
    X[['a_x_1', 'a_y_1', 'a_x_2', 'a_y_2', 'a_x_3', 'a_y_3']] = X.apply(rowAccelerations, axis=1, result_type='expand')
    return X

def rowPairDistances(row) :
    coords1 = [row['x_1'], row['y_1']]
    coords2 = [row['x_2'], row['y_2']]
    coords3 = [row['x_3'], row['y_3']]
    return [math.dist(coords1, coords2), math.dist(coords1, coords3), math.dist(coords2, coords3)]

def addPairDistancesFeatures(X) :
    X[['d_1_2', 'd_1_3', 'd_2_3']] = X.apply(rowPairDistances, axis=1, result_type='expand')
    return X

def safeDiv2(n, d) :
    return n / d if d else 0.0

def rowPairPotentials(row) :
    d_1_2, d_1_3, d_2_3 = rowPairDistances(row)
    return [safeDiv2(1, d_1_2**2), safeDiv2(1, d_1_3**2), safeDiv2(1, d_2_3**2)]

def addPairPotentialsFeatures(X) :
    X[['p_1_2', 'p_1_3', 'p_2_3']] = X.apply(rowPairPotentials, axis=1, result_type='expand')
    return X

def toONNX(pipeline, name) :
    initial_type = [('numfeat', FloatTensorType([None, 25]))]
    model_onnx = sklearn_to_onnx(pipeline, initial_types=initial_type)

    with open("model_acc_ip_pd_MAbsS_d3.onnx", "wb") as f:
        f.write(model_onnx.SerializeToString())

    sess = rt.InferenceSession("{}.onnx".format(name))#, providers=["CPUExecutionProvider"])

    input_data = np.array(X_test, dtype=np.float32)
    input_data = {'numfeat': input_data}
    label_name = sess.get_outputs()[0].name
    output = sess.run([label_name], input_data)
    pred_onx = pd.DataFrame(output[0], columns=y.columns)

    print("\nY:\n  {}".format(y_test))
    print("\nskl_predict:\n  {}".format(y_predicted))
    print("\nomnx_predict:\n {}".format(pred_onx))

"""
# Formatting for original X_train.csv file to work with model
X_train = pd.read_csv("project1/X_train.csv").drop(columns=['Id'])
X = format_X_Train_for_input(X_train)
y = addFeatureEngineering(X_train)
"""

# Custom files used because FunctionTransformers take too long
X = pd.read_csv("project1/X_train_a_d_p.csv")
y = pd.read_csv("project1/y_train_a_d_p.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

print("Poly3 MaxAbsScaler Linear:")
pipeline = make_pipeline(#FunctionTransformer(addAccelerationsFeatures), FunctionTransformer(addPairDistancesFeatures), FunctionTransformer(addPairPotentialsFeatures),
    PolynomialFeatures(3), MaxAbsScaler(), LinearRegression()) 
pipeline.fit(X_train, y_train)
y_predicted = pd.DataFrame(pipeline.predict(X_test), columns=y.columns)
rmse = math.sqrt(mean_squared_error(y_test[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']], y_predicted[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]))
print("RMSE: {}".format(rmse))

name = 'poly3_maxabs_linear_{}'.format(rmse)
predictAndGenerateSubmissionCsv(name)
toONNX(pipeline, name)

"""
print("StandardScaler KNN Regression:")
pipeline = make_pipeline(#FunctionTransformer(addAccelerationsFeatures), FunctionTransformer(addPairDistancesFeatures), FunctionTransformer(addInteractionPotentialsFeatures),
    StandardScaler(), KNeighborsRegressor()) 
pipeline.fit(X_train, y_train)
y_predicted = pd.DataFrame(pipeline.predict(X_test), columns=y.columns)
print("RMSE: {}".format(math.sqrt(mean_squared_error(y_test[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']], y_predicted[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]))))
"""


