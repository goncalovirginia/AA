import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def getArrayColumn(array, column) :
    return np.array([x[column] for x in array])

lines = np.loadtxt("lab2/aa.ssdi.di.fct.unl.pt_files_bluegills.txt", skiprows=1)

ages = getArrayColumn(lines, 0).reshape(-1, 1)
lengths = getArrayColumn(lines, 1)
trainingAges, testingAges, trainingLengths, testingLengths = train_test_split(ages, lengths)

plt.figure("Blue gill size (Error vs Lambda)")
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.xlim(0, 10)

lambdaVals = np.geomspace(start=0.001, stop=10, num=100)
errors = []

for lambdaVal in lambdaVals :
    pipeline = make_pipeline(PolynomialFeatures(5), Ridge(alpha=lambdaVal))
    pipeline.fit(trainingAges, trainingLengths)

    testingError = mean_squared_error(testingLengths, pipeline.predict(testingAges))

    errors.append(testingError)

plt.scatter(lambdaVals, errors, s=1)
plt.show()