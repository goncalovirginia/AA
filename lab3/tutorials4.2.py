import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lines = np.loadtxt("lab3/aa.ssdi.di.fct.unl.pt_files_yield.txt", skiprows=1)

# Shuffles and splits data into Training (1/2), Validation (1/4) and Testing (1/4) sets
# Custom version of sklearn.model_selection.train_test_split(array)
def shuffleAndSplitData(array) :
    permutation = np.random.permutation(array)
    split = np.array_split(permutation, 2)
    split2 = np.array_split(split[1], 2)
    return [split[0], split2[0], split2[1]]

def separateToXy(array) :
    return [np.array([x[0] for x in array]).reshape(-1, 1), np.array([x[1] for x in array])]

trainingData, validationData, testingData = shuffleAndSplitData(lines)

trainingTemp, trainingYield = separateToXy(trainingData)
validationTemp, validationYield = separateToXy(validationData)
testingTemp, testingYield = separateToXy(testingData)

plt.figure("Yield from Temperature")
plt.xlabel('Temperature')
plt.ylabel('Yield')
plt.scatter(trainingTemp, trainingYield, s=5, c="b", label="Training Temps")
plt.scatter(validationTemp, validationYield, s=1, c="r", label="Validation Temps")
plt.scatter(testingTemp, testingYield, s=1, c="g", label="Testing Temps")

allTemp, allYield = separateToXy(lines)
xSeq = np.linspace(allTemp.min(), allTemp.max(), 1000).reshape(-1, 1)

for polynomialDegree in range(1, 7) :
    pipeline = make_pipeline(PolynomialFeatures(polynomialDegree), LinearRegression())
    pipeline.fit(trainingTemp, trainingYield)

    trainingError = mean_squared_error(trainingYield, pipeline.predict(trainingTemp))
    validationError = mean_squared_error(validationYield, pipeline.predict(validationTemp))
    testingError = mean_squared_error(testingYield, pipeline.predict(testingTemp))

    plt.plot(xSeq, pipeline.predict(xSeq), linewidth=0.75, label="{} / {} / {} / {}".format(polynomialDegree, round(trainingError, 4), round(validationError, 4), round(testingError, 4)))

plt.legend(loc="best")
plt.show()