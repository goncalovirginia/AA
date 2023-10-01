import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

lines = np.loadtxt("lab2/aa.ssdi.di.fct.unl.pt_files_bluegills.txt", skiprows=1)

# Shuffles and splits data into Training (1/2), Validation (1/4) and Testing (1/4) sets
# Custom version of: train_test_split(array)
def shuffleAndSplitData(array) :
    permutation = np.random.permutation(array)
    split = np.array_split(permutation, 2)
    split2 = np.array_split(split[1], 2)
    return [split[0], split2[0], split2[1]]

def getArrayColumn(array, column) :
    return np.array([x[column] for x in array])

trainingData, validationData, testData = shuffleAndSplitData(lines)

### Standardize data (z-score) ###

scaler = StandardScaler()

# Calculate mean and std based on the training data
# agesMean, lengthsMean = np.mean(lines, axis=0)
# agesStd, lengthsStd = np.std(lines, axis=0)
scaler.fit(trainingData)

# Transform the original data into their corresponding z-scores (based on the training mean)
trainingDataZScores = scaler.transform(trainingData)
validationDataZScores = scaler.transform(validationData)
testDataZScores = scaler.transform(testData)

# Separate ages and lengths columns into their own arrays
trainingAgesZScore = getArrayColumn(trainingDataZScores, 0)
trainingLengthsZScore = getArrayColumn(trainingDataZScores, 1)
validationAgesZScore = getArrayColumn(validationDataZScores, 0)
validationLengthsZScore = getArrayColumn(validationDataZScores, 1)
testAgesZScore = getArrayColumn(testDataZScores, 0)
testLengthsZScore = getArrayColumn(testDataZScores, 1)

# Build graph with training, validation and test data points
plt.figure("Blue gill size")
plt.xlabel('Age')
plt.ylabel('Length')
plt.scatter(trainingAgesZScore, trainingLengthsZScore, s=5, c="b")
plt.scatter(validationAgesZScore, validationLengthsZScore, s=1, c="r")
plt.scatter(testAgesZScore, testLengthsZScore, s=1, c="g")

# Create evenly spaced numbers on the x-axis (age) for smooth plotting
xSeq = np.linspace(trainingAgesZScore.min(), trainingAgesZScore.max(), 1000).reshape(-1, 1)

bestPolynomialDegree = 1
lowestValidationError = math.inf

# Create and plot polynomial (regression) curves from degrees 1 to 6, calculating their training and validation errors
for polynomialDegree in range(1, 7) :
    coefficients = np.polyfit(trainingAgesZScore.flatten(), trainingLengthsZScore.flatten(), polynomialDegree)

    trainingError = mean_squared_error(trainingLengthsZScore, np.polyval(coefficients, trainingAgesZScore))
    validationError = mean_squared_error(validationLengthsZScore, np.polyval(coefficients, validationAgesZScore))

    plt.plot(xSeq, np.polyval(coefficients, xSeq), linewidth=0.75, label="{} / {} / {}".format(polynomialDegree, round(trainingError, 4), round(validationError, 4)))

    if validationError < lowestValidationError :
        lowestValidationError = validationError
        bestPolynomialDegree = polynomialDegree

print("Best polynomial degree for regression: {}".format(bestPolynomialDegree))
print("Validation error: {}".format(lowestValidationError))
testError = mean_squared_error(testLengthsZScore, np.polyval(coefficients, testAgesZScore))
print("Test error: {}".format(testError))

plt.legend(loc="lower right")
plt.show()
