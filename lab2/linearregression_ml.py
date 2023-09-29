# Starter colab file is located at https://colab.research.google.com/drive/1nEYXLGoZ7-e2nnMcecdslh_UsfnH_CyG?usp=sharing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

numSamples = 100
X, y = make_regression(n_samples=numSamples, n_features=1, noise=15.0)

# Manual linear regression

# New matrix [1 X]
Xnew = np.hstack((np.ones((numSamples, 1)), X))

# Calculate regression weights
# a = (Xnew^T * Xnew)^−1 * Xnew^T * y
# a[0] = offset, a[1] = slope
a = np.linalg.pinv(Xnew.T @ Xnew) @ Xnew.T @ y

# Plot calculated linear regression
plt.figure("Manual linear regression")
plt.scatter(X, y, color='b', s=5)
plt.xlabel('X')
plt.ylabel('y')
plt.plot([X.min(), X.max()], [a[1]*X.min() + a[0], a[1]*X.max() + a[0]], color='r')

# Scikit-learn linear regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X, y)

plt.figure("Scikit-learn linear regression")
plt.scatter(X, y, color='b', s=5)
plt.xlabel('X')
plt.ylabel('y')
plt.plot([X.min(), X.max()], [reg.coef_*X.min() + reg.intercept_, reg.coef_*X.max() + reg.intercept_], color='r')

# Predict y from learned regression
yPrediction = reg.predict(X)
reg.fit(X, yPrediction)

plt.figure("Scikit-learn predicted y")
plt.scatter(X, yPrediction, color='b', s=5)
plt.xlabel('X')
plt.ylabel('y')
plt.plot([X.min(), X.max()], [reg.coef_*X.min() + reg.intercept_, reg.coef_*X.max() + reg.intercept_], color='r')

plt.show()