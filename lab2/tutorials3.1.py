import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

data = np.loadtxt("lab2/planets.csv", delimiter=",", skiprows=1, usecols=range(1,3))
distanceAU = np.vstack([x[0] for x in data])
years = np.array([x[1] for x in data])

polynomialDegree = 2
xSeq = np.linspace(distanceAU.min(), distanceAU.max(), 100).reshape(-1, 1)

# Scikit-Learn model

polyreg = make_pipeline(PolynomialFeatures(polynomialDegree), LinearRegression())
polyreg.fit(distanceAU, years)

plt.figure("Scikit-Learn model")
plt.xlabel('Distance (AU)')
plt.ylabel('Time (Earth Years)')
plt.scatter(distanceAU, years, color="r")
plt.plot(xSeq, polyreg.predict(xSeq), color="b")

# Numpy model

coefs = np.polyfit(distanceAU.flatten(), years.flatten(), polynomialDegree)

plt.figure("Numpy model")
plt.xlabel('Distance (AU)')
plt.ylabel('Time (Earth Years)')
plt.scatter(distanceAU, years, color="r")
plt.plot(xSeq, np.polyval(coefs, xSeq), color="b")

plt.show()

