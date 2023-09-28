import matplotlib.pyplot as plt
import numpy as np

# Function
def f(x): 
    return x**3 - 1/x

# Function derivative
def fDerivative(x): 
    return 3*(x**2) + 1/(x**2)

# Tangent line (y = m*(x - x1) + y1)
def tangentLine(xRange, x):
    return fDerivative(x)*(xRange - x) + f(x)

# Tangent line range
def xRangeTangent(x) :
    return np.linspace(x-1, x+1);

def plotTangentLine(x) : 
    plt.scatter(x, f(x), color='r', s=25)
    plt.plot(xRangeTangent(x), tangentLine(xRangeTangent(x), x), 'r--')

# x axis range
xRange = np.linspace(0.1, 3)

# Plot f
plt.plot(xRange, f(xRange))

# Plot tangent lines
plotTangentLine(1)
plotTangentLine(2)

plt.show()