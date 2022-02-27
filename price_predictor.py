import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = y.shape[0]
    J = 0
    diffs = [np.dot(np.transpose(theta), X[i]) - y[i] for i in range(0,m)]
    squared_diffs = [np.power(diffs, 2)]
    J = 1/(2*m) * np.sum(squared_diffs)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    theta_buffer = np.zeros(theta.size)
    for i in range(0, iterations):
        for j in range(0, theta.size):
            diffs = [(np.dot(np.transpose(theta), X[i]) - y[i]) * X[i][j] for i in range(0, m)]
            theta_buffer[j] = theta[j] - alpha * 1/m * np.sum(diffs)
        theta = theta_buffer
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

def featureNormalization(X):
    X = X.copy()
    m = X.shape[0]
    n = X.shape[1]
    X = np.transpose(X)
    means = []
    ranges = []
    for i in range(0, n):
        mean = np.mean(X[i])
        means.append(mean)
        rangeX = np.max(X[i]) - np.min(X[i])
        ranges.append(rangeX)
        for j in range(0, m):
            X[i][j] = (X[i][j] - mean) / rangeX
    return np.transpose(X), means, ranges

def predictionNormalization(prediction, means, ranges):
    norm_prediction = []
    for i in range(0, len(means)):
        norm_prediction.append((prediction[i] - means[i])/ranges[i])
    norm_prediction.insert(0, 1)
    return norm_prediction

def plotData(x, y, x_label, y_label):
    pyplot.figure()  # open a new figure
    pyplot.plot(x, y, 'ro', ms=2, mec='k')
    pyplot.ylabel(y_label)
    pyplot.xlabel(x_label)
    plt.show()

#Initializing the data 
data = np.loadtxt('housing_halifax.txt', delimiter=',')
X = data[:, :3]
y = data[:, 3]
alpha = 0.03
#Initializing theta -> 
theta = np.zeros(4)

#normalizing the values of the features
X, means, ranges = featureNormalization(X)

X = np.column_stack((np.ones(y.shape[0]), X))
theta, J_history = gradientDescent(X, y, theta, alpha, 10000)
plotData([i for i in range(0, 10000)], J_history, 'iterations', 'cost')

prediction1 = np.dot(predictionNormalization([4,4,5324], means, ranges), theta)
print(f'4 bedrooms, 4 bathrooms, 5324 sqft price prediction = {prediction1}')