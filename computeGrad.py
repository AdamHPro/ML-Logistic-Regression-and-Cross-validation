import numpy as np
from sigmoid import sigmoid

def computeGrad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.
    m = X.shape[0] # number of training examples
    grad = np.zeros(theta.shape) # initialize gradient

    Xtheta = np.matmul(X, theta)
    for j in range(theta.shape[0]) :
        delta = 0
        for i in range(m) :
            delta += (sigmoid(Xtheta[i]) - y[i])*X[i][j]
        grad[j] = delta/m

    return grad
