import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    # Predict whether the label is 0 or 1 using learned logistic 
    # regression parameters theta. The threshold is set at 0.5
    m = X.shape[0] # number of test examples
    c = np.zeros(m) # predicted classes of training examples
    p = np.zeros(m) # logistic regression outputs of training examples
    
    Xtheta = np.matmul(X, theta)
    
    for i in range(m) :
        p[i] = sigmoid(Xtheta[i])
        if p[i] <= 0.5 :
            c[i] = 0
        else :
            c[i] = 1
    
    
    
    
    
    
    
    
    # =============================================================
    return c

