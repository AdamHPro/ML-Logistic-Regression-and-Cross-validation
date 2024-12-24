import numpy as np
from sigmoid import sigmoid

def computeCost(theta, X, y): 
    # Computes the cost using theta as the parameter 
    # for logistic regression. 
    m = X.shape[0] # number of training examples
    
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment 
    #               for more details).
    J = 0
    
    Xtheta = np.matmul(X, theta)
    for i in range(m) :
        J += y[i]*np.log(sigmoid(Xtheta[i])) + (1-y[i])*np.log(1-sigmoid(Xtheta[i]))
    J = -J/m
        
        
    
    
    

    
    
    
    # =============================================================
    return J
