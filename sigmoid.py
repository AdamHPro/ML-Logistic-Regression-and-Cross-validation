import numpy as np

def sigmoid(z):
    # Computes the sigmoid of z.
    g = 1/(1+np.exp(-z))
  
    return g
