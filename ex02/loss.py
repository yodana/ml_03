import numpy as np
import matplotlib.pyplot as plt

def loss_(y, y_hat):
    j = 0
    if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        y = np.squeeze(y)
        y_hat = np.squeeze(y_hat)
        if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
            m = y_hat - y
            j = m.dot(m)
            return float(j / (2*len(y)))
    return None

if __name__ == '__main__':
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    # Example 1:
    print(loss_(X, Y))
    # Output:
    # Example 2:
    print(loss_(X, X))    
