import numpy as np
import matplotlib.pyplot as plt


def predict_(x, theta):
    if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True):
        x = np.c_[ np.ones(len(x)) , x]
        y = x.dot(theta)
        return y
    return None

def gradient(x, y, theta):
    if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        theta = np.squeeze(theta)
        y = np.squeeze(y)
        if (x.shape[0] != 0 and y.shape[0] != 0 and theta.shape == (x.shape[1] + 1,) and y.shape[0] == x.shape[0] and y.ndim == 1):
            xt = np.c_[ np.ones(len(x)) , x]
            r = (xt.T.dot(predict_(x, theta) - y)) / len(x)
            return r
    return None

if __name__ == '__main__':
    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([0, 3,0.5,-6]).reshape((-1, 1))
    # Example :
    print(gradient(x, y, theta1))
    # Output:
    #array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])
    # Example :
    theta2 = np.array([0,0,0,0]).reshape((-1, 1))
    print(gradient(x, y, theta2))
    # Output:
    #array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]]) 
