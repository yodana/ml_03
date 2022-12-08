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

def fit_(x, y, theta, alpha, max_iter):
    if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True or isinstance(y, np.ndarray) == True or isinstance(alpha, float) == True or isinstance(max_iter, int) == True):
        theta = np.squeeze(theta)
        x = np.squeeze(x)
        y = np.squeeze(y)
        if (x.shape[0] != 0 and y.shape[0] != 0 and theta.shape == (x.shape[1] + 1,) and y.shape[0] == x.shape[0] and y.ndim == 1):
            while(max_iter != 0):
                g = gradient(x, y, theta)
                theta = theta - alpha*g
                max_iter = max_iter - 1
            return theta
    return None

if __name__ == '__main__':
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, 0.0005, 42000)
    print(theta2)
    # Output:
    #array([[41.99..],[0.97..], [0.77..], [-1.20..]])
    # Example 1:
    print(predict_(x, theta2))
    # Output:
    #array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])