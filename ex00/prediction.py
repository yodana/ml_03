import numpy as np

def simple_predict(x, theta):
    if (isinstance(x, np.ndarray) == True or isinstance(theta, np.ndarray) == True):
        theta = np.squeeze(theta)
        y = []
        if (x.shape[0] != 0 and theta.shape == (x.shape[1] + 1,)):
            for X in x:
                y_hat = theta[0]
                for i in range(0, len(X)):
                    y_hat = y_hat + X[i]*theta[i + 1]
                y.append(y_hat)
            return np.array(y)
    return None


if __name__ == '__main__':
    x = np.arange(1,13).reshape((4,-1))
    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta1))
    # Do you understand why y_hat contains only 5’s here?
    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta2))
    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(simple_predict(x, theta3))
    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(simple_predict(x, theta4))