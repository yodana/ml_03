import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyLinearRegression():
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
    

    def gradient(self, x, y):
        if (isinstance(x, np.ndarray) == True or isinstance(self.theta, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            self.theta = np.squeeze(self.theta)
            y = np.squeeze(y)
            if (x.shape[0] != 0 and y.shape[0] != 0 and self.theta.shape == (x.shape[1] + 1,) and y.shape[0] == x.shape[0] and y.ndim == 1):
                xt = np.c_[ np.ones(len(x)) , x]
                r = (xt.T.dot(self.predict_(x) - y)) / len(x)
                return r
        return None

    def mse_(self, y, y_hat):
        j = 0
        if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            y = np.squeeze(y)
            y_hat = np.squeeze(y_hat)
            if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
                m = y_hat - y
                j = m.dot(m)
                return float(j / (len(y)))
        return None
    
    def fit_(self, x, y):
        save = self.max_iter
        if (isinstance(x, np.ndarray) == True or isinstance(self.theta, np.ndarray) == True or isinstance(y, np.ndarray) == True or isinstance(self.alpha, float) == True or isinstance(self.max_iter, int) == True):
            self.theta = np.squeeze(self.theta)
            y = np.squeeze(y)
            if (x.shape[0] != 0 and y.shape[0] != 0 and self.theta.shape == (x.shape[1] + 1,) and y.shape[0] == x.shape[0] and y.ndim == 1):
                while(self.max_iter != 0):
                    g = self.gradient(x, y)
                    self.theta = self.theta - self.alpha*g
                    self.max_iter = self.max_iter - 1
                self.max_iter = save
                return self.theta
        return None
    
    def predict_(self, x):
        if (isinstance(x, np.ndarray) == True or isinstance(self.theta, np.ndarray) == True):
            x = np.c_[ np.ones(len(x)) , x]
            y = x.dot(self.theta)
            return y
        return None
    
    def loss_elem_(self, y, y_hat):
        j = []
        if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
                for i in range(0, len(y)):
                    j.append((y_hat[i] - y[i])**2)
                return np.array(j)
        return None

    def loss_(self, y, y_hat):
        j = 0
        if (isinstance(y_hat, np.ndarray) == True or isinstance(y, np.ndarray) == True):
            y = np.squeeze(y)
            y_hat = np.squeeze(y_hat)
            if (y.ndim == 1 and y_hat.ndim == 1 and len(y) == len(y_hat)):
                m = y_hat - y
                j = m.dot(m)
                return float(j / (2*len(y)))
        return None

def add_polynomial_features(x, power):
    j = 0
    if (isinstance(x, np.ndarray) == True and isinstance(power, int)):
        x = np.squeeze(x)
        if (x.ndim == 1 and len(x) != 0):
            r = []
            X = []
            for e in x:
                for i in range(1, power+1):
                    X.append(e**i)
                r.append(X)
                X = []
            return np.array(r)
    return None

if __name__ == '__main__':
    data = pd.read_csv("are_blue_pills_magics.csv")
    X = np.array(data[["Micrograms"]])
    Y = np.array(data[["Score"]])
    y_pred = []
    mse = []
    X1 = add_polynomial_features(X, 1)
    theta1 = np.array([[-1],[ 2]]).reshape(-1,1)
    lgr = MyLinearRegression(theta1, 0.001, 10000)
    lgr.fit_(X1, Y)
    y_hat1 = lgr.predict_(X1)
    y_pred.append(y_hat1)
    print(lgr.mse_(Y, y_hat1))
    mse.append(lgr.mse_(Y, y_hat1))
    X1 = add_polynomial_features(X, 2)
    theta1 = np.array([[-2],[4],[16]]).reshape(-1,1)
    lgr = MyLinearRegression(theta1, 0.001, 10000)
    lgr.fit_(X1, Y)
    y_hat1 = lgr.predict_(X1)
    y_pred.append(y_hat1)
    print(lgr.mse_(Y, y_hat1))
    mse.append(lgr.mse_(Y, y_hat1))
    X1 = add_polynomial_features(X, 3)
    theta1 = np.array([[-20],[ 160],[ -80],[ 10]]).reshape(-1,1)
    lgr = MyLinearRegression(theta1, 0.00001, 10000)
    lgr.fit_(X1, Y)
    y_hat1 = lgr.predict_(X1)
    y_pred.append(y_hat1)
    print(lgr.mse_(Y, y_hat1))
    mse.append(lgr.mse_(Y, y_hat1))
    X1 = add_polynomial_features(X, 4)
    theta1 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
    lgr = MyLinearRegression(theta1, 0.0000001, 10000)
    lgr.fit_(X1, Y)
    y_hat1 = lgr.predict_(X1)
    y_pred.append(y_hat1)
    print(lgr.mse_(Y, y_hat1))
    mse.append(lgr.mse_(Y, y_hat1))
    X1 = add_polynomial_features(X, 5)
    theta1 = np.array([[[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]]).reshape(-1,1)
    lgr = MyLinearRegression(theta1, 0.00000001, 10000)
    lgr.fit_(X1, Y)
    y_hat1 = lgr.predict_(X1)
    y_pred.append(y_hat1)
    print(lgr.mse_(Y, y_hat1))
    mse.append(lgr.mse_(Y, y_hat1))
    X1 = add_polynomial_features(X, 6)
    theta1 = np.array(([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]])).reshape(-1,1)
    lgr = MyLinearRegression(theta1, 0.000000001, 10000)
    lgr.fit_(X1, Y)
    y_hat1 = lgr.predict_(X1)
    y_pred.append(y_hat1)
    print(lgr.mse_(Y, y_hat1))
    mse.append(lgr.mse_(Y, y_hat1))
    plt.bar(["1", "2", "3", "4", "5", "6"], mse)
    plt.ylabel("Mse score")
    plt.xlabel("Nbrs polynomial")
    plt.show()
    plt.plot()
    plt.scatter(X, Y)
    for i in range(0, 6):
        plt.plot(X, y_pred[i])
    plt.legend(["data points", "1", "2", "3", "4", "5", "6"])
    plt.show()

