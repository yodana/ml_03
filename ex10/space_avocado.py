import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class MyLinearRegression():
    def __init__(self, theta, alpha=0.1, max_iter=10):
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


def data_spliter(x, y, proportion):
    if (isinstance(x, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        y = np.squeeze(y)
        if (y.ndim == 1 and len(x) == len(y) and len(x) != 0 and (proportion > 0 and proportion < 1)):
            t = np.c_[ y , x]
            np.random.shuffle(t)
            x = t.T[1:]
            x_train = x.T[:int(len(x[0]) * proportion)]
            x_test = x.T[int(len(x[0]) * proportion):]
            y_train = y[:int(len(x[0]) * proportion)]
            y_test = y[(int(len(x[0]) * proportion)):]
            return x_train, x_test, y_train, y_test
    return None

def min_max(x):
    r = np.zeros(x.T.shape[0])
    for l in x:
        r = np.c_[ r , (l - min(l)) / (max(l) - min(l))]
    r = r[:, 1:]
    return r

def scaler_y_params(x, mean, std):
    r = (x - mean) / std
    return r

def scaler_params(x, mean, std):
    r = np.zeros(x.T.shape[0])
    i = 0
    for l in x:
        r = np.c_[ r , (l - mean[i]) / std[i]]
        i = i + 1
    r = r[:, 1:]
    return r

def reverse_scaler_y(x, mean, std):
    r = x * std + mean
    return r

if __name__ == '__main__':
    models = pd.read_csv("models.csv")
    data = pd.read_csv("space_avocado.csv")
    np.random.seed(1)
    y_pred = []
    mse = []
    X = np.array([data["weight"], data["prod_distance"],data["time_delivery"]])
    Y = np.array([data["target"]])
    x_train,x_test, y_train, y_test = data_spliter(X.T,Y,0.80)
    x_test = scaler_params(x_test.T, models.iloc[:,-2][np.isfinite(models.iloc[:,-2])], models.iloc[:,-1][np.isfinite(models.iloc[:,-1])])
    y_test = scaler_y_params(y_test, models.iloc[:,-2][np.isfinite(models.iloc[:,-2])].iloc[-1], models.iloc[:,-1][np.isfinite(models.iloc[:,-1])].iloc[-1])
    p = x_test
    r = x_train
    lgr= MyLinearRegression(models.iloc[:,-6][np.isfinite(models.iloc[:,-6])].T)
    y_hat1 = lgr.predict_(x_test)
    y_pred.append(y_hat1)
    mse.append(lgr.mse_(y_test, y_hat1))
    r = x_test
    for x in r.T:
        p = np.c_[ p, add_polynomial_features(x, 2)]
        p = p[:, 1:]
    lgr = MyLinearRegression(models.iloc[:,-5][np.isfinite(models.iloc[:,-5])])
    y_hat1 = lgr.predict_(p)
    y_pred.append(y_hat1)
    mse.append(lgr.mse_(y_test, y_hat1))
    p = x_test
    r = x_test
    for x in r.T:
        p = np.c_[ p , add_polynomial_features(x, 3)]
        p = p[:, 1:]
    lgr = MyLinearRegression(models.iloc[:,-4][np.isfinite(models.iloc[:,-4])])
    y_hat1 = lgr.predict_(p)
    y_pred.append(y_hat1)
    mse.append(lgr.mse_(y_test, y_hat1))
    p = x_test
    r = x_test
    for x in r.T:
        p = np.c_[ p, add_polynomial_features(x, 4)]
        p = p[:, 1:]
    lgr = MyLinearRegression(models.iloc[:,-3][np.isfinite(models.iloc[:,-3])])
    y_hat1 = lgr.predict_(p)
    y_pred.append(y_hat1)
    mse.append(lgr.mse_(y_test, y_hat1))
    y_test = reverse_scaler_y(y_test, models.iloc[:,-2][np.isfinite(models.iloc[:,-2])].iloc[-1], models.iloc[:,-1][np.isfinite(models.iloc[:,-1])].iloc[-1])
    y_pred[3] = reverse_scaler_y(y_pred[3], models.iloc[:,-2][np.isfinite(models.iloc[:,-2])].iloc[-1], models.iloc[:,-1][np.isfinite(models.iloc[:,-1])].iloc[-1])
    plt.plot(["1", "2", "3", "4"], mse)
    plt.ylabel("Mse score")
    plt.xlabel("Nbrs polynomial")
    plt.show()
    for x in x_test.T:
        plt.scatter(x, y_test, color="blue")
        plt.scatter(x, y_pred[3], color="red")
        plt.ylabel("Price")
        plt.xlabel("Feature")
        plt.show()

