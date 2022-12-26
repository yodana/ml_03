import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class MyLinearRegression():
    def __init__(self, theta, alpha=0.1, max_iter=100000):
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

def scaler(x):
    r = np.zeros(x.T.shape[0])
    for l in x:
        r = np.c_[ r , (l - l.mean()) / l.std()]
    r = r[:, 1:]
    return r

def scaler_y(x):
    r = (x - x.mean()) / x.std()
    return r

def min_max(x):
    r = np.zeros(x.T.shape[0])
    for l in x:
        r = np.c_[ r , (l - min(l)) / (max(l) - min(l))]
    r = r[:, 1:]
    return r

def min_max_y(x):
    r = (x - min(x)) / (max(x) - min(x))
    return r

if __name__ == '__main__':
    '''data = pd.read_csv("space_avocado.csv")
    np.random.seed(1)
    X = np.array([data["weight"], data["prod_distance"],data["time_delivery"]])
    Y = np.array([data["target"]])
    x_train,x_test, y_train, y_test = data_spliter(X.T,Y,0.80)
    df = pd.concat([pd.DataFrame(x_train.T[0]), pd.DataFrame(x_train.T[1]), pd.DataFrame(x_train.T[2]), pd.DataFrame(y_train)], axis=1)
    df.columns = ['weight', 'prod_distance', "time_delivery", "target"]
    df.to_csv("avocado_train.csv")
    df = pd.concat([pd.DataFrame(x_test.T[0]), pd.DataFrame(x_test.T[1]), pd.DataFrame(x_test.T[2]), pd.DataFrame(y_test)], axis=1)
    df.columns = ['weight', 'prod_distance', "time_delivery", "target"]
    df.to_csv("avocado_test.csv")'''
    data = pd.read_csv("avocado_train.csv")
    x_train = np.array([data["weight"], data["prod_distance"],data["time_delivery"]])
    y_train = np.array([data["target"]])
    data = pd.read_csv("avocado_test.csv")
    x_test = np.array([data["weight"], data["prod_distance"],data["time_delivery"]])
    y_test = np.array([data["target"]])
    np.random.seed(1)
    means = np.array([x_train[0].mean(), x_train[1].mean(), x_train[2].mean(), y_train.mean()])
    stds = np.array([x_train[0].std(), x_train[1].std(), x_train[2].std(), y_train.std()])
    x_train = scaler(x_train)
    y_train = scaler(y_train)
    p = x_train
    t = x_test
    mylr1 = MyLinearRegression([[1], [1], [-1], [1]])
    mylr1.fit_(p, y_train)
    #test with poly 2
    p = x_train
    r = x_train
    for x in r.T:
        p = np.c_[ p , add_polynomial_features(x, 2)]
        p = p[:, 1:]
    mylr2 = MyLinearRegression([[1], [1], [-1], [1], [1], [-1], [1]])
    mylr2.fit_(p, y_train)
    #test with poly 3
    p = x_train
    r = x_train
    for x in r.T:
        p = np.c_[ p , add_polynomial_features(x, 3)]
        p = p[:, 1:]
    mylr3 = MyLinearRegression([[1], [-1], [1], [-1], [1], [-1], [1], [-1],[1], [-1]])
    mylr3.fit_(p, y_train)
    #test with poly 4
    p = x_train
    r = x_train
    for x in r.T:
        p = np.c_[ p , add_polynomial_features(x, 4)]
        p = p[:, 1:]
    mylr4 = MyLinearRegression([[1], [-1], [1], [-1], [1],[-1],[-1],[-1],[-1], [1], [-1], [1],[1]])
    mylr4.fit_(p, y_train)
    df = pd.concat([pd.DataFrame(mylr1.theta), pd.DataFrame(mylr2.theta), pd.DataFrame(mylr3.theta), pd.DataFrame(mylr4.theta), pd.DataFrame(means), pd.DataFrame(stds)], axis=1)
    df.to_csv("models.csv", index=False)





