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

if __name__ == '__main__':
    #ex01
    '''data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[["Age"]])
    Y = np.array(data[["Sell_price"]])
    myLR_age = MyLinearRegression(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
    myLR_age.fit_(X[:,0].reshape(-1,1), Y)
    y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
    plt.scatter(X, Y, color="blue")
    plt.scatter(X, y_pred, color="deepskyblue")
    plt.xlabel("x: age (in years)")
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(["Sell price", "Predicted sell price"])
    print(myLR_age.theta)
    print(myLR_age.mse_(y_pred,Y))
    plt.show()

    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[["Thrust_power"]])
    Y = np.array(data[["Sell_price"]])
    myLR_thrust = MyLinearRegression(theta = [[100.0], [100.0]], alpha = 2.5e-5, max_iter = 100000)
    myLR_thrust.fit_(X[:,0].reshape(-1,1), Y)
    y_pred = myLR_thrust.predict_(X[:,0].reshape(-1,1))
    plt.scatter(X, Y, color="green")
    plt.scatter(X, y_pred, color="springgreen")
    plt.xlabel("x: thrust power (in 10km/s)")
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(["Sell price", "Predicted sell price"])
    print(myLR_thrust.theta)
    print(myLR_thrust.mse_(y_pred,Y))
    plt.show()

    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[["Terameters"]])
    Y = np.array(data[["Sell_price"]])
    myLR_distance = MyLinearRegression(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
    myLR_distance.fit_(X[:,0].reshape(-1,1), Y)
    y_pred = myLR_distance.predict_(X[:,0].reshape(-1,1))
    plt.scatter(X, Y, color="darkviolet")
    plt.scatter(X, y_pred, color="violet")
    plt.xlabel("x: distance totalizer value of spacecraft (in Temeters)")
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(["Sell price", "Predicted sell price"])
    print(myLR_distance.theta)
    print(myLR_distance.mse_(y_pred,Y))
    plt.show()'''

    #ex02
    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[["Age","Thrust_power","Terameters"]])
    Y = np.array(data[["Sell_price"]])
    X_age = np.array(data[["Age"]])
    my_lreg = MyLinearRegression(theta = [1.0, 1.0, 1.0, 1.0], alpha = 5e-5, max_iter = 600000)
    my_lreg.fit_(X,Y)
    y_pred = my_lreg.predict_(X)
    print(my_lreg.theta)
    print(my_lreg.mse_(Y,y_pred))
    plt.scatter(X_age, Y, color="blue")
    plt.scatter(X_age, y_pred, color="deepskyblue")
    plt.xlabel("x: age (in years)")
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(["Sell price", "Predicted sell price"])
    plt.show()
    X_thrust = np.array(data[["Thrust_power"]])
    plt.scatter(X_thrust, Y, color="green")
    plt.scatter(X_thrust, y_pred, color="springgreen")
    plt.xlabel("x: thrust power (in 10km/s)")
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(["Sell price", "Predicted sell price"])
    plt.show()
    X_terameters = np.array(data[["Terameters"]])
    plt.scatter(X_terameters, Y, color="darkviolet")
    plt.scatter(X_terameters, y_pred, color="violet")
    plt.xlabel("x: distance totalizer value of spacecraft (in Temeters)")
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(["Sell price", "Predicted sell price"])
    plt.show()


