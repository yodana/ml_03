import numpy as np

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

if __name__ == '__main__':
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    #array([[8.], [48.], [323.]])
    # Example 1:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    #array([[225.], [0.], [11025.]])
    # Example 2:
    print(mylr.loss_(Y, y_hat))
    # Output:
    #1875.0
    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output:
    #array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
    # Example 4:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    #array([[23.417..], [47.489..], [218.065...]])
    # Example 5:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
   # array([[0.174..], [0.260..], [0.004..]])
    # Example 6:
    print(mylr.loss_(Y, y_hat))
    # Output:
    #0.0732..