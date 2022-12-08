import numpy as np
import matplotlib.pyplot as plt

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
    x = np.arange(1,6).reshape(-1, 1)
    print(add_polynomial_features(x, 3))
    print(add_polynomial_features(x, 6))