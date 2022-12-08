import numpy as np

def data_spliter(x, y, proportion):
    if (isinstance(x, np.ndarray) == True or isinstance(y, np.ndarray) == True):
        y = np.squeeze(y)
        if (y.ndim == 1 and len(x) == len(y) and len(x) != 0 and (p > 0 and p < 1)):
            t = np.c_[ y , x]
            np.random.shuffle(t)
            print(t)
            y = t.T[0]
            x = t.T[1:]
            
            return x, y
    return None


if __name__ == '__main__':
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    print(data_spliter(x1, y, 10.0))
