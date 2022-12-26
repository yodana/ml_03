import numpy as np

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


if __name__ == '__main__':
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    print(data_spliter(x1, y, 0.5))
    x2 = np.array([[ 1, 42],
    [300, 10],
    [ 59, 1],
    [300, 59],
    [ 10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # Example 3:
    print(data_spliter(x2, y, 0.8))
