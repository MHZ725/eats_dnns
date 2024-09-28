from termcolor import colored
import numpy as np

def printWithColor(string,colors):
    print(colored(string,colors))

# 混洗数据
def shuffle_data(X, Y, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    return X, Y