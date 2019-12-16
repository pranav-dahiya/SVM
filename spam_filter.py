import glob
import pickle
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from multiprocessing import Pool
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, "rb") as f:
        return [pickle.load(f), -2*int("spm" in filename) + 1]


def read_data(folder):
    with open("vocabulary.pickle", "rb") as f:
        data = [np.ones((len(folder), len(pickle.load(f))+1)), np.zeros(len(folder))]
    pool = Pool()
    temp = [None for i in range(len(folder))]
    for i, filename in enumerate(folder):
        temp[i] = pool.apply_async(read_file, (filename, ))
    pool.close()
    pool.join()
    for i, val in enumerate(temp):
        data[0][i,:-1] = val.get()[0]
        data[1][i] = val.get()[1]
    return data


def dual_coordinate_descent(K, y, Q, U, epochs, plot):
    alpha = np.zeros(y.shape)
    z = list()
    for epoch in range(epochs):
        for i, (a, q) in enumerate(zip(alpha, Q)):
            G = np.inner(q, alpha) - 1
            if a == 0:
                PG = min(G, 0)
            elif a == U:
                PG = max(G, 0)
            else:
                PG = G
            if PG != 0:
                alpha[i] = min(max(a-G/q[i], 0), U)
        if plot:
            z.append(np.sum(np.less(y*np.dot(alpha*y, K), np.zeros(y.shape))))
    if plot:
        plt.plot(range(1, epochs+1), z)
        plt.xlabel("epoch")
        plt.ylabel("mispredictions")
        plt.show()
    return alpha


def poly(X, Y, order):
    if order == None:
        order = 3
    return np.power(np.dot(X, np.transpose(Y)) + 1, order)


def gaussian(X, Y, l):
    if l == None:
        l = 1
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            z = x - y
            K[i,j] = np.inner(z, z)
    K = np.exp(-l*K)
    return K


def linear(X, Y, args):
    return poly(X, Y, 1)


def L1_SVM(K, y, C, epochs=0, plot=False):
    Q = np.multiply(np.outer(y, y), K)
    return dual_coordinate_descent(K, y, Q, C, epochs, plot)


def L2_SVM(K, y, C, epochs=700, plot=False):
    D = np.diag(np.ones(y.shape)/(2*C))
    Q = np.multiply(np.outer(y, y), K)
    return dual_coordinate_descent(K, y, Q+D, float("inf"), epochs, plot)


def test(K, y_train, y_test, alpha):
    predictions = np.greater_equal(np.dot(K, alpha*y_train), np.zeros(y_test.shape))
    TP, FP, TN, FN = 0, 0, 0, 0
    for prediction, label in zip(predictions, y_test):
        if prediction:
            if label == 1:
                TN += 1
            else:
                FN += 1
        else:
            if label == 1:
                FP += 1
            else:
                TP += 1
    print(TP, ",", FP, ",", TN, ",", FN)
    return (TP+TN)/(TP+TN+FP+FN)


def SVM(X_train, y_train, X_test, y_test, func, C, epochs, kernel=None, kernel_args=None, verbose=False):
    if kernel == None:
        K_train = np.dot(X_train, np.transpose(X_train))
        K_test = np.dot(X_test, np.transpose(X_train))
    else:
        K_train = kernel(X_train, X_train, kernel_args)
        K_test = kernel(X_test, X_train, kernel_args)
    alpha = func(K_train, y_train, C, epochs)
    test_accuracy = test(K_test, y_train, y_test, alpha)
    if verbose:
        return (1-np.sum(np.less(y_train*np.dot(K_train, alpha*y_train),
                    np.zeros(y_train.shape)))/y_train.shape[0], test_accuracy)


def flatten(data):
    X, y = [], []
    for [Xi, yi] in data:
        X.extend(Xi)
        y.extend(yi)
    return np.array(X), np.array(y)


def cross_validate(func=L1_SVM, C=500, epochs=1000, kernel=None, kernel_args=None):
    data = [read_data(glob.glob("lingspam/part"+str(i)+"_tfidf/*.pickle")) for i in range(1,11)]
    pool = Pool()
    for i, [X_test, y_test] in enumerate(data):
        X_train, y_train = flatten([val for j, val in enumerate(data) if j != i])
        pool.apply_async(SVM, args=(X_train, y_train, X_test, y_test, func, C, epochs, kernel, kernel_args))
    pool.close()
    pool.join()


if __name__ == '__main__':
    print("SVM without kernel")
    cross_validate()
    print("SVM with linear kernel")
    cross_validate(kernel=linear)
    print("SVM with polynomial kernel")
    cross_validate(kernel=poly)
    print("SVM with gaussian kernel")
    cross_validate(kernel=gaussian)
