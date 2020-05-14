import numpy as np
import gzip

def load_mnnist(train_x, train_y, test_x, test_y):
    testX = np.loadtxt(gzip.open(test_x, "rb"), delimiter=",")
    testY = [parse(y) for y in np.loadtxt(gzip.open(test_y, "rb"))]
    trainingX = np.loadtxt(gzip.open(train_x, "rb"), delimiter=",")
    trainingY = [parse(y) for y in np.loadtxt(gzip.open(train_y, "rb"))]
    return trainingX, trainingY, testX, testY


def parse(val):
    data = np.zeros(10)
    data[int(val)] = 1.0
    return data
