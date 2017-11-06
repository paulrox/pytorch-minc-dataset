import numpy as np


def confusionMatrix(mat, predicted, actual):

    for i in range(len(actual)):
        mat[int(round(predicted[i])), actual[i]] += 1
    return mat


def getAccuracy(matrix):
    # sum(diag(mat))/(sum(mat))
    sumd = np.sum(np.diagonal(matrix))
    sumall = np.sum(matrix)
    sumall = np.add(sumall, 0.00000001)
    return sumd/sumall


def getPrecision(matrix):
    # diag(mat) / rowSum(mat)
    sumrow = np.sum(matrix, axis=1)
    sumrow = np.add(sumrow, 0.00000001)
    precision = np.divide(np.diagonal(matrix), sumrow)
    return np.sum(precision)/precision.shape[0]


def getRecall(matrix):
    # diag(mat) / colsum(mat)
    sumcol = np.sum(matrix, axis=0)
    sumcol = np.add(sumcol, 0.00000001)
    recall = np.divide(np.diagonal(matrix), sumcol)
    return np.sum(recall)/recall.shape[0]


def get2f(matrix):
    # 2*precision*recall/(precision+recall)
    precision = getPrecision(matrix)
    recall = getRecall(matrix)
    return (2*precision*recall)/(precision+recall)
