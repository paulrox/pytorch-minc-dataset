import sys
import numpy as np


def getCM(mat, predicted, actual):

    for i in range(len(actual)):
        mat[int(round(predicted[i])), actual[i]] += 1
    return mat


# https://i.stack.imgur.com/AuTKP.png
class MulticlassStat:

    def __init__(self, matrix=None, n_class=None, pred=None, actual=None):
        if matrix is None:
            # Compute the confusion matrix
            if pred is not None and actual is not None and \
               n_class is not None:
                self.matrix = np.zeros([n_class, n_class])
                self.matrix = getCM(self.matrix, pred, actual)
            else:
                sys.exit("MulticlassStat: Missing arguments")
        else:
            self.matrix = matrix

        sumall = np.sum(self.matrix)
        sumall = np.add(sumall, 0.00000001)  # TP+FP+TN+FN

        TP = np.diagonal(self.matrix)

        sumrow = np.sum(self.matrix, axis=1)
        sumrow = np.add(sumrow, 0.00000001)  # TP+FP
        uprecision = np.divide(TP, sumrow)  # TP/(TP+FP)

        sumcol = np.sum(self.matrix, axis=0)
        sumcol = np.add(sumcol, 0.00000001)  # TP+FN
        urecall = np.divide(TP, sumcol)  # TP/(TP+FN)

        FP = sumrow-TP
        FN = sumcol-TP
        # TN = sumall-FP-FN-TP

        ufpr = np.divide(FP, sumrow)  # FP/(FP+FN)

        self.TP = TP
        self.FP = FP
        self.FN = FN
        # self.TN = TN
        self.accuracy = np.sum(TP)/sumall  # (TP+TN)/all
        # TP/(TP+FP) aka positive predictive value PPV
        self.precision = np.sum(uprecision)/uprecision.shape[0]
        # TP / (TP+FN)  aka sensitivity aka hit rate aka true positive rate TPR
        self.recall = np.sum(urecall)/urecall.shape[0]
        self.fpr = np.sum(ufpr)/ufpr.shape[0]
        # TN/(TN+FP) aka true negative rate (TNR) === 1-FPR ==  fall out or
        # false positive rate FP/(FP+TP)
        self.specificity = 1-self.fpr
        # 2*precision*recall/(precision+recall)
        self.Fscore = (2*self.precision*self.recall) / \
                      (self.precision+self.recall)
        self.fpr_all = ufpr
        self.tpr_all = urecall
        self.tpr = self.recall
