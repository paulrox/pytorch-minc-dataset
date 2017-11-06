import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

"""
References:
https://i.stack.imgur.com/AuTKP.png
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
"""


def updateCM(mat, predicted, actual):

    for i in range(len(actual)):
        mat[int(round(predicted[i])), actual[i]] += 1
    return mat


class MulticlassStat:

    def __init__(self, matrix=None, n_class=None, pred=None, actual=None):
        if matrix is None:
            # Compute the confusion matrix
            if pred is not None and actual is not None and \
               n_class is not None:
                self.matrix = np.zeros([n_class, n_class])
                self.matrix = updateCM(self.matrix, pred, actual)
            else:
                sys.exit("MulticlassStat: Missing arguments")
        else:
            self.matrix = matrix

        self.n_class = self.matrix.shape[0]
        self.precision = dict()
        self.recall = dict()
        self.Fscore = dict()

        sumall = np.sum(self.matrix)
        sumall = np.add(sumall, 0.00000001)  # TP+FP+TN+FN

        TP = np.diagonal(self.matrix)

        sumrow = np.sum(self.matrix, axis=1)
        sumrow = np.add(sumrow, 0.00000001)  # TP+FP
        self.precision["micro"] = np.divide(TP, sumrow)  # TP/(TP+FP)

        sumcol = np.sum(self.matrix, axis=0)
        sumcol = np.add(sumcol, 0.00000001)  # TP+FN
        self.recall["micro"] = np.divide(TP, sumcol)  # TP/(TP+FN)

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
        self.precision["macro"] = np.sum(self.precision["micro"]) / \
            self.precision["micro"].shape[0]
        # TP / (TP+FN)  aka sensitivity aka hit rate aka true positive rate TPR
        self.recall["macro"] = np.sum(self.recall["micro"]) / \
            self.recall["micro"].shape[0]
        self.fpr = np.sum(ufpr)/ufpr.shape[0]
        # TN/(TN+FP) aka true negative rate (TNR) === 1-FPR ==  fall out or
        # false positive rate FP/(FP+TP)
        self.specificity = 1-self.fpr
        # 2*precision*recall/(precision+recall)
        self.Fscore["macro"] = (2*self.precision["macro"] *
                                self.recall["macro"]) / (
                                self.precision["macro"] +
                                self.recall["macro"])
        self.Fscore["micro"] = (2*self.precision["micro"] *
                                self.recall["micro"]) / (
                                self.precision["micro"] +
                                self.recall["micro"])
        self.fpr_all = ufpr
        self.tpr_all = self.recall["micro"]
        self.tpr = self.recall["macro"]

    def confusion_matrix_to_roc(self):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.n_class):
            # emulate h
            fpr[i] = [0, self.fpr_all[i], 1]
            tpr[i] = [0, self.tpr_all[i], 1]
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"] = [0, self.fpr, 1]
        tpr["micro"] = [0, self.tpr, 1]
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        xn_classes = range(self.n_class)
        all_fpr = np.unique(np.concatenate([fpr[i] for i in xn_classes]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in xn_classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(xn_classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return dict(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    def oneclass_decision_function_to_roc(self, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Binarize labels in a one-vs-all fashion
        y_bin = label_binarize(y_test, range(self.n_class))

        for i in range(self.n_class):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return dict(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    def plotmulticlass(self, fpr, tpr, roc_auc):
        # Compute macro-average ROC curve and ROC area
        lw = 2
        plt.figure()

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_class), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC micro curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('')
        plt.legend(loc="lower right")
        plt.show()
