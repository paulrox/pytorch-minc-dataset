import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import interp
import visdom


def updateCM(mat, predicted, actual):

    for i in range(len(actual)):
        mat[actual[i], int(round(predicted[i]))] += 1
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
        self.tpr = dict()
        self.fpr = dict()

        self.tpr["bin"] = dict()
        self.fpr["bin"] = dict()

        # scalar containing TP + FP + TN + FN
        sumall = np.sum(self.matrix)
        sumall = np.add(sumall, 0.00000001)
        # sumrow[i] = TP[i] + FN[i]
        sumrow = np.sum(self.matrix, axis=1)
        sumrow = np.add(sumrow, 0.00000001)
        # sumcol[i] = TP[i] + FP[i]
        sumcol = np.sum(self.matrix, axis=0)
        sumcol = np.add(sumcol, 0.00000001)

        # TP[i] = True positives for class 'i'
        TP = np.diagonal(self.matrix)
        # FN[i] = False negatives for class 'i'
        FN = sumrow-TP
        # FP[i] = False positives for class 'i'
        FP = sumcol-TP
        # TN[i] = True negatives for class 'i'
        TN = sumall-FP-FN-TP

        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN

        # Binary (one-vs-all) class recall (aka sensitivity, TPR, hit rate)
        # recall[i] = TP[i] / (TP[i] + FN[i])
        recall = np.divide(TP, sumrow)
        # Binary (one-vs-all) class precision (aka PPV - positive
        # predictive value)
        # precision[i] = TP[i] / (TP[i] + FP[i])
        precision = np.divide(TP, sumcol)
        # Binary (one-vs-all) class fall-out (aka FPR - false positive rate)
        # fall_out[i] = FP[i] / (FP[i] + TN[i])
        fall_out = np.divide(FP, FP + TN)

        # Multi-Class measures
        self.accuracy = np.sum(TP) / sumall  # (TP + TN) / all
        self.avg_accuracy = np.sum(np.divide(TP + TN, sumall)) / self.n_class
        self.precision["micro"] = np.sum(TP) / np.sum(sumcol)
        self.precision["macro"] = np.sum(precision) / self.n_class
        self.recall["micro"] = np.sum(TP) / np.sum(sumrow)
        self.recall["macro"] = np.sum(recall) / self.n_class
        self.Fscore["micro"] = (2*self.precision["micro"] *
                                self.recall["micro"]) / (
                                self.precision["micro"] +
                                self.recall["micro"])
        self.Fscore["macro"] = (2*self.precision["macro"] *
                                self.recall["macro"]) / (
                                self.precision["macro"] +
                                self.recall["macro"])

        # ROC measures (multi-class classifier)
        self.tpr["micro"] = self.recall["micro"]
        self.tpr["macro"] = self.recall["macro"]
        self.fpr["micro"] = np.sum(FP) / np.sum(FP + TN)
        self.fpr["macro"] = np.sum(fall_out) / self.n_class
        # ROC measures (binary classifiers)
        self.tpr["bin"] = recall
        self.fpr["bin"] = fall_out

    def plot_multi_roc(self):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr["bin"] = dict()
        tpr["bin"] = dict()
        roc_auc["bin"] = dict()

        for i in range(self.n_class):
            # emulate h
            fpr["bin"][i] = [0, self.fpr["bin"][i], 1]
            tpr["bin"][i] = [0, self.tpr["bin"][i], 1]
            roc_auc["bin"][i] = auc(fpr["bin"][i], tpr["bin"][i])

        fpr["micro"] = [0, self.fpr["micro"], 1]
        tpr["micro"] = [0, self.tpr["micro"], 1]
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr["macro"] = [0, self.fpr["macro"], 1]
        tpr["macro"] = [0, self.tpr["macro"], 1]
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        self.plotmulticlassvis(fpr, tpr, roc_auc)
        # self.plotmulticlass(fpr, tpr, roc_auc)

        return roc_auc

    def plot_scores_roc(self, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr["bin"] = dict()
        tpr["bin"] = dict()
        roc_auc["bin"] = dict()

        # Binarize labels in a one-vs-all fashion
        y_bin = label_binarize(y_test, range(self.n_class))

        for i in range(self.n_class):
            fpr["bin"][i], tpr["bin"][i], _ = roc_curve(y_bin[:, i],
                                                        y_score[:, i])
            roc_auc["bin"][i] = auc(fpr["bin"][i], tpr["bin"][i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(),
                                                  y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        xn_classes = range(self.n_class)
        all_fpr = np.unique(np.concatenate([fpr["bin"][i]
                                            for i in xn_classes]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in xn_classes:
            mean_tpr += interp(all_fpr, fpr["bin"][i], tpr["bin"][i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_class

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        self.plotmulticlassvis(fpr, tpr, roc_auc)
        # self.plotmulticlass(fpr, tpr, roc_auc)

        return roc_auc

    def plotmulticlassvis(self, fpr, tpr, roc_auc):
        vis = visdom.Visdom()

        # Visdom windows to draw the training graphs
        roc_window = vis.line(X=np.array([0, 1]),
                              Y=np.array([0, 1]),
                              opts=dict(xlabel='False Positive Rate',
                                        ylabel='True Positive Rate',
                                        title='ROC',
                                        legend=['random']))
        for i in range(self.n_class):
            vis.updateTrace(
                X=np.array(fpr["bin"][i]),
                Y=np.array(tpr["bin"][i]),
                win=roc_window,
                name=str(i))

        vis.updateTrace(
            X=np.array(fpr['micro']),
            Y=np.array(tpr['micro']),
            win=roc_window,
            name='micro')

        vis.updateTrace(
            X=np.array(fpr['macro']),
            Y=np.array(tpr['macro']),
            win=roc_window,
            name='macro')

    def plotmulticlass(self, fpr, tpr, roc_auc):
        # Compute macro-average ROC curve and ROC area
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, len(roc_auc["bin"]) + 2))
        lw = 2
        plt.figure()

        for i, color in zip(range(self.n_class), colors):
            plt.plot(fpr["bin"][i], tpr["bin"][i], color=color, lw=lw,
                     label='class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc["bin"][i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.plot(fpr["micro"], tpr["micro"], color=colors[-2], lw=lw,
                 linestyle=':', label='micro-average (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]))
        plt.plot(fpr["macro"], tpr["macro"], color=colors[-1], lw=lw,
                 linestyle=':', label='macro-average (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('')
        plt.legend(loc="lower right")
        plt.show()

    def get_stats_dict(self, precision=4):
        stats = dict()

        stats["accuracy"] = round(self.accuracy, precision)
        stats["avg_accuracy"] = round(self.avg_accuracy, precision)
        stats["precision_u"] = round(self.precision["micro"], precision)
        stats["precision_M"] = round(self.precision["macro"], precision)
        stats["recall_u"] = round(self.recall["micro"], precision)
        stats["recall_M"] = round(self.recall["macro"], precision)
        stats["Fscore_u"] = round(self.Fscore["micro"], precision)
        stats["Fscore_M"] = round(self.Fscore["macro"], precision)

        return stats

    def print_stats(self):
        print('Accuracy (from CM): %.2f %%'
              % (self.accuracy * 100))
        print('Average Accuracy: %.2f %%'
              % (self.avg_accuracy * 100))
        print('Precision (macro): %.2f %%'
              % (self.precision["macro"] * 100))
        print('Recall (macro): %.2f %%'
              % (self.recall["macro"] * 100))
        print('Fscore (macro): %.2f %%'
              % (self.Fscore["macro"] * 100))
        print('Precision (micro): %.2f %%'
              % (self.precision["micro"] * 100))
        print('Recall (micro): %.2f %%'
              % (self.recall["micro"] * 100))
        print('Fscore (micro): %.2f %%'
              % (self.Fscore["micro"] * 100))
