
import sys
import numpy as np 
import json
import base64
import StringIO
from itertools import cycle
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_curve, auc
from scipy import interp
from confusionmatrixinfo import *
plt.rcParams["font.family"] = "Times New Roman"

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def oneclass_decision_function_to_roc(y_test,y_score,n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    return dict(fpr=fpr,tpr=tpr,roc_auc=roc_auc)


def build_micro_macro(n_classes,fpr,tpr,roc_auc):
    xn_classes = range(n_classes)
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
    return dict(fpr=fpr,tpr=tpr,roc_auc=roc_auc)

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plotmulticlass(n_classes,fpr,tpr,roc_auc):
    # Compute macro-average ROC curve and ROC area
    lw = 2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(n_classes, colors):
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

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def confusion_matrix_to_roc(cm):
    # Compute ROC curve and ROC area for each class
    mm = MulticlassStat(cm)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = cm.shape[0]
    for i in range(n_classes):
        # emulate h
        fpr[i] = [0,mm.fpr_all[i],1]
        tpr[i] = [0,mm.tpr_all[i],1]
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"] = [0,mm.fpr,1]
    tpr["micro"] = [0,mm.tpr,1]
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return dict(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
    
#https://www.iterm2.com/utilities/imgls
def encode_iterm2_image(data,height=None):
    if height is None:
        height = "auto"
    else:
        height = "%spx" % height
    return ("\x1B]1337;File=loss.jpg;width=auto;height=%s;inline=1;size=%dpreserveAspectRatio=1:" % (height,len(data))) + base64.b64encode(data) + "\a\033\\"


def main():
    lista = ["20171103T151842.769-3054.json","20171103T160748.100-1270.json","3bbee473-c096-11e7-b515-60f81dbb5784.json"]
    dd =dict(tpr={},fpr={},roc_auc={})
    names = []
    for x in lista:
        f = json.load(open(x,"rb"))
        cmf = x+".cm.txt"
        cm = np.loadtxt(cmf).astype(np.int32)

        #print mm.fpr_all,mm.tpr_all
        cd = confusion_matrix_to_roc(cm)
        name = "%s-%s" % (f["test"],f["implementation"])
        names.append(name)
        cd = build_micro_macro(10,cd["fpr"],cd["tpr"],cd["roc_auc"])
        #build micro and macro per-case

        # only micro for each entry
        dd["fpr"][name] = cd["fpr"]["micro"]
        dd["tpr"][name] = cd["tpr"]["micro"]
        dd["roc_auc"][name] = cd["roc_auc"]["micro"]
        print cmf,"gpu" if f["gpu"] else "","singlecore" if f.get("single_core",False) else ""
    plotmulticlass(names,dd["fpr"],dd["tpr"],dd["roc_auc"])
    buf = StringIO.StringIO()
    plt.savefig(buf,format="png")
    plt.savefig("roc.pdf",format="pdf")
    print encode_iterm2_image(buf.getvalue())

if __name__ == '__main__':
    main()