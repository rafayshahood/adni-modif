import logging
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


class Mode(str, Enum):
    """"
    A helper class to differentiate between different stages: training and evaluation
    """
    training = 'training'
    evaluation = 'evaluation'
    independent_evaluation = 'independent_evaluation'


def plot_density(data, filename):
    data.loc[data['Group'] == 0, 'Group'] = "Control"
    data.loc[data['Group'] == 1, 'Group'] = "Patient"
    sns.set(rc={'figure.figsize': (10, 10)})
    ax = sns.displot(data, x="Preds", hue="Group", kind="kde", fill=True, height=8, rug=True)
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axvline(0, color='red', linestyle='--')
    plt.gcf().subplots_adjust(bottom=0.1, left=0.1)
    plt.savefig(filename)
    plt.close()


def plot_cm(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def plot_cm_mlxtend(conf_matrix, title):
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title, fontsize=18)
    plt.show()


def plot_roc(y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    logging.info('NOptimal threshold=%.3f' % (optimal_threshold))

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y))]

    ns_auc = roc_auc_score(y, ns_probs)
    lr_auc = roc_auc_score(y, probs)
    # summarize scores
    logging.info('Random Classifier: ROC AUC=%.3f' % (ns_auc))
    logging.info('Disease Classifier: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y, probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Classifier (area = 0.5)')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Disease Classifier (area = %.3f)' % (lr_auc))
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


plot_cm_mlxtend(np.array([[357, 52, 25, 79],
                          [19, 131, 1, 30],
                          [0, 2, 30, 0],
                          [87, 63, 3, 94]]
                         ),
                "Confusion matrix: \n0 - CN, 1 - AD, 2 - BV, 3 - MCI")
