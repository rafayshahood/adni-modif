import logging
import os
import random
from enum import Enum
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
import torch
from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score
from torch.backends import cudnn


class Mode(str, Enum):
    """"
    A helper class to differentiate between different stages: training and evaluation
    """
    training = 'training'
    evaluation = 'evaluation'
    independent_evaluation = 'independent_evaluation'


def set_logging(seed, suffix):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("/mnt/ssd2/ClinicNET/log/debug_{}_{}.log".format(seed, suffix)),
            logging.StreamHandler()
        ]
    )


def set_seeds(seed):
    """
    Set seed for reproducable results.
    :param seed: int value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


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


def mean_confidence_interval(data, confidence=0.95):
    # taken from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    print("Mean: {}; SE: {}".format(m, se))
    return m, m - h, m + h, se

# mcc = [0.4294, 0.4318, 0.4167]
# precision = [0.5729, 0.5766, 0.5583]
# recall = [0.6978, 0.6970, 0.6861]
# mean_confidence_interval(mcc)


# input_dir = "/mnt/ssd2/ClinicNET/data/adni3/CAPS/subjects/"
# output_dir = "/mnt/ssd2/ClinicNET/data/temp/ADNI3/"
# for root, dirs, files in os.walk(input_dir):
#     for name in files:
#         if "ses-M00" in name and name.endswith(".pt"):
#             file_path = os.path.join(os.path.abspath(root), name)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             copy2(file_path, output_dir)
