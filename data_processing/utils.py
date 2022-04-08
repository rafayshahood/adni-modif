import logging
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as TSNE_sklearn
from tsnecuda import TSNE
import scipy.stats
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


def mean_confidence_interval(data, confidence=0.95):
    # taken from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print("Mean: {}; SE: {}".format(m, se))
    return m, m-h, m+h, se


def plot_2d_tsne(data: np.array, class_mapping: dict, n_vis: int = 20):
    """
    Creates a t-SNE visualisation for 4 perplexity values.
    :param data: Array containing data and class
    :param class_mapping: a mapping from int ot str
    :param n_vis: samples per each class that are used for visualisation
    """
    X = data[:, :-1]  #

    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(X)
        X = pca.transform(X)

    Y = data[:, -1]  # class labels
    C = len(np.unique(Y))

    perplexities = [5, 30, 50, 100]

    x_embeded_list = []
    for i, perplexity in enumerate(perplexities):
        x_embeded = TSNE(perplexity=perplexity, n_iter=10000).fit_transform(X)
        df = pd.DataFrame(x_embeded, columns=["X1", "X2"])
        df["Perplexity"] = perplexity
        df["Y"] = Y
        x_embeded_list.append(df)

    df = pd.concat(x_embeded_list)
    df.replace({"Y": class_mapping}, inplace=True)  # map int values to str values
    df.groupby('Y').apply(lambda x: x.sample(n_vis)).reset_index(drop=True, inplace=True)  # sample n samples for viualisation
    g = sns.FacetGrid(df, col="Perplexity", col_wrap=2, height=2)
    g.map_dataframe(sns.scatterplot, x="X1", y="X2", hue="Y", palette=sns.color_palette("hls", C), s=5, style="Y")
    g.add_legend()
    g.set(xticks=[])
    g.set(yticks=[])
    g.set(xlabel=None)
    g.set(ylabel=None)
    plt.savefig("../images/tsne_2D.pdf", format="pdf")


def plot_3d_tsne(data: np.array, class_mapping: dict, n_vis: int = None):
    """
    Creates a t-SNE visualisation for 4 perplexity values.
    :param data: Array containing data and class
    :param class_mapping: a mapping from int ot str
    :param n_vis: samples per each class that are used for visualisation
    """
    X = data[:, :-1]  # features

    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(X)
        X = pca.transform(X)

    Y = data[:, -1]  # class labels
    projections = TSNE_sklearn(n_components=3, perplexity=100, n_iter=10000).fit_transform(X)
    df = pd.DataFrame(projections, columns=["X", "W", "Z"])
    df["Y"] = Y

    df.replace({"Y": class_mapping}, inplace=True)  # map int values to str values
    if n_vis is not None:
        df.groupby('Y').apply(lambda x: x.sample(n_vis)).reset_index(drop=True,
                                                                     inplace=True)  # sample n samples for viualisation
    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=df.Y, labels={'color': 'Y'}
    )
    fig.update_traces(marker_size=4)
    fig.show()


data = np.load("/mnt/ssd2/ClinicNET/features/nifd_adni_aibl_test.npy")
plot_2d_tsne(data, {0:'CN', 1:'AD', 2:'BV', 3:'MCI'})
plot_3d_tsne(data, {0:'CN', 1:'AD', 2:'BV', 3:'MCI'})



# mcc = [0.4294, 0.4318, 0.4167]
# precision = [0.5729, 0.5766, 0.5583]
# recall = [0.6978, 0.6970, 0.6861]
# mean_confidence_interval(mcc)


# plot_cm_mlxtend(np.array([[360, 47, 20, 86],
#                           [20, 125, 6, 30],
#                           [0, 0, 32, 0],
#                           [85, 71, 4, 87]]
#                          ),
#                 "Confusion matrix: \n0 - CN, 1 - AD, 2 - BV, 3 - MCI")
