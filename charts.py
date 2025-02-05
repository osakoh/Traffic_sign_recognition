import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from func import class_dict, y_train, y_val, y_test, x_train, x_val, x_test
from keras.models import load_model

name_dict = {
    0: "LeNet Architecture",
    1: "2-Conv 2-pooling layers",
    2: "4-Conv 2-pooling layers",
    3: "5-Conv 2-pooling layers augmented",
    4: "5-Conv 2-pooling layers augmented stopping",
}

author_dict = {
    0: "Horak, Cip and Davidek, 2016",
    1: "Zang et al., 2016",
    2: "Jmour, Zayen and Abdelkrim, 2018",
    3: "Madan et al., 2019",
    4: "Yin et al., 2017b",
    5: "State-of-the-art",
    6: "Proposed classifier"

}

loss_scores = [0.3597, 0.3524, 0.2077, 0.1521, 0.0099]
accuracy_scores = [91.30, 91.20, 95.00, 95.50, 97.20]
paper_accuracy_scores = [93.00, 98.10, 93.33, 99.48, 98.96, 99.71, 97.20]


# TODO: Summary of loss plots
def plot_loss_summary(graph_plt, loss_score):
    """
    """
    # list comprehension
    x = [i for i in name_dict.values()]

    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(21, 10))  # width, height
    graph_plt.bar(x, loss_score)

    # adding text to the plot
    for i in range(0, 5):
        graph_plt.text(i, loss_score[i], loss_score[i], ha='center', weight='bold', fontsize=15)

    # graph_plt.xticks(rotation="vertical")  # rotate the horizontal(x) axis
    graph_plt.title("Summary of loss plots", fontsize=25, weight='bold')
    graph_plt.xlabel("Implemented Architectures", fontsize=22, weight='bold')
    graph_plt.ylabel("Loss", fontsize=22, weight='bold')
    # graph_plt.show()
    return fig_save.savefig('plot/summary_loss_plots.png')


# TODO: Summary of accuracy plots

def plot_accuracy_summary(graph_plt, acc_score):
    """
    :param num_sam: reference to number of samples from the plot
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param num_of_classes: number of classes in the GTSRB
    :return: a plot of the num
    """
    # list comprehension
    x = [i for i in name_dict.values()]

    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(21, 10))  # width, height
    graph_plt.bar(x, acc_score)

    # adding text to the plot
    for i in range(0, 5):
        graph_plt.text(i, acc_score[i], acc_score[i], ha='center', weight='bold', fontsize=15)

    # graph_plt.xticks(rotation="vertical")  # rotate the horizontal(x) axis
    graph_plt.title("Summary of Accuracy plots", fontsize=25, weight='bold')
    graph_plt.xlabel("Implemented Architectures", fontsize=22, weight='bold')
    graph_plt.ylabel("Accuracy", fontsize=22, weight='bold')
    # graph_plt.show()
    return fig_save.savefig('plot/summary_accuracy_plots.png')


def classifier_accuracy_plot(graph_plt, paper_score):
    """
    :param num_sam: reference to number of samples from the plot
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param num_of_classes: number of classes in the GTSRB
    :return: a plot of the num
    """
    # list comprehension
    x = [i for i in author_dict.values()]

    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(21, 10))  # width, height
    graph_plt.bar(x, paper_score)

    # adding text to the plot
    for i in range(0, len(paper_score)):
        graph_plt.text(i, paper_score[i], paper_score[i], ha='center', weight='bold', fontsize=15)

    # graph_plt.xticks(rotation="vertical")  # rotate the horizontal(x) axis
    graph_plt.title("Comparison of different classifiers", fontsize=25, weight='bold')
    graph_plt.xlabel("Authors", fontsize=22, weight='bold')
    graph_plt.ylabel("Accuracy", fontsize=22, weight='bold')
    # graph_plt.show()
    return fig_save.savefig('plot/different_classifier_comparison.png')


classifier_accuracy_plot(plt, paper_accuracy_scores)
