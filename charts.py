import matplotlib.pyplot as plt

name_dict = {
    0: "LeNet Architecture",
    1: "2-Conv 2-pooling layers",
    2: "4-Conv 2-pooling layers",
    3: "5-Conv 2-pooling layers augmented",
    4: "5-Conv 2-pooling layers augmented stopping",
}

loss_scores = [0.3597, 0.3524, 0.2077, 0.1521, 0.0099]
accuracy_scores = [91.30, 91.20, 95.00, 95.50, 97.20]


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


plot_accuracy_summary(plt, accuracy_scores)

# TODO:
# TODO:
# TODO:
