import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd
import seaborn as sns
import cv2

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

img_shape = (32, 32, 3)
data_path = "data"
class_dict = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of Speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycle crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep Left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing veh over 3.5 tons"}


def load_data(file_path):
    """
    :return: the training, validation and test data
    """
    with open(file_path + '/train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open(file_path + '/valid.p', 'rb') as f:
        val_data = pickle.load(f)
    with open(file_path + '/test.p', 'rb') as f:
        test_data = pickle.load(f)

    if train_data is None or test_data is None or val_data is None:
        return "Incomplete data, can't load"
    else:
        return train_data, val_data, test_data


# train, valid, test = load_data()[0], load_data()[1], load_data()[2]


def read_csv(csv_file):
    """
    :param csv_file: takes a CSV file containing the ClassId and SignName of the dataset
    :return: a CSV file read using the Pandas library
    """
    return pd.read_csv(csv_file)


sign_names = read_csv('data/signnames.csv')


def split_data(tr_data, vl_data, ts_data):
    """
    :param tr_data: reference to train data
    :param vl_data: reference to validation data
    :param ts_data: reference to test data
    :return: a tuple of features(x_train, x_val, x_test) and labels(y_train, y_val, y_test)
    """
    x_train, y_train = tr_data['features'], tr_data['labels']
    x_val, y_val = vl_data['features'], vl_data['labels']
    x_test, y_test = ts_data['features'], ts_data['labels']
    assert (x_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
    assert (x_train.shape[1:] == (
        img_shape[0], img_shape[1], img_shape[2])), "The dimensions of the images are not 32 x 32 x 3."

    assert (x_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
    assert (x_val.shape[1:] == (
        img_shape[0], img_shape[1], img_shape[2])), "The dimensions of the images are not 32 x 32 x 3."

    assert (x_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
    assert (x_test.shape[1:] == (
        img_shape[0], img_shape[1], img_shape[2])), "The dimensions of the images are not 32 x 32 x 3."

    return x_train, y_train, x_val, y_val, x_test, y_test


X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(load_data(data_path)[0], load_data(data_path)[1],
                                                            load_data(data_path)[2])

# print(X_train.shape[0] == y_train.shape[0])
# print(X_train.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(X_val.shape[0] == y_val.shape[0])
# print(X_val.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(X_test.shape[0] == y_test.shape[0])
# print(X_test.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(f"\nX_train height(rows): {X_train.shape[0]}\nX_train dimensions(columns): {X_train.shape[1:]}")
# print(f"y_train : {y_train.shape}")

# print(f"\nX_val height(rows): {X_val.shape[0]}\nX_val dimensions(columns): {X_val.shape[1:]}")
# print(f"X_val : {X_val.shape}")

# print(f"\nX_test height(rows): {X_test.shape[0]}\nX_test dimensions(columns): {X_test.shape[1:]}")
# print(f"y_test : {y_test.shape}")

print(f"\nTotal = {X_train.shape[0] + X_test.shape[0] + X_val.shape[0]}")


def plot_traffic_signs(cols, num_of_classes, graph_plt):
    """
    save a png file of randomly selected images and their sign names from each class
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param cols: number of columns
    :param num_of_classes: number of classes in dataset
    :return: number of samples in each class
    """
    samples = []
    fig, axs = graph_plt.subplots(nrows=num_of_classes, ncols=cols, figsize=(10, 50))
    fig.tight_layout()

    for i in range(cols):
        for j, row in sign_names.iterrows():  # iterrows: iterate over the df_data as (index, series) pair
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :],
                             cmap=graph_plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + " : " + row["SignName"])
                samples.append(len(x_selected))

    return samples, graph_plt.savefig('plot/ts_plot.png')


num_of_samples = (plot_traffic_signs(5, 43, plt)[0])


def plot_data_variations(graph_plt, num_of_classes):
    """
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param num_of_classes: number of classes in the GTSRB
    :return: a plot of the num
    """
    x = [i for i in class_dict.values()]
    graph_plt.figure(figsize=(40, 28))  # width, height
    graph_plt.bar(x, num_of_samples)

    # adding text to the plot
    for i in range(0, num_of_classes):
        graph_plt.text(i, num_of_samples[i], num_of_samples[i], ha='center', weight='bold')

    graph_plt.xticks(rotation="vertical")  # rotate the horizontal(x) axis
    # plt.bar(range(0, num_of_classes), num_of_samples)
    graph_plt.title("Distribution of classes in the training dataset", fontsize=12)
    graph_plt.xlabel("Class number", fontsize=12)
    graph_plt.ylabel("Number of images", fontsize=12)
    return graph_plt.savefig('plot/data_variation_sns.png')


plot_data_variations(plt, 43)


def unprocessed_rand_image():
    pass


def processed_rand_image():
    pass
