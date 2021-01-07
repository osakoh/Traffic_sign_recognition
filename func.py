import tensorflow.compat.v1 as tf

tf.Session()

sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))

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


train, valid, test = load_data(data_path)[0], load_data(data_path)[1], load_data(data_path)[2]


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
    :return: splits data into x_train, x_val, x_test) and labels(y_train, y_val, y_test)
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


X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(train, valid, test)

# Before preprocessing
# print(X_train.shape[0] == Y_train.shape[0])
# print(X_train.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(X_val.shape[0] == Y_val.shape[0])
# print(X_val.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(X_test.shape[0] == Y_test.shape[0])
# print(X_test.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))
print("_________________________________ Before preprocessing______________________________")
print(f"X_train height(rows): {X_train.shape[0]}\nX_train dimensions(columns): {X_train.shape[1:]}")
print(f"y_train : {Y_train.shape}")

print(f"\nX_val height(rows): {X_val.shape[0]}\nX_val dimensions(columns): {X_val.shape[1:]}")
print(f"X_val : {X_val.shape}")

print(f"\nX_test height(rows): {X_test.shape[0]}\nX_test dimensions(columns): {X_test.shape[1:]}")
print(f"y_test : {Y_test.shape}")

print(f"\nTotal = {X_train.shape[0] + X_test.shape[0] + X_val.shape[0]}")
print("_________________________________ Before preprocessing______________________________")


def plot_traffic_signs(cols, num_of_classes, graph_plt):
    """
    save a png file of randomly selected images and their sign names from each class
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param cols: number of columns
    :param num_of_classes: number of classes in dataset
    :return: number of samples in each class
    """

    samples = []

    # Setting the figure size - width, height
    fig, axs = graph_plt.subplots(nrows=num_of_classes, ncols=cols, figsize=(15, 50))
    fig.tight_layout()

    for i in range(cols):
        for j, row in sign_names.iterrows():  # iterrows: iterate over the df_data as (index, series) pair
            x_selected = X_train[Y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :],
                             cmap=graph_plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + " : " + row["SignName"])
                samples.append(len(x_selected))

    return samples, fig.savefig('plot/dataset_img.png')


# num_of_samples = (plot_traffic_signs(5, 43, plt)[0])


def plot_data_variations(graph_plt, num_of_classes, num_sam):
    """
    :param num_sam: reference to number of samples from the plot
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param num_of_classes: number of classes in the GTSRB
    :return: a plot of the num
    """
    x = [i for i in class_dict.values()]

    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(40, 25))  # width, height
    graph_plt.bar(x, num_sam)

    # adding text to the plot
    for i in range(0, num_of_classes):
        graph_plt.text(i, num_sam[i], num_sam[i], ha='center', weight='bold')

    graph_plt.xticks(rotation="vertical")  # rotate the horizontal(x) axis
    # plt.bar(range(0, num_of_classes), num_of_samples)
    graph_plt.title("Distribution of classes in the training dataset", fontsize=12)
    graph_plt.xlabel("Class number", fontsize=12)
    graph_plt.ylabel("Number of images", fontsize=12)
    return fig_save.savefig('plot/data_variation.png')


# plot_data_variations(plt, 43, plot_traffic_signs(5, 43, plt)[0])


def unprocessed_rand_image(graph_plt):
    """
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :return: an  image
    """
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(28, 28))  # width, height

    # img = X_train[random.randint(0, len(X_train) - 1)]
    img = X_train[3000]
    graph_plt.imshow(img)
    graph_plt.axis('off')
    # save the plot
    return fig_save.savefig('plot/rand_unprocessed_img.png')


def plot_grayscale(graph_plt):
    """
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :return: an  image
    """
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(20, 20))  # width, height

    # img = X_train[random.randint(0, len(X_train) - 1)]
    img = X_train[3000]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    graph_plt.imshow(img, cmap=graph_plt.get_cmap('gray'))
    graph_plt.axis('off')
    # save the plot
    return fig_save.savefig('plot/grayscale1.png')


def plot_equalise(graph_plt):
    """
    works only on grayscale(1 channel) images
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :return: an  image
    """
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(28, 28))  # width, height
    # get random image
    # img = X_train[random.randint(0, len(X_train) - 1)]
    img = X_train[3000]
    # convert image to grayscale because 'equalizeHist' takes a grayscale image as an argument
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equalises the most frequent intensity values
    img = cv2.equalizeHist(img)
    graph_plt.imshow(img, cmap=graph_plt.get_cmap('gray'))
    graph_plt.axis('off')
    # save the plot
    return fig_save.savefig('plot/equalise.png')


def grayscale(img):
    """
    Reduces the depth of an image from 3 to 1 - network will have fewer parameters when training
    :param img: takes an image in the form of a numpy array
    :return: an image converted to grayscale - just one channel
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(img):
    """
    Spreads the distribution of the most frequent intensity values.
    ie areas of low contrast will have a higher contrast
    :param img: takes an image in the form of a numpy array
    :return: an image converted to grayscale - just one channel
    """
    img = cv2.equalizeHist(img)
    return img


def normalise(img):
    """
    Divides the pixel values by 255 - makes the pixel values to be normalised between 0 and 1
    :param img: takes an image as an argument
    :return: a normalised image
    """
    img = img / 255
    return img


def preprocess(img):
    img = grayscale(img)
    img = equalise(img)
    img = img / 255
    return img


def preprocessed_img(img_train, img_val, img_test):
    """
    :param img_train: reference to X_train
    :param img_val: reference to X_val
    :param img_test: reference to X_test
    :return: a list of preprocessed images
    """
    img_train = np.array(list(map(preprocess, img_train)))
    img_val = np.array(list(map(preprocess, img_val)))
    img_test = np.array(list(map(preprocess, img_test)))
    return img_train, img_val, img_test


x_train, x_val, x_test = preprocessed_img(X_train, X_val, X_test)


def processed_rand_image(graph_plt, process_img):
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(28, 28))  # width, height
    img = process_img[3000]
    graph_plt.imshow(img, cmap=graph_plt.get_cmap('gray'))
    graph_plt.axis('off')
    # save the plot
    return fig_save.savefig('plot/processed_img.png')


print("\n_________________________________ After preprocessing______________________________")
print(f"\npreprocessed x_train height(rows): {x_train.shape[0]}\nX_train dimensions(columns): {x_train.shape[1:]}")
print(f"\npreprocessed x_val height(rows): {x_val.shape[0]}\nX_val dimensions(columns): {x_val.shape[1:]}")
print(f"\npreprocessed x_test height(rows): {x_test.shape[0]}\nX_test dimensions(columns): {x_test.shape[1:]}")

print(f"\nTotal = {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]}")
print("_________________________________ After preprocessing______________________________")
sess.close()
