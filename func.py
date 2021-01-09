import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd
import cv2
import tensorflow.compat.v1 as tf

from keras.utils.np_utils import to_categorical

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# ensures that the random numbers are predictable. Resetting the seed everytime makes the random numbers predictable
# If this isn't done, different results will be gotten
np.random.seed(0)

# image shape
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


# print normalise x_train
# print("\n___________________________________ Before normalisation _________________________________")
# print(X_train)
# print("___________________________________ Before normalisation  _________________________________")

# Before preprocessing
# print(X_train.shape[0] == Y_train.shape[0])
# print(X_train.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(X_val.shape[0] == Y_val.shape[0])
# print(X_val.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))

# print(X_test.shape[0] == Y_test.shape[0])
# print(X_test.shape[1:] == (img_shape[0], img_shape[1], img_shape[2]))
# print("_________________________________ Before preprocessing______________________________")
# print(f"X_train height(rows): {X_train.shape[0]}\nX_train dimensions(columns): {X_train.shape[1:]}")
# print(f"y_train : {Y_train.shape}")
#
# print(f"\nX_val height(rows): {X_val.shape[0]}\nX_val dimensions(columns): {X_val.shape[1:]}")
# print(f"X_val : {X_val.shape}")
#
# print(f"\nX_test height(rows): {X_test.shape[0]}\nX_test dimensions(columns): {X_test.shape[1:]}")
# print(f"y_test : {Y_test.shape}")
#
# print(f"\nTotal = {X_train.shape[0] + X_test.shape[0] + X_val.shape[0]}")
# print("_________________________________ Before preprocessing______________________________")


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
    # list comprehension
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

def plot_histogram(graph_plt, img_train):
    """
    :param img_train: reference to X_train
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :return: the histogram of an image
    """
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(28, 28))  # width, height

    # extract image
    img = img_train[3000]
    # cv2.calcHist(images,channels,mask,histSize,ranges)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    graph_plt.figure()
    graph_plt.title("Grayscale Histogram of Image", fontsize=12)
    graph_plt.xlabel("Bins", fontsize=12)
    graph_plt.ylabel("Number of of Pixels", fontsize=12)
    graph_plt.plot(hist)
    graph_plt.xlim([0, 256])
    return fig_save.savefig('plot/equalised_processed.png')
    # return graph_plt.savefig('plot/hist_unprocessed.png')


# plot_histogram(plt, X_train)


def unprocessed_rand_image(graph_plt):
    """
    Histogram shows the distribution of pixel intensities (grayscale and coloured)
    :return: an  image
    """
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(28, 28))  # width, height

    # img = X_train[random.randint(0, len(X_train) - 1)]
    img = X_train[3000]
    graph_plt.imshow(img)
    graph_plt.axis('off')
    # save the plot
    return fig_save.savefig('plot/unprocessed_img.png')


# unprocessed_rand_image(plt)


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
    return fig_save.savefig('plot/grayscale.png')


# plot_grayscale(plt)


def plot_processed_histogram(graph_plt, img_file):
    """
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param img_file: image file
    :return: histogram plot
    """
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(10, 8))  # width, height
    img = cv2.imread(img_file, 0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    graph_plt.title("Grayscale Histogram of Image - Processed", fontsize=12)
    graph_plt.xlabel("Bins", fontsize=12)
    graph_plt.ylabel("Number of of Pixels", fontsize=12)
    graph_plt.plot(hist)

    graph_plt.xlim([0, 256])
    return fig_save.savefig('plot/equalised_hist.png')


# plot_processed_histogram(plt, 'plot/equalise.png')

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
    :param img: takes a grayscale image in the form of a numpy array
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


# print normalise x_train
# print("\n___________________________________ After normalisation  _________________________________")
# print(x_train)
# print("___________________________________ After normalisation _________________________________")


def processed_rand_image(graph_plt, process_img):
    # Setting the figure size
    fig_save = graph_plt.figure(figsize=(28, 28))  # width, height
    img = process_img[3000]
    graph_plt.imshow(img, cmap=graph_plt.get_cmap('gray'))
    graph_plt.axis('off')
    # save the plot
    return fig_save.savefig('plot/processed_img.png')


# processed_rand_image(plt, x_train)

# print("\n_________________________________ After preprocessing______________________________")
# print(f"\npreprocessed x_train height(rows): {x_train.shape[0]}\nx_train dimensions(columns): {x_train.shape[1:]}")
# print(f"\npreprocessed x_val height(rows): {x_val.shape[0]}\nx_val dimensions(columns): {x_val.shape[1:]}")
# print(f"\npreprocessed x_test height(rows): {x_test.shape[0]}\nx_test dimensions(columns): {x_test.shape[1:]}")
# print(f"\nTotal = {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]}")
# print("_________________________________ After preprocessing______________________________")


# One-hot encoding
"""
Carried out only the labels. It is the process of converting the labels into a vector. From the name, in a vector
only one element is hot and the rest are cold. That is just one element has a value of 1 while the rest are zero. 
The hot element (1) represents the location of the label in a vector and other available locations.
For example, the GTSRB has 43 labels (0 to 42). The vector will have 43 values, and the actual label will be one as
shown in the table below.
ClassId 	SignName                               One-hot encoded values
    0  "Speed limit (20km/h)"                   1 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0
    1 "Speed limit (30km/h)"                   0 1 0 0 0 0 0 0 0 0 .... 0 0 0 0 0
    2 "Speed limit (50km/h)"                   0 0 1 0 0 0 0 0 0 0 .... 0 0 0 0 0
    3 "Speed limit (60km/h)"                   0 0 0 1 0 0 0 0 0 0 .... 0 0 0 0 0
    4 "Speed limit (70km/h)"                   0 0 0 0 1 0 0 0 0 0 .... 0 0 0 0 0
    .
    .
    39 "Keep Left"                              0 0 0 0 0 0 0 0 0 0 .... 0 1 0 0 0
    40 "Roundabout mandatory"                   0 0 0 0 0 0 0 0 0 0 .... 0 0 1 0 0
    41 "End of no passing"                      0 0 0 0 0 0 0 0 0 0 .... 0 0 0 1 0
    42 "End of no passing veh over 3.5 tons"    0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 1
"""


# print("\n____________________ Before Encoding ______________________________")
# print(f"Training label: {Y_train}")
# print(f"Val label: {Y_val}")
# print(f"Test label: {Y_test}")
# print("____________________ Before Encoding ______________________________")


def one_hot_encode(train_label, val_label, test_label, num_of_classes):
    """
    One-hot encodes each of the labels
    :param train_label: reference to the train label
    :param val_label: reference to the validation label
    :param test_label: reference to the test label
    :param num_of_classes: reference to the number of classes present in the dataset
    :return: labels that are one-hot encoded
    """
    train_label = to_categorical(train_label, num_of_classes)
    val_label = to_categorical(val_label, num_of_classes)
    test_label = to_categorical(test_label, num_of_classes)
    return train_label, val_label, test_label


y_train, y_val, y_test = one_hot_encode(Y_train, Y_val, Y_test, 43)


# print("\n____________________ After Encoding ______________________________")
# print(f"Training label: {y_train}")
# print(f"Val label: {y_val}")
# print(f"Test label: {y_test}")
# print("____________________ After Encoding ______________________________")

def channel_depth(train_data, val_data, test_data):
    """
    this adds a depth of each of the training data
    :param train_data: reference to the train data
    :param val_data: reference to the train data
    :param test_data: reference to the train data
    :return: training data with a depth of one
    """
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
    val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], 1)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
    return train_data, val_data, test_data


x_train, x_val, x_test = channel_depth(x_train, x_val, x_test)
