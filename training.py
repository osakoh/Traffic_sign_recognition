import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from func import x_train, x_val, x_test, y_test, y_val, y_train

tf.disable_v2_behavior()

# start Tensorflow-gpu session
gpu_opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))


# print("\n___________________________________ After reshape _______________________________")
# print(f"shape of x_train: {x_train.shape}")
# print(f"shape of x_val: {x_val.shape}")
# print(f"shape of x_val{x_test.shape}")
# # print(f"Total = {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]}")
# print("___________________________________ After reshape _______________________________")


# print("\n____________________ After Encoding ______________________________")
# print(f"Training label: {y_train}")
# print(f"Val label: {y_val}")
# print(f"Test label: {y_test}")
# print("____________________ After Encoding ______________________________")


def cnn_model():
    img_shape = (32, 32, 1)
    conv_filter1 = 30  # kernels
    conv_filter2 = 15  # kernels
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_classes = 43
    no_of_nodes = 500
    # stride = 0
    # padding = 1

    model = Sequential()
    model.add((Conv2D(conv_filter1, size_of_filter1, input_shape=(img_shape[0],
                                                                  img_shape[1],
                                                                  img_shape[2]), activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(conv_filter2, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Flatten())
    model.add(Dense(no_of_nodes, activation='relu'))
    model.add(Dropout(0.5))  # half of input nodes are dropped at each update
    model.add(Dense(no_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


my_model = cnn_model()
my_model.summary()

# TODO: training the model
# x_train: training images.
# y_train: training labels.
# used to validate the model: x_test, y_test
# total number of training examples in a single batch
# since the entire training data (34799) cannot pass through the CNN at once, the dataset is divided into number of batches
# shuffle: randomly shuffles the training data before each epoch
# total number of training examples in a single batch. Meaning it will take about 7 iterations to complete
# 1 epoch since there is about 35000 samples in the training set
batch_size = 400
# epoch: when the complete dataset is passed forward and backward through the CNN just once
no_of_epochs = 10  # no of times the dataset will go through the CNN

history = my_model.fit(x_train, y_train, epochs=no_of_epochs,
                       batch_size=batch_size, verbose=1,
                       validation_data=(x_val, y_val), shuffle=1)


def acc_plot(graph_plt, hist):
    """
    Plots the Accuracy graph
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param hist: reference to  a  history.history
    """
    # Setting the figure size
    fig_save = graph_plt.figure()  # width, height
    history_dict = hist.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)
    line1 = graph_plt.plot(epochs, val_acc_values, label='Validation/Test accuracy')
    line2 = graph_plt.plot(epochs, acc_values, label='Training accuracy')
    graph_plt.setp(line1, linewidth=2.0, marker='x', markersize=5.0)
    graph_plt.setp(line2, linewidth=2.0, marker='*', markersize=5.0)
    graph_plt.xlabel('Epochs')
    graph_plt.ylabel('Accuracy')
    graph_plt.grid(True)
    graph_plt.legend()
    graph_plt.title('Plot for model accuracy')
    graph_plt.show()
    # fig_save.savefig('plot/change_accuary.png')


# acc_plot(plt, history)


def loss_plot(graph_plt, hist):
    """
    Plots the Accuracy graph
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :param hist: reference to  a  history.history
    """
    # Setting the figure size
    fig_save = graph_plt.figure()  # width, height
    history_dict = hist.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    line1 = graph_plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
    line2 = graph_plt.plot(epochs, loss_values, label='Training Loss')
    graph_plt.setp(line1, linewidth=2.0, marker='x', markersize=5.0)
    graph_plt.setp(line2, linewidth=2.0, marker='*', markersize=5.0)
    graph_plt.xlabel('Epochs')
    graph_plt.ylabel('Loss')
    graph_plt.grid(True)
    graph_plt.legend()
    graph_plt.title('Plot for model loss')
    # fig_save.savefig('plot/change_loss.png')


# loss_plot(plt, history)

#### EVALUATE USING TEST IMAGES
score = my_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Score (Loss) = {score[0]}")
print(f"Test Accuracy = {score[1] * 100}")

traffic_signs = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of Speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing veh over 3.5 tons",
    "Right-of-way at intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Veh > 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycle crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End speed + passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep Left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing veh over 3.5 tons"]

# end Tensorflow-gpu session
sess.close()
