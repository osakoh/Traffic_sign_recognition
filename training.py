import tensorflow.compat.v1 as tf
import pickle
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
# import tensorflow.keras.backend as K

from keras.models import Sequential
from tensorflow.keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator  # for data augmentation
from tensorflow.keras import layers

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from func import x_train, x_val, x_test, y_train, y_val, y_test, class_dict, preprocess


def cnn_model():
    img_shape = (32, 32, 1)
    conv_filter1 = 60  # kernels
    conv_filter2 = 30  # kernels
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
    model.add((Conv2D(conv_filter1, size_of_filter1, activation='relu')))
    model.add((Conv2D(conv_filter1, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))  # half of input nodes are dropped at each update

    model.add((Conv2D(conv_filter2, size_of_filter2, activation='relu')))
    model.add((Conv2D(conv_filter2, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))  # half of input nodes are dropped at each update

    model.add(Flatten())
    model.add(Dense(no_of_nodes, activation='relu'))
    model.add(Dropout(0.5))  # half of input nodes are dropped at each update
    model.add(Dense(no_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


my_model = cnn_model()
# my_model.summary()

# TODO: data augmentation
data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                              shear_range=0.1, rotation_range=10, zoom_range=0.2)

# TODO: training the model
# x_train: training images.
# y_train: training labels.
# used to validate the model: x_test, y_test
# total number of training examples in a single batch
# the entire training data (34799) cannot go through the CNN at once, the dataset is divided into number of batches
# shuffle: randomly shuffles the training data before each epoch
# total number of training examples in a single batch. Meaning it will take about 100 iterations to complete
# 1 epoch since there is about 35000 samples in the training set
train_batch_size = 350
gen_batch_size = 200  # generates 200 additional images along with their labels during training
# epoch: when the complete dataset is passed forward and backward through the CNN just once
no_of_epochs = 20  # no of times the dataset will go through the CNN
epoch_steps = 2000

stop = EarlyStopping(monitor='val_loss', min_delta=0.0004, patience=4, verbose=1)

# TODO: call to image generator(iterator); that is it only returns batches of images when needed
data_gen.fit(x_train)
batches = data_gen.flow(x_train, y_train, batch_size=gen_batch_size)
x_batch, y_batch = next(batches)


def plot_batch(graph_plt):
    """
    Plots a graph of 15 augmented images
    :param graph_plt: reference to  a plotting library - matplotlib.pyplot
    :return:
    """
    fig, axs = graph_plt.subplots(1, 15, figsize=(20, 3))
    fig.tight_layout()  # so the axis doesn't overlap

    for i in range(15):
        axs[i].imshow(x_batch[i].reshape(32, 32), cmap=graph_plt.get_cmap('gray'))
        axs[i].axis('off')
    return fig.savefig('plot/aug_img.png')


# TODO: train without data augmentation: takes about 1:30mins
# history = my_model.fit(x_train, y_train, epochs=no_of_epochs,
#                        batch_size=train_batch_size, verbose=1,
#                        validation_data=(x_val, y_val), shuffle=1, callbacks=[stop])


# TODO: train using data augmentation: takes about 10mins
# history = my_model.fit_generator(data_gen.flow(x_train, y_train, batch_size=50),
#                                  steps_per_epoch=epoch_steps, epochs=no_of_epochs,
#                                  validation_data=(x_val, y_val), shuffle=1, callbacks=[stop])


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
    fig_save.savefig('plot/early_stopping_accuracy.png')


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
    fig_save.savefig('plot/early_stopping_loss.png')


# TODO:evaluate using test images
score = my_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Score (Loss) = {score[0]}")
print(f"Test Accuracy = {score[1] * 100}")

# TODO: save model
# early_stopping
# my_model.save('model/early_stopping.h5')
# print("Saved successful")

# TODO: load saved model to visualise network architecture
# this_model = load_model('model/early_stopping.h5')
# print("\nModel Loaded successfully\n")

# TODO: visualise network architecture
# plot_model(this_model, to_file='model/early_stopping.png', show_shapes=True)