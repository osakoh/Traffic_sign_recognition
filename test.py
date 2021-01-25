import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import cv2
import glob
import seaborn as sns
from keras.models import load_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
from PIL import Image
from sklearn.utils.multiclass import unique_labels

from func import class_dict, y_train, y_val, y_test, x_train, x_val, x_test

url = []
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


# TODO: testing using request
# r = requests.get(url, stream=True)
# img = Image.open(r.raw)
# plt.imshow(img, cmap=plt.get_cmap('gray'))

# pickle_in = open("model/model_trained_10_epoch.p", "rb")
# model_load = pickle.load(pickle_in)

def pre_processing(img):
    """
    :param img: takes an image as an argument
    :return: a processed(grayscale, equalised, normalised) image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def predict_img(img, the_model, threshold):
    """
    predict(): to get the probability and predict_classes(),to get the label
    :param threshold: the minimum value that must be reached to predict an image
    :param the_model: saved trained model
    :param img: img to test
    """
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = pre_processing(img)
    img = img.reshape(1, 32, 32, 1)

    # predicted class
    pred = the_model.predict_classes(img)  # first predict the class ID
    prob_val = np.amax(pred)  # returns the maximum element in an array/list

    if prob_val > threshold:
        print(f"\nPredicted sign: {class_dict[int(pred)]} Class: {int(pred)}\n"
              f"Actual sign: {class_dict[int(prob_val)]} Class: {prob_val}")


# TODO: load saved model
this_model = load_model('model/final_no_callback.h5')
print("\nModel Loaded successfully\n")

# img_path = Image.open(r'pred_test/cars-and-automobiles-must-turn-left-ahead-sign.jpg')

# TODO: test images
# gets all images in a folder with the .jpg extension
images = [cv2.imread(file) for file in glob.glob("pred_test/*.jpg")]
# check the number of images in the folder
print(f"Folder contains {len(images)} images")
for img in images:
    predict_img(img, this_model, 0.90)


y_pred = this_model.predict(x_test)
# print(type(y_pred))
# np.set_printoptions(threshold=sys.maxsize)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
# print(cm)

# TODO: view accuracy report
pred = this_model.predict_classes(x_test)

# print(classification_report(y_test, pred, target_names=traffic_signs))

# TODO: plot confusion matrix
# classes = 43
con_matrix = confusion_matrix(pred, y_test)


def plot_cm(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots(figsize=(35, 35))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=25, rotation=45)
    plt.yticks(tick_marks, fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    plt.ylabel('True label', fontsize=25)
    plt.title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #            title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontsize=20,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('plot/confusion_matrix_early_stopping_with_names.png')
    # plt.show()
    return ax


