import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import cv2
import glob
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from PIL import Image

from func import class_dict, y_train, y_val, y_test, x_train, x_val, x_test

url = ['https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg',
       'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg',
       'https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign'
       '-slippery-road.jpg',
       'https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg',
       'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg']
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


# r = requests.get(url[3], stream=True)
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

img_path = Image.open(r'pred_test/cars-and-automobiles-must-turn-left-ahead-sign.jpg')
images = [cv2.imread(file) for file in glob.glob("pred_test/*.jpg")]
# print(len(images))

# TODO: test images
# for img in images:
#     predict_img(img, this_model, 0.90)

# TODO: view convolutional filter outputs
# layer = [layer for layer in this_model.layers]
# print(type(layer))  # 13 layers in total
# print(len(layer))  # shows the number of layers in the model
# filters, biases = this_model.layers[1].get_weights()


# print(layer[1].name, filters.shape)

# for i in range(len(layer)):
#     print(layer[i].name)


# activation_model = models.Model(input_shape=this_model.input, outputs= layer_outputs)
# activations = activation_model.predict(img_path)
# print(activations)

y_pred = this_model.predict(x_test)
# print(type(y_pred))
# np.set_printoptions(threshold=sys.maxsize)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)

# TODO: view accuracy report
pred = this_model.predict_classes(x_test)
print(classification_report(y_test, pred, target_names=traffic_signs))


# TODO: plot confusion matrix
# classes = 43
# confusion_matrix = confusion_matrix(y_pred, y_test)
#
# plt.figure(figsize=(25, 25))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# tick_marks = np.arange(classes)
#
# plt.xticks(range(classes), range(classes))
# plt.yticks(range(classes), range(classes))
#
#
# plt.xticks(rotation=90)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel('Predicted', fontsize=24)
# plt.ylabel('True', fontsize=24)
# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, cm[i, j], horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")
# plt.savefig('plot/confusion_matrix_early_stopping.png')
# plt.show()
