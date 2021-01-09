import tensorflow.compat.v1 as tf
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

from func import class_dict, y_train, y_val, y_test

# Function call stack: keras_scratch_graph Error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

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

r = requests.get(url[3], stream=True)
img = Image.open(r.raw)
# plt.imshow(img, cmap=plt.get_cmap('gray'))

pickle_in = open("model/model_trained_10_epoch.p", "rb")
model_load = pickle.load(pickle_in)


def pre_processing(img):
    """
    :param img: takes an image as an argument
    :return: a processed(grayscale, equalised, normalised) image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def predict_img(img, graph_plt):
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = pre_processing(img)
    graph_plt.show()
    img = img.reshape(1, 32, 32, 1)

    pred = (model_load.predict_classes(img))  # first predict the class ID
    actual = model_load.predict(img)
    print(f"\npredicted sign: {class_dict[int(pred)]}-{int(pred)} | actual sign: {actual}")

img_path = Image.open(r'pred_test/cars-and-automobiles-must-turn-left-ahead-sign.jpg')
predict_img(img, plt)


