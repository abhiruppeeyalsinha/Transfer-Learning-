# from inspect import classify_class_attrs
# from re import S
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pathlib
import numpy as np
import cv2, pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.models import  Sequential
# from keras


Image_shape = (224,224)
img_label = []

classifier = Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
 input_shape=Image_shape+(3,))])

# img_file = cv2.imread(r"E:\CNN Project\features of car cnn filters\water-lily.jpg")
img_file = cv2.imread("Gold_fish.jpg")

img_file = cv2.resize(img_file,(Image_shape))
img_file = np.array(img_file)/255
image_file = np.expand_dims(img_file,0)
# print(image_file.shape)

result = classifier.predict(image_file)
# print(f"{result}")


predicted_label_index = np.argmax(result)
print(f"values:- {predicted_label_index}")


with open("ImageNetLabels.txt","r") as f:
    img_label = f.read().splitlines()
print(img_label[predicted_label_index])

