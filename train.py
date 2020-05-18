import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissor": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)

#eg. mapper of rock it will give you 0
def mapper(val):
    return CLASS_MAP[val]

#add on to the squezenet neural n/w having different layers
def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),#3 channels(rgb)
        Dropout(0.5),#to prevent overfitting,we have 50% dropout rate
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),#rectified linear unit
        GlobalAveragePooling2D(),#calculates avg of each feature map
        Activation('softmax')#The softmax function is an activation function that turns numbers into probabilities which sum to one
    ])
    return model


# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))#to load the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#convert bgr to rgb
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

'''
dataset = [
    [[...], 'rock'],
    [[...], 'paper'],
    ...
]
'''


#these labels need to be mapped as integers as they are in human readable language
data, labels = zip(*dataset)#to unpack dataset:img data in data & labels in label list
labels = list(map(mapper, labels))


'''
labels: rock,paper,paper,scissors,rock...
one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
'''

# one hot encode the labels
labels = np_utils.to_categorical(labels)

# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),#adam optimiser with learning rate 0.0001
    loss='categorical_crossentropy',#as we are doing classification, categorical_crossentropy is the best to find the loss
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=10)# Epoch is when an ENTIRE dataset is passed forward and backward through the neural network

# save the model for later use
model.save("rock-paper-scissors-model.h5")
