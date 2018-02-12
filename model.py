#
#Steering angle prediction model
#
# Import all packages
#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
%matplotlib inline
from PIL import Image
import keras
import math
import random
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam, RMSprop
from scipy.misc.pilutil import imresize
from sklearn.utils import shuffle
import cv2

import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint

import csv
import cv2
import numpy as np
#
# Read the data from the driving_log.csv
#
def bgr2rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r,g,b])
dir = 'mydata4'
lines = []
with open(dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)
imagesC = []
imagesL = []
imagesR = []
measurements = []
i = 0;
for line in lines:
    source_path = line[0]
    filenameC = source_path.split('/')[-1]
    current_path = dir + '/IMG/' + filenameC
    imageC = bgr2rgb(cv2.imread(current_path))
    imagesC.append(imageC)
    source_pathL = line[1]
    filenameL = source_pathL.split('/')[-1]
    current_path = dir + '/IMG/' + filenameL
    imageL = bgr2rgb(cv2.imread(current_path))
    imagesL.append(imageL)
    source_pathR = line[2]
    filenameR = source_pathR.split('/')[-1]
    current_path = dir + '/IMG/' + filenameR
    imageR = bgr2rgb(cv2.imread(current_path))
    imagesR.append(imageR)
    measurement = float(line[3])
    measurements.append(measurement)
#
# Simple setup of the training samples without the use of the generator.
#
X_t = []
m_t = []
X_train = []
correction = 0.25
zero_steer = 0.1
dsize = len(imagesC)
#
# For all the images read in from the driving_log.csv
#
for i in range(0, dsize):
    #
    # Give priority to all the images when the steering angle is above a threshold
    #
    sa = measurements[i]
    good_steer = (sa < -zero_steer and sa > float('-inf') ) or (sa > zero_steer and sa < float('inf'))
    if(good_steer or random.random() > 0.9) :
        #
        # Use cases where the steering angle is above threshold or cases with zero steer with a probability of 10%
        #
        X_t.append(imagesC[i])
        m_t.append(sa)
        X_t.append(flip_image(imagesC[i]))
        m_t.append(-sa)
        X_t.append(imagesL[i])
        m_t.append(sa+correction)
        X_t.append(imagesR[i])
        m_t.append(sa-correction)
    else:
        #
        # When the steering angle is close to zero, add a random translation to the left or right and use it.
        #
        use_image, use_angle = trans_image(imagesC[i], sa)
        X_t.append(use_image)
        m_t.append(use_angle)

X_train = np.array(X_t)
y_train = np.array(m_t)
#
# Set this variable to start a model if it is set to false it will load the current model and improve it.
#
make_model = True
#
# Set the cropping window size
#
chs, rows, cols = 3, 90, 320  # Trimmed image format

#
# Make the model resemble the NVIDIA model
#
model = Sequential()
#
# Crop the image to only consider the region of interest which concentrates on the road ahead.
#
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#
# Normalize all the pixels to -0.5 to +0.5
#
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(rows, cols, chs), output_shape=(rows, cols, chs)))
#
# Add a 5x5 Convolution layer and MaxPool the output.
#
model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='same', activation="relu"))
model.add(MaxPooling2D())
#
# Add a second 5x5 Convolution layer.
#
model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='same', activation="relu"))
#model.add(MaxPooling2D())
#
# Add a third 5x5 Convolution layer.
#
model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='same', activation="relu"))
model.add(MaxPooling2D())
#
# Add a fourth 3x3 Convolution layer.
#
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same', activation="relu"))
#
# Add a fifth 3x3 Convolution layer.
#
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same', activation="relu"))
#
# Flatten the image from the convolution layer.
#
model.add(Flatten())
#
# Add a dropout layer to make the learning robust to missing data.
#
model.add(Dropout(0.5))
#
# Add a Dense layer with output size 1164 feeding into an exponential linear unit
#
model.add(Dense(1164))
model.add(ELU())
#
# Add a dropout layer to make the learning robust to missing data.
#
model.add(Dropout(0.5))
#
# Add a Dense layer with output size 100 feeding into an exponential linear unit
#
model.add(Dense(100))
model.add(ELU())
#
# Add a dropout layer to make the learning robust to missing data.
#
model.add(Dropout(0.5))
#
# Add a Dense layer with output size 50 feeding into an exponential linear unit
#
model.add(Dense(50))
model.add(ELU())
#
# Final output layer
#
model.add(Dense(1))
#
# Use this branch when starting out with no model.
#
if(make_model):
    model.compile(loss='mse', optimizer='adam')
else:
    #
    # This code is used when the model is already present and needs to be improved with the new data.
    #
    from keras.models import load_model
    model = load_model('model-mydata-best.h5')
    keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mse', optimizer='adam')
#
# Fit the data into the model and save the history of the loss for both training and validation to graph later.
#
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
#
# Save the model.
#
model.save('model.h5')
