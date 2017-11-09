#Implement simple Keras Conv NN
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import json
import os

np.random.seed(1234)

#Data Files
DATA_DIR = "Data/"

# Training data
train_raw = open(DATA_DIR + "train.json", "r").read()
train = json.loads(train_raw)

# Testing data
test_raw = open(DATA_DIR + 'test.json', 'r').read()
test = json.loads(test_raw)

# Getting the images' pixel arrays out of the lists
train_band_2s = np.array([train[x]['band_2'] for x in range(len(train))])
train_band_1s = np.array([train[x]['band_1'] for x in range(len(train))])

train_labels = np.array([train[x]['is_iceberg'] for x in range(len(train))])

test_band_2s = np.array([test[x]['band_2'] for x in range(len(test))])
test_band_1s = np.array([test[x]['band_1'] for x in range(len(test))])

train_band_1s = train_band_1s.reshape(train_band_1s.shape[0], 75, 75, 1)
train_band_2s = train_band_2s.reshape(train_band_2s.shape[0], 75, 75, 1)

test_band_1s = test_band_1s.reshape(test_band_1s.shape[0], 75, 75, 1)
test_band_2s = test_band_2s.reshape(test_band_2s.shape[0], 75, 75, 1)

# Changing number types
train_band_1s = train_band_1s.astype('float32')
train_band_2s = train_band_2s.astype('float32')
test_band_1s = test_band_1s.astype('float32')
test_band_2s = test_band_2s.astype('float32')

# pixel values spectrum of [0, 1]
train_band_1s /= 255
train_band_2s /= 255
test_band_1s /= 255
test_band_2s /= 255

train_labels = np_utils.to_categorical(train_labels, 2)

#Model
model = Sequential()

#Conv Layer 1
model.add(Convolution2D(128, (3, 3), activation='relu', input_shape=(75, 75, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Conv Layer 2
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Conv Layer 3
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Conv Layer 4
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Falttening data for Dense layers
model.add(Flatten())

#Dense Layer 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

#Dense Layer 2
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=['accuracy'])
model.fit(train_band_1s, train_labels, batch_size=24, epochs=50, verbose=1)

scores = model.predict(test_band_1s)
scores = [scores[x][1] for x in range(len(scores))]
ids = [test[x]['id'] for x in range(len(test))]

scores = pd.DataFrame(scores)
ids = pd.DataFrame(ids)
frames = [ids, scores]
preds = pd.concat(frames, axis=1)
preds.columns = ['id', 'is_iceberg']

preds.to_csv("keras_cnn_3-preds.csv", sep=',', index=False)
model.save("keras_cnn_3.h5")