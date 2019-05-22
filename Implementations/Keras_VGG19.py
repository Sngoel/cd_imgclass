#Mandatory imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
from mpl_toolkits.mplot3d import Axes3D
import pylab

#Import Keras.
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop, Adam, SGD, Adamax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import load_model

train = pd.read_json("Data/train.json")
target_train = train['is_iceberg']
test = pd.read_json("Data/test.json")

#Generate the Train Data
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_band_3 = (X_band_1+X_band_2)/2
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis]
						, X_band_2[:, :, :, np.newaxis]
						, X_band_3[:, :, :, np.newaxis]], axis=-1)

#Generating the Test Data
X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3 = (X_band_test_1+X_band_test_2)/2
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
						, X_band_test_2[:, :, :, np.newaxis]
						, X_band_test_3[:, :, :, np.newaxis]], axis=-1)

#VGG19 Model
def getVggModel():
	base_model = VGG19(weights = 'imagenet', include_top=False, 
						input_shape = X_train.shape[1:], classes=1)

	x = base_model.get_layer('block5_pool').output
	x = GlobalMaxPooling2D()(x)
	x = Dense(512, activation = 'relu', name = 'fc1')(x)
	x = Dropout(0.2)(x)
	x = Dense(256, activation = 'relu', name = 'fc2')(x)
	x = Dropout(0.2)(x)
	x = Dense(128, activation = 'relu', name = 'fc3')(x)
	x = Dropout(0.2)(x)

	predictions = Dense(1, activation = 'sigmoid')(x)

	model = Model(input = base_model.input, output = predictions)    
	model.compile(loss ='binary_crossentropy',
					optimizer = 'Adam',
					metrics = ['accuracy'])
	return model

#Base CV Structure
def get_callbacks(filepath, patience = 2):
	print('\n')
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, mode = "min")
	print('\n')
	model_check = ModelCheckpoint(filepath, save_best_only = True)
	return [early_stop, model_check]

#Using K-fold Cross Validation.
def myBaseCrossTrain(X_train, target_train):
	folds = list(StratifiedKFold(n_splits = 4, shuffle = True, random_state = 16).split(X_train, target_train))
	y_test_pred_log = 0
	y_valid_pred_log = 0.0*target_train
	for j, (train_idx, test_idx) in enumerate(folds):
		print('\nFOLD = ',j)

		X_train_cv = X_train[train_idx]
		y_train_cv = target_train[train_idx]
		X_holdout = X_train[test_idx]
		Y_holdout= target_train[test_idx]
		file_path = "%s_model_weights.hdf5"%j

		callbacks = get_callbacks(filepath = file_path, patience = 5)
		galaxyModel = getVggModel()

		galaxyModel.fit(X_train_cv, y_train_cv,
						batch_size = 32,
						epochs = 100,
						verbose = 1,
						validation_data = (X_holdout, Y_holdout),
						callbacks = callbacks)

		#Getting the Best Model
		galaxyModel.load_weights(filepath = file_path)

		#Getting Training Score
		print('\n')
		score = galaxyModel.evaluate(X_train_cv, y_train_cv)
		print('Train loss:', score[0])
		print('Train accuracy:', score[1])

		#Getting Test Score
		print('\n')
		score = galaxyModel.evaluate(X_holdout, Y_holdout)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		#Getting validation Score.
		pred_valid = galaxyModel.predict(X_holdout)
		y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

		#Getting Test Scores
		temp_test = galaxyModel.predict(X_test)
		y_test_pred_log += temp_test.reshape(temp_test.shape[0])

	y_test_pred_log=y_test_pred_log/4
	print('\nLog Loss Validation= ',log_loss(target_train, y_valid_pred_log))
	return y_test_pred_log

preds = myBaseCrossTrain(X_train, target_train)

#Submission for each day.
submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = preds
submission.to_csv('Predictions.csv', index=False)