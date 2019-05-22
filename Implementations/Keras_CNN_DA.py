import numpy as np
import pandas as pd
import h5py
import time

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten, Activation
from keras.layers.merge import Concatenate, add
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


def build_model():

    image_input = Input(shape = (75, 75, 3), name = 'images')
    angle_input = Input(shape = [1], name = 'angle')
    activation = 'elu'
    bn_momentum = 0.99

    img_1 = Conv2D(32, kernel_size = (3, 3), activation = activation, padding = 'same') ((BatchNormalization(momentum=bn_momentum)) (image_input))
    img_1 = MaxPooling2D((2,2)) (img_1)
    #img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(128, kernel_size = (3, 3), activation = activation, padding = 'same') ((BatchNormalization(momentum=bn_momentum)) (img_1))
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.25) (img_1) 
    img_1 = Conv2D(128, kernel_size = (3, 3), activation = activation, padding = 'same') ((BatchNormalization(momentum=bn_momentum)) (img_1))
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(0.25) (img_1) 

    # Residual block
    img_2 = Conv2D(32, kernel_size = (3, 3), activation = activation, padding = 'same') ((BatchNormalization(momentum=bn_momentum)) (img_1))
    img_2 = MaxPooling2D((2,2)) (img_2)
    #img_2 = Dropout(0.25) (img_2)
    img_2 = Conv2D(64, kernel_size = (3, 3), activation = activation, padding = 'same') ((BatchNormalization(momentum=bn_momentum)) (img_2))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dropout(0.25) (img_2)
    img_2 = Conv2D(128, kernel_size = (3, 3), activation = activation, padding = 'same') ((BatchNormalization(momentum=bn_momentum)) (img_2))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dropout(0.25) (img_2)

    img_res = add([img_1, img_2])

    # Filter resudial output
    img_res = Conv2D(128, kernel_size = (3, 3), activation = activation) ((BatchNormalization(momentum=bn_momentum)) (img_res))
    img_res = MaxPooling2D((2,2)) (img_res)
    img_res = Dropout(0.2)(img_res)
    img_res = GlobalMaxPooling2D() (img_res)

    cnn_out = (Concatenate()([img_res, BatchNormalization(momentum=bn_momentum)(angle_input)]))

    dense_layer = Dropout(0.5) (BatchNormalization(momentum=bn_momentum) (Dense(512, activation = activation) (cnn_out)))
    dense_layer = Dropout(0.5) (BatchNormalization(momentum=bn_momentum) (Dense(256, activation = activation) (dense_layer)))
    dense_layer = Dropout(0.5) (BatchNormalization(momentum=bn_momentum) (Dense(128, activation = activation) (dense_layer)))

    output = Dense(1, activation = 'sigmoid') (dense_layer)

    model = Model([image_input, angle_input], output)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model

def get_callbacks( weight_save_path, no_improv_epochs = 10, min_delta = 1e-4 ):
    es = EarlyStopping( 'val_loss', patience = no_improv_epochs, mode = 'min', min_delta = min_delta )
    ms = ModelCheckpoint( weight_save_path, 'val_loss', save_best_only = True )

    return [ es, ms ]

def generate_data( data ):
    X_band_1=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_1']] )
    X_band_2=np.array( [np.array(band).astype(np.float32).reshape(75, 75) 
                        for band in data['band_2']] )
    X = np.concatenate( [X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], \
                        ((X_band_1 + X_band_2)/2)[:, :, :, np.newaxis]], axis=-1 )
    return X

def augment_data( generator, X1, X2, y, batch_size = 32 ):
    generator_seed = np.random.randint( 9999 )
    gen_X1 = generator.flow( X1, y, batch_size = batch_size, seed = generator_seed )
    gen_X2 = generator.flow( X1, X2, batch_size = batch_size, seed = generator_seed )

    while True:
        X1i = gen_X1.next()
        X2i = gen_X2.next()

        yield [X1i[0], X2i[1]], X1i[1]
    
Train = 'Data/train.json'
Test = 'Data/test.json'
Weights = 'model_weights.hdf5'

Seed = np.random.randint(9999)
Batch_Size = 32
Epochs = 1

train_data = pd.read_json(Train)
train_data['inc_angle'] = train_data['inc_angle'].replace('na', 0)
train_data['inc_angle'] = train_data['inc_angle'].astype(float).fillna(0.0)

X = generate_data(train_data)
X_angle = train_data['inc_angle']
X_iceberg = train_data['is_iceberg']

X_train, X_val, X_angle_train, X_angle_val, X_iceberg_train, X_iceberg_val = train_test_split(X, X_angle, X_iceberg, train_size = .8, random_state = Seed)
callback_list = get_callbacks(Weights, 20)

model = build_model()
start_time = time.time()

image_augmentation = ImageDataGenerator(rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        fill_mode='nearest')

input_generator = augment_data(image_augmentation, X_train, X_angle_train, X_iceberg_train, batch_size = Batch_Size)

model.fit_generator(input_generator, steps_per_epoch = 4096/Batch_Size, epochs = Epochs, 
                    callbacks = callback_list, verbose = 1,
                    validation_data = augment_data(image_augmentation, X_val, X_angle_val, X_iceberg_val, batch_size = Batch_Size),
                    validation_steps = len(X_val)/Batch_Size)

m, s = divmod(time.time() - start_time, 60)
print('Model fitting done. Total time: {}m {}s'.format(int(m), int(s)))

model.load_weights(Weights)

train_score = model.evaluate([X_train, X_angle_train], X_iceberg_train, verbose = 1)
print('Train score: ', train_score[0])
print('Train accuracy: ', train_score[1])

val_score = model.evaluate([X_val, X_angle_val], X_iceberg_val, verbose = 1)
print('Validation loss: ', val_score[0])
print('Validation accuracy: ', val_score[1])

print('Loading and evaluating on test data')
test_data = pd.read_json(Test)
X_test = generate_data(test_data)
X_a_test = test_data['inc_angle']
test_predictions = model.predict([X_test, X_a_test], verbose = 1)
test_score = model.evaluate([X_test, X_a_test], verbose = 1)
print('Test loss: ', test_score[0])
print('Test accuracy: ', test_score[1])
submission = pd.DataFrame()
submission['id'] = test_data['id']
submission['is_iceberg'] = test_predictions.reshape((test_predictions.shape[0]))
submission.to_csv('Predictions.csv', index = False)