import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

train = pd.read_csv((sys.argv[1]);

train_x = train.feature.str.split(' ').tolist()
train_x = np.array(train_x).astype('float32')
train_x = train_x / 255

train_y = np.array(train.label).astype('float32')

num_classes = 7

valid_num = 5000

train_x = train_x[valid_num:]
train_y = train_y[valid_num:]

valid_x = train_x[:valid_num]
valid_y = train_y[:valid_num]

train_x = train_x.reshape(train_x.shape[0], 48, 48, 1)
valid_x = valid_x.reshape(valid_x.shape[0], 48, 48, 1)

train_y = np_utils.to_categorical(train_y,num_classes)
valid_y = np_utils.to_categorical(valid_y,num_classes)

batch_size = 128
epoch = 100

model = Sequential()

model.add(Conv2D(64,(5,5),input_shape=(48,48,1),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2,padding='same'))
# model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.3))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)

datagen = ImageDataGenerator(
        rotation_range=20,                  
        width_shift_range=0.2,              
        height_shift_range=0.1,            
        zoom_range=[0.9, 1.1],
        shear_range=0.1,
        horizontal_flip=True,
        )

# datagen.fit(train_x)

callbacks = []
callbacks.append(ModelCheckpoint('model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))
    

model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
                    steps_per_epoch=train_x.shape[0]/batch_size,
                    epochs=epoch,
                    validation_data=(valid_x, valid_y),
                    callbacks=callbacks,
                    )