import numpy as np

np.random.seed(0)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import h5py


def training():
    train_data_dir = './data/train'
    validation_data_dir = './data/test'

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 16
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255)
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            train_data_dir,  # this is the target directory
            target_size=(512, 512),  # all images will be resized to 512x512
            batch_size=batch_size,
            class_mode='binary',   # since we use binary_crossentropy loss, we need binary labels
            shuffle=False)
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(512, 512),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False)

    model.fit_generator(
            train_generator,
            steps_per_epoch=320 // batch_size,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=80 // batch_size)

    # save model to json
    model_json = model.to_json()
    with open("./models/model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('./models/first_try.h5')  # always save your weights after training or during training
    print("Saved model %s" % './models/first_try.h5')
