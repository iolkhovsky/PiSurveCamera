from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization


def make_classifier(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 64 x 32 x 32

    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 256 x 16 x 16

    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 512 x 8 x 8

    model.add(BatchNormalization())
    model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 1024 x 4 x 4

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model
