"""
tf2-chinese-mnist
model.py

Created by Justine Paul Sanchez Vitan.
Copyright © 2021 Justine Paul Sanchez Vitan. All rights reserved.
"""

from tensorflow.keras import Sequential
from tensorflow.keras import layers


def create_model():
    model = Sequential()

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                            input_shape=(64, 64, 1)))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1024, activation='relu'))
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=15, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
