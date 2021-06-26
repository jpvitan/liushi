"""
tf2-chinese-mnist
model.py

Created by Justine Paul Sanchez Vitan.
Copyright © 2021 Justine Paul Sanchez Vitan. All rights reserved.
"""

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


def create_model():
    model = Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                            input_shape=(64, 64, 1)))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1024, activation='relu'))
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=15, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(data, epochs, checkpoint_location):
    def plot_history(history):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(history['loss'], label='training loss')
        axs[0].plot(history['val_loss'], label='validation loss')
        axs[0].legend(loc='upper left')
        axs[0].set_title('training data vs validation data')

        axs[1].plot(history['accuracy'], label='testing accuracy')
        axs[1].plot(history['val_accuracy'], label='validation accuracy')
        axs[1].set_ylim([0, 1])
        axs[1].legend(loc='upper left')
        axs[1].set_title('accuracy')

        axs.flat[0].set(xlabel='epochs', ylabel='loss')
        axs.flat[1].set(xlabel='epochs', ylabel='accuracy')

        plt.show()

    model = create_model()

    training_feature = data[0]
    training_label = data[1]
    validation_feature = data[2]
    validation_label = data[3]

    callback = ModelCheckpoint(checkpoint_location, save_weights_only=True, verbose=1)

    history = model.fit(training_feature, training_label, validation_data=(validation_feature, validation_label),
                        epochs=epochs, callbacks=[callback], verbose=1).history
    plot_history(history)

    return model


def load_model(checkpoint_location):
    model = create_model()

    model.load_weights(checkpoint_location)

    return model
