"""
tf2-chinese-mnist
workspace.py

Created by Justine Paul Sanchez Vitan.
Copyright © 2021 Justine Paul Sanchez Vitan. All rights reserved.
"""

import numpy as np

import data
import model

checkpoint_location = 'data/checkpoint/train.ckpt'


def train():
    img_data = data.Data()
    model.train_model(img_data.validation_split(0.2), 3, checkpoint_location)


def predict(img_location):
    cnn = model.load_model(checkpoint_location)

    img_ndarray = data.img_to_ndarray(img_location).reshape(-1, 64, 64, 1)
    prediction = cnn.predict(img_ndarray)
    argmax = np.argmax(prediction)
    probability = prediction[0][argmax]

    data.inspect_img(img_ndarray,
                     'Predicted Value: {:d}'.format(data.denormalize_value(argmax)))


def evaluate(checkpoint_location, count):
    img_data = data.Data(size_limit=count).extract_data()

    cnn = model.load_model(checkpoint_location)
    cnn.evaluate(img_data[0], img_data[1])


predict('data/test/test1.jpg')
