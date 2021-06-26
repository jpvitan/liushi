"""
tf2-chinese-mnist
data.py

Created by Justine Paul Sanchez Vitan.
Copyright © 2021 Justine Paul Sanchez Vitan. All rights reserved.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


class Data:

    def __init__(self, shuffle=True, size_limit=None):
        img_folder = 'data/train'
        img_info = 'data/train/info.csv'

        img_info_df = pd.read_csv(img_info)
        img_info_df_rows = img_info_df.shape[0]

        if size_limit is not None:
            img_info_df_rows = size_limit

        sequence = np.random.permutation(img_info_df_rows)

        if not shuffle:
            np.arange(0, img_info_df_rows)

        feature_list = []
        label_list = []

        for i in sequence:
            suite_id = img_info_df['suite_id'][i]
            sample_id = img_info_df['sample_id'][i]
            code = img_info_df['code'][i]

            img_filename = '{:s}/input_{:d}_{:d}_{:d}.jpg'.format(img_folder, suite_id, sample_id, code)

            feature_list.append(img_to_ndarray(img_filename))
            label_list.append(normalize_value(img_info_df['value'][i]))

        self.feature_ndarray = np.array(feature_list).reshape(-1, 64, 64, 1)
        self.label_ndarray = np.array(label_list)

    def extract_data(self, size_limit=None):
        feature = self.feature_ndarray
        label = self.label_ndarray

        if size_limit is not None:
            feature = feature[:size_limit]
            label = label[:size_limit]

        return feature, label

    def validation_split(self, validation_ratio):
        testing_size = int(self.label_ndarray.shape[0] * validation_ratio)
        training_feature = self.feature_ndarray[testing_size:]
        training_label = self.label_ndarray[testing_size:]
        validation_feature = self.feature_ndarray[:testing_size]
        validation_label = self.label_ndarray[:testing_size]

        return training_feature, training_label, validation_feature, validation_label


def normalize_value(value):
    if value <= 10:
        return value
    elif value == 100000000:
        return 14
    return int(math.log(value, 10) + 9)


def denormalize_value(normalized_value):
    if normalized_value <= 10:
        return normalized_value
    elif normalized_value == 14:
        return 100000000
    return int(math.pow(10, normalized_value) / math.pow(10, 9))


def img_to_ndarray(img_location):
    return np.array(list(Image.open(img_location).getdata())) / 255


def inspect_img(img_ndarray, title=None):
    plt.figure()
    plt.imshow(img_ndarray.reshape(64, 64))
    plt.colorbar()
    plt.title(title)
    plt.show()
