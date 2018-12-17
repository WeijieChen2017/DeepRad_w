#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import datetime
from keras import backend as K


def mean_squared_error_1e12(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)*1e12


def mean_squared_error_1e6(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)*1e6


def mean_absolute_error_1e6(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)*1e6


def psnr(y_true, y_pred):
#     return -10.0*K.log(1.0/(K.mean(K.square(y_pred - y_true))))/K.log(10.0)
    mse = K.mean(K.square(y_pred - y_true))
    return (20 - 10 * K.log(mse)/K.log(10.0))*1e3


def mse1e12_weighted(y_true, y_pred):
    diff = np.dot(K.square(y_pred - y_true), y_pred)
    loss = K.mean(diff, axis=-1)
    return loss*1e12


def output_dataset(filename, list_train, list_val):
    file_name = filename + "dataset.txt"
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")

    with open(file_name, "w") as text_file:
        print("Date: ", date, file=text_file)
        print("Produced by Winston Chen", file=text_file)
        print("Number of the training set: ", 14, file=text_file)
        print("Number of the testing set: ", 5, file=text_file)
        print(' ', file=text_file)
        print("The training set names:", file=text_file)
        for i in list_train:
            print(i, file=text_file)
        print(' ', file=text_file)
        print("The validation set names:", file=text_file)
        for i in list_val:
            print(i, file=text_file)
        print(' ', file=text_file)
        print("The testing set names:", file=text_file)
        # print(list_patient[LOOCV], file=text_file)
