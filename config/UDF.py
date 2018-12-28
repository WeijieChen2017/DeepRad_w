#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import datetime
from keras import backend as K
from GL.w_global import GL_get_value


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


def Gray_White_CSF(y_true, y_pred):


    W_PGWC = (GL_get_value("W_PGWC"))

    # [PET, Gray, White, CSF]
    weight = [int(W_PGWC[0]),
              int(W_PGWC[1]),
              int(W_PGWC[2]),
              int(W_PGWC[3])]

    # k1 = np.array([[-1, -1, -1],
    #                [0, 0, 0],
    #                [1, 1, 1]], dtype=np.float32)
    # k2 = np.array([[-1, 0, 1],
    #                [-1, 0, 1],
    #                [-1, 0, 1]], dtype=np.float32)
    # k1 = K.reshape(k1, (3, 3, 1, 1))
    # k2 = K.reshape(k2, (3, 3, 1, 1))
    # pet_true = K.reshape(y_true[:, :, :, 0], (1, 512, 512, 1))
    # pet_pred = K.reshape(y_pred[:, :, :, 0], (1, 512, 512, 1))
    # norm1 = K.conv2d(pet_true, k1, strides=(1, 1), padding='same', dilation_rate=(1, 1))
    # norm2 = K.conv2d(pet_true, k2, strides=(1, 1), padding='same', dilation_rate=(1, 1))
    #     grads = K.sqrt(K.square(norm1) + K.square(norm2))

    pet_error = K.mean(K.square(y_pred[:, :, :, 0] - y_true[:, :, :, 0]), axis=-1)
    #     pet_error = K.mean(K.square(y_pred[:,:,:,0] - y_true[:,:,:,0]), axis=-1)*((-grads)/3500 + 10)

    #     pet_loss = mean_squared_error(pet_true, pet_pred)

    csf_mask = y_true[:, :, :, 1]
    csf_true = csf_mask * y_true[:, :, :, 0]
    csf_pred = csf_mask * y_pred[:, :, :, 0]
    #     csf_error = K.mean(K.square(csf_pred - csf_true), axis=-1)
    csf_error = K.max(K.square(csf_pred)) - K.mean(K.square(csf_true))

    gm_mask = y_true[:, :, :, 2]
    gm_true = gm_mask * y_true[:, :, :, 0]
    gm_pred = gm_mask * y_pred[:, :, :, 0]
    gm_error = K.mean(K.square(gm_pred - gm_true), axis=-1)

    wm_mask = y_true[:, :, :, 3]
    # wm_mask = np.bitwise_and(wm_mask == 1, y_true[:, :, :, 0] < MRI_TH)
    wm_true = wm_mask * y_true[:, :, :, 0]
    wm_pred = wm_mask * y_pred[:, :, :, 0]
    # wm_sum = K.sum( wm_pred, axis=-1 )
    #     wm_error = K.mean(K.square(wm_pred - wm_true), axis=-1)
    wm_error = K.max(K.square(wm_pred)) - K.mean(K.square(wm_true))
    return pet_error * weight[0] + gm_error * weight[1] + wm_error * weight[2] + csf_error * weight[3]


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
