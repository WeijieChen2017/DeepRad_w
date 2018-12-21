#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np


def data_pre_PVC(data_mri, data_pet):

    # global IMG_ROWS, IMG_COLS
    # global IDX_SLICE, FA_NORM

    IMG_ROWS = 512
    IMG_COLS = 512
    IDX_SLICE = 150
    FA_NORM = 35000.0

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 4))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 4))

    data_pet = np.divide(data_pet, FA_NORM)

    X[0, :, :, 0] = data_pet[:, :, IDX_SLICE]
    X[0, :, :, 1] = data_mri[:, :, IDX_SLICE] == 1
    X[0, :, :, 2] = data_mri[:, :, IDX_SLICE] == 2
    X[0, :, :, 3] = data_mri[:, :, IDX_SLICE] == 3

    Y = X

    return X, Y
