#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np
from GL.w_global import GL_get_value


def data_pre_PVC(data_mri, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IDX_SLICE = GL_get_value("IDX_SLICE")
    FA_NORM = GL_get_value("FA_NORM")
    MRI_TH = float(GL_get_value("MRI_TH"))

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 4))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 4))
    Z = np.zeros((1, IMG_ROWS, IMG_COLS, 2))

    data_pet = np.divide(data_pet, FA_NORM)

    Z[0, :, :, 0] = data_pet[:, :, IDX_SLICE] <= MRI_TH
    Z[0, :, :, 1] = data_mri[:, :, IDX_SLICE] == 3

    X[0, :, :, 0] = data_pet[:, :, IDX_SLICE]
    X[0, :, :, 1] = data_mri[:, :, IDX_SLICE] == 1
    X[0, :, :, 2] = data_mri[:, :, IDX_SLICE] == 2
    X[0, :, :, 3] = Z[0, :, :, 0]*Z[0, :, :, 1]

    Y = X

    return X, Y
