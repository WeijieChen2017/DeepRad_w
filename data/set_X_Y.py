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

    # if GL_get_value("flag_reg"):
    #     Y = X.flatten()
    # else:
    #     Y = X

    Y = X

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    return X, Y

def data_pre_seg(data_mri, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IMG_DEPT = GL_get_value("IMG_DEPT")
    FA_NORM = GL_get_value("FA_NORM")

    X = np.zeros((IMG_DEPT, IMG_ROWS, IMG_COLS, 1))
    Y = np.zeros((IMG_DEPT, IMG_ROWS, IMG_COLS, 3))

    for i in range(IMG_DEPT):
        X[i, :, :, :] = data_pet[:, :, i]
        Y[0, :, :, 0] = data_mri[:, :, i] == 1
        Y[0, :, :, 1] = data_mri[:, :, i] == 2
        Y[0, :, :, 2] = data_mri[:, :, i] == 3

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y

