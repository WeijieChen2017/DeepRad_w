#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np
import gc
from GL.w_global import GL_get_value, GL_set_value


def data_pre_PVC(data_mri, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IDX_SLICE = GL_get_value("IDX_SLICE")
    # FA_NORM = GL_get_value("FA_NORM")

    FA_NORM = np.amax(data_pet[:, :, IDX_SLICE])
    GL_set_value('FA_NORM', FA_NORM)
    mri_th = float(GL_get_value("mri_th"))

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 4))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 4))
    Z = np.zeros((1, IMG_ROWS, IMG_COLS, 4))

    data_pet = np.divide(data_pet, FA_NORM)

    Z[0, :, :, 0] = data_pet[:, :, IDX_SLICE] <= mri_th
    Z[0, :, :, 1] = data_mri[:, :, IDX_SLICE] == 3
    Z[0, :, :, 2] = data_mri[:, :, IDX_SLICE] != 0
    Z[0, :, :, 2] = Z[0, :, :, 2].astype(bool).astype(int)
    Z[0, :, :, 3] = data_pet[:, :, IDX_SLICE] > mri_th

    X[0, :, :, 0] = data_pet[:, :, IDX_SLICE] * Z[0, :, :, 2]  # PET
    X[0, :, :, 1] = data_mri[:, :, IDX_SLICE] == 1  # CSF
    X[0, :, :, 2] = data_mri[:, :, IDX_SLICE] == 2  # gray matter
    X[0, :, :, 2] = (Z[0, :, :, 3] + X[0, :, :, 2]).astype(bool).astype(int) # gray matter
    X[0, :, :, 3] = Z[0, :, :, 0]*Z[0, :, :, 1]  # white matter

    # if GL_get_value("flag_reg"):
    #     Y = X.flatten()
    # else:
    #     Y = X

    Y = X
    del Z
    gc.collect()
    # print("X shape:", X.shape)
    # print("Y shape:", Y.shape)
    return X, Y

def data_pre_seg(data_mri, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IMG_DEPT = GL_get_value("IMG_DEPT")
    IDX_SLICE = GL_get_value("IDX_SLICE")
    FA_NORM = GL_get_value("FA_NORM")

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 1))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 3))

    data_pet = np.divide(data_pet, FA_NORM)

    X[0, :, :, 0] = data_pet[:, :, IDX_SLICE]
    Y[0, :, :, 0] = data_mri[:, :, IDX_SLICE] == 1
    Y[0, :, :, 1] = data_mri[:, :, IDX_SLICE] == 2
    Y[0, :, :, 2] = data_mri[:, :, IDX_SLICE] == 3

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y


def data_pre_breast(data_mri_water, data_mri_fat, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IDX_SLICE = GL_get_value("IDX_SLICE")

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 3))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 3))

    data_pet = np.divide(data_pet, np.amax(data_pet))
    data_mri_water = np.divide(data_mri_water, np.amax(data_mri_water))
    data_mri_fat = np.divide(data_mri_fat, np.amax(data_mri_fat))

    X[0, :, :, 0] = data_pet[:, :, IDX_SLICE]
    X[0, :, :, 1] = data_mri_water[:, :, IDX_SLICE]
    X[0, :, :, 2] = data_mri_fat[:, :, IDX_SLICE]

    Y = X

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y

