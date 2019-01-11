#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np
import gc
from GL.w_global import GL_get_value, GL_set_value
import skimage.morphology as sm

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

    GL_set_value("FA_NORM", np.amax(data_pet))

    img_p = data_pet[:, :, IDX_SLICE]
    mask = np.asarray([img_p > 350]).reshape((256, 256)).astype(int)

    data_pet = np.divide(data_pet, np.amax(data_pet))
    data_mri_water = np.divide(data_mri_water, np.amax(data_mri_water))
    data_mri_fat = np.divide(data_mri_fat, np.amax(data_mri_fat))

    X[0, :, :, 0] = data_pet[:, :, IDX_SLICE]
    img_w = data_mri_water[:, :, IDX_SLICE]
    img_f = data_mri_fat[:, :, IDX_SLICE]

    img_sum = img_f + img_w + 1e-6
    # img_sum = img_sum / np.amax(img_sum)


    img_ff = np.divide(img_f, img_sum) * mask
    # img_f = img_f / np.amax(img_f)
    #img_f[img_f <= 0.95] = 0

    X[0, :, :, 2] = img_w
    X[0, :, :, 1] = img_w
    print(img_ff.shape)
    GL_set_value("img_ff", img_ff)
    Y = X

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y


def data_pre_breast_practical(data_mri_water, data_mri_fat, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IDX_SLICE = GL_get_value("IDX_SLICE")

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 3))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 3))

    GL_set_value("FA_NORM", np.amax(data_pet))

    img_p = data_pet[:, :, IDX_SLICE]
    mask_pet = np.asarray([img_p > 350]).reshape((256, 256)).astype(int)

    # data_pet = np.divide(data_pet, np.amax(data_pet))
    # data_mri_water = np.divide(data_mri_water, np.amax(data_mri_water))
    # data_mri_fat = np.divide(data_mri_fat, np.amax(data_mri_fat))


    img_w = data_mri_water[:, :, IDX_SLICE]
    img_f = data_mri_fat[:, :, IDX_SLICE]

    img_sum = img_f + img_w + 1e-6
    mask_sum = np.asarray([img_sum > 150]).reshape((256, 256)).astype(int)
    mask = mask_pet * mask_sum
    mask = sm.opening(mask, sm.disk(5))
    mask = sm.closing(mask, sm.square(5))

    # img_sum = img_sum / np.amax(img_sum)

    img_ff = np.divide(img_f, img_sum) * mask
    # img_f = img_f / np.amax(img_f)
    # img_ff[img_ff <= 0.8] = 0

    X[0, :, :, 0] = np.divide(img_p, np.amax(data_pet))
    X[0, :, :, 2] = np.divide(img_w, np.amax(data_mri_water))
    X[0, :, :, 1] = np.divide(img_w, np.amax(data_mri_water))
    print(img_ff.shape)
    GL_set_value("img_ff", img_ff)
    Y = X

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y


def data_pre_breast_p2p(data_mri_water, data_mri_fat, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IDX_SLICE = GL_get_value("IDX_SLICE")

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 1))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 1))

    GL_set_value("FA_NORM", np.amax(data_pet))

    # input
    img_p = data_pet[:, :, IDX_SLICE]
    X[0, :, :, 0] = np.divide(img_p, np.amax(data_pet))
    Y = X

    # water/fat fraction
    img_w = data_mri_water[:, :, IDX_SLICE]
    img_f = data_mri_fat[:, :, IDX_SLICE]
    img_sum = img_f + img_w + 1e-6

    # mask
    mask_pet = np.asarray([img_p > 350]).reshape((256, 256)).astype(int)
    mask_sum = np.asarray([img_sum > 150]).reshape((256, 256)).astype(int)
    mask = mask_pet * mask_sum
    mask = sm.opening(mask, sm.disk(5))
    mask = sm.closing(mask, sm.square(5))

    # water/fat fraction
    img_ff = np.divide(img_f, img_sum) * mask
    img_wf = np.divide(img_w, img_sum) * mask
    # img_ff[img_ff <= 0.8] = 0

    GL_set_value("img_ff", img_ff)
    GL_set_value("img_wf", img_wf)
    GL_set_value("mask_pet", mask_pet)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y

def data_pre_breast_m2p(data_mri_water, data_mri_fat, data_pet):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    IDX_SLICE = GL_get_value("IDX_SLICE")

    X = np.zeros((1, IMG_ROWS, IMG_COLS, 1))
    Y = np.zeros((1, IMG_ROWS, IMG_COLS, 1))

    GL_set_value("FA_NORM", np.amax(data_pet))

    # input
    img_p = data_pet[:, :, IDX_SLICE]
    Y[0, :, :, 0] = np.divide(img_p, np.amax(data_pet))

    # water/fat fraction
    img_w = data_mri_water[:, :, IDX_SLICE]
    img_f = data_mri_fat[:, :, IDX_SLICE]
    img_sum = img_f + img_w + 1e-6

    X[0, :, :, 0] = np.divide(img_sum, np.amax(img_sum))

    # mask
    mask_pet = np.asarray([img_p > 350]).reshape((256, 256)).astype(int)
    mask_sum = np.asarray([img_sum > 150]).reshape((256, 256)).astype(int)
    mask = mask_pet * mask_sum
    mask = sm.opening(mask, sm.disk(5))
    mask = sm.closing(mask, sm.square(5))

    # water/fat fraction
    img_ff = np.divide(img_f, img_sum) * mask
    img_wf = np.divide(img_w, img_sum) * mask
    # img_ff[img_ff <= 0.8] = 0

    GL_set_value("img_ff", img_ff)
    GL_set_value("img_wf", img_wf)
    GL_set_value("mask_pet", mask_pet)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y