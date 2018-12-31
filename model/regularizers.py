#!/usr/bin/python
# -*- coding: UTF-8 -*-


from keras import regularizers
from keras.layers import Dense, Flatten
from GL.w_global import GL_get_value


def add_regularizer(model):

    flag_wr = GL_get_value("flag_wr")
    flag_yr = GL_get_value("flag_yr")
    para_wr = GL_get_value("para_wr")
    para_yr = GL_get_value("para_yr")
    n_pixel = GL_get_value("IMG_ROWS") * GL_get_value("IMG_COLS")

    if flag_wr == 'l2':
        model.add(Flatten())
        model.add(Dense(n_pixel, input_dim=n_pixel, activation='sigmoid', kernel_regularizer=regularizers.l2(para_wr)))
    if flag_wr == 'l1':
        model.add(Flatten())
        model.add(Dense(n_pixel, input_dim=n_pixel, activation='sigmoid', kernel_regularizer=regularizers.l1(para_wr)))
    if flag_yr == 'l2':
        model.add(Flatten())
        model.add(Dense(n_pixel, input_dim=n_pixel, activation='sigmoid', activity_regularizer=regularizers.l2(para_yr)))
    if flag_yr == 'l1':
        model.add(Flatten())
        model.add(Dense(n_pixel, input_dim=n_pixel, activation='sigmoid', activity_regularizer=regularizers.l1(para_yr)))
