#!/usr/bin/python
# -*- coding: UTF-8 -*-


from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose
from keras.layers import UpSampling2D, Dropout, BatchNormalization


from keras import regularizers
from keras.layers import Dense, Flatten
from GL.w_global import GL_get_value


'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''


def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m


def unet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)

    flag_reg = GL_get_value("flag_reg")
    type_wr = GL_get_value("flag_wr")
    type_yr = GL_get_value("flag_yr")
    para_wr = GL_get_value("para_wr")
    para_yr = GL_get_value("para_yr")

    if flag_reg:
        if type_wr == 'l2':
            o = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(para_wr))(o)
        if type_wr == 'l1':
            o = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(para_wr))(o)
        if type_yr == 'l2':
            o = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(para_yr))(o)
        if type_yr == 'l1':
            o = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l1(para_yr))(o)

    return Model(inputs=i, outputs=o)
