#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
from keras.optimizers import Adam

from model.unet import unet
from config.UDF import mean_squared_error_1e6, loss_breast,\
                       loss_breast_practical, loss_breast_p2p
from config.UDF import Gray_White_CSF, Gray_White_CSF_soomth
from config.callbacks import set_checkpoint
from GL.w_global import GL_get_value


def set_configuration(n_epoch=500, flag_aug=False):

    IMG_ROWS = GL_get_value("IMG_ROWS")
    IMG_COLS = GL_get_value("IMG_COLS")
    MODEL_ID = GL_get_value("MODEL_ID")

    model = None
    opt = None
    loss = None

    # logs
    log_path = '.\\logs\\' + MODEL_ID + "\\"
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    # set traininig configurations
    conf = {"image_shape": (IMG_ROWS, IMG_COLS, 1),
            "out_channel": 1,
            "filter": GL_get_value("n_filter"),
            "depth": GL_get_value("depth"),
            "inc_rate": 2,
            "activation": 'relu',
            "dropout": GL_get_value("flag_Dropout"),
            "batchnorm": GL_get_value("flag_BN"),
            "maxpool": True,
            "upconv": True,
            "residual": True,
            "shuffle": True,
            "augmentation": False,
            "learning_rate": 1e-5,
            "decay": 0.0,
            "epsilon": 1e-8,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epochs": n_epoch,
            "loss": 'loss_breast_p2p',
            "metric": "mse",
            "optimizer": 'Adam',
            "batch_size": 10}
    if GL_get_value("flag_smooth"):
        conf["loss"] = conf["loss"]+'_smooth'
    np.save(log_path + 'info.npy', conf)

    if flag_aug:
        # set augmentation configurations
        conf_a = {"rotation_range": 15, "shear_range": 10,
                  "width_shift_range": 0.33, "height_shift_range": 0.33, "zoom_range": 0.33,
                  "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'nearest',
                  "seed": 314, "batch_size": conf["batch_size"]}
        np.save(log_path + 'aug.npy', conf_a)

    # build up the model
    model = unet(img_shape=conf["image_shape"],
                 out_ch=conf["out_channel"],
                 start_ch=conf["filter"],
                 depth=conf["depth"],
                 inc_rate=conf["inc_rate"],
                 activation=conf["activation"],
                 dropout=conf["dropout"],
                 batchnorm=conf["batchnorm"],
                 maxpool=conf["maxpool"],
                 upconv=conf["upconv"],
                 residual=conf["residual"])

    # Adam optimizer
    if conf["optimizer"] == 'Adam':
        opt = Adam(lr=conf["learning_rate"], decay=conf["decay"],
                   epsilon=conf["epsilon"], beta_1=conf["beta_1"], beta_2=conf["beta_2"])
    if conf["loss"] == 'mse1e6':
        loss = mean_squared_error_1e6
    if conf["loss"] == 'Gray_White_CSF':
        loss = Gray_White_CSF
    if conf["loss"] == 'Gray_White_CSF_smooth':
        loss = Gray_White_CSF_soomth
    if conf["loss"] == 'loss_breast':
        loss = loss_breast
    if conf["loss"] == 'loss_breast_practical':
        loss = loss_breast_practical
    if conf["loss"] == 'loss_breast_p2p':
        loss = loss_breast_p2p

    # callback
    callbacks_list = set_checkpoint(log_path=log_path, MODEL_ID=MODEL_ID, batch_size=conf["batch_size"])

    return model, opt, loss, callbacks_list, conf
