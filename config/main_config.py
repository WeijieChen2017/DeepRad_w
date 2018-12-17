#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
from keras.optimizers import rmsprop, Adam

from model.unet import unet
from config.UDF import mean_squared_error_1e6


def config(MODEL_ID):


    # logs
    log_path = '.\\logs\\' + MODEL_ID + "\\"
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    # set traininig configurations
    conf = {"image_shape": (192, 192, slice_x), "out_channel": 1, "filter": n_fliter, "depth": depth,
            "inc_rate": 2, "activation": 'relu', "dropout": True, "batchnorm": True, "maxpool": True,
            "upconv": True, "residual": True, "shuffle": True, "augmentation": True,
            "learning_rate": 1e-5, "decay": 0.0, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999,
            "validation_split": 0.2632, "batch_size": batch_size, "epochs": epochs,
            "loss": loss, "metric": "mse", "optimizer": optimizer, "LOOCV": LOOCV, "model_type": model_type}
    np.save(log_path + 'info.npy', conf)


    # set augmentation configurations
    conf_a = {"rotation_range": 15, "shear_range": 10,
              "width_shift_range": 0.33, "height_shift_range": 0.33, "zoom_range": 0.33,
              "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'nearest',
              "seed": 314, "batch_size": conf["batch_size"]}
    np.save(log_path + 'aug.npy', conf_a)

    # build up the model
    model = unet(img_shape=conf["image_shape"], out_ch=conf["out_channel"],
                 start_ch=conf["filter"], depth=conf["depth"],
                 inc_rate=conf["inc_rate"], activation=conf["activation"],
                 dropout=conf["dropout"], batchnorm=conf["batchnorm"],
                 maxpool=conf["maxpool"], upconv=conf["upconv"],
                 residual=conf["residual"])

    # Adam optimizer
    if conf["optimizer"] == 'Adam':
        opt = Adam(lr=conf["learning_rate"], decay=conf["decay"],
                   epsilon=conf["epsilon"], beta_1=conf["beta_1"], beta_2=conf["beta_2"])
    if conf["loss"] == 'mse1e6':
        loss = mean_squared_error_1e6
