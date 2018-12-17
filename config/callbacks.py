#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard


def set_checkpoint(log_path, MODEL_ID, batch_size):


    tensorboard = TensorBoard(log_dir=log_path, batch_size=batch_size,
                              write_graph=True, write_grads=True,
                              write_images=True)

    # checkpoint
    check_path = '.\\training_models\\' + MODEL_ID + '\\'
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    check_path = check_path + 'model.hdf5' # _{epoch:03d}_{val_loss:.4f}
    checkpoint1 = ModelCheckpoint(check_path, monitor='val_psnr',
                                  verbose=1, save_best_only=True, mode='max')
#     checkpoint2 = ModelCheckpoint(check_path, period=100)
    callbacks_list = [checkpoint1, tensorboard]

    return callbacks_list
