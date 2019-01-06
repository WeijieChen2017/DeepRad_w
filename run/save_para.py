#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
from GL.w_global import GL_all, GL_get_value


def save_all():

    model_id = GL_get_value("MODEL_ID")

    # logs
    log_path = '.\\logs\\' + model_id + "\\"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    global_dict = GL_all()
    np.save(log_path+'global.npy', global_dict)
