#!/usr/bin/python
# -*- coding: UTF-8 -*-


import gc
import sys
import getopt
import datetime
import numpy as np
from config.main_config import set_configuration
from data.load_data import set_dataset
from data.set_X_Y import data_pre_PVC
from run.run_pvc import w_train
from GL.w_global import GL_set_value



global IMG_ROWS, IMG_COLS
global IDX_SLICE, FA_NORM

GL_set_value("IMG_ROWS", 512)
GL_set_value("IMG_COLS", 512)
GL_set_value("IDX_SLICE", 142)
GL_set_value("FA_NORM", 35000.0)

np.random.seed(591)

def usage():
    print("Error in input argv")


def main(argv):
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'm:p:e:i:t:w:',
                                   ['mri=', 'pet=', 'epoch=', 'id=', 'th=', 'weight_PGWC='])
    except getopt.GetoptError:
        usage()
        sys.exit()

    for opt, arg in opts:
        if opt in ['-m', '--mri']:
            dir_mri = arg
        elif opt in ['-p', '--pet']:
            dir_pet = arg
        elif opt in ['-e', '--epoch']:
            n_epoch = int(arg)+1
        elif opt in ['-i', '--id']:
            model_id = arg
        elif opt in ['-t', '--th']:
            MRI_TH = arg
        elif opt in ['-w', '--weight_PGWC']:
            W_PGWC = arg
        else:
            print("Error: invalid parameters")

    dir_mri = './/files//'+dir_mri+'_mri.nii.gz'
    dir_pet = './/files//'+dir_pet+'_pet.nii.gz'

    # print('Number of arguments:', len(argv), 'arguments.')
    # print('Argument List:', str(argv))
    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    # print("------------------------------------------------------------------")
    # print("MRI_dir: ", dir_mri)
    # print("PET_dir: ", dir_pet)
    # print("n_EPOCH: ", n_epoch)
    print("MODEL_ID: ", model_id+time_stamp)
    # print("------------------------------------------------------------------")
    # print("Build a U-Net:")

    GL_set_value("MODEL_ID", model_id+time_stamp)
    GL_set_value("MRI_TH", MRI_TH)
    GL_set_value("W_PGWC", W_PGWC)

    model, opt, loss, callbacks_list, conf = set_configuration(n_epoch=n_epoch, flag_aug=False)
    data_mri, data_pet = set_dataset(dir_mri=dir_mri, dir_pet=dir_pet)
    X, Y = data_pre_PVC(data_mri=data_mri, data_pet=data_pet)
    # model.summary()

    model.compile(opt, loss)

    w_train(model=model, X=X, Y=Y, n_epoch=n_epoch)

    del model
    del data_mri
    del data_pet
    gc.collect()


if __name__ == "__main__":
    main(sys.argv)
