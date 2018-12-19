#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import getopt
import datetime
from config.main_config import config


def usage():
    print("Error in input argv")


def main(argv):
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:p:e:i:', ['mri=', 'pet=', 'epoch=', 'id='])
    except getopt.GetoptError:
        usage()
        sys.exit()

    for opt, arg in opts:
        if opt in ['-m', '--mri']:
            dir_mri = arg
        elif opt in ['-p', '--pet']:
            dir_pet = arg
        elif opt in ['-e', '--epoch']:
            n_epoch = arg
        elif opt in ['-i', '--id']:
            model_id = arg
        else:
            print("Error: invalid parameters")

    # print('Number of arguments:', len(argv), 'arguments.')
    # print('Argument List:', str(argv))
    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    print("------------------------------------------------------------------")
    print("MRI_dir: ", dir_mri)
    print("PET_dir: ", dir_pet)
    print("n_EPOCH: ", n_epoch)
    print("MODEL_ID: ", model_id+time_stamp)
    print("------------------------------------------------------------------")
    print("Build a U-Net:")
    model, opt, loss = config(model_id, n_epoch=n_epoch)
    model.summary()

if __name__ == "__main__":
    main(sys.argv)
