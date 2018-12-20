#!/usr/bin/python
# -*- coding: UTF-8 -*-


import gc
import sys
import getopt
import datetime
from config.main_config import set_configuration
from data.load_data import set_dataset


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

    dir_mri = './/files//'+dir_mri+'_mri.nii.gz'
    dir_pet = './/files//'+dir_pet+'_pet.nii.gz'

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
    model, opt, loss, callbacks_list, conf = set_configuration(MODEL_ID=model_id,
                                                               n_epoch=n_epoch,
                                                               flag_aug=False)
    data_mri, data_pet = set_dataset(dir_mri=dir_mri, dir_pet=dir_pet)
    model.summary()

    del model
    del data_mri
    del data_pet
    gc.collect()


if __name__ == "__main__":
    main(sys.argv)
