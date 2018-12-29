#!/usr/bin/python
# -*- coding: UTF-8 -*-


import gc
import sys
import getopt
import datetime
import argparse
import numpy as np
from config.main_config import set_configuration
from data.load_data import set_dataset
from data.set_X_Y import data_pre_PVC
from run.run_pvc import w_train
from GL.w_global import GL_set_value


GL_set_value("IMG_ROWS", 512)
GL_set_value("IMG_COLS", 512)
GL_set_value("IDX_SLICE", 142)
GL_set_value("FA_NORM", 35000.0)

np.random.seed(591)

def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('-m', '--mri', metavar='', type=str, default="subj01", help='Name of MRI subject.')
    parser.add_argument('-p', '--pet', metavar='', type=str, default="subj01", help='Name of PET subject.')
    parser.add_argument('-e', '--epoch', metavar='', type=int, default=2001, help='Number of epoches of training.')
    parser.add_argument('-i', '--id', metavar='', type=str, default="eeVee", help='ID of the current model.')
    parser.add_argument('-t', '--th_wm', metavar='', type=float, default=0.6, help='Threshold of white matter.')
    parser.add_argument('-w', '--w_pgwc', metavar='', type=str, default="5115", help='Weight of PET/Gm/Wm/CSF')
    args = parser.parse_args()

    dir_mri = './/files//'+args.mri+'_mri.nii.gz'
    dir_pet = './/files//'+args.pet+'_pet.nii.gz'
    n_epoch = args.epoch + 1
    model_id = args.id
    MRI_TH = args.th_wm
    W_PGWC = args.w_pgwc

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    GL_set_value("MODEL_ID", model_id+time_stamp)
    GL_set_value("MRI_TH", MRI_TH)
    GL_set_value("W_PGWC", W_PGWC)

    print("------------------------------------------------------------------")
    print("MRI_dir: ", dir_mri)
    print("PET_dir: ", dir_pet)
    print("n_EPOCH: ", n_epoch)
    print("MODEL_ID: ", model_id+time_stamp)
    print("------------------------------------------------------------------")
    print("Build a U-Net:")

    # model establishment
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
    main()
