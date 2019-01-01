#!/usr/bin/python
# -*- coding: UTF-8 -*-


import gc
# import sys
# import getopt
import datetime
import argparse
import numpy as np
from config.main_config import set_configuration
from data.load_data import set_dataset
from data.set_X_Y import data_pre_PVC, data_pre_seg
from run.run_pvc import w_train
from GL.w_global import GL_set_value


GL_set_value("IMG_ROWS", 512)
GL_set_value("IMG_COLS", 512)
GL_set_value("IMG_DEPT", 284)
GL_set_value("IDX_SLICE", 142)
GL_set_value("FA_NORM", 35000.0)

np.random.seed(591)

def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('-m', '--mri', metavar='', type=str, default="subj01",
                        help='Name of MRI subject.(subj01)')
    parser.add_argument('-p', '--pet', metavar='', type=str, default="subj01",
                        help='Name of PET subject.(subj01)')
    parser.add_argument('-e', '--epoch', metavar='', type=int, default=2000,
                        help='Number of epoches of training.(2000)')
    parser.add_argument('-i', '--id', metavar='', type=str, default="eeVee",
                        help='ID of the current model.(eeVee)')
    parser.add_argument('-t', '--th_wm', metavar='', type=float, default=0.571,
                        help='Threshold of white matter.(0.571)')
    parser.add_argument('-w', '--w_pgwc', metavar='', type=str, default="5115",
                        help='Weight of PET/Gm/Wm/CSF(5115)')
    parser.add_argument('--flag_BN', metavar='', type=bool, default=True,
                        help='Flag of BatchNormlization(True)')
    parser.add_argument('--flag_Dropout', metavar='', type=bool, default=True,
                        help='Flag of Dropout(True)')
    parser.add_argument('--flag_reg', metavar='', type=bool, default=False,
                        help='Flag of regularizer(False)')
    parser.add_argument('--type_wr', metavar='', type=str, default='None',
                        help='Flag of weight regularizer(l2/l1)')
    parser.add_argument('--type_yr', metavar='', type=str, default='None',
                        help='Flag of y regularizer(l2/l1)')
    parser.add_argument('--para_wr', metavar='', type=float, default=0.01,
                        help='Para of weight regularizer(0.01)')
    parser.add_argument('--para_yr', metavar='', type=float, default=0.01,
                        help='Para of y regularizer(0.01)')
    args = parser.parse_args()

    dir_mri = './/files//'+args.mri+'_mri.nii.gz'
    dir_pet = './/files//'+args.pet+'_pet.nii.gz'
    n_epoch = args.epoch + 1

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    GL_set_value("MODEL_ID", args.id+time_stamp)
    GL_set_value("MRI_TH", args.th_wm)
    GL_set_value("W_PGWC", args.w_pgwc)
    GL_set_value("flag_BN", args.flag_BN)
    GL_set_value("flag_Dropout", args.flag_Dropout)
    GL_set_value("flag_reg", args.flag_reg)
    GL_set_value("flag_wr", args.type_wr)
    GL_set_value("flag_yr", args.type_yr)
    GL_set_value("para_wr", args.para_wr)
    GL_set_value("para_yr", args.para_yr)


    print("------------------------------------------------------------------")
    print("MRI_dir: ", dir_mri)
    print("PET_dir: ", dir_pet)
    print("n_EPOCH: ", n_epoch)
    print("MODEL_ID: ", args.id+time_stamp)
    print("------------------------------------------------------------------")
    print("Build a U-Net:")

    # model establishment
    model, opt, loss, callbacks_list, conf = set_configuration(n_epoch=n_epoch, flag_aug=False)
    # add_regularizer(model)
    data_mri, data_pet = set_dataset(dir_mri=dir_mri, dir_pet=dir_pet)
    # X, Y = data_pre_PVC(data_mri=data_mri, data_pet=data_pet)
    X, Y = data_pre_seg(data_mri=data_mri, data_pet=data_pet)
    model.summary()
    model.compile(opt, loss)
    w_train(model=model, X=X, Y=Y, n_epoch=n_epoch)

    del model
    del data_mri
    del data_pet
    gc.collect()


if __name__ == "__main__":
    main()
