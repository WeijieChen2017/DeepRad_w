#!/usr/bin/python
# -*- coding: UTF-8 -*-


import gc
import os
# import sys
# import getopt
import datetime
import argparse
import numpy as np
from config.main_config import set_configuration
from data.load_data import set_dataset_brest
from data.set_X_Y import data_pre_breast
from run.run_pvc import w_train, w_pred
from GL.w_global import GL_set_value, GL_get_value
from eval.output import w_output
from run.save_para import save_all


GL_set_value("IMG_ROWS", 512)
GL_set_value("IMG_COLS", 512)
GL_set_value("IMG_DEPT", 284)
GL_set_value("FA_NORM", 35000.0)

np.random.seed(591)

def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--mri_water', metavar='', type=str, default="subj02",
                        help='Name of MRI water subject.(subj02)')
    parser.add_argument('--mri_fat', metavar='', type=str, default="subj02",
                        help='Name of MRI fat subject.(subj02)')
    parser.add_argument('-p', '--pet', metavar='', type=str, default="subj02",
                        help='Name of PET subject.(subj02)')
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
    parser.add_argument('--n_filter', metavar='', type=int, default=16,
                        help='The initial filter number')
    parser.add_argument('--depth', metavar='', type=int, default=4,
                        help='The depth of U-Net')
    parser.add_argument('--gap_flash', metavar='', type=int, default=100,
                        help='How many epochs between two flash shoot')
    parser.add_argument('--flag_whole', metavar='', type=bool, default=False,
                        help='Whether process the whole PET image')
    parser.add_argument('--idx_slice', metavar='', type=int, default=142,
                        help='The idx to be processed.')
    parser.add_argument('--flag_smooth', metavar='', type=bool, default=False,
                        help='Flag of Smooth loss function')

    args = parser.parse_args()

    dir_mri_water = './/files//'+args.mri_water+'_water.nii'
    dir_mri_fat = './/files//' + args.mri_fat + '_pet.nii'
    dir_pet = './/files//'+args.pet+'_pet.nii'
    n_epoch = args.epoch + 1

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    GL_set_value("MODEL_ID", args.id+time_stamp)
    GL_set_value("mri_th", args.th_wm)
    GL_set_value("W_PGWC", args.w_pgwc)
    GL_set_value("flag_BN", args.flag_BN)
    GL_set_value("flag_Dropout", args.flag_Dropout)
    GL_set_value("flag_reg", args.flag_reg)
    GL_set_value("flag_wr", args.type_wr)
    GL_set_value("flag_yr", args.type_yr)
    GL_set_value("para_wr", args.para_wr)
    GL_set_value("para_yr", args.para_yr)
    GL_set_value("n_filter", args.n_filter)
    GL_set_value("depth", args.depth)
    GL_set_value("gap_flash", args.gap_flash)
    GL_set_value("flag_whole", args.flag_whole)
    GL_set_value("IDX_SLICE", args.idx_slice)
    GL_set_value("flag_smooth", args.flag_smooth)

    # model establishment
    if args.flag_whole:
        GL_set_value("MODEL_ID", args.id)

    model, opt, loss, callbacks_list, conf = set_configuration(n_epoch=n_epoch, flag_aug=False)
    # add_regularizer(model)
    data_mri_water, data_mri_fat, data_pet = set_dataset_brest(dir_mri_water=dir_mri_water,
                                                               dir_mri_fat=dir_mri_fat,
                                                               dir_pet=dir_pet)

    GL_set_value("IDX_SLICE", args.idx_slice)

    X, Y = data_pre_breast(data_mri_water=data_mri_water,
                           data_mri_fat=data_mri_fat,
                           data_pet=data_pet)
    model.summary()
    model.compile(opt, loss)

    if args.flag_whole:
        w_pred(model=model, X=X, Y=Y, n_epoch=n_epoch)
        print('The slice has been completed. ' + str(args.idx_slice))
    else:
        w_train(model=model, X=X, Y=Y, n_epoch=n_epoch)

    save_all()
    del model
    del data_mri
    del data_pet
    gc.collect()


if __name__ == "__main__":
    main()

