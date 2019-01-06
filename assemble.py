#!/usr/bin/python
# -*- coding: UTF-8 -*-


import glob
import numpy as np
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(
    description='''This is a beta script for assemble results or PVC. ''',
    epilog="""All's well that ends well.""")
parser.add_argument('--id', metavar='', type=str, default="d4f64PET", help='Name of target subject.(d4f64PET)')
args = parser.parse_args()

ID = args.id

path = '.\\mid_results\\' + ID + "\\*.npy"
list_nii = glob.glob(path)
print(list_nii)