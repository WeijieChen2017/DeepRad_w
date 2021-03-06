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
parser.add_argument('--pet', metavar='', type=str, default="subj01", help='Name of PET subject.(subj01)')
args = parser.parse_args()

dir_pet = './/files//'+args.pet+'_pet.nii'
pet_file = nib.load(dir_pet)
pet_img = pet_file.get_fdata()

header = pet_file.header
affine = pet_file.affine

ID = args.id
data = np.zeros(pet_img.shape)

path = '.\\mid_results\\' + ID + "\\*.npy"
list_nii = glob.glob(path)

for idx in range(len(list_nii)):
    curr_path = list_nii[idx]
    curr_data = np.load(curr_path).reshape((pet_img.shape[0], pet_img.shape[1]))
    data[:, :, idx] = curr_data

data = np.array(data)
print(data.shape)

nii_file = nib.Nifti1Image(data, affine, header)
save_path = '.\\mid_results\\' + ID + "\\"
nib.save(nii_file, save_path+'PVC_PET.nii.gz')
print('Assembling complete.')
