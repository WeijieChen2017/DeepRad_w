#!/usr/bin/python
# -*- coding: UTF-8 -*-


import glob
import numpy as np
import nibabel as nib
import argparse
from scipy import stats

parser = argparse.ArgumentParser(
    description='''This is a beta script for assemble results or PVC. ''',
    epilog="""All's well that ends well.""")
parser.add_argument('--id', metavar='', type=str, default="d4f64PET", help='Name of target subject.(d4f64PET)')
parser.add_argument('--pet', metavar='', type=str, default="subj01", help='Name of PET subject.(subj01)')
args = parser.parse_args()

dir_pet = './/files//'+args.pet+'_pet.nii.gz'
pet_file = nib.load(dir_pet)
pet_data = pet_file.get_fdata()

header = pet_file.header
affine = pet_file.affine

ID = args.id
data = np.zeros((512, 512, 284))

path = '.\\mid_results\\' + ID + "\\*.npy"
list_nii = glob.glob(path)

for idx in range(len(list_nii)):
    curr_path = list_nii[idx]
    curr_data = np.load(curr_path).reshape((512, 512))

    # hist0 = np.histogram(pet_data[:,:,idx], bins=64)
    # max_idx0 = np.argmax(hist0[0])
    # factor0 = (hist0[1][max_idx0]+hist0[1][max_idx0+1]) / 2
    hist1 = np.histogram(curr_data, bins=64)
    max_idx1 = np.argmax(hist1[0])
    factor1 = (hist1[1][max_idx1]+hist1[1][max_idx1+1]) / 2

    # factor = factor1 / factor0
    curr_data[curr_data < factor1] = 0
    curr_data[curr_data >= factor1] -= factor1
    curr_data = curr_data / np.amax(curr_data) * np.amax(pet_data[:,:,idx])
    factor = np.sum(curr_data) / np.sum(pet_data[:,:,idx])
    data[:, :, idx] = curr_data / factor

data = np.array(data)
print(data.shape)



nii_file = nib.Nifti1Image(data, affine, header)
save_path = '.\\mid_results\\' + ID + "\\"
nib.save(nii_file, save_path+'PVC_PET_norm.nii.gz')
print('Assembling complete.')
