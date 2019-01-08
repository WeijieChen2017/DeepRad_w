#!/usr/bin/python
# -*- coding: UTF-8 -*-

import nibabel as nib
from GL.w_global import GL_set_value


def set_dataset(dir_mri, dir_pet):
    mri_file = nib.load(dir_mri)
    pet_file = nib.load(dir_pet)

    header = pet_file.header
    affine = pet_file.affine
    GL_set_value("header", header)
    GL_set_value("affine", affine)

    data_mri = mri_file.get_fdata()
    data_pet = pet_file.get_fdata()

    print("MRI_img shape:", data_mri.shape)
    print("PET_img shape:", data_pet.shape)

    return data_mri, data_pet


def set_dataset_brest(dir_mri_water, dir_mri_fat, dir_pet):

    mri_water_file = nib.load(dir_mri_water)
    mri_fat_file = nib.load(dir_mri_fat)
    pet_file = nib.load(dir_pet)

    header = pet_file.header
    affine = pet_file.affine
    GL_set_value("header", header)
    GL_set_value("affine", affine)

    data_water = mri_water_file.get_fdata()
    data_fat = mri_fat_file.get_fdata()
    data_pet = pet_file.get_fdata()

    print("WATER_img shape:", data_water.shape)
    print("FAT_img shape:", data_fat.shape)
    print("PET_img shape:", data_pet.shape)

    return data_water, data_fat, data_pet
