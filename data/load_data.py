#!/usr/bin/python
# -*- coding: UTF-8 -*-

import nibabel as nib

def set_dataset(dir_mri, dir_pet):
    mri_file = nib.load(dir_mri)
    pet_file = nib.load(dir_pet)

    data_mri = mri_file.get_fdata()
    data_pet = pet_file.get_fdata()

    print("MRI_img shape:", data_mri.shape)
    print("PET_img shape:", data_pet.shape)

    return data_mri, data_pet
