#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
import nibabel as nib
from GL.w_global import GL_get_value


def w_output():

    header = GL_get_value("header")
    affine = GL_get_value("affine")

    save_path = '.\\mid_results\\' + GL_get_value("MODEL_ID") + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    nii = np.asarray(GL_get_value("nii"))
    nii_file_0 = nib.Nifti1Image(nii, affine, header)
    nib.save(nii_file_0, save_path + 'nii_syn.nii.gz')
