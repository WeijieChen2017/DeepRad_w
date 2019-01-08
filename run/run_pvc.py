#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from GL.w_global import GL_get_value, GL_set_value


def w_train(model, X, Y, n_epoch):

    header = GL_get_value("header")
    affine = GL_get_value("affine")

    fig = plt.figure(figsize=(10, 5))
    fig.show(False)

    save_path = '.\\mid_results\\' + GL_get_value("MODEL_ID") + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx_epoch in range(n_epoch):

        curr_loss = model.train_on_batch(X, Y)

        if idx_epoch % GL_get_value("gap_flash") == 0:
            fig.clf()
            # a = fig.add_subplot(1, 3, 1)
            # plt.imshow(np.rot90(X[0, :, :, 0]), cmap='gray')
            # a.axis('off')
            # a.set_title('X')
            a = fig.add_subplot(1, 2, 1)
            plt.imshow(np.rot90(Y[0, :, :, 0], 3), cmap='gray')
            a.axis('off')
            a.set_title('Y')
            Y_ = model.predict(X)
            a = fig.add_subplot(1, 2, 2)
            plt.imshow(np.rot90(Y_[0, :, :, 0], 3), cmap='gray')
            a.axis('off')
            a.set_title('\^Y')
            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig(save_path+'progress_dip_{0:05d}.jpg'.format(idx_epoch))
            fig.canvas.flush_events()

            # Y = model.predict(X)
            # nii_file_0 = nib.Nifti1Image(Y[:, :, :, 0], affine, header)
            # nii_file_1 = nib.Nifti1Image(Y[:, :, :, 1], affine, header)
            # nii_file_2 = nib.Nifti1Image(Y[:, :, :, 2], affine, header)
            # nib.save(nii_file_0, save_path+'progress_dip_{0:05d}_0.nii.gz'.format(idx_epoch))
            # nib.save(nii_file_1, save_path + 'progress_dip_{0:05d}_1.nii.gz'.format(idx_epoch))
            # nib.save(nii_file_2, save_path + 'progress_dip_{0:05d}_2.nii.gz'.format(idx_epoch))


def w_pred(model, X, Y, n_epoch):

    FA_NORM = GL_get_value("FA_NORM")

    for idx_epoch in range(n_epoch):
        curr_loss = model.train_on_batch(X, Y)

    fig = plt.figure(figsize=(10, 5))
    fig.show(False)
    save_path = '.\\mid_results\\' + GL_get_value("MODEL_ID") + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.clf()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(np.rot90(Y[0, :, :, 0]), cmap='gray')
    a.axis('off')
    a.set_title('Y')
    Y_ = model.predict(X)
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(np.rot90(Y_[0, :, :, 0]), cmap='gray')
    a.axis('off')
    a.set_title('\^Y')
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(save_path + 'progress_dip_{0:03d}.jpg'.format(GL_get_value("IDX_SLICE")))
    fig.canvas.flush_events()

    np.save(save_path + 'progress_dip_{0:03d}.nii'.format(GL_get_value("IDX_SLICE")), Y_*FA_NORM)
    # nii = GL_get_value("nii")
    # nii.append(Y_)
    # GL_set_value("nii", nii)

    return None


