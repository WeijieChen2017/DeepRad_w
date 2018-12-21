#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
fig.show(False)


def w_train(model, X, Y, n_epoch):

    for idx_epoch in range(n_epoch):

        curr_loss = model.train_on_batch(X, Y)

        if idx_epoch % 100 == 0:
            fig.clf()
            a = fig.add_subplot(1, 3, 1)
            plt.imshow(np.rot90(X[0, :, :, 0]), cmap='gray')
            a.axis('off')
            a.set_title('X')
            a = fig.add_subplot(1, 3, 2)
            plt.imshow(np.rot90(Y[0, :, :, 0]), cmap='gray')
            a.axis('off')
            a.set_title('Y')
            Y_ = model.predict(X)
            a = fig.add_subplot(1, 3, 3)
            plt.imshow(np.rot90(Y_[0, :, :, 0]), cmap='gray')
            a.axis('off')
            a.set_title('\^Y')
            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig('progress_dip_{0:05d}.jpg'.format(it))
            fig.canvas.flush_events()
