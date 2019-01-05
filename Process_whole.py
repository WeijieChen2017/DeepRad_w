#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

static = 'python Helium.py --w 1110 --n_filter 64 --epoch 40 --id d4f64PET --flag_whole True --idx_slice '
for idx in range(284):
    static_curr = static+str(idx)
    print(static_curr)
    os.system(static_curr)
