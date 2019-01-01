#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

static = 'python Helium.py --flag_reg True -e 700 --type_wr l1 --type_yr l1'
for idx_yr in range(9):
    for idx_wr in range(9):
        yr = '1e-'+str(idx_yr+1)
        wr = '1e-'+str(idx_wr+1)
        # if (idx_yr > 5)or(idx_wr > 5):
        ID = ' --para_yr '+yr+' --para_wr '+wr+' --id yr'+str(idx_yr+1)+'_wr'+str(idx_wr+1)+'_l1'
        static_curr = static+ID
        print(static_curr)
        os.system(static_curr)
