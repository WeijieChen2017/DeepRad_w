#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

static = 'python Helium.py --flag_reg True -e 700 --type_wr l2 --type_yr l2'
for idx_yr in range(6):
    for idx_wr in range(6):
        yr = '1e-'+str(idx_yr+1)
        wr = '1e-'+str(idx_wr+1)
        ID = ' --para_yr '+yr+'--para_wr '+wr+' --id yr'+str(idx_yr+1)+'_wr'+str(idx_wr+1)+'_l2'
        static_curr = static+ID
        print(static_curr)
        # os.system(static_curr)
