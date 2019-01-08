#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

# 4
static4 = 'python Lithium.py --id d4f64e2000_Zelda --flag_whole True --idx_slice '
for idx in range(80):
    static_curr = static4+str(idx)
    print(static_curr)
    # os.system(static_curr)
os.system('python assemble.py --id d4f64e2000_Zelda --pet subj02')