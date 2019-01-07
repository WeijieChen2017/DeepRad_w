#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

# static0 = 'python Helium.py --w 1110 --n_filter 64 --epoch 40 --id d4f64_Kirby' \
#          ' --flag_whole True --flag_reg True --type_yr l2 --para_yr 1e-9 ' \
#          '--idx_slice '
#
# for idx in range(284):
#     static_curr = static0+str(idx)
#     print(static_curr)
#     os.system(static_curr)

# 1
# static1 = 'python Helium.py --w 1110 --epoch 180 --id d4f16e180_Kirby --flag_whole True' \
#          ' --flag_whole True --flag_reg True --type_yr l2 --para_yr 1e-6 ' \
#          '--idx_slice '
# for idx in range(284):
#     static_curr = static1+str(idx)
#     print(static_curr)
#     os.system(static_curr)
# os.system('python assemble.py --id d4f16e180_Kirby')
#
# # 2
# static2 = 'python Helium.py --w 1110 --epoch 2000 --id d1f16e2000_Kirby --flag_whole True --depth 1 --idx_slice '
# for idx in range(284):
#     static_curr = static2+str(idx)
#     print(static_curr)
#     os.system(static_curr)
# os.system('python assemble.py --id d1f16e2000_Kirby')
#
# # 3
# static3 = 'python Helium.py --w 1110 --epoch 500 --id d4f8e500_Kirby --flag_whole True --n_filter 8 --idx_slice '
# for idx in range(284):
#     static_curr = static3+str(idx)
#     print(static_curr)
#     os.system(static_curr)
# os.system('python assemble.py --id d4f8e500_Kirby')

# 4
static4 = 'python Helium.py --w 1110 --epoch 200 --id d4f64e200_Kirby --flag_whole True' \
         ' --flag_whole True --flag_reg True --type_yr l1 --para_yr 1e-9 --n_filter 64' \
         '--idx_slice '
for idx in range(284):
    static_curr = static4+str(idx)
    print(static_curr)
    os.system(static_curr)
os.system('python assemble.py --id d4f64e200l1_Kirby')