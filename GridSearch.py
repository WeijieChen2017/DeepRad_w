#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

static = 'python Hydrogen.py -m subj01 -p subj01 -e 2000'
for th_idx in range(10):
    th = th_idx/10
    for w0 in ['1', '5']:
        for w1 in ['1', '5']:
            for w2 in ['1', '5']:
                for w3 in ['1', '5']:
                    w = w0+w1+w2+w3
                    ID = 'w'+w+'_th'+"%02d"%th_idx
                    static_curr = static+' -t '+str(th)+' -w '+w+' -i '+ID
                    print(static_curr)
                    # os.system(static_curr)
