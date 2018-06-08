#!/usr/bin/python3
import os
#
def examine(fname):
    if os.path.exists(fname):
        print('{:s} exists'.format(fname))
    else:
        print('I did not find {:s} in front of my face.'.format(fname))
##
for fname in ['exists.txt', 'pent_df_0_exg.csv']:
    examine(fname)
    
