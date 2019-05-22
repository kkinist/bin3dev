#!/usr/bin/python3
#
# Report irreducible representations of occupied orbitals, as listed
#   in a Gaussian output file
# KKI 12/19/2016
#
import sys
import re
import pandas as pd
sys.path.append('/media/sf_GitHub/BEB')
from g09_subs3 import *
##
if len(sys.argv) < 2:
    sys.exit( 'Usage:  occup.py <outputfile.out>' )
fname = sys.argv[1]
fgout = open(fname, 'r')
print('*** {:s} ***'.format(fname))
df = read_orbital_irreps(fgout)
print('--Number of occupied orbitals of each irrep--')
# process each set of orbitals
for iset in range(len(df.index)):
    print('{:s} orbitals'.format(df.iloc[iset]['Type']))
    dforbs = df.iloc[iset]['Orbs']
    # restrict interest to occupied orbitals
    dfocc = dforbs[dforbs.Occup == True]
    # get the list of all irreps reported by Gaussian
    irrep_list = sorted(set(dfocc['Irrep']))
    # count the number of spins (1 or 2)
    spin_list = sorted(set(dfocc['Spin']))
    nspin = len(spin_list)
    dfsums = dfocc.groupby(['Spin', 'Irrep']).sum()
    # print column headings: list of irreps
    for irr in irrep_list:
        print('\t{:s}'.format(irr), end='')
    print()
    # print totals for each orbital spin
    for sp in spin_list:
        if sp != 'both':
            print('{:s}'.format(sp), end='')
        for irr in irrep_list:
            idx = (sp, irr)
            print('\t{:d}'.format(int(dfsums.loc[idx]['Occup'])), end='')
        print()
    if nspin > 1:
        # also print the (alpha-beta) difference
        print('diff', end='')
        for irr in irrep_list:
            ida = ('alpha', irr)
            idb = ('beta', irr)
            dif = int(dfsums.loc[ida]['Occup']) - int(dfsums.loc[idb]['Occup'])
            print('\t{:d}'.format(dif), end='')
        print()
