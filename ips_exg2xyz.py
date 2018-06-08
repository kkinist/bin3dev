#!/usr/bin/python3
# Analysis of trajectory files from IPS
# Karl Irikura, NIST 2017
#
import numpy as np
import pandas as pd
import sys, os, glob
from chem_subs import *
from qm_subs import *
from ips_subs import *
""" Convert an EXG file to an XYZ file """
try:
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        # maybe user did not include the file suffix
        input_file += '.ips'
    ips_input = read_input(input_file)
    exg_file = sys.argv[2]
    if not os.path.isfile(exg_file):
        # maybe user did not include the file suffixes
        exg_file += '_exg.csv'
    fexg = open(exg_file, 'r')
    xyz_file = sys.argv[3]
    fxyz = open(xyz_file, 'w')
except BaseException as err:
    sys.exit('Usage: ips_exg2xyz.py <name of IPS input file> <EXG file> <XYZ output file>\n' + str(err))
print('Reading input file "{:s}"'.format(input_file))
# read the IPS input file
ips_input = parse_input(ips_input)
if True:
    for key in ips_input:
        print('{:s}: '.format(key), ips_input[key])
# read the input EXG file
print('Reading EXG file "{:s}"'.format(exg_file))
dfexg = pd.read_csv(fexg)
fexg.close()
# delete the gradient and connection columns
ncol = dfexg.shape[1]
ndrop = ncol // 2
dfexg.drop(dfexg.columns[range(ndrop, ncol)], axis=1, inplace=True)
ncol = dfexg.shape[1]
if False:
    print(dfexg)
    print('Modified shape is ', dfexg.shape)
# convert to XYZ form
coordtype = ips_input['coordtype']
molec = ips_input['molecule']
Coord = ips_input[coordtype].copy()  # reference coordinates object
unitS = ['angstrom', 'radian']  # assume these are the units
istep = 0
for row in dfexg.values.tolist():
    E = row[0]  # relative energy, kJ/mol
    comment = 'Erel = {:.1f} kJ/mol for step {:d} of molecule {:s}'.format(E, istep, molec)
    Coord.fromVector(np.array(row[1:]), unitS)
    xyzstring = Coord.XmolXYZ(comment)
    fxyz.write(xyzstring)
    istep += 1 
fexg.close()
print('File {:s} written and closed.'.format(xyz_file))
