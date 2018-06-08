#!/usr/bin/python3
#   Test piecewise optimization of dissociated geometries
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, glob, multiprocessing
from chem_subs import *
from qm_subs import *
from ips_subs import *
#
ips_input = read_input('test_minimize.ips')
ips_input = parse_input(ips_input)
# replace structure with a dissociated one
G, natom, comment = readXmol('test_minimize.xyz', handle=False)
ips_input['cartesian'] = G
X0 = G.toVector()
#
Efrag, optfrag, superGeom = spread_opt(ips_input, 'test_spread')
#superGeom.printXYZ(fname='test_spread.xyz', comment='testing spread_opt()')
print('Fragment energies: ', ', '.join(['{:6f}'.format(E) for E in Efrag]))
if Efrag is not None:
    for frag in optfrag:
        frag.printXYZ()
    superGeom.printXYZ()
else:
    print('There were no fragments.')
