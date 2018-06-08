#!/usr/bin/python3
#   Test usage of scipy.optimize.minimize for 
#   simple and constrained geometry optimization
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, glob, multiprocessing
from chem_subs import *
from qm_subs import *
from ips_subs import *
import scipy.optimize
#
# First use Gaussian09 optimizer
ips_input = read_input('test_minimize.ips')
ips_input = parse_input(ips_input)
# add 'geom=nocrowd' option 
ips_input['code_options']['geom'] = ['command', 'nocrowd']
# replace structure with a dissociated one
G, natom, comment = readXmol('test_minimize.xyz', handle=False)
ips_input['cartesian'] = G
X0 = G.toVector()
#E0, Struct0, IDx = qm_function(ips_input, 'minimize', verbose=False, ID=0, fileroot='test_minimize')
#print('QM optimizer converged to E = {:.6f}'.format(E0))
#
# Use the scipy function
#
def testfunc(x, ips_input, ID, comment):
    # test function + gradient 
    # just quadratic function (x-const)**2 (vectors)
    const = ips_input['cartesian'].toVector()
    y = x - const
    val = np.dot(y, y)
    grad = 2 * y
    return val, grad
##
x0 = np.random.random(len(X0))
print('const = ', X0)
# Wrapper for QM gradient calculation
def qmgrad_wrap(X, ips_input, ID, fileroot):
    # X is a flattened vector of cartesian atomic coordinates
    #
    # insert X into the molecular structure
    ips_input['cartesian'].fromVector(X, 'angstrom')
    E, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', ID=ID, fileroot=fileroot)
    print('!!! type(E) = ', type(E), ' type(grad) = ', type(grad))
    if type(grad) is list:
        print('!!!!! grad = ', grad)
    return E, grad
##
#
method = 'BFGS'
options = {'maxiter': 50, 'disp': True}

args = (ips_input, 9898, 'meaningless string')
result = scipy.optimize.minimize(testfunc, x0, args=args, method=method, jac=True, options=options)
print(result)


print('Applying scipy.optimize.minimize using method {:s}'.format(method))
args = (ips_input, 0, 'test_minim')
result = scipy.optimize.minimize(qmgrad_wrap, X0, args=args, method=method, jac=True, options=options)
print(result)

