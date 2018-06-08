#!/usr/bin/python3
# testing various sizes of matrices for speed
import numpy as np
import time
##
dimen = int(input('dimension? '))
np.random.seed(99)
A = np.random.random(size=(dimen,dimen))
t0 = time.time()
vals, vecs = np.linalg.eig(A)
t1 = time.time()
print('{:d} eigenvalues/vectors in {:.2f} sec'.format(len(vals), t1-t0))
print('first eigval = ', vals[0])
