#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as MP
import time

def f(id, x):
    print('{:d} : {:.2f}'.format(id, x))
    return '{:.4f}'.format(x)

def fmany(n):
    np.random.seed(seed=int(time.time()))
    vals = np.random.rand(n)
    tasks = []
    for i in range(n):
        tasks.append( (i, vals[i]) )
    pool = MP.Pool(4)
    results = [pool.apply_async(f, t) for t in tasks]
    output = []
    for result in results:
        output.append(result.get())
    return output

if __name__ == '__main__':
    output = fmany(12)

print('summary:')
for s in output:
    print(s)

