#!/usr/bin/python3
# Plot summary data for an IPS walker's trajectory
# KKI 8/15/2017
#
import sys, os
import matplotlib.pyplot as plt
import pandas as pd
try:
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        # maybe use did not include the file suffix
        input_file += '.nrm'
    traj = pd.read_csv(input_file, sep=' ', header=None, names=['step','E', 'X', 'G'])
except IOError:
    print('*** Failure reading input file')
    sys.exit('Usage: ips_plotnrm.py <name of IPS walker summary file>')
# plot relative energy as a function of step 
traj.plot(x='step', y='E', title='relative energy (kJ/mol)', legend=False)
plt.show()
# plot distance from origin
traj.plot(x='step', y='X', title='distance from origin', legend=False)
plt.show()
# plot gradient norm
traj.plot(x='step', y='G', title='gradient norm', legend=False)
plt.show()
# plot gradient norm against distance from origin 
traj.plot(x='X', y='G', title='gradient norm', kind='scatter', legend=False, alpha=0.5)
plt.show()
