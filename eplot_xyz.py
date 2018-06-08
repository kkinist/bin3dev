#!/usr/bin/python3
# Extract energy from multi-structure XYZ file(s), plot
#   assume multiple files are for the same trajectory
#
import re, sys
import matplotlib.pyplot as plt
import numpy as np
#
infiles = sys.argv[1:]
#
regx = re.compile(r'bead (\d+), Erel = (.*) kJ')
print()
for input_file in infiles:
    print('Reading {:s}'.format(input_file))
    bead = []
    E = []
    with open(input_file) as finp:
        for line in finp:
            m = regx.search(line)
            if m:
                bead.append(int(m.group(1)))
                E.append(float(m.group(2)))
    print('\tMean energy = {:.1f}'.format(np.mean(E)))
    print('\tHighest energy = {:.1f}'.format(np.max(E)))
    plt.plot(bead, E, label=input_file)
plt.gca().legend(loc='center')
plt.show()
