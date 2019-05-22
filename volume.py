# Read volume data from a Gaussian output file
# especially for output file with many repetitions of the same
# Monte-Carlo calculation
import re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'C:/Users/irikura/Documents/bin3')
import chem_subs
import g09_subs3
##
if len(sys.argv) < 2:
    sys.exit('Usage: volume.py <Gaussian volume output file>')
#
def charsub(line):
    # given a line of text, replace certain characters to simplify parsing
    # then split the line and return the fields
    line = line.replace('D', 'E')
    for c in ['=', '(', ')']:
        line = line.replace(c, ' ')
    return line.split()
#
def parse_block(block):
    # return dict of selected data
    retval = {}
    rxdens = re.compile(r'Number of points per bohr')
    rxnpt = re.compile(r'There are\s+(\d+) points\.  Will hold')
    rxcheck = re.compile(r'Integrated density=')
    rxvol = re.compile(r'Molar volume =')
    rxa0 = re.compile(r'Recommended a0 for SCRF calculation')
    for line in block:
        if rxdens.search(line):
            # the number of points per cubic bohr
            # replace D with E for scientific notation
            field = charsub(line)
            retval['pointdens'] = int(field[-3])
            retval['cutoff'] = float(field[-1])
        m = rxnpt.search(line)
        if m:
            # the number of sampling points?
            retval['npt'] = int(m.group(1))
        if rxcheck.search(line):
            # the integrated electron density as check of MC 
            field = charsub(line)
            dens = float(field[2])
            err = float(field[-1])
            pct = 100 * err / dens
            retval['errpct'] = pct
        if rxvol.search(line):
            # the molar volume in two units
            field = charsub(line)
            retval['au/mol'] = float(field[2])
            retval['cc/mol'] = float(field[-2])
        if rxa0.search(line):
            field = charsub(line)
            retval['a0ang'] = float(field[-4])
    return retval
##
# Find the volume-related output and copy it to a buffer for parsing
rxblock = re.compile(r'Monte-Carlo method of calculating molar volume:')
rxblockend = re.compile(r'Recommended a0 for SCRF calculation')
inblock = False
block = []  # line buffer for volume section of output
results = []
with open(sys.argv[1]) as fout:
    print(sys.argv[1])
    for line in fout:
        if inblock:
            block.append(line)
            if rxblockend.search(line):
                inblock = False
                blockdata = parse_block(block)
                results.append(blockdata)
                # prepare for next block
                block = []
        if rxblock.search(line):
            inblock = True
# calculate statistics
df = pd.DataFrame(results)
print(df)
Npt = df.shape[0]
meanvol = df['cc/mol'].mean()
stdvol = df['cc/mol'].values.std(ddof=1)
print('\nMean vol (cc/mol) = {:.1f}'.format(meanvol))
print('Stds     (cc/mol) = {:.1f}'.format(stdvol))
print('Stdev of mean     = {:.1f}'.format(stdvol/np.sqrt(Npt)))
print('\nMeans:')
print(df.mean())