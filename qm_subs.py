# Interface routines for quantum-chemistry codes
# Python3 and pandas
# Karl Irikura, NIST 2017
####
import sys, re, os
import subprocess
import pandas as pd
import numpy as np
from chem_subs import *
##
def run_qm_job(code, inlist, outlist=None):
    #   *** THIS ROUTINE MUST BE TAILORED TO YOUR COMPUTER SYSTEM ***
    #   * change the *CMD commands below so that they work on your machine
    #
    # Run one quantum chemistry calculation
    #   'code':    QM code to run (string or list of strings)
    #   'infile':  input file(s) (string or list of strings)
    #   'outlist': output file(s) (string or list of strings)
    #              will be created if not supplied 
    #
    G09_CMD = 'g09'
    QE_PW_CMD = 'pw.x'
    QE_PROJWFC_CMD = 'projwfc.x'
    #
    code = conv_list_scalar(code, 'list') # make it a list
    inlist = conv_list_scalar(inlist, 'list')
    # create name of primary output file, if it was not provided as outlist[0]
    if outlist is None:
        # output file name(s) not provided
        outlist = [ supply_qm_filename_suffix(code[0], \
            os.path.splitext(inlist[0])[0], 'output') ]
    outlist = conv_list_scalar(outlist, 'list')
    FDin = open(inlist[0], 'r')
    FDout = open(outlist[0], 'w')
    if code[0] == 'gaussian09':
        # if a checkpoint file is used, append it to the list of 
        #   output files
        chkfile = ''
        regxchk = re.compile(r'%chk=(\S+)', re.IGNORECASE)
        with open(inlist[0]) as f:
            for line in f:
                m = regxchk.match(line)
                if m:
                    chkfile = m.group(1)
                    # a checkpoint file is specified; add it to the list of output file
                    outlist.append(chkfile)
        # run the calculation and wait for it to complete
        rcode = subprocess.call([G09_CMD], stdin=FDin, stdout=FDout, stderr=FDout)
    elif code[0] == 'quantum-espresso':
        # run the specific code specified by code[1]
        if len(code) < 2:
            print_err('', 'Must specify module within Quantum Espresso package')
        if code[1].lower() == 'pw':
            rcode = subprocess.call([QE_PW_CMD], stdin=FDin, stdout=FDout, stderr=FDout)
        elif code[1].lower() == 'projwfc':
            rcode = subprocess.call([QE_PROJWFC_CMD], stdin=FDin, stdout=FDout, stderr=FDout)
        else:
            print_err('code', '{:s} within Quantum Espresso package'.format(code[1]))
    else:
        print_err('code', str(code))
    FDin.close()
    FDout.close()
    return outlist
##
def run_qm_task(code, qminp, task):
    # invoke run_qm_job() as appropriate
    #    (maybe more than once)
    # 'code' and 'qminp' may be lists 
    # return a list of output files
    inlist = conv_list_scalar(qminp, 'list')
    outlist = []
    code = assign_qm_codename(code, task)
    if len(inlist) == 1:
        # only one calculation to run
        outlist = run_qm_job(code, inlist[0])
    else:
        njob = len(inlist)
        for ijob in range(njob):
            # two-name code specification
            code2 = [code[0], code[ijob+1]]
            qmout = run_qm_job(code2, inlist[ijob])
            outlist.extend(qmout)
    return outlist
##
def assign_qm_codename(code, task='energy'):
    # 'code': string naming QM package
    # 'task': calculation type 
    # return a list with 'code' as the first element and any added elements
    #   specifying code(s) within the QM package
    if code == 'gaussian09':
        # there is only one invocation
        return [code]
    elif code == 'quantum-espresso':
        if task in ['minimize', 'gradient', 'force', 'energy']:
            return [code, 'PW']
        elif task == 'charges':
            return [code, 'PW', 'PROJWFC']
        else:
            # the requested task is not recognized
            print_err('task', task)
    else:
        print_err('code', code)
##
def identify_qm_code_out(outfile):
    # return a list of the name of the quantum code and any version info
    # codes recognized: 'gaussian09', 'quantum-espresso'
    regx_g09 = re.compile(' Gaussian 09:\s+\S+Rev([A-Z]\.\d+)\s+(\d+-\w+-\d+)')
    regx_qe = re.compile('This program is part of the open-source ' +
        'Quantum ESPRESSO suite')
    regx_qe_module = re.compile('\s+Program ([A-Z]+)\s+v\.(\d+\.\d+) starts on')
    code = []
    with open(outfile) as fout:
        for line in fout:
            # check for gaussian09
            m = regx_g09.match(line)
            if m:
                return(['gaussian09', m.group(1), m.group(2)])
            # check for quantum-espresso and component codes
            m = regx_qe_module.match(line)
            if m:
                code.extend([m.group(1), m.group(2)])
            if regx_qe.search(line):
                code.insert(0, 'quantum-espresso')
                # this line appears after the 'PWSCF' line
                return(code)                
    # failure
    return False
##
def read_qm_ZMgradient(code, qmoutfile, ZM, nlines=False):
    # Read energy gradient expressed in z-matrix coordinates
    # the argument 'ZM' is a ZMatrix object that contains the
    #   variable names and definitions
    # Return a list of [dict of variable values]
    # units are hartree, bohr, radian
    grads = []
    linenos = []
    lcount = 0  # line counter
    if code == 'gaussian09':
        # avoid the section with explicit z-mat variable names
        #   because it has only five decimal digits instead of six
        regxStart = re.compile(r'\s+Internal Coordinate Forces\s+.Hartree.Bohr or radian')
        regxEnd = re.compile(r'\s*Internal\s+Forces:\s+Max')
        regxAtom = re.compile(r'\s+(\d+)\s+([A-Za-z]+)')
        regxDel = re.compile(r'\(\s+\d+\)(\s+0)?|^\s*\d+\s+')  # trash to delete from data lines
        regxInt = re.compile(r'^\d+$')
        scaling = -1  # Gaussian reports force, not gradient
        # lines look like this:
        #    5  C        4   0.023718(     4)      3   0.015120(    18)      2   0.000000(    31)     0
    elif code == 'someothercode':
        pass
    else:
        print_err('code', code)
    ingrad = False
    with open(qmoutfile) as f:
        for line in f:
            lcount += 1
            if ingrad:
                # inside a forces block
                m = regxAtom.match(line)
                if m:
                    # a data line
                    atno += 1
                    if atno == 0:
                        # the first atom; no information here
                        continue
                    line = regxDel.sub('', line).strip()  # clean up the line
                    words = line.split()
                    if words[0] != ZM.el[atno]:
                        # this is not the element expected
                        print_err('atom_order', '{:s} != {:s}'.format(words[1], ZM.el[atno]))
                    reflist = []  # list of reference atoms in this line
                    vallist = []  # list of coordinate values in this line
                    for w in words[1:]:
                        if regxInt.match(w):
                            # a reference atom
                            reflist.append(int(w) - 1)   # adjust index to be zero-based
                        else:
                            # a zvar value
                            vallist.append(float(w))
                    # check that reflist[] is the same as expected
                    if not reflist == ZM.refat[atno]:
                        # this is not the expected list
                        print_err('atom_order', 'reflist {:s} != {:s}'.format(str(reflist), str(ZM.refat[atno])))
                    # record the values of the gradients
                    for j in range(len(reflist)):
                        gradIC[ZM.var[atno][j]] = vallist[j] * scaling
                if regxEnd.search(line):
                    # end of coordinate block
                    ingrad = False
                    # add it to the list
                    grads.append(gradIC)
            if regxStart.match(line):
                # beginning of a force block
                ingrad = True
                gradIC = {} # dict of internal-coordinate derivatives
                linenos.append(lcount)
                atno = -1
    if nlines:
        return grads, linenos
    else:
        return grads
##
def read_qm_gradient(code, qmoutfile, nlines=False):
    # Read cartesian energy gradient from quantum chemistry output file
    # Return a list of [numpy array of [x,y,z] triples]
    # If nlines is True, also return the corresponding line numbers
    # units are hartree, bohr
    grads = []
    linenos = []
    lcount = 0  # line counter
    if code == 'gaussian09':
        # units from Gaussian are hartree, bohr
        regxStart = re.compile(r'\s*Center\s+Atomic\s+Forces')
        regxEnd = re.compile(r'\s*Cartesian Forces:\s+Max')
        regxAtom = re.compile(r'\s+(\d+)\s+(\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)')
        # regxAtom match-groups are: center no., Z, -dX, -dY, -dZ
        scaling = -1.  # Gaussian reports force, not gradient
    elif code == 'quantum-espresso':
        # units from Quantum Espress are Ryd/bohr
        regxStart = re.compile(r'\s*Forces acting on atoms \(cartesian axes, Ry/au')
        regxEnd = re.compile(r'\s*Total force = ')
        regxAtom = re.compile(r'\s*atom\s+(\d+)\s+type\s+(\d+)\s+force =\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')
        # regxAtom match-groups are: atom no., type no., -dX, -dY, -dZ
        scaling = -0.5  # force not gradient; Ryd not hartree
    else:
        print_err('code', code)
    incoord = False
    with open(qmoutfile) as f:
        for line in f:
            lcount += 1
            if incoord:
                # inside a forces block
                m = regxAtom.match(line)
                if m:
                    # a data line; ignore the two int fields
                    xyzlist.append(m.group(3,4,5))
                if regxEnd.search(line):
                    # end of coordinate block
                    incoord = False
                    # convert to a numpy array and add it to the list
                    grads.append(scaling * np.array(xyzlist, dtype=np.float64))
            if regxStart.match(line):
                # beginning of a force block
                incoord = True
                xyzlist = []  # list of gradient triples
                linenos.append(lcount)
    if nlines:
        return grads, linenos
    else:
        return grads
##
def qm_calculation_failure(codename, task, outlist):
    # Did quantum chemistry calculation succeed at 'task'?
    # return 0 for success, else return the number of the bad output file (usually 1)
    # task options: 'minimize', 'gradient', 'energy'
    code = conv_list_scalar(codename, 'string')
    qmoutfile = conv_list_scalar(outlist, 'string')
    if code == 'gaussian09':
        regxOpt = re.compile(r'\s*Optimization completed\.')
        regxForce = re.compile(r'\s*Maximum Force\s+\d+\.\d+')
        regxSCF = re.compile(r'\s*SCF Done:\s+E\(')
        # scan only one output file for Gaussian09
    elif code == 'quantum-espresso':
        regxOpt = re.compile(r'\s*End of BFGS Geometry Optimization')
        regxForce = re.compile(r'\s*Forces acting on atoms \(cartesian axes, Ry/au\):')
        regxSCF = re.compile(r'\s*End of self-consistent calculation')
        regxCharge = re.compile(r'Lowdin Charges:')
    else:
        print_err('code', code)
    if task == 'minimize':
        # ordinary geometry optimization
        regx = regxOpt
    elif task == 'gradient':
        # ordinary energy gradient
        regx = regxForce
    elif task in ['energy', 'charges']:
        # ordinary SCF energy
        regx = regxSCF
    success = False
    with open(qmoutfile) as f:
        for line in f:
            if regx.match(line):
                success = True
                break
    if not success:
        # failed already
        return 1
    if (code == 'quantum-espresso') and (task == 'charges'):
        # there is a second output file to scan
        success = False
        with open(outlist[1], 'r') as f:
            for line in f:
                if regxCharge.match(line):
                    success = True
                    break
        if success:
            return 0
        else:
            return 2
    return 0
##
def read_qm_ZMatrix(code, qmoutfile, nline=False, unitR='angstrom'):
    # Read internal coordinates from quantum chemistry output file
    # Return a list of ZMatrix objects (distances in requested unitR)
    # If nline is True, also return a list of the corresponding
    #   line numbers. 
    if code == 'gaussian09':
        regxStart = re.compile(r'^\s*(Symbolic|Final structure in terms of initial) Z-matrix:')
        regxEnd = re.compile(r'(\s*$|\s*1[\\\|]1[\\\|])')  # either blank line or archive block
        regxMid = re.compile(r'^\s+Variables:\s*$')   # between atom and variable sections
        regIgnore = re.compile(r'^\s*Charge .*Multiplicity ')  # ignore this line
        regxStrip = re.compile(r'(,0| 0)$')  # remove this from all data lines
        unitR = 'angstrom'
        unitA = 'degree'
    elif code == 'someothercode':
        pass
    else:
        print_err('code', code)
    zmlists = []  # list of [list of lines of each z-matrix]
    zmat = []  # list of lines in a z-matrix
    linenos = []
    lcount = 0  # line counter
    inzmat = 0   # ternary flag (0, 1, 2) for (not, top, bottom) of z-matrix
    with open(qmoutfile) as f:
        for line in f:
            lcount += 1
            if inzmat:
                if regIgnore.match(line):
                    # ignore this line
                    continue
            if inzmat == 1:
                # atom-definition part of z-matrix
                if regxMid.match(line):
                    # done with top part
                    inzmat = 2
                else:
                    # add this line to the list
                    line = line.strip()
                    line = regxStrip.sub('', line)
                    #line = line.strip('0,')   # Gaussian09 have have trailing ' 0' or ',0'
                    zmat.append(line.rstrip())
            elif inzmat == 2:
                # variable assignment part of z-matrix
                if regxEnd.match(line):
                    # end of this z-matrix; add it to the list
                    inzmat = 0
                    zmlists.append(zmat.copy())
                    zmat = []
                else:
                    # add this line to the list
                    zmat.append(line.strip())
            if regxStart.match(line):
                # beginning of a z-matrix
                inzmat = 1
                linenos.append(lcount)
    # convert zmat lists into proper ZMatrix objects
    geoms = []
    for zmat in zmlists:
        geom = parse_ZMatrix(zmat, unitR=unitR, unitA=unitA)
        geoms.append(geom)
    if nline:
        return geoms, linenos
    else:
        return geoms
##
def read_qm_Geometry(codename, qmoutfile, nline=False, units='angstrom'):
    # Read atomic coordinates from quantum chemistry output file
    # Return a list of Geometry objects
    # If nline is True, also return a list of the corresponding
    #   line numbers. 
    code = conv_list_scalar(codename, 'string')
    if code == 'gaussian09':
        regxStart = re.compile(r'\s+(Standard|Input) orientation:')
        regxEnd = re.compile(r'-{60}')
        nEnd = 3  # need to match regxEnd three times before reaching end of coordinates
        regxAtom = re.compile(r'\s+\d+\s+(\d+)\s+\d+\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)')
        # regxAtom match-groups are: center no., Z, X, Y, Z
        qm_unit = 'angstrom'
    elif code == 'quantum-espresso':
        regxStart = re.compile(r'^ATOMIC_POSITIONS \((angstrom|bohr)\)')
        regxEnd = re.compile(r'^\s*$')
        nEnd = 1
        regxAtom = re.compile(r'(^[A-Z][a-zA-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')
        # regxAtom match-groups are: element symbol, X, Y, Z
    else:
        print_err('code', code)
    geoms = []
    linenos = []
    lcount = 0  # line counter
    countEnd = 0  # regxEnd match counter
    incoord = False
    with open(qmoutfile) as f:
        for line in f:
            lcount += 1
            if incoord:
                # inside a ccoordinate block
                m = regxAtom.match(line)
                if m:
                    # a data line
                    a = Atom(m.group(1), m.group(2,3,4))
                    atlist.append(a)
                if regxEnd.search(line):
                    countEnd += 1
                    if countEnd == nEnd:
                        # end of coordinate block
                        incoord = False
                        countEnd = 0
                        # convert the list of Atoms to a Geometry and add it to the list
                        Geom = Geometry(atlist, intype='atlist', units=qm_unit)
                        if qm_unit != units:
                            # convert to requested units
                            if units == 'angstrom':
                                Geom.toAngstrom()
                            else:
                                Geom.toBohr()
                        geoms.append(Geom)
            m = regxStart.match(line)
            if m:
                # beginning of a coordinate block
                incoord = True
                linenos.append(lcount)
                atlist = []  # list of atoms
                if code == 'quantum-espresso':
                    qm_unit = m.group(1)
    if nline:
        return geoms, linenos
    else:
        return geoms
##
def read_qm_E_scf(code, qmoutfile, nline=False, ncycle=False):
    # Read SCF energ(ies) from quantum chemistry output file 'qmoutfile'
    #   generated by quantum chemistry code 'code'
    # Return a list of energies (as float)
    # if 'lineno' is True, also return a list of the corresponding line numbers
    # if 'ncycle' is True, also return a list of the number of SCF cycles
    if code == 'gaussian09':
        regx = re.compile(r'\s*SCF Done:\s+E\((\S+)\)\s+=\s+(\S+)\s+A\.U\. after\s+(\d+) cycle')
        igrp_e = 2
    elif code == 'quantum-espresso':
        regx = re.compile(r'!\s+total energy\s+=\s+(-?\d+\.\d+) Ry')
        igrp_e = 1
        regx_iter = re.compile('convergence has been achieved in\s(\d+)\s+iterations')
        igrp_cyc = 1
    else:
        print_err('code', code)
    escf = []
    linenos = []
    cycles = []
    lcount = 0  # line counter
    with open(qmoutfile) as f:
        for line in f:
            lcount += 1
            m = regx.match(line)
            if m:
                escf.append(float(m.group(igrp_e)))
                linenos.append(lcount)
                if code in ['gaussian09']:
                    # read number of SCF cycles from same line
                    cycles.append(int(m.group(3)))
            if code in ['quantum-espresso']:
                # read number of SCF cycles from a different line
                m = regx_iter.search(line)
                if m:
                    cycles.append(int(m.group(igrp_cyc)))
    if nline:
        if ncycle:
            return escf, linenos, ncycle
        else:
            return escf, linenos
    else:
        if ncycle:
            return escf, ncycle
        else:
            return escf
##
def write_qm_input(filename, code, contents):
    # using information in dict 'contents', write a QM input file
    #   appropriate for quantum chemistry code 'code'
    try:
        qmfile = open(filename, 'w')
    except:
        print_err('write_fail', filename)
    if code == 'gaussian09':
        # expect 'contents' to have keys:
        #   header; command; comment; charge; spinmult; coordinates; trailer
        #   of type:
        #   dict  ; str    ; str    ; int   ; int     ; str        ; str
        if 'header' in contents:
            # not always present
            for key in contents['header']:
                val = contents['header'][key]
                qmfile.write('%{:s}={:s}\n'.format(key, str(val)))
        qmfile.write('{:s}\n\n'.format(contents['command']))  # blank line after commands
        qmfile.write('{:s}\n\n'.format(contents['comment']))
        qmfile.write('{:d} {:d}\n'.format(contents['charge'], contents['spinmult']))
        qmfile.write(contents['coordinates'] + '\n')
        if 'trailer' in contents:
            # not always present
            qmfile.write(contents['trailer'] + '\n')
    elif code == 'quantum-espresso':
        # expect 'contents' to include keys 'amp' and 'noamp',
        # both of type 'str'
        qmfile.write(contents['amp'])
        qmfile.write(contents['noamp'])
    else:
        print_err('code', code)
    qmfile.close()
    return
##
def supply_qm_filename_suffix(code, filename, io):
    # if 'filename' lacks a suffix, add one appropriate for the
    #   specified QM code
    if not re.search(r'\.\w+$', filename):
        # there is no suffix; add one
        if io == 'input':
            # input file for QM code
            suffix = {'default': 'inp', 'gaussian09': 'gjf',
                'quantum-espresso': 'in'}
        elif io == 'output':
            # output file for QM code
            suffix = {'default': 'out', 'gaussian09': 'out',
                'quantum-espresso': 'out'}
        else:
            print_err('io', io)
        # add the suffix
        if code in suffix:
            filename += '.{:s}'.format(suffix[code])
        else:
            filename += '.{:s}'.format(suffix['default'])
    return filename
##
def format_qm_coordinates(code, crds, intype='auto'):
    # given coordinates in some format:
    #   'Geometry' (a Geometry object--see chem_subs.py)
    #   'cartesian' (a list of chemical elements and list/array of cartesians)
    #   'ZMatrix' (a ZMatrix object--see chem_subs.py)
    #   'auto' (detect type)
    # return a string suitable for the coordinates section of 
    #   an input file for the specified quantum-chemistry code
    if intype == 'auto':
        # detect the data type
        intype = typeCoord(crds)
    coord = ''
    Gcopy = crds.copy()  # units may be changed
    if code in ['gaussian09', 'quantum-espresso']:
        # el  x  y  z
        if intype == 'Geometry':
            Gcopy.toAngstrom()  # make sure units are Angstrom
            for at in Gcopy.atom:
                coord += at.printstr() + '\n'
        elif intype == 'cartesian':
            # no information about units
            elem = Gcopy[0]  # element symbols first
            natom = len(elem)
            xyz = np.array(Gcopy[1])  # cartesian coordinates
            if len(xyz.shape) == 1:
                # coordinate array is flattened
                xyz = xyz.reshape((natom, 3))
            for i in range(natom):
                coord += '{:s}  {:11.5f}  {:11.5f}  {:11.5f}\n'.format(elem[i], xyz[i][0], xyz[i][1], xyz[i][2])
        elif intype == 'ZMatrix':
            # z-matrix with both constant and symbolic variables
            Gcopy.toAngstrom() 
            Gcopy.toDegree()  # want angles in degrees
            coord = Gcopy.printstr()
        else:
            print_err('coordtype', intype)
    else:
        print_err('code', code)
    return coord 
##
def qm_diagnose(codename, task, qmoutfile):
    # Figure out what went wrong with a failed QM calculation
    problem = 'unidentified problem with task "{:s}'.format(task)
    code = conv_list_scalar(codename, 'string')
    if code == 'gaussian09':
        if task == 'minimize':
            # Was there a coordinate failure?
            regxCoord = re.compile('FormBX had a problem')
            # Did we exceed the iteration limit?
            regxIter = re.compile(r'Number of steps exceeded,\s+NStep=\s+(\d+)')
            with open(qmoutfile, 'r') as f:
                for line in f:
                    if regxCoord.search(line):
                        problem = 'coordinate failure'
                        break
                    if regxIter.search(line):
                        problem = 'iteration limit'
                        break
    elif code == 'quantum-espresso':
        if task == 'minimize':
            # Did we exceed the iteration limit?
            regxLim = re.compile(r'\s*nstep\s+=\s+(\d+)')
            regxActual = re.compile(r'\s*number of bfgs steps\s+=\s+(\d+)')
            with open(qmoutfile) as f:
                for line in f:
                    m = regxLim.match(line)
                    if m:
                        nstep = int(m.group(1))
                        niter = 0
                    m = regxActual.match(line)
                    if m:
                        # QE sometimes repeats step numbers! Have to count.
                        niter += 1
            try:
                if niter >= nstep:
                    problem = 'iteration limit'
            except:
                pass
    else:
        print_err('code', code)
    print('******* nstep = {}, niter = {}'.format(nstep, niter))
    return problem
##
def read_Znuc(fname):
    # return a pandas DataFrame with one row per atom:
    #    element symbol, Z, and Zeff
    # If multiple blocks are in the file, only read the first one
    code = identify_qm_code_out(fname)
    Z = []
    Zeff = []
    symb = []
    if code[0] == 'gaussian09':
        df = read_g09_Znuc(fname)
        return df
    elif code[0] == 'quantum-espresso':
        # Zeff values are listed by element; atoms are listed 
        #   individually in a different place
        regxStart = re.compile(r'atomic species   valence    mass     pseudopotential')
        regxEnd = re.compile(r'^\s*$')
        # columns are: symbol, valence, mass, and PP info
        # only want columns 1 and 2
        # each element is listed once, regardless of the number of instances
        #   within the molecule
        regx = re.compile(r'\s+([A-Z][a-z]?)\s+(\d+)\.00\s+(?:\d+\.\d+)')
        inblock = False
        nval = {}
        # also must find table of individual atoms
        regxAStart = re.compile(r'\s+site n\.\s+atom\s+positions')
        regxA = re.compile(r'\s+(?:\d+)\s+([A-Z][a-z]?)\s+tau')
        inA = False
        with open(fname) as f:
            for line in f:
                if inblock:
                    m = regx.match(line)
                    if m:
                        nval[m.group(1)] = int(m.group(2))
                    if regxEnd.match(line):
                        inblock = False
                        continue
                if regxStart.search(line):
                    if len(nval) == 0:
                        inblock = True
                    else:
                        # ignore all but first block
                        continue
                if inA:
                    m = regxA.match(line)
                    if m:
                        symb.append(m.group(1))
                    if regxEnd.match(line):
                        inA = False
                        continue
                if regxAStart.match(line):
                    if len(symb) == 0:
                        inA = True
                    else:
                        # ignore all but first block
                        continue
        for el in symb:
            Z.append(elz(el, 'Z'))
            Zeff.append(nval[el])
        data = {'Elem': pd.Series(symb),
                'Z'   : pd.Series(Z),
                'Zeff': pd.Series(Zeff)}
        df = pd.DataFrame(data)
        return df[['Elem', 'Z', 'Zeff']]
    else:
        print_err('code', code)
    return None
##
def read_g09_Znuc(fname):
    # return a pandas DataFrame with one row per atom:
    #    element symbol, Z, and Zeff
    # If multiple blocks are in the file, only read the first one
    Z = []
    Zeff = []
    symb = []
    # Z values are in one or two places
    regxZStart = re.compile(r'(Input|Standard) orientation:')
    regxZEnd = re.compile(r'------------------------')
    endZcount = 3  # number of lines that must match regxEnd 
    # columns in g09 output: center no., atomic no., atomic type,
    #   x, y, z coordinates; ignore all but column 2
    s = r'\s+(?:\d+)\s+(\d+)\s+(?:\d+)' + r'\s+(?:-?\d+\.\d+)' * 3
    regxZ = re.compile(s)
    inZ = False
    # pseudopotential info (if any) is elsewhere
    regxPPStart = re.compile(r'\s+Pseudopotential Parameters')
    regxPPEnd = re.compile(r'================')
    endPPcount = 3
    # only PP atoms have more than two columns
    # first three columns: center no., atomic no., [val. electrons]
    regxPP = re.compile(r'\s+(?:\d+)\s+(\d+)\s*(\d+)?$')
    inPP = False
    nval = {}  # number of explicit electrons
    # search the file
    with open(fname) as f:
        for line in f:
            # look for Z data
            if inZ:
                if regxZEnd.search(line):
                    endZcount -= 1
                if endZcount < 1:
                    # done reading Z data
                    inZ = False
                    continue
                m = regxZ.search(line)
                if m:
                    zval = int(m.group(1))
                    Z.append(zval)
                    symb.append(elz(zval, 'symbol'))
            if regxZStart.search(line):
                if len(Z) > 0:
                    # already have data; ignore subsequent blocks
                    continue
                else:
                    # prepare to read block
                    inZ = True
            # look for PP data
            if inPP:
                if regxPPEnd.search(line):
                    endPPcount -= 1
                if endPPcount < 1:
                    # done reading PP data
                    inPP = False
                    continue
                m = regxPP.match(line)
                if m:
                    zval = int(m.group(1))
                    if m.group(2) is None:
                        # no PP on this element
                        nval[zval] = zval
                    else:
                        nval[zval] = int(m.group(2))
            if regxPPStart.match(line):
                if len(nval) > 0:
                    # ignore all but first block
                    continue
                else:
                    # prepare to read block
                    inPP = True
    # transfer nval data to Zeff[]
    for zval in Z:
        if zval in nval:
            Zeff.append(nval[zval])
        else:
            Zeff.append(zval)
    data = {'Elem': pd.Series(symb),
            'Z'   : pd.Series(Z),
            'Zeff': pd.Series(Zeff)}
    df = pd.DataFrame(data)
    return df[['Elem', 'Z', 'Zeff']]
##
def read_atomic_charges(fnames):
    # read atomic charges from minimum number of output files
    # return charge type ('Mulliken' or 'Lowdin') and a pandas
    #   DataFrame with a row for each atom
    fname = conv_list_scalar(fnames, 'string')
    code = identify_qm_code_out(fname)
    if code[0] == 'gaussian09':
        # Gaussian09 prints net atomic charges
        df = read_Mulliken_charges(fname)['Mulliken'].iloc[-1]
        return df, 'Mulliken'
    elif code[0] == 'quantum-espresso':
        # Quantum Espresso (projwfc.x) prints electron populations, not charges
        if (type(fnames) is not list) or (len(fnames) != 2):
            print_err('', 'need two output files: from pw.x and from projwc.x')
        dfZ = read_Znuc(fnames[0])
        dfq, qtype = read_atomic_pops(fnames[1])
        if (dfZ is None) or (dfq is None):
            return None, None
        elems = dfZ['Elem'].values
        charges = dfq['Pop'].values - dfZ['Zeff'].values
        data = {'Element': elems, 'Charge': charges}
        df = pd.DataFrame(data)
        return df[['Element', 'Charge']], qtype
    else:
        print_err('code', code)
##
def read_atomic_pops(fname):
    # read atomic electron populations from minimum number of output files
    # return charge type ('Mulliken' or 'Lowdin') and a pandas
    #   DataFrame with a row for each atom
    code = identify_qm_code_out(fname)
    if code[0] == 'gaussian09':
        # Gaussian09 prints net atomic charges, not populations
        dfq = read_Mulliken_charges(fname)['Mulliken'].iloc[-1]
        dfz = read_g09_Znuc(fname)
        data = {'Element': dfq['Element'].values,
                'Pop'    : dfz['Zeff'].values + dfq['Charge'].values}
        df = pd.DataFrame(data)
        return df[['Element', 'Pop']], 'Mulliken'
    elif code[0] == 'quantum-espresso':
        # Quantum Espresso (projwfc.x) prints electron populations
        regxStart = re.compile(r'Lowdin Charges:')
        regxEnd = re.compile(r'Spilling Parameter:\s+(-?\d+\.\d+)')
        regxQ = re.compile(r'Atom #\s+(\d+): total charge =\s+(-?\d+\.\d+),')
        inQ = False
        with open(fname) as f:
            for line in f:
                if inQ:
                    if regxEnd.search(line):
                        inQ = False
                    m = regxQ.search(line)
                    if m:
                        anum = int(m.group(1))
                        q = float(m.group(2))
                        if anum not in atno:
                            atno.append(anum)
                            epop.append(q)
                else:
                    if regxStart.search(line):
                        # initialize lists
                        atno = []
                        epop = []
                        inQ = True
        try:
            data = {'Atom#': atno, 'Pop': epop}
            df = pd.DataFrame(data)
            return df[['Atom#', 'Pop']], 'Lowdin'
        except:
            # probably no data in this file
            return None, None
    else:
        print_err('code', code)
##
def read_Mulliken_charges(fname, sumH=False):
    ## only for Gaussian09 ##
    # Read Mulliken charges (and spin densities, if present).  
    # If sumH==True, read "with hydrogens summed into heavy atoms".
    # Return a pandas DataFrame with a row for each set of populations like:
    #   (1) line number,
    #   (2) a DataFrame with columns:
    #       (a) 'Element' (element symbol)
    #       (b) 'Charge' (Mulliken atomic charge)
    #       (c) 'SpinDensity' (if data are found)
    fline = []
    Mulq = []
    lineno = 0
    inblock = False
    if sumH:
        regstart = re.compile('Mulliken charges( and spin densities)? with hydrogens summed')
    else:
        regstart = re.compile('Mulliken charges( and spin densities)?:')
    regspin = re.compile('and spin densit')
    regdata = re.compile(r'\s+\d+\s+[A-Z][a-z]?\s+')    # atom number, element symbol
    regheader = re.compile(r'^[\s\d]+$')
    with open(fname, 'r') as fhandl:
        for line in fhandl:
            lineno += 1
            if inblock:
                m = regdata.match(line)
                if m:
                    # save the element symbol and the charge
                    # line example: " 1  Cl   0.069672   0.086170" (last float is spin density)
                    fields = line.split()
                    elem.append(fields[1])
                    charge.append(float(fields[2]))
                    if haveSpin:
                        spin.append(float(fields[3]))
                else:
                    m = regheader.match(line)
                    if m:
                        # just skip this line
                        continue
                    # end of data block
                    if haveSpin:
                        data = list(zip(elem, charge, spin))
                        cols = ['Element', 'Charge', 'SpinD']
                    else:
                        data = list(zip(elem, charge))
                        cols = ['Element', 'Charge']
                    df = pd.DataFrame(data, columns=cols)
                    Mulq.append(df)
                    inblock = False
            m = regstart.search(line)
            if m:
                # found a block of Mulliken charges
                fline.append(lineno)
                inblock = True
                elem = []
                charge = []
                haveSpin = False
                if regspin.search(line):
                    haveSpin = True
                    spin = []
    data = list(zip(fline, Mulq))
    cols = ['line', 'Mulliken']
    df = pd.DataFrame(data, columns=cols)
    return df
##
def read_g09_atom_masses(fname):
    # read atom masses from the main output text
    # return a simple list of atom masses
    # only read first instance
    regx = re.compile(r' Atom\s+\d+ has atomic number\s+\d+ and mass\s+(\d+\.\d+)')
    regend = re.compile(r' Molecular mass:\s\d+\.\d+ amu')
    masses = []
    with open(fname) as fhandl:
        for line in fhandl:
            if regend.match(line):
                # done reading atom masses
                break
            m = regx.match(line)
            if m:
                # add this mass to the list
                masses.append(float(m.group(1)))
    return masses
##
def read_g09_archive_block(fname):
    # read the last archive block in a Gaussian09 output file
    # return a list of lists of lists; no parsing here!
    regx = re.compile(r' 1[\|\\]1[\|\\](.*)')
    regblank = re.compile('^\s*$')
    inarch = False
    with open(fname) as fhandl:
        for line in fhandl:
            if inarch:
                if regblank.match(line):
                    # end of archive block
                    inarch = False
                    continue
                archstring += line[1:-1]
            m = regx.match(line)
            if m:
                inarch = True
                archstring = m.group(1)   # this erases any previous data
    # first break the long string at the double delimiters
    arch = re.split(r'\|\||\\\\', archstring)
    # break each major field into minor fields, at the single delimiters
    regx = re.compile(r'\||\\')
    for i in range(len(arch)):
        field = regx.split(arch[i])
        arch[i] = field  # replace by a list of sub-fields
    return arch
##
def parse_g09_archive_block(arch):
    # attempt to interpret the fields read by read_g09_archive_block()
    # return a dict with (mis?)informative keys
    # WARNING--CONTAINS GUESSING
    retval = {}
    retval['metadata'] = arch[0]
    retval['command'] = arch[1]
    retval['comment'] = arch[2]
    # expect arch[3] to be (charge,mult) and coords in 'Input Orientation'
    t = arch[3][0].split(',')
    retval['charge'] = int(t[0])
    retval['multiplicity'] = int(t[1])
    # coordinates may be cartesians or z-matrix
    cartesians = ('Version' in arch[4][0])
    if not cartesians:
        # arch[4] is z-matrix variable values
        retval['zmatrix'] = arch[3][1:]
        retval['zvars'] = arch[4]
        natom = len(retval['zmatrix'])
        i = 5
    else:
        retval['cartesians'] = arch[3][1:]
        natom = len(retval['cartesians'])
        i = 4
    retval['natom'] = natom
    # expect next field to include several properties expressed using '='
    #  its first field is the version number; move that into the 'metadata'
    retval['metadata'].append(arch[i].pop(0))
    retval['properties'] = {}
    for field in arch[i]:
        # split key from value at '='
        [key, val] = field.split('=')
        retval['properties'][key] = val
    i += 1
    if '@' in arch[i][-1]:
        # no more to read
        return retval
    # expect hessian next (lower triangle)
    fc = [float(x) for x in arch[i][0].split(',')]
    ndof = 3 * natom
    ntriangle = ndof * (ndof + 1) // 2
    if ntriangle != len(fc):
        print_err('', 'Expected {:d} elements '.format(ntriangle) +
            'of triangular hessian but found {:d}'.format(len(fc)),
            halt=False)
        return retval
    else:
        # convert to a square matrix
        il = np.tril_indices(ndof)
        mat = np.zeros((ndof, ndof))
        mat[il] = fc
        utri = np.tril(mat, -1).T
        hess = mat + utri
        #iu = np.triu_indices(ndof)
        #mat = np.zeros((ndof, ndof))
        #mat[iu] = fc
        #hess = np.tril(mat.T, -1) + mat
        retval['hessian'] = hess
    i += 1
    # expect nuclear gradient next
    retval['gradient'] = [float(x) for x in arch[i][0].split(',')]
    return retval
##
