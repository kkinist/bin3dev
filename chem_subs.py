# Routines for general quantum chemistry (no particular software package)
# Python3 and pandas
# Karl Irikura 
#
import re, sys
import string, copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
#
# don't use the lower-case constants anymore
amu_au = 1 / 5.4857990946E-4    # amu expressed in a.u. (viz., electron masses)
amu_kg = 1.660539040e-27        # amu (aka u) expressed in kg
au_wavenumber = 219474.6313708  # hartree expressed in wavenumbers
au_joule = 4.35974434E-18       # hartree expressed in joule
avogadro = 6.02214129e23 # Avogadro constant
planck = 6.62606957e-34  # Planck constant (h) in J.s
clight = 299792458.     # speed of light (c)in m/s
bohr = 0.5291772109     # Bohr radius (a0) in angstrom
boltzmann = 1.38064852e-23  # Boltzmann constant (k) in J/K
Rgas = avogadro * boltzmann # in J / mol.K
eV_per_hartree = 27.21138602  # from NIST website 10/25/2016
au_kjmol = au_joule * avogadro / 1000   # hartree expressed in kJ/mol
ev_wavenumber = au_wavenumber / eV_per_hartree      # eV expressed in cm**-1
#
AVOGADRO = 6.02214129e23 # Avogadro constant
PLANCK = 6.62606957e-34  # Planck constant (h) in J.s
HBAR = PLANCK / (2 * np.pi) 
CLIGHT = 299792458.     # speed of light (c)in m/s
BOLTZMANN = 1.38064852e-23  # Boltzmann constant (k) in J/K
RGAS = AVOGADRO * BOLTZMANN # in J / mol.K
GOLD = (1 + np.sqrt(5))/2  # golden ratio
#
AMU_AU = 1 / 5.4857990946E-4    # amu expressed in a.u. (viz., electron masses)
AMU_KG = 1.660539040e-27        # amu (aka u) expressed in kg
AU_WAVENUMBER = 219474.6313708  # hartree expressed in wavenumbers
AU_JOULE = 4.35974434E-18       # hartree expressed in joule
EV_PER_HARTREE = 27.21138602  # from NIST website 10/25/2016
AU_KJMOL = AU_JOULE * AVOGADRO / 1000   # hartree expressed in kJ/mol
EV_WAVENUMBER = AU_WAVENUMBER / EV_PER_HARTREE      # eV expressed in cm**-1
KJMOL_WAVENUMBER = AU_WAVENUMBER / AU_KJMOL  # kJ/mol expressed in cm**-1
BOHR = 0.5291772109     # Bohr radius (a0) in angstrom
KCAL_KJ = 4.184  # kcal expressed in kJ
EV_KJMOL = AU_KJMOL / EV_PER_HARTREE  # eV expressed in kJ/mol
#
def convert_unit(quantity, target_unit):
    # unit conversion; 'quantity' is a dict with 'value' (or 'how_much') and 'unit'
    # 'target_unit' is a string in one of the lists below
    # return value: another dict with the new units
    #
    try:
        v1 = quantity['value']
        vkey = 'value'
    except KeyError:
        v1 = quantity['how_much']
        vkey = 'how_much'
    except:
        print_err('', 'unable to convert units for quantity: ' + str(quantity))
    u1 = quantity['unit'].lower()
    u2 = target_unit.lower()
    qconv = quantity.copy()
    if u1 == u2:
        # no conversion needed
        return qconv
    #
    # the order of the unit names and the regex's must be consistent
    regxE = [re.compile(s) for s in \
             ['har',     'ev', 'kj',     'cm',   'kcal']]
    energy = ['hartree', 'ev', 'kj/mol', 'cm-1', 'kcal/mol']
    regxD = [re.compile(s) for s in ['ang',      'bohr']]
    distance =                      ['angstrom', 'bohr']
    # 'me' unit is electron mass (atomic unit of mass)
    regxM = [re.compile(s) for s in ['u',   'kg', 'me']]
    mass =                          ['amu', 'kg', 'me']
    # special treatment for mass unit 'amu'/'u'
    regxM[0] = re.compile(r'u\b')
    # construct boolean arrays of unit matching
    # need 'search' method for mass because of 'u'/'amu'
    bu1E = np.array([bool(regx.match(u1)) for regx in regxE])
    bu1D = np.array([bool(regx.match(u1)) for regx in regxD])
    bu1M = np.array([bool(regx.search(u1)) for regx in regxM])
    bu2E = np.array([bool(regx.match(u2)) for regx in regxE])
    bu2D = np.array([bool(regx.match(u2)) for regx in regxD])
    bu2M = np.array([bool(regx.search(u2)) for regx in regxM])
    # construct boolean lists of quantity type
    b1 = np.array([b.any() for b in [bu1E, bu1D, bu1M]])
    b2 = np.array([b.any() for b in [bu2E, bu2D, bu2M]])
    if (b1.any() and b2.any()) and \
        (np.argwhere(b1)[0][0] == np.argwhere(b2)[0][0]):
        # units are known and are compatible
        itype = np.argwhere(b1)[0][0]
    else:
        # units are problematic
        print_err('', 'Unable to convert ' +
            '{} to {}'.format(quantity['unit'], target_unit))
    # do the conversion
    if itype == 0:
        # energy units
        i1 = np.argwhere(bu1E)[0][0]
        i2 = np.argwhere(bu2E)[0][0]
        # change 'u1' and 'u2' to the full unit names
        u1 = energy[i1]
        u2 = energy[i2]
        if i1 == i2:
            # no conversion needed, but rename the unit
            qconv['unit'] = u2
            return qconv
        if u2 == 'hartree':
            # converting to hartree
            if u1 == 'ev':
                v2 = v1 / EV_PER_HARTREE
            elif u1 == 'kj/mol':
                v2 = v1 / AU_KJMOL
            elif u1 == 'cm-1':
                v2 = v1 / AU_WAVENUMBER
            elif u1 == 'kcal/mol':
                v2 = v1 * KCAL_KJ / AU_KJMOL
            else:
                # should never get here
                v2 = None
        elif u2 == 'ev':
            if u1 == 'hartree':
                v2 = v1 * EV_PER_HARTREE
            elif u1 == 'kj/mol':
                v2 = v1 / EV_KJMOL
            elif u1 == 'cm-1':
                v2 = v1 / EV_WAVENUMBER
            elif u1 == 'kcal/mol':
                v2 = v1 * KCAL_KJ / EV_KJMOL
            else:
                v2 = None
        elif u2 == 'kj/mol':
            if u1 == 'hartree':
                v2 = v1 * AU_KJMOL
            elif u1 == 'ev':
                v2 = v1 * EV_KJMOL
            elif u1 == 'cm-1':
                v2 = v1 / KJMOL_WAVENUMBER
            elif u1 == 'kcal/mol':
                v2 = v1 * KCAL_KJ
            else:
                v2 = None
        elif u2 == 'cm-1':
            if u1 == 'hartree':
                v2 = v1 * AU_WAVENUMBER
            elif u1 == 'ev':
                v2 = v1 * EV_WAVENUMBER
            elif u1 == 'kj/mol':
                v2 = v1 * KJMOL_WAVENUMBER
            elif u1 == 'kcal/mol':
                v2 = v1 * KCAL_KJ * KJMOL_WAVENUMBER
            else:
                v2 = None
        elif u2 == 'kcal/mol':
            if u1 == 'hartree':
                v2 = v1 * AU_KJMOL / KCAL_KJ
            elif u1 == 'ev':
                v2 = v1 * EV_KJMOL / KCAL_KJ
            elif u1 == 'kj/mol':
                v2 = v1 / KCAL_KJ
            elif u1 == 'cm-1':
                v2 = v1 / KJMOL_WAVENUMBER / KCAL_KJ
            else:
                v2 = None
        else:
            # should never get here
            v2 = None
    elif itype == 1:
        # distance units
        i1 = np.argwhere(bu1D)[0][0]
        i2 = np.argwhere(bu2D)[0][0]
        u1 = distance[i1]
        u2 = distance[i2]
        if i1 == i2:
            # no conversion needed, but rename the unit
            qconv['unit'] = u2
            return qconv
        if u2 == 'angstrom':
            if u1 == 'bohr':
                v2 = v1 * BOHR
            else:
                v2 = None
        elif u2 == 'bohr':
            if u1 == 'angstrom':
                v2 = v1 / BOHR
            else:
                v2 = None
        else:
            v2 = None
    elif itype == 2:
        # mass units
        i1 = np.argwhere(bu1M)[0][0]
        i2 = np.argwhere(bu2M)[0][0]
        u1 = mass[i1]
        u2 = mass[i2]
        if i1 == i2:
            # no conversion needed, but rename the unit
            qconv['unit'] = u2
            return qconv
        if u2 == 'amu':
            if u1 == 'kg':
                v2 = v1 / AMU_KG
            elif u1 == 'me':
                v2 = v1 / AMU_AU
            else:
                v2 = None
        elif u2 == 'kg':
            if u1 == 'amu':
                v2 = v1 * AMU_KG
            elif u1 == 'me':
                v2 = v1 / AMU_AU * AMU_KG
            else:
                v2 = None
        elif u2 == 'me':
            if u1 == 'amu':
                v2 = v1 * AMU_AU
            elif u1 == 'kg':
                v2 = v1 / AMU_KG * AMU_AU
            else:
                v2 = None
        else:
            v2 = None
    else:
        v2 = None
    if v2 is None:
        # failure
        print_err('', 'unable to convert ({} {}) to {}'.format(v1, 
            quantity['unit'], target_unit))
    qconv['unit'] = u2
    qconv[vkey] = v2
    return qconv
##
def RRHO_symmtop(freqs, Emax, binwidth, ABC_GHz, Bunit='GHz'):
    # RRHO with symmetric-top approximation.
    # Use Stein-Rabinovitch counting method (less roundoff error than 
    #   with Beyer-Swinehart)
    # ** Does not account for any symmetry **
    n = int(Emax/binwidth)  # number of bins
    nos = np.zeros(n)  # number of states in each bin
    nos[0] = 1  # the zero-point level
    for freq in freqs:
        Eladder = np.arange(freq, Emax+binwidth, freq)
        iladder = np.rint(Eladder / binwidth).astype(int)
        miyo = nos.copy()  # temporary copy of 'nos'
        # add each value in ladder to existing count in 'nos'
        for irung in iladder:
            for ibin in range(irung, n):
                miyo[ibin] += nos[ibin - irung]
        nos = miyo.copy()
    # Do similar thing for the rotational levels.
    E_rot, g_rot = rotational_levels_symmtop(ABC_GHz, Emax, Bunit=Bunit)
    ilist = np.rint(E_rot / binwidth).astype(int).reshape(-1)
    miyo = nos.copy()
    for idx in range(1, len(ilist)):
        # Loop over this index, instead of the 'iladder' values,
        #   to find the matching rotational degeneracies.
        # Start from 1 instead of 0 to skip the (non-degenerate) J=0
        irung = ilist[idx]
        degen = g_rot[idx]
        # vectorized version
        binrange = np.arange(irung, n).astype(int)
        miyo[binrange] = miyo[binrange] + nos[binrange - irung] * degen
    nos = miyo.copy()
    # find centers of energy bins
    centers = binwidth * (0.5 + np.arange(n))
    return nos, centers
    
##
def rotational_levels_symmtop(ABC, Emax, Bunit='cm-1'):
    # Rigid-rotor levels for a symmetric top
    # Return two arrays: energies (in cm^-1) and degeneracies
    # 'ABC' are the three rotational constants, either in GHz or cm^-1
    # 'Emax' is the upper bound on energy, in cm^-1
    ABC = np.array(ABC)
    ABC[::-1].sort()  # sort in descending order
    if Bunit.lower() == 'ghz':
        # convert ABC to cm^-1
        ABC *= 1.0e7 / CLIGHT
    if (ABC[0]-ABC[1] > ABC[1]-ABC[2]):
        # call it prolate
        B = np.sqrt(ABC[1]*ABC[2])  # geometric mean; "perpendicular"
        A = ABC[0]
        Jmax = int(-0.5 + 0.5 * np.sqrt(1 + 4*Emax/B))
    else:
        # call it oblate
        B = np.sqrt(ABC[1]*ABC[0])  # geometric mean; "perpendicular"
        A = ABC[2]
        Jmax = int( (-B + np.sqrt(B*B+4*A*Emax)) / (2*A) )
    J = np.arange(Jmax+1)  # all allowed values of J, including Jmax
    # K = 0 cases
    E = B * J * (J + 1)
    degen = 2*J + 1
    # K != 0 cases
    C = A-B
    for J in range(1,Jmax+1):
        # now J is a scalar
        K = np.arange(1, J+1)
        Kstack = B*J*(J+1) + C * K * K
        g = 2 * (2*J+1) * np.ones_like(K)
        E = np.concatenate((E, Kstack))
        degen = np.concatenate((degen, g))
    # sort by increasing energy
    idx = np.argsort(E)
    E = E[idx]
    degen = degen[idx]
    # filter out energies that exceed Emax
    idx = np.argwhere(E <= Emax)
    return E[idx], degen[idx]
##
def rotational_levels_spherical(B, Emax, Bunit='cm-1'):
    # Rigid-rotor levels for a spherical top
    # Return two arrays: energies (in cm^-1) and degeneracies
    # 'B' is the rotational constant, either in GHz or cm^-1
    # 'Emax' is the upper bound on energy, in cm^-1
    if Bunit.lower() == 'ghz':
        # convert B to cm^-1
        B *= 1.0e7 / CLIGHT
    Jmax = int(-0.5 + 0.5 * np.sqrt(1 + 4*Emax/B))
    J = np.arange(Jmax+1)  # all allowed values of J, including Jmax
    E = B * J * (J+1)
    degen = 2*J + 1
    degen *= degen  # this line is the only difference from the linear case
    return E, degen
##
def rotational_levels_linear(B, Emax, Bunit='cm-1'):
    # Rigid-rotor levels for a linear molecule
    # Return two arrays: energies (in cm^-1) and degeneracies
    # 'B' is the rotational constant, either in GHz or cm^-1
    # 'Emax' is the upper bound on energy, in cm^-1
    if Bunit.lower() == 'ghz':
        # convert B to cm^-1
        B *= 1.0e7 / CLIGHT
    Jmax = int(-0.5 + 0.5 * np.sqrt(1 + 4*Emax/B))
    J = np.arange(Jmax+1)  # all allowed values of J, including Jmax
    E = B * J * (J+1)
    degen = 2*J + 1
    return E, degen
##
def Beyer_Swinehart(freqs, Emax, binwidth):
    # Return a harmonic vibrational density of states (numpy array)
    #   whose index is the energy bin number.
    # Also return an array of the bin center energies.
    # Not vectorized
    n = int(Emax/binwidth)  # number of bins
    nos = np.zeros(n)  # number of states in each bin
    nos[0] = 1  # the zero-point level
    for freq in freqs:
        # outer loop in BS paper
        ifreq = np.rint(freq/binwidth).astype(int)
        for ibin in range(ifreq, n):
            # inner loop
            nos[ibin] += nos[ibin - ifreq]
    # find centers of energy bins
    centers = binwidth * (0.5 + np.arange(n))
    return nos, centers
##
def thermo_RRHO(T, freqs, symno, ABC_GHz, mass, pressure=1.0e5, deriv=0):
    # Return S, Cp, and [H(T)-H(0)] at the specified temperature
    lnQ = lnQvrt(T, freqs, symno, ABC_GHz, mass)
    d = lnQvrt(T, freqs, symno, ABC_GHz, mass, deriv=1)  # derivative of lnQ
    deriv = T * d + lnQ  # derivative of TlnQ
    S = RGAS * (deriv - np.log(AVOGADRO) + 1)
    d2 = lnQvrt(T, freqs, symno, ABC_GHz, mass, deriv=2)  # 2nd derivative of lnQ
    deriv2 = 2 * d + T * d2  # 2nd derivative of TlnQ
    Cp = RGAS + RGAS * T * deriv2
    ddH = RGAS * T * (1 + T * d) / 1000
    return (S, Cp, ddH)
##
def lnQvrt(T, freqs, symno, ABC_GHz, mass, pressure=1.0e5, deriv=0):
    # Return the total (vib + rot + transl) ln(Q) partition function
    #   or a derivative. RRHO approximation
    lnQv = lnQvib(T, freqs, deriv=deriv)
    lnQr = lnQrot(T, symno, ABC_GHz, deriv=deriv)
    lnQt = lnQtrans(T, mass, pressure=pressure, deriv=deriv)
    lnQ = lnQv + lnQr + lnQt
    return lnQ
##
def lnQtrans(T, mass, pressure=1.0e5, deriv=0):
    # Given a temperature (in K), a molecular mass (in amu),
    #   and optionally a pressure (in Pa), return ln(Q), where
    #   Q is the ideal-gas translational partition function.
    # If deriv > 0, return a (1st or 2nd) derivative of TlnQ
    #   instead of lnQ. 
    if deriv == 1:
        # return (d/dT)lnQ = (3/2T)
        return (1.5 / T)
    if deriv == 2:
        # return (d2/dT2)lnQ = -(3/2T**2)
        return (-1.5 / (T*T))
    kT = BOLTZMANN * T  # in J
    m = mass * AMU_KG   # in kg
    V = RGAS * T / pressure  # in m**3
    lnQ = 1.5 * np.log(2 * np.pi * m * kT)
    lnQ -= 3 * np.log(PLANCK)
    lnQ += np.log(V)
    return lnQ
##
def lnQrot(T, symno, ABC_GHz, deriv=0):
    # Given a temperature (in K), symmetry number, and list of
    #   rotational constants (in GHz), return ln(Q), where Q is
    #   the rigid-rotor partition function.
    n = len(ABC_GHz)
    if n == 0:
        # atom; no rotations possible
        return 0.
    if deriv == 1:
        # first derivative of lnQ depends only on temperature
        if n < 3:
            # linear case
            return (1/T)
        else:
            # non-linear
            return (1.5/T)
    if deriv == 2:
        # second derivative of lnQ 
        if n < 3:
            # linear case
            return (-1 / (T*T))
        else:
            # non-linear
            return (-1.5 / (T*T))
    ln_kTh = np.log(T) + np.log(BOLTZMANN) - np.log(PLANCK)  # ln(kT/h) expressed in ln(Hz)
    if n < 3:
        # linear molecule
        B = ABC_GHz[0] * 1.0e9  # convert to Hz
        lnQ = ln_kTh - np.log(symno * B)
    else:
        # polyatomic molecule with 3 constants
        lnQ = 1.5 * ln_kTh + 0.5 * np.log(np.pi) - np.log(symno)
        for c in ABC_GHz:
            B = c * 1.0e9 # convert to Hz
            lnQ -= 0.5 * np.log(B)
    return lnQ
##
def lnQvib(T, freqs, deriv=0):
    # Given a temperature (in K) and array of vibrational 
    #   frequencies (in cm^-1), return ln(Q) where Q is
    #   the harmonic-oscillator partition function.
    kTh = T * BOLTZMANN / PLANCK  # kT/h expressed in Hz
    lnQ = 0.
    nu = freqs * 100 # convert to m^-1 (as array)
    nu = nu * CLIGHT # convert to Hz
    fred = nu / kTh # reduced frequencies
    x = np.exp(-fred)  # exponentiated, reduced frequencies
    xm1 = 1 - x
    if deriv == 1:
        # derivative of lnQ
        term = nu * x / xm1
        d = term.sum()
        return (d / (kTh*T))
    if deriv == 2:
        # 2nd derivative of lnQ
        t1 = nu * (1/xm1 - 1)
        sum1 = -2 * t1.sum() / (kTh * T * T)
        t2 = nu * nu * x / (xm1 * xm1)
        sum2 = t2.sum() / (kTh * kTh * T * T)
        return (sum1 + sum2)
    # return lnQ itself
    lnq = np.log(xm1)
    lnQ = -1 * lnq.sum()
    return lnQ
##
def typeCoord(crds):
    #   'Geometry' (a Geometry object)
    #   'cartesian' (a list of elements and list/array of cartesians)
    #   'ZMatrix' (a ZMatrix object)
    if isinstance(crds, Geometry):
        intype = 'Geometry'
    elif isinstance(crds, ZMatrix):
        intype = 'ZMatrix'
    elif isinstance(crds, list) and (len(crds) == 2) and (
        (len(crds[0]) == len(crds[1])) or (len(crds[0]) * 3 == len(crds[1])) ):
        # 'cartesian' is plausible
        intype = 'cartesian'
    else:
        print_err('autodetect')
    return intype
##
def parse_ZMatrix(zlist, unitR='angstrom', unitA='degree'):
    # Given a list of all the lines of a z-matrix, 
    # return a ZMatrix object 
    el = []
    refat = []
    var = []
    val = {}
    # split lines on whitespace, comma, or equals
    regexSplit = re.compile('[\s,=]+')
    iline = 0
    intop = True
    for line in zlist:
        line = line.strip()
        words = list(filter(None, regexSplit.split(line)))
        nwords = len(words)
        # check for expected number of words
        if intop:
            # atom definitions
            if nwords < 1:
                # blank line marks end of atom definitions
                intop = False
                continue
            # atom definition: expect up to seven words
            xwords = min(2 * iline + 1, 7)
        else:
            # inside variable-definitions block
            if nwords < 1:
                # blank line marks end of input
                continue
            # variable definition: expect two words
            xwords = 2
        if nwords != xwords:
            print_err('zmatrix', 'expected {:d} words, got {:d} in {:s}'.format(xwords,
                nwords, str(words)))
        if intop:
            # list of atoms and variable names (or floats)
            # add element symbol
            el.append(words[0])
            # add variable (str|float)'s
            var.append([])
            for i in range(2, nwords, 2):
                try:
                    var[-1].append(float(words[i]))
                except:
                    # symbolic z-matrix variable (str type)
                    var[-1].append(words[i])
            # add list of atoms to which variables refer
            refat.append([])
            for i in range(1, nwords, 2):
                refat[-1].append(int(words[i]) - 1)  # subtract one from user-viewed index
        else:
            # values of any z-matrix variables
            val[words[0]] = float(words[1])
        iline += 1
    ZM = ZMatrix(el, refat, var, val, unitR=unitR, unitA=unitA)
    return ZM
##
class ZMatrix(object):
    # symbolic or numerical z-matrix
    # initialize empty and then add to it
    # indices are zero-based but user will be one-based
    def __init__(self, el=[], refat=[], var=[], val={}, vtype={}, unitR='angstrom', unitA='radian'):
        # this structure corresponds with the usual way of writing
        #   a z-matrix, with one atom defined per line
        self.el = el  # element symbols; should be in correct order
        self.refat = refat  # list of [list of ref. atoms that define position of this atom]
        self.var = var  # list of [list of z-matrix vars/constants that define this atom pos.]
        self.val = val  # dict of float values of any symbolic z-matrix variables
        self.vtype = vtype # dict of names of variable types ('distance', 'angle', 'dihedral')
        self.unitR = unitR # for distances
        self.unitA = unitA # for angles and dihedrals ('radian' or 'degree')
        self.coordtype = 'ZMatrix'
        self.charge = None      # optional
        self.spinmult = None    # optional
        if len(val) != len(vtype):
            # generate the vtype's automatically
            self.vtypeBuild()
    def vtypeBuild(self):
        # categorize the variables
        # this is important because they have different units
        category = ['distance', 'angle', 'dihedral']
        for iat in range(self.natom()):
            # loop over atoms
            for ivar in range(len(self.var[iat])):
                # loop over names of z-matrix variables for this atom
                # it's left-to-right, so vars are in the order in 'category'
                v = self.var[iat][ivar]  # name of a variable
                if ivar > 2:
                    self.vtype[v] = 'unknown'
                else:
                    self.vtype[v] = category[ivar]
        return
    def varMask(self, varlist):
        # given a list of z-matrix variable names, return a numpy array of Boolean
        #   showing which indices [from ZMatrix.fromVector()] correspond
        blist = []
        for var in sorted(self.val):
            blist.append(var in varlist)
        return np.array(blist)
    def canonical_angles(self):
        # shift all dihedral angles into the range (-pi, pi]
        for varname in self.val:
            if self.vtype[varname] == 'dihedral':
                self.val[varname] = angle_canon(self.val[varname], unit=self.unitA)
        return
    def cap_angles(self):
        # force all bond angles to be in the range (0, pi)
        for varname in self.val:
            if self.vtype[varname] == 'angle':
                if self.unitA == 'degree':
                    if self.val[varname] >= 180.:
                        self.val[varname] = 179.9
                    if self.val[varname] < 0.:
                        self.val[varname] = 0.1
                else:
                    # radian
                    if self.val[varname] >= np.pi:
                        self.val[varname] = np.pi - 0.0002
                    if self.val[varname] < 0.:
                        self.val[varname] = 0.0002
        return
    def adjust_dTau(self, dX):
        # given a vector of coordinate differences, move
        #   dihedral angle differences into the range (-pi, pi]
        i = 0
        for k in sorted(self.val):
            if self.vtype[k] == 'dihedral':
                dX[i] = angle_canon(dX[i], unit=self.unitA)
            i += 1
        return dX
    def toRadian(self):
        # make sure all angles/dihedrals are in radian
        if self.unitA == 'degree':
            for v in self.val:
                if self.vtype[v] in ['angle', 'dihedral']:
                    self.val[v] = np.deg2rad(self.val[v])
            self.unitA = 'radian'
        return
    def toDegree(self):
        # make sure all angles/dihedrals are in degree
        if self.unitA == 'radian':
            for v in self.val:
                if self.vtype[v] in ['angle', 'dihedral']:
                    self.val[v] = np.rad2deg(self.val[v])
            self.unitA = 'degree'
        return
    def toAngstrom(self):
        # make sure all distances are in angstrom
        if self.unitR == 'bohr':
            for v in self.val:
                if self.vtype[v] == 'distance':
                    self.val[v] *= BOHR
            self.unitR = 'angstrom'
        return
    def toBohr(self):
        # make sure all distances are in bohr
        if self.unitR == 'angstrom':
            for v in self.val:
                if self.vtype[v] == 'distance':
                    self.val[v] /= BOHR
            self.unitR = 'bohr'
        return
    def unitX(self):
        # return (tuple) of units
        return (self.unitR, self.unitA)
    def toUnits(self, unitS):
        # given (unitR, unitA), in either order, convert to those units
        if 'angstrom' in unitS:
            self.toAngstrom()
        if 'bohr' in unitS:
            self.toBohr()
        if 'degree' in unitS:
            self.toDegree()
        if 'radian' in unitS:
            self.toRadian()
        return
    def varlist(self):
        # return a list of the variable names in standard (sorted) order
        vlist = [k for k in sorted(self.val)]
        return vlist
    def toVector(self):
        # return a numpy array containing the values of the coordinates
        # they are sorted according to their names
        vec = [self.val[k] for k in sorted(self.val)]
        return np.array(vec)
    def dict2vector(self, dictin):
        # given a dict with keys that are the z-matrix variables,
        #   return a numpy array of the values (after sorting by name)
        #   there is no checking!
        vec = [dictin[k] for k in sorted(self.val)]
        return np.array(vec)
    def vector2dict(self, vecin):
        # given a vector, return a dict that has keys that
        #   are the z-matrix variables (sorted by name)
        #   No checking!
        i = 0
        dictout = {}
        for k in sorted(self.val):
            dictout[k] = vecin[i]
            i += 1
        return dictout
    def fromVector(self, vec, unitS, add=False):
        # replace current coordinates with those in 'vec' (list-like)
        # if 'add' is true, add to coordinates instead of replacing
        if unitS != self.unitX(): 
            # convert ZMatrix units, then convert back
            old_units = self.unitX()
            self.toUnits(unitS)
            unitS = False  # use as a flag
        i = 0
        for k in sorted(self.val):
            if add:
                self.val[k] += vec[i]
            else:
                self.val[k] = vec[i]
            i += 1
        if unitS == False:
            # convert units back
            self.toUnits(old_units)
        return
    def toGeometry(self):
        # generate Cartesian coordinates; return a Geometry object
        # assume that the z-matrix makes sense; no checking!
        newGeom = Geometry(units=self.unitR) # empty
        #newGeom.units = self.unitR   # angstrom or bohr
        for i in range(self.natom()):
            elem = self.el[i]
            if i == 0:
                # place first atom at the origin
                newGeom.addatom(Atom(elem, [0.,0.,0.]))
            elif i == 1:
                # place second atom on the z-axis
                zvar = self.var[i][0]
                z = self.val[zvar]
                newGeom.addatom(Atom(elem, [0.,0.,z]))
            elif i == 2:
                # place third atom in XZ plane
                zvar = self.var[i][0]  # distance
                r = self.val[zvar]
                rprev = [z, r]         # for later use
                zvar = self.var[i][1]  # angle
                theta = self.val[zvar]
                if self.unitA == 'degree':
                    theta = np.deg2rad(theta)
                z += -r * np.cos(theta) # displace from second atom
                x = r * np.sin(theta)
                newGeom.addatom(Atom(elem, [x,0.,z]))
            else:
                zvar = self.var[i][0]  # distance
                r = self.val[zvar]
                zvar = self.var[i][1]  # angle
                theta = self.val[zvar]
                zvar = self.var[i][2]  # dihedral
                phi = self.val[zvar]
                if self.unitA == 'degree':
                    theta = np.deg2rad(theta)
                    phi = np.deg2rad(phi)
                # find the three connected atoms (D-C-B-A) and get their coordinates
                C = self.refat[i][0]  # index of bonded atom
                B = self.refat[i][1]
                A = self.refat[i][2]
                C = newGeom.atom[C].xyz
                B = newGeom.atom[B].xyz
                A = newGeom.atom[A].xyz
                BC = C - B   # vector from B to C
                BA = A - B   # vector from B to A
                N = np.cross(BC, BA)    # normal to plane ABC
                # construct position for new atom 
                xp = normalize(np.cross(N, BC))  # unit vector toward A perp. to BC
                yp = normalize(N)
                dp = xp * np.cos(phi) + yp * np.sin(phi)  # within plane perp. to BC
                dp *= np.sin(theta)
                zp = normalize(BC)
                dp -= zp * np.cos(theta)
                D = normalize(dp, length=r) + C
                newGeom.addatom(Atom(elem, D))
        return newGeom
    def copy(self):
        return copy.deepcopy(self)
    def natom(self):
        # number of atoms
        return len(self.el)
    def nDOF(self):
        # number of degrees of freedom
        return len(self.val)
    def checkVals(self, verbose=True):
        # check that all variables are defined
        # print error message(s) if 'verbose' is True
        errcount = 0
        for v in [varname for varlist in self.var for varname in varlist]:
            # loop over all variable names
            if not v in self.val:
                # missing variable
                errcount += 1
                if verbose:
                    print('*** Missing value for variable {:s} in Z-matrix'.format(v))
        return errcount
    def printstr(self, unitR='angstrom', unitA='degree'):
        # print to a string, in specified units
        pstr = ''
        # first the list of atoms and variable names
        for i in range(self.natom()):
            pstr += self.el[i]   # element symbol
            for j in range(len(self.refat[i])):
                pstr += ' {:d}'.format(self.refat[i][j] + 1)  # +1 index offset for user viewing
                try:
                    pstr += ' {:f}'.format(self.var[i][j]).rstrip('0')  # omit trailing zeros
                except:
                    # not a float; should be str
                    pstr += ' {:s}'.format(self.var[i][j])
            pstr += '\n'
        # last the list of variable values in requested units
        pstr += '\n'  # blank line
        # find longest variable name, just to make the output pretty
        wlong = max([len(varname) for varname in self.val])
        for v in [varname for varlist in self.var for varname in varlist]:
            # loop over all variable names, in order by atom
            if v in self.val:
                value = self.val[v]
                if self.vtype[v] in ['angle', 'dihedral']:
                    if self.unitA != unitA:
                        # convert to requested unit for display
                        if unitA == 'degree':
                            value = np.rad2deg(value)
                        else:
                            value = np.deg2rad(value)
                else:
                    # distance variable
                    if self.unitR != unitR:
                        # convert unit
                        if unitR == 'angstrom':
                            value *= BOHR
                        else:
                            value /= BOHR
                pstr += '{:{width}s} {:f}'.format(v, value, width=wlong).rstrip('0') + '\n' # keep the decimal point
        return pstr
    def print(self):
        # print to stdout
        print(self.printstr())
        return
    def print_gradient(self, grad):
        # assuming alphabetical ordering of variable names, print gradient
        wlong = max([len(varname) for varname in self.val])
        ivar = 0
        for varname in sorted(self.val):
            print('{:{width}s}  {:f}'.format(varname, grad[ivar], width=wlong))
            ivar += 1
    def connection_table(self, tol=1.3):
        # return a connection table
        return self.toGeometry().connection_table(tol=tol)
    def extended_connection_table(self, tol=1.3):
        # return an extended connection table
        return self.toGeometry().extended_connection_table(tol=tol)
    def Coulomb_mat(self, select=0, bondtol=1.3):
        # return a (possibly restricted) Coulomb matrix
        return self.toGeometry().Coulomb_mat(select=select, bondtol=bondtol)
    def separateNonbonded(self, tol=1.3):
        # return a list of Geometry objects that are completely connected
        return self.toGeometry().separateNonbonded(tol=tol)
    def printXYZ(self, fname='', comment=''):
        # write an Xmol XYZ file
        self.toGeometry().printXYZ(fname, comment=comment)
        return
    def XmolXYZ(self, comment=''):
        # return a string in Xmol's XYZ format
        return self.toGeometry().XmolXYZ(comment)
##
def elz(ar, choice=''):
    # return atomic number given an elemental symbol, or
    # return elemental symbol given an atomic number 
    # If 'choice' is specified as 'symbol' or 'Z', return that.
    # if 'ar' is a list, then return a corresponding list
    symb = ['n',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
        'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba',
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra',
            'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
                 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
               'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt']
    if type(ar) == str and not re.match(r'^\d+$', ar):
        # this looks like an element symbol
        ar = ar.title()  # Title Case
        if choice == 'symbol':
            return ar
        else:
            return symb.index(ar)
    if type(ar) == list:
        # process a list of atoms
        vals = []
        for el in ar:
            vals.append(elz(el, choice))
        return vals
    # if we got here, the argument is an atomic number
    try:
        Z = int(ar)
    except:
        print('Error taking int of ar = in elz()', ar, type(ar))
        return None
    if choice == 'Z':
        return Z
    else:
        return symb[Z]
##
def n_core(atno, code=''):
    # given Z value (or element symbol) return number of core electrons
    # if 'atno' is a stoichiometric dict of {'el' : number}, then return the sum for
    #   the whole molecule
    # if the optional argument, 'code', is specified, the number will be the default
    #   for that quantum chemistry code
    ncore = 0
    if type(atno) == str:
        # convert symbol to Z value
        atno = elz(atno)
    if type(atno) == dict:
        # a molecular formula
        for el, natom in atno.items():
            ncore += n_core(el) * natom
        return ncore
    if code == 'gaussian09':
        # default for Gaussian09 frozen-core calculations
        core = {
            # these are the minimum atomic numbers (Z) that have
            #   the given number of core elecrons (Z : ncore)
            3  :  2,
            11 : 10,
            19 : 18,
            37 : 36,
            55 : 54, # this is a guess
            87 : 86  # this is a guess
        }
    else:
        core = {
            # these are the minimum atomic numbers (Z) that have
            #   the given number of core elecrons (Z : ncore)
            3  :  2,
            11 : 10,
            19 : 18,
            31 : 28,
            37 : 36,
            49 : 46,
            55 : 54,
            81 : 78,
            87 : 86
        }
    for ki in sorted(core):
        if atno >= ki:
            ncore = core[ki]
    return ncore
##
def read_regex(regex, fhandl, idx=1):
    # Return something from a line matchine a regular expression.
    #   First arg is the regular expression; idx is the match-group
    #	to return.  Return a list of values from all matching lines. 
    fhandl.seek(0)
    matches = []
    regx = re.compile(regex)
    for line in fhandl:
        mch = regx.search(line)
        if mch:
            matches.append(mch.group(idx))
    return matches
##
def spinname(m):
    # given a spin multiplity (m = 2S+1), return the text name (or the reverse)
    name = [ 'spinless', 'singlet', 'doublet', 'triplet', 'quartet', 'quintet', 'sextet',
        'septet', 'octet', 'nonet', 'decet', 'undecet', 'duodecet' ]
    try:
        m = int(m)
        if m in range(12):
            return name[m]
        else:
            return str(m) + '-tet'
    except:
        # convert a string into the corresponding multiplicity
        return name.index(m)
##
def max_not_exceed(bigser, target):
    # args are: (1) a pandas Series
    #           (2) a target value
    # return the largest value in 'bigser' that does not exceed 'target'
    # This is useful for matching up line numbers.
    smaller = bigser[bigser <= target]
    return smaller.max()
##
def hartree_eV(energy, direction='to_eV', multiplier=1):
    # convert from hartree to eV or the reverse (if direction == 'from_eV')
    if direction == 'to_eV':
        return multiplier * energy * EV_PER_HARTREE
    elif direction == 'from_eV':
        return multiplier * energy / EV_PER_HARTREE
    else:
        # illegal direction
        return 'unrecognized direction = {:s} in routine hartree_eV'.format(direction)
##
def starting_n(Ltype, nppe=0):
    # given an orbital-angular momentum type ('s', 'p', etc.), 
    # return the lowest possible principal quantum number (1, 2, etc.)
    # The optional second argument is the number of electrons that have
    #   been replaced by an ECP/pseudopotential
    # This routine only handles the common cases
    nmin = {'s': 1, 'p': 2, 'd': 3, 'f': 4, 'g': 5, 'h': 6}
    cases = [2, 10, 18, 28, 36, 46, 54, 60, 68, 78, 92]
    if nppe > 0:
        # Some electrons have been replaced by ECP; adjust the explicit
        #   shell numbers accordingly
        if (not nppe in cases):
            print('*** Unhandled number of ECP-replaced electrons ***')
            print('\tnppe = {:d} in routine "starting_n"'.format(nppe))
            # But go ahead and apply the algorithm, anyway!
        # determine number of shells replaced
        rcore = {'s': 0, 'p': 0, 'd': 0, 'f':0}
        resid = nppe
        nf = (resid - 28) // 32 # number of f shells replaced
        if nf > 0:
            rcore['f'] = nf
            resid -= nf * 14
        nd = (resid - 10) // 18 # number of d shells replaced
        if nd > 0:
            rcore['d'] = nd
            resid -= nd * 10
        np = (resid - 2) // 8  # number of p shells replaced
        if np > 0:
            rcore['p'] = np
            resid -= np * 6
        ns = resid // 2  # number of s shells replaced
        rcore['s'] = ns
        resid -= ns * 2
        if resid != 0:
            print('*** Unexpected residual electrons in routine "starting_n" ***')
        for L in rcore:
            nmin[L] += rcore[L]
    return nmin[Ltype.lower()]
##
def L_degeneracy(Ltype):
    # given an orbital-angular momentum type ('s', 'p', etc.), 
    # return the degeneracy (1, 3, etc.)
    degen = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13}
    return degen[Ltype.lower()]
##
def combine_MOspin(df, col1='Orbital', col2='Spin', colnew='MO'):
    # Given a pandas DataFrame, combine a numeric 'Orbital' field with
    #   a 'Spin' field ('alpha' or 'beta') to create a new 'MO' field
    #   that is a combination like '1a' or '5b'.
    # Return that new DataFrame.
    abbrev = {'alpha': 'a', 'beta': 'b', 'both': ''}
    dfret = df.copy()
    dfret[colnew] = df.apply(lambda x: str(x[col1])+abbrev[x[col2]], axis=1)
    return dfret
##
class Atom(object):
    # element symbol + cartesian coordinates + optional mass (default = 0)
    def __init__(self, el, xyz, mass=0):
        # 'el' : Element symbol or atomic number
        # 'xyz': cartesian coordinates as list or numpy array
        # 'mass':  atomic mass in amu
        self.el = elz(el, choice='symbol')
        self.xyz = np.array(xyz, dtype=np.float64)
        self.mass = mass
    def Z(self):
        # atomic number
        return elz(self.el, 'Z')
    def copy( self ):
        if type(self).__name__ == 'LabeledAtom':
            newatom = LabeledAtom(self.el, self.xyz, self.mass, self.label)
        else:
            # regular Atom object
            newatom = Atom(self.el, self.xyz, self.mass)
        return newatom
    def newxyz(self, triple):
        # replace current coordinates
        self.xyz = triple
        return
    def addxyz(self, triple):
        # add to current coordinates
        self.xyz = self.xyz + triple
        return
    def rotate(self, Rmat):
        # multipy the coordinates by the specified matrix
        self.xyz = Rmat.dot(self.xyz)
        return
    def printstr( self ):
        # print to a string (exclude mass)
        return '{:s}\t{:9.5f}\t{:9.5f}\t{:9.5f}'.format(self.el, self.xyz[0], self.xyz[1], self.xyz[2])
    def set_mass(self, m): 
        # set atom mass: either a number (in amu) or an option string
        try:
            m = float(m)
            self.mass = m
        except:
            if m == 'atomic_weight':
                self.mass = atomic_weight(self.el)
            else:
                print_err('', 'Unrecognized option, m = {:s}'.format(str(m)))
        return
    def print(self):
        # print to stdout (including mass)
        print(self.printstr())
        return
##
class LabeledAtom(Atom):
    # like an Atom, but carrying a label
    def __init__(self, el, xyz, mass=0, label='label'):
        Atom.__init__(self, el, xyz, mass)
        # label each atom simply with its ordinal number
        self.label = label
    def printstr(self):
        # print to a string (exclude mass)
        return '{:s}\t{:9.5f}\t{:9.5f}\t{:9.5f}\t{:s}'.format(self.el, self.xyz[0], self.xyz[1], self.xyz[2], str(self.label))
    def print(self):
        # print to stdout (including mass)
        print(self.printstr())
        return
    def fromAtom(atom, label='label'):
        # create from unlabeled Atom
        newLA = LabeledAtom(atom.el, atom.xyz, atom.mass, label)
        return newLA
    def setLabel(self, label=''):
        # change the label
        self.label = label
        return
##
def distance(pos1, pos2):
    # return distance between two vectors (numpy)
    # return NaN if the vectors have different dimensionality
    if len(pos1) != len(pos2):
        print('Unequal vector dimensions in "distance": dim1 = {:d}, dim2 = {:d}')
        return np.nan 
    v = pos2 - pos1
    d = np.linalg.norm(v)
    return d
##
def structure_distance(Struct1, Struct2, align=True):
    # Return "distance" between two structure objects
    #   return Nan if they are incompatible
    if Struct1.coordtype != Struct2.coordtype:
        # different types; distance does not make sense
        return np.nan
    if Struct1.natom() != Struct2.natom():
        # different atom counts; distance does not make sense
        return np.nan
    v1 = Struct1.toVector()
    if align:
        v2 = RMSD_align(Struct2, Struct1).toVector()
    else:
        v2 = Struct2.toVector()
    d = distance(v1, v2)  # cartesian distance
    return d
##
def angleabc(a, b, c, unit='radian'):
    # return the angle a-b-c, where all are numpy arrays
    v1 = a - b
    v2 = c - b
    s = np.dot( v1, v2 )
    s /= np.linalg.norm(v1)
    s /= np.linalg.norm(v2)
    theta = np.arccos(s)
    if unit == 'degree':
        # requested unit is degrees
        theta = np.rad2deg(theta)
    return theta
##
class Geometry(object):
    # a list of Atoms
    #   constructor does not accept masses
    def __init__(self, *args, intype='1list', units='angstrom'):
        # three input types are recognized:
        #   '2lists'    : a list of elements and a list of coordinate triples
        #   '1list'     : a list of [el, x, y, z] quadruples
        #   'atlist'    : a list of Atoms
        #   'DataFrame' : a pandas DataFrame with four columns (Z, x, y, z)
        self.coordtype = 'Geometry'
        self.atom = []
        self.units = units
        self.charge = None      # optional
        self.spinmult = None    # optional
        if len(args) == 0:
            # return an empty Geometry
            return
        if intype == 'atlist':
            # argument is already a list of Atoms
            self.atom = list(args[0])
            return
        if intype == '1list':
            # argument is a list of quadruples, [el, x, y, z]
            for quad in args[0]:
                at = Atom(quad[0], quad[1:4])
                self.atom.append(at)
            return
        if intype == '2lists':
            # first argument is a list of elements
            # second argument is a list of triples
            nsymb = len(args[0])
            nxyz = len(args[1])
            if nsymb != nxyz:
                print('*** Inconsistent #symb = {:d} and #xyz = {:d} in Geometry initialization'.format(nsymb, nxyz))
                return  # empty 
            for iat in range(nsymb):
                at = Atom(args[0][iat], args[1][iat])
                self.atom.append(at)
            return
        if intype == 'DataFrame':
            # argument is a four-column pandas DataFrame (Z, x, y, z)
            for iat in range(len(args[0].index)):
                elxyz = args[0].iloc[iat]
                at = Atom(elxyz[0], elxyz[1:].tolist())
                self.atom.append(at)
    def addatom(self, atom):
        self.atom.append(atom)
        return
    def randomize_atom_numbering(self):
        # re-number atoms randomly; may be useful for software testing
        idx = np.random.permutation(self.natom())
        self.atom = [self.atom[i] for i in idx]
        return
    def delatom(self, iatom):
        del self.atom[iatom]
        return
    def copy(self, elements=[], atoms=[]):
        # A restrictive list of elements or atom numbers may be provided
        newgeom = self.__class__()
        newgeom.units = self.units
        newgeom.coordtype = self.coordtype
        if len(elements) > 0:
            # copy only specified elements
            for a in self.atom:
                if (a.el in elements):
                    newgeom.addatom(a.copy())
        elif len(atoms) > 0:
            # copy only specified atoms (by index)
            for i in atoms:
                newgeom.addatom(self.atom[i].copy())
        else:
            # copy all atoms
            for a in self.atom:
                newgeom.addatom(a.copy())
            # copy any charge or spin multiplicity
            try:
                newgeom.charge = self.charge
            except:
                pass
            try:
                newgeom.spinmult = self.spinmult
            except:
                pass
        return newgeom
    def natom(self):
        return len(self.atom)
    def nDOF(self):
        # number of degrees of freedom
        return 3 * self.natom()
    def set_masses(self, mlist):
        # given a list of atom masses, assign these to the constituent Atoms
        # If 'mlist' is a string, get masses elsewhere
        try:
            if len(mlist) == self.natom():
                for i in range(self.natom()):
                    self.atom[i].set_mass(mlist[i])
            else:
                print('Expected {:d} atom masses but received only {:d} in Geometry.set_masses()'.format(self.natom(), len(mlist)))
        except:
            # 'mlist' is a string
            for i in range(self.natom()):
                self.atom[i].set_mass(mlist)
        return
    def translate(self, vector):
        # given a 3-vector, translate all atoms
        for i in range(self.natom()):
            self.atom[i].addxyz(vector)
        return
    def rotational(self):
        # return rotational constants, moments of inertia, and principal axes
        ### around the center of mass ###
        centered = self.copy()
        centered.center()
        imat = centered.inertia_tensor()
        moment, axes = np.linalg.eigh( imat )
        # convert moment to kg.m^2, assuming distances in angstrom and masses in u
        moment /= 1.0e20 * AVOGADRO * 1000.0
        convt = PLANCK / ( 8 * np.pi * np.pi * CLIGHT )
        # compute rotational constants, allowing for zero moments
        rotconst = np.zeros_like(moment) + np.inf
        for i in range(len(moment)):
            if moment[i] > 0:
                rotconst[i] = convt / moment[i]
        #rotconst = PLANCK / ( 8 * np.pi * np.pi * CLIGHT * moment )   # now in units (1/m)
        rotconst *= CLIGHT * 1.0e-9      # now in GHZ
        return rotconst, moment, axes        
    def center(self, origin=np.zeros(3), use_masses=True):
        # translate molecule to set center of mass at 'origin'
        # if use_masses is False, the use geometric centroid instead of COM
        C = self.COM(use_masses=use_masses)
        vec = origin - C
        self.translate(vec)
        return
    def rotate(self, Rmat):
        # given a 3x3 rotation matrix, multiply all atomic coords
        for A in self.atom:
            A.rotate(Rmat)
        return
    def mass(self):
        # sum of masses of constituent atoms
        m = 0
        for a in self.atom:
            m += a.mass
        return m
    def COM(self, use_masses=True):
        # center of mass
        com = np.zeros(3)
        if use_masses:
            # ordinary center of mass
            for a in self.atom:
                com += a.xyz * a.mass
            com /= self.mass()
        else:
            # geometric center (no masses)
            for a in self.atom:
                com += a.xyz
            com /= self.natom()
        return com
    def massVector(self, tripled=False):
        # return 1D vector of atomic masses
        # if 'tripled', repeat each mass three times (to match coordinates)
        n = 1
        if tripled:
            n = 3
        vmass = [[a.mass]*n for a in self.atom]
        vmass = np.array(vmass).flatten()
        return vmass
    def inertia_tensor(self):
        # return 3x3 inertia tensor
        mvec = self.massVector()
        elem, triples = self.separateXYZ()
        inertia = inertia_tensor(mvec, triples)
        return inertia
    def suppress_translation(self, direction):
        # given a displacement vector, remove net translation and return the adjusted vector
        # construct vector of masses
        vmass = self.massVector(tripled=True)
        if np.any(vmass <= 0.):
            print_err('', 'an atom has non-positive mass')
        transl = np.multiply(vmass, direction) / self.mass()
        transl = transl.reshape(-1, 3)
        center = transl.sum(axis=0)
        # subtract this 'center' from the input direction
        dnew = direction.reshape(-1,3) - center
        return dnew.flatten()
    def suppress_rotation(self, direction, thresh=0.001, maxiter=1000):
        # given a displacement vector, suppress net rotation and return the adjusted vector
        # crummy iterative method
        v = direction.reshape(-1,3)
        r = self.toVector().reshape(-1,3)  # atomic positions
        m = self.massVector()  # atomic masses
        I = ( (r*r).T * m ).T.sum()  # total moment of inertia
        iter = 0
        while True:
            L = angular_momentum(m, r, v)
            Lnorm = np.linalg.norm(L)
            #print('Lnorm = {:.4f} at iteration {:d}'.format(Lnorm, iter))
            if Lnorm < thresh:
                return v.flatten()
            w = L/I  # angular velocity
            u = np.cross(r, w)  # velocity adjustment
            v += u
            iter += 1
            if iter > maxiter:
                print('*** warning:  maxiter = {:d} exceeded in calm_rotation()'.format(maxiter))
    def toAngstrom(self):
        # ensure that units are angstrom
        if self.units == 'bohr':
            # multiply all coordinates by 'bohr' constant 
            for a in self.atom:
                a.xyz *= BOHR
            self.units = 'angstrom'
        return
    def toBohr(self):
        # ensure that units are bohr
        if self.units == 'angstrom':
            # divide all coordinates by 'bohr' constant 
            for a in self.atom:
                a.xyz /= BOHR
            self.units = 'bohr'
        return
    def toUnits(self, unitS):
        # given tuple of units, convert to those units
        if 'angstrom' in unitS:
            self.toAngstrom()
        if 'bohr' in unitS:
            self.toBohr()
        return
    def unitX(self):
        # return (tuple) of units
        return (self.units,)
    def print(self):
        # printing routine
        if type(self).__name__ == 'LabeledGeometry':
            header = 'el\t     x\t\t     y\t\t     z\t\tlabel'
        else:
            # regular Geometry object
            header = 'el\t     x\t\t     y\t\t     z'
        if self.units == 'bohr':
            header += '\t(units=bohr)'
        print(header)
        for atom in self.atom:
            atom.print()
        # print any charge and spin multiplicity
        try:
            print('charge = {:.1f}'.format(self.charge))
        except:
            # not a problem
            pass
        try:
            print('spinmult = {:.1f}'.format(self.spinmult))
        except:
            # not a problem
            pass
        return
    def XmolXYZ(self, comment=''):
        # return a string in Xmol's XYZ format
        if comment == '':
            # supply a default comment line
            comment = 'molecular composition is {:s}'.format(self.stoichiometry())
        if self.units == 'bohr':
            comment += '\t(units=bohr)'      
        xstr = '{:d}\n{:s}\n'.format(self.natom(), comment)
        for a in self.atom:
            xstr += '{:s}\t{:9.5f}\t{:9.5f}\t{:9.5f}\n'.format(a.el, a.xyz[0], a.xyz[1], a.xyz[2])
        return xstr
    def printXYZ(self, fname='', comment='', handle=False):
        # print a string in Xmol's XYZ format, to file or stdout
        if handle:
            # 'fname' is a file pointer
            fname.write(self.XmolXYZ(comment=comment))
        else:
            # 'fname' is the name of a file or blank
            if len(fname) > 0:
                # print to specified file; over-write existing data
                with open(fname, 'w') as f:
                    f.write(self.XmolXYZ(comment=comment))
            else:
                # print to stdout
                print(self.XmolXYZ(comment=comment))
        return
    def separateXYZ(self):
        # return a list with two elements: 
        #   [element symbols]; [array of cartesian triples]
        elem = []
        triples = []
        for a in self.atom:
            elem.append(a.el)
            triples.append(a.xyz)
        return [elem, np.array(triples)]
    def varlist(self):
        # return a list of (formal) variable names
        vlist = []
        for i in range(self.natom()):
            n = str(i)
            vlist += ['x_'+n, 'y_'+n, 'z_'+n]
        return vlist
    def toVector(self):
        # return a numpy array with all coordinates
        elem, triples = self.separateXYZ()
        return triples.flatten()
    def fromVector(self, vec, unitS, add=False):
        # given a flat vector of coordinates, replace the current coordinates
        # unitS[0] is the distance unit of the vector
        # if 'add' is True, then add to the current coordinates instead
        #   of replacing them
        if unitS[0] != self.units:
            # convert vector to Geometry units
            if self.units == 'angstrom':
                if unitS[0] == 'bohr':
                    vec *= BOHR
                else:
                    print('** unrecognized units: unitS[0] = {:s}'.format(unitS[0]))
            elif self.units == 'bohr':
                if unitS[0] == 'angstrom':
                    vec /= BOHR
                else:
                    print('** unrecognized units: unitS[0] = {:s}'.format(unitS[0]))
            else:
                print("** I don't recognize my own units! self.units = {:s}".format(self.units))                    
        triples = np.array(vec).reshape((-1,3))
        for i in range(self.natom()):
            if add:
                self.atom[i].addxyz(triples[i])
            else:
                self.atom[i].newxyz(triples[i])
        return
    def stoichiometry(self, as_dict=False):
        # stoichiometry string (without charge or spin multiplicity)
        order = ['C', 'H', 'N', 'O', 'F', 'Cl', 'S', 'P']
        # build hash of elements and their atom counts
        acount = {}
        for a in self.atom:
            try:
                acount[a.el] += 1
            except:
                acount[a.el] = 1
        if as_dict:
            return acount
        stoich = ''
        for e in order:
            if e in acount:
                stoich += '{:s}{:d}'.format(e, acount[e])
        # alphabetical for elements not specified in 'order'
        others = []
        for e in acount.keys():
            if not e in order:
                others.append(e)
        if len(others):
            for e in sorted(others):
                stoich += "{:s}{:d}".format(e, acount[e])
        return stoich
    def distance(self, i, j, unit=''):
        # distance between atoms i and j
        # use unit if requested; default is not to change units
        try:
            d = distance(self.atom[i].xyz, self.atom[j].xyz)
        except IndexError:
            s = '*** Illegal atom number in Geometry.distance(): ' + \
                'i = {:d}, j = {:d}'.format(i, j)
            print(s)
            return np.nan
        if unit == 'angstrom' and self.units == 'bohr':
            d *= BOHR  # convert bohr to angstrom
        if unit == 'bohr' and self.units == 'angstrom':
            d /= BOHR  # convert angstrom to bohr
        return d
    def vec(self, i, j, norm=None):
        # return the vector pointing from atom i to atom j
        #   is 'norm' is not None, then normalize the vector
        #   length to 'norm'
        v = self.atom[j].xyz - self.atom[i].xyz
        if norm is None:
            return v
        else:
            # normalize to specified length
            return normalize(v, norm)
    def angle(self, i, j, k, unit='degree'):
        # bond (or other) angle defined by atoms i, j, k
        try:
            a = angleabc(self.atom[i].xyz, self.atom[j].xyz, self.atom[k].xyz, unit=unit)
            return a
        except IndexError:
            s = '*** Illegal atom number in Geometry.angle(): ' + \
                'i = {:d}, j = {:d}, k = {:d}'.format(i, j, k)
            print(s)
            return np.nan
    def dihedral(self, i, j, k, l, typ='linear', unit='radian'):
        # calculate dihedral angle in radians (optionally in 'degree')
        # typ='linear'   :  connectivity is i-j-k-l
        #   dihedral is between planes ijk and jkl
        # typ='branched' :  connectivity is i-j<kl (i, k and l all bonded to j)
        #   dihedral is between planes ijk and jkl (conforming with Avogadro)
        a = self.vec(j, i)
        b = self.vec(j, k)
        c = self.vec(k, l)
        if typ == 'branched':
            c = self.vec(j, l)
        b = normalize(b) 
        x = a - b * np.dot(a, b) # component of a normal to b
        z = c - b * np.dot(c, b)
        x = normalize(x)
        z = normalize(z)
        if ( np.linalg.norm(x) == 0.0) or ( np.linalg.norm(z) == 0.0):
            # something is linear; dihedral is undefined
            return np.nan
        phi = np.arccos( np.dot(x,z) )  # in range [0, pi]
        s = np.cross(x, z)  # vector cross-product to get sign of dihedral
        s = np.sign( np.dot(s,b) )  # parallel or antiparallel to b
        phi *= s        # include sign (right-handed definition)
        if s == 0:
            # x and z are parallel
            if np.dot(x, z) > 0:
                phi = 0
            else: 
                phi = np.pi
        if unit == 'degree':
            phi *= 180 / np.pi
        return phi
    def simple_dihedrals(self, bondtol=1.3, unit='radian'):
        # Return a list of all (redundant) linear dihedral angles. 
        #   Each list element is a tuple:
        #       ( (i,j,k,l), angle_value )
        xconn = self.extended_connection_table(bondtol)
        pairs14 = np.argwhere(xconn == 3)  # pairs of atoms 3 bonds apart
        aldihe = []
        for il in pairs14:
            [i, l] = il.tolist()
            if l < i:
                # list each dihedral only once
                continue
            j = np.intersect1d( (np.argwhere(xconn[i,:] == 1)), (np.argwhere(xconn[l,:] == 2)) ).min()
            k = np.intersect1d( (np.argwhere(xconn[i,:] == 2)), (np.argwhere(xconn[l,:] == 1)) ).tolist()
            blist = np.where(xconn[j,:] == 1)[0]
            k = np.intersect1d(k, blist).min()
            ang = self.dihedral(i, j, k, l, 'linear', unit)
            aldihe.append( ((i,j,k,l), ang) )
        return aldihe
    def find_methyls(self, bondtol=1.3):
        # return list of tuples of atom numbers (C, H, H, H)
        mlist = []
        conn = self.connection_table(bondtol)
        for i in range(self.natom()):
            if self.atom[i].Z() == 6:
                # a carbon atom
                h = np.argwhere(conn[i,:] == 1).flatten()
                if len(h) == 4:
                    # tetravalent carbon
                    hlist = []
                    for j in h:
                        if self.atom[j].Z() == 1:
                            # hydrogen atom
                            hlist.append(j)
                    if len(hlist) == 3:
                        # a methyl group; add to list
                        mlist.append( (i, *hlist) )
        return mlist
    def bonded(self, i, j, tol=1.3):
        # return True if bonded, else False (based on distance only) (3/2/10)
        # 'tol' tolerated amount of bond stretching
        r0 = r0_ref(self.atom[i].el, self.atom[j].el)
        if self.distance(i, j, unit='angstrom') < r0 * tol:
            return True
        return False
    def bonded_list(self, tol=1.3):
        # return a list of lists of bonded atoms (by index)
        natom = self.natom()
        connex = self.connection_table(tol=tol)
        bonded = [ np.argwhere(connex[i,:]).flatten() for i in range(natom) ]
        return bonded
    def distmat(self, unit=''):
        # 2D array of interatomic distances
        # use unit if specified
        xyz = [a.xyz for a in self.atom]
        dmat = cdist(xyz, xyz, metric='euclidean')
        if unit == 'angstrom' and self.units == 'bohr':
            dmat *= BOHR  # convert bohr to angstrom
        if unit == 'bohr' and self.units == 'angstrom':
            dmat /= BOHR  # convert angstrom to bohr
        return dmat
    def connection_table(self, tol=1.3):
        # return a connection table:  a 2D array indicating bonded distances (= 0 or 1)
        # 'tol' is bond-stretch tolerance
        dmat = self.distmat(unit='angstrom') / tol
        connex = np.zeros_like(dmat, dtype=int)
        for i in range(self.natom()):
            for j in range(i):
                # j < i
                if dmat[i][j] < r0_ref(self.atom[i].el, self.atom[j].el):
                    connex[i][j] = 1
                    connex[j][i] = 1
        return connex
    def extended_connection_table(self, tol=1.3):
        # return a 2D array where A_ij is the number of bonded
        #   links to get from atom i to atom j
        # Zeros on the diagonal and for unconnected atom pairs
        xconn = self.connection_table(tol)
        natom = xconn.shape[0] 
        changed = True
        nbond = 1
        while changed:
            changed = False
            for i in range(natom):
                for j in range(natom):
                    if xconn[i][j] == nbond:
                        # j is 'nbonds' from i
                        # find atoms k that are bonded to j
                        for k in range(natom):
                            if (k != i) and (k != j) and (xconn[j][k] == 1) and (xconn[i][k] == 0):
                                # record this distance
                                xconn[i][k] = xconn[k][i] = nbond + 1
                                changed = True
            nbond += 1
        return xconn
    def Coulomb_mat(self, select=0, bondtol=1.3):
        # return a Coulomb matrix (atomic units)
        # if 'select' != 0, then the matrix is zero
        #   except for atom pairs separated by 'select' bonds
        # when 'select' == 0, 'bondtol' is irrelevant
        zvals = [a.Z() for a in self.atom]
        zmat = np.outer(zvals, zvals)
        xconn = self.extended_connection_table()
        nat = xconn.shape[0]
        if select >= nat:
            print('Warning: select = {:d} exceeds atom limit in Coulomb_mat(); setting to zero'.format(select))
            select = 0
        dmat = self.distmat('bohr')
        if select > 0:
            # destroy values at wrong bonded distances
            dmat[np.where(xconn != select)] = np.inf
        else:
            # set only diagonal to inf (so that reciprocal will be zero)
            np.fill_diagonal(dmat, np.inf)
        return zmat/dmat
    def subMolecules(self, lolist, ltype='index'):
        # return a list of sub-molecules
        # arg 'lolist' is a list of lists
        # 'ltype' indicates whether 'index' number or 'label' are specified in lolist
        #     'label' only makes sense for LabeledGeometry
        geomlist = []
        for lol in lolist:
            # create an empty object for each list in lolist
            newG = self.__class__()
            newG.units = self.units
            if ltype == 'index':
                # sort indices to preserve atom ordering
                for i in sorted(lol):  
                    # 'i' is just the index in self.atom[]
                    newG.addatom(self.atom[i])
            elif (ltype == 'label') and (type(self).__name__ == 'LabeledGeometry'):
                for i in lol:
                    # 'i' is the label; add all matching atoms
                    m = False # flag
                    for at in self.atom:
                        if at.label == i:
                            newG.addatom(at)
                            m = True
                    if not m:
                        # no matching atom found
                        print('Found no atoms with label {:s} in LabeledGeometry.subMolecules()'.format(str(i)))
            else:
                print('Unrecognized ltype =', ltype, 'in LabeledGeometry.subMolecules()')
                return None
            geomlist.append(newG)
        return geomlist    
    def separateNonbonded(self, tol=1.3):
        # return a list of Geometry objects for all disconnected fragments
        fragments = self.find_fragments(tol=tol)
        # create the sub-molecules
        submols = self.subMolecules(fragments, ltype='index')
        return submols
    def fragment_distances(self, loc='nearest', tol=1.3):
        # Identify non-bonded fragments, then
        #   return the matrix of inter-fragment distances and 
        #   another item (depending upon 'loc' value)
        #   loc == 'nearest' : minimal interatomic distance
        #   loc == 'center'  : between geometric centers (no masses)
        fragments = self.find_fragments(tol=tol)
        nfrag = len(fragments)
        sep = np.zeros((nfrag, nfrag))  # matrix of inter-fragment distances
        if nfrag == 1:
            # there is nothing to do (still return two values)
            return sep, sep.tolist()
        if loc == 'nearest':
            # find the nearest atoms between all pairs of fragments
            ijDist = self.distmat()
            ijNearest = np.zeros((nfrag, nfrag)).tolist()  # for storing the (i,j) atom numbers
            for ifrag in range(nfrag):
                mindist = np.inf
                minj = mini = -1
                for jfrag in range(ifrag):
                    for iat in fragments[ifrag]:
                        for jat in fragments[jfrag]:
                            if ijDist[iat][jat] < mindist:
                                # new closest pair
                                minj = jat
                                mini = iat
                                mindist = ijDist[iat][jat]
                    # record the closest atom pair for these two fragments
                    ijNearest[ifrag][jfrag] = (mini, minj)
                    ijNearest[jfrag][ifrag] = (minj, mini)
                    sep[ifrag][jfrag] = mindist
                    sep[jfrag][ifrag] = mindist
            return sep, ijNearest
        elif loc == 'center':
            # find the distance between geometric centers
            #   (without mass-weighting)
            cent = np.zeros((nfrag, 3))  # coordinates of fragment centers
            # compute fragment centers
            for ifrag in range(nfrag):
                for iat in fragments[ifrag]:
                    cent[ifrag] += self.atom[iat].xyz
                cent[ifrag] /= len(fragments[ifrag])
            # compute distances between centers
            for ifrag in range(nfrag):
                for jfrag in range(ifrag):
                    sep[ifrag][jfrag] = np.linalg.norm(cent[jfrag] - cent[ifrag])
                    sep[jfrag][ifrag] = sep[ifrag][jfrag]
            return sep, cent
        else:
            print_err('option', 'loc = {:s}'.format(loc))
    def spread_fragments(self, dist=5.0, tol=1.3):
        # displace fragments away from each other along
        #   closest inter-atom vectors, to distance 'dist'
        # Return value is the number of fragments detected
        sep, ijNearest = self.fragment_distances(loc='nearest', tol=tol)
        nfrag = sep.shape[0]
        if nfrag < 2:
            # nothing to do
            return nfrag
        # compute the translation vectors
        # each row in 'transl' is the translation to apply to all
        #   atoms in one fragment
        transl = np.zeros( (nfrag, 3) )
        for ifrag in range(nfrag):
            for jfrag in range(ifrag):
                (iat, jat) = ijNearest[ifrag][jfrag]
                v12 = (self.atom[iat].xyz - self.atom[jat].xyz)
                # adjust length of translation vector
                curlen = np.linalg.norm(v12)
                v12 = normalize(v12, (dist-curlen)/2)
                transl[ifrag] += v12  # move fragment i away from fragment j
                transl[jfrag] -= v12  # move fragment j away from fragment i
        # apply the translations
        fragments = self.find_fragments(tol=tol)
        for ifrag in range(nfrag):
            for iat in fragments[ifrag]: 
                self.atom[iat].addxyz(transl[ifrag])
        return nfrag
    def find_fragments(self, tol=1.3):
        # return a list of [list of atom numbers] that are connected
        natom = self.natom()
        bonded = self.bonded_list(tol=tol)
        # bonded[i] is the list of atoms that are connected to atom i (indices, not labels)
        bunch = []  # list of lists; atom "bunches" that are intact molecules
        remaining = list(range(natom))  # the indices of the atoms not yet assigned to a bunch
        moved = False  # a flag
        while(len(remaining)):
            if not moved:
                # no atoms were moved last round; start a new bunch
                seed = remaining.pop(0)
                bunch.append([seed])
                moved = True
            for i in bunch[-1]:
                moved = False
                for j in bonded[i]:
                    if not j in bunch[-1]:
                        # move this atom into the current bunch
                        bunch[-1].append(j)
                        remaining.remove(j)
                        moved = True
        return bunch
    def assignTerminality(self, tol=1.3):
        # assign a 'terminality' number to each atom;
        #  it's the number of iterations that the atom survives,
        #  where one iteration removes all terminal atoms
        # Return a list of terminality numbers
        # Atoms in rings get terminality = -1
        natom = self.natom()
        terminality = np.zeros(natom, dtype=int)
        remaining = np.arange(natom)  # the indices of the atoms not yet removed
        round = 0  # counter
        while(len(remaining)):
            # find the terminal atoms
            buff = self.copy(atoms=remaining)
            # count bonds
            connex = buff.connection_table(tol=tol)
            numbond = connex.sum(axis=0)
            nonterminal = np.argwhere(numbond >= 2).flatten() # non-bonded is considered terminal
            remaining = remaining[nonterminal]
            terminality[remaining] += 1
            round += 1
            if len(remaining) == natom:
                # no atoms were eliminated; only rings must remain
                terminality[remaining] = -1
                break
            else:
                natom = len(remaining)
        return terminality
    def findRings(self, tol=1.3):
        # return a list of lists
        #   each sub-list is the indices of atoms in one ring
        termy = self.assignTerminality(tol=tol)
        ringAtom = [i for i in range(len(termy)) if termy[i] == -1]
        bonded = self.bonded_list(tol=tol)
        natom = len(ringAtom)  # number of ring atoms
        rings = []  # list of rings
        assigned = 0  # counter; we're done when it reaches 'natom'
        # need an algorithm here!
        
        return ringAtom
##
class LabeledGeometry(Geometry):
    # like a Geometry, but composed of LabeledAtom instead of Atom
    def __init__(self, *args, intype='atlist', labels='', units='angstrom'):
        Geometry.__init__(self, *args, intype=intype, units=units)
        natom = self.natom()
        for i in range(natom):
            # replace each Atom with a LabeledAtom
            if len(labels) >= natom:
                # user-supplied list of atom labels
                self.atom[i] = LabeledAtom.fromAtom(self.atom[i], labels[i])
            else:
                # use the atom number as the label
                self.atom[i] = LabeledAtom.fromAtom(self.atom[i], i)
    def setLabels(self, labels):
        # change the labels on the LabeledAtoms
        natom = self.natom()
        if len(labels) != natom:
            # this is not allowed; make no changes
            print('Expected {:d} but received {:d} labels in LabeledGeometry.setLabels()'.format(natom, len(labels)))
            return
        else:
            # change the labels
            for i in range(natom):
                self.atom[i].setLabel(label[i])
            return
    def fromGeometry(geom, labels=''):
        # create from unlabled Geometry
        Lmolec = LabeledGeometry(geom.atom, intype='atlist', labels=labels, units=geom.units)
        return Lmolec
##
def atomic_weight(iz):
    # return atomic weight given Z (3/21/2012) or elemental symbol (9/16/2014)
    # values are from the NIST 2003 periodic table
    # units are u (amu)
    wt = [ 0, 1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994, 18.9984032, 20.1797,
        22.989770, 24.3050, 26.981538, 28.0855, 30.973761, 32.076, 35.453, 39.948,
        39.0983, 40.078, 44.955910, 47.867, 50.9415, 51.9961, 54.938049, 55.845, 58.933200, 58.6934,
        63.546, 65.409, 69.723, 72.64, 74.92160, 78.96, 79.904, 83.798,
        85.4678, 87.62, 88.90585, 91.224, 92.90638, 95.94, 98, 101.07, 102.90550, 106.42,
        107.8682, 112.411, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.293,
        132.90545, 137.327,
        138.9055, 140.116, 140.90765, 144.24, 145, 150.36, 151.964, 157.25, 158.92534,
        162.500, 164.93032, 167.259, 168.93421, 173.04, 174.967,
        178.49, 180.9479, 183.84, 186.207, 190.23, 192.217, 195.078,
        196.96655, 200.59, 204.3833, 207.2, 208.98038, 209, 210, 222,
        223, 226,
        227, 232.0381, 231.03588, 238.02891, 237, 244, 243, 247, 247,
        251, 252, 257, 258, 259, 262,
        261, 262, 266, 264, 277, 268 ]
    if type( iz ) == int:
        return wt[iz]
    else:
        # probably an elemental symbol
        z = elz(iz)
        return wt[z]
##
def xyz2Atom(atno, xyz):
	# given Z value (or element symbol) and list [x, y, z], return an Atom
    if type(atno) == int:
        el = elz(atno)
    else:
        # we were probably given an element symbol, not an atomic number
        el = atno
        atno = elz(el)
    m = atomic_weight(atno)
    return Atom(el, xyz[0], xyz[1], xyz[2], m)
##
def xyz2Geometry(atnos, xyzs, units='angstrom'):
    # args: list of atomic numbers; list of coordinates [x1, y1, z1, x2, y2, z2,...]
    # return a Geometry
    # 9/16/2014
    #
    # check for compatible list lengths
    natom = len(atnos)
    nxyz = len(xyzs)
    if nxyz != 3 * natom:
        print('Incompatible numbers of atoms and of coordinates:')
        print('natom = {:d}, nxyz = {:d} in xyz2Geometry()'.format(natom, nxyz))
        return Null
    # build Geometry one Atom at a time
    molecule = Geometry(units=units)
    for i in range(natom):
        atno = atnos[i]
        xyz = xyzs[3*i:3*i+3]
        atom = xyz2Atom(atno, xyz)
        molecule.addatom(atom)
    return molecule
##
def JSdm(P, Q, base=4):
    # Jensen-Shannon divergence metric; base=4 gives range = [0, 1]
    # P and Q are *discrete* PDFs (with same data type)
    # Allowed data types: tuple; list; dict; 1D numpy array
    # P and Q must be same length, except when dict
    # Return:
    #   (1) metric (float)
    #   (2) messages (list of string)
    #
    message = []
    if type(P) != type(Q):
        print('*** P and Q must be same data type in routine JSdm() ***')
        return (None, None)
    if (type(P) == list) or (type(P) == tuple) or (type(P) == np.ndarray):
        P = np.array(P).astype(float)
        Q = np.array(Q).astype(float)
        allkeys = []   # length will be tested later, to infer input type
    elif type(P) == dict:
        # make a sorted list of all the keys
        allkeys = sorted(set(list(P.keys()) + list(Q.keys())))
        Plist = []
        Qlist = []
        for key in allkeys:
            try:
                Plist.append(P[key])
            except:
                # probably key is not present in this dict
                Plist.append(0)
            try:
                Qlist.append(Q[key])
            except:
                Qlist.append(0)
        if P.keys() != Q.keys():
            message.append('Different key lists merged for P and Q')
        # convert list to numpy array
        P = np.array(Plist).astype(float)
        Q = np.array(Qlist).astype(float)
    else:
        print('*** Unhandled data type in routine JSdm():', type(P))
        return (None, None)
    # No negative values are allowed
    if len(np.where(P < 0)[0]) or len(np.where(Q < 0)[0]):
        print('*** Negative values not allowed in routine JSdm() ***')
        return (None, None)
    # P and Q must have the same number of elements
    if len(P) != len(Q):
        print('*** P and Q must have same length in routine JSdm() ***')
        return (None, None)
    # Normalize both PDFs (L1-normalization)
    Plen = P.sum()
    Qlen = Q.sum()
    if (Plen == 0) or (Qlen == 0):
        print('*** P and Q may not be all zeros in routine JSdm() ***')
        return (None, None)
    P /= Plen
    Q /= Qlen
    pqsum = P + Q
    # find any zeros in (P+Q) and delete corresponding elements in P, Q, and P+Q
    nullidx = np.where(pqsum == 0)[0]
    if len(nullidx > 0):
        # delete the troublesome elements
        if len(allkeys) > 0:
            # input was dict
            message.append('Deleted null elements with indices ' + str([allkeys[i] for i in nullidx]))
        else:
            # input was list-like
            message.append('Deleted null elements with indices ' + str(nullidx))
        P = np.delete(P, nullidx)
        Q = np.delete(Q, nullidx)
        pqsum = np.delete(pqsum, nullidx)
    # compute the JSDM
    # P or Q may still contain zeros, so don't take straight logarithm
    #   instead, use x*ln(y) = ln(y**x) and convention 0**0 = 1
    s1 = 2 * P / pqsum
    s2 = 2 * Q / pqsum
    s1 = s1 ** P
    s2 = s2 ** Q
    s1 = np.log(s1) / np.log(base)
    s2 = np.log(s2) / np.log(base)
    dsq = (s1 + s2).sum()
    return np.sqrt(dsq), message
##
def AOpopdiffmats(df1, df2):
    # Compare two pandas DataFrames with Mulliken population data,
    #   as returned by routine 'read_AOpop_in_MOs()' in 'g09_subs.py'
    # Return two numpy 2D-arrays:
    #   (1) JSdm() differences in AO populations (Jensen-Shannon divergence metric)
    #   (2) (E2-E1) orbital energy differences
    # Also return two lists of MO numbers:
    #   (3) MO labels in df1 (rows of matrices)
    #   (4) MO labels in df2 (columns of matrics)
    MOlist1 = sorted(set(df1.MO))
    MOlist2 = sorted(set(df2.MO))
    nmo1 = len(MOlist1)
    nmo2 = len(MOlist2)
    dPmat = np.zeros((nmo1, nmo2))
    dEmat = np.zeros((nmo1, nmo2))
    for imo in MOlist1:
        # looping over MOs in first set
        idx = MOlist1.index(imo)  # row number in returned matrices
        orb1 = df1[df1.MO == imo]
        E1 = orb1.iloc[0]['Energy']
        # convert AO populations into a dict
        mulpop1 = {}
        # create a label for each AO that looks like '#5-p' for a p-orbital on atom #5
        for ao in orb1.index:
            s = '#{:d}-{:s}'.format(orb1.loc[ao]['Atom#'], orb1.loc[ao]['L'])
            c = orb1.loc[ao]['Contrib']
            if c < 0:
                # treat negative AO pop as a new variable (by changing its label)
                s += '-neg'
                c = abs(c)
            mulpop1[s] = c
        # loop over orbitals in second set
        for jmo in MOlist2:
            jdx = MOlist2.index(jmo)  # column number in returned matrices
            orb2 = df2[df2.MO == jmo]
            E2 = orb2.iloc[0]['Energy']
            dEmat[idx, jdx] = E2 - E1  # signed difference
            # construct dict of AO populations as above
            mulpop2 = {}
            for ao in orb2.index:
                s = '#{:d}-{:s}'.format(orb2.loc[ao]['Atom#'], orb2.loc[ao]['L'])
                c = orb2.loc[ao]['Contrib']
                if c < 0:
                    # negative AO pop
                    s += '-neg'
                    c = abs(c)
                mulpop2[s] = c
            # get JSdm distance between the two AO population vectors
            dist = JSdm(mulpop1, mulpop2)
            dPmat[idx, jdx] = dist[0]
    return dPmat, dEmat, MOlist1, MOlist2
##
def orbitalPopMatch(df1, df2, Eweight=0.1, diagBias=0.001):
    # Find which MOs correspond between two calculations. 
    # Note: Cannot distinguish degenerate orbitals! 
    # Compare two pandas DataFrames with Mulliken population data,
    #   as returned by routine 'read_AOpop_in_MOs()' in 'g09_subs.py'
    # Argument 'Eweight' is the weight to give to energy differences.
    # Argument 'diagBias' is the preference to give to the existing
    #   orbital numbering.
    # Return a dict of MO number correspondences. The dict only includes
    #   orbitals that appear to be mismatched. 
    #   Keys are MO labels in df2, values are MO labels in df1.
    # Do not mix alpha with beta orbitals.
    #
    momap = {}
    if (df1['Spin']  == 'alpha').any() & (df1['Spin'] == 'beta').any():
        # this is a UHF case; keep alpha and beta orbitals separate
        for sp in ['alpha', 'beta']:
            set1 = df1[df1['Spin'] == sp]
            set2 = df2[df2['Spin'] == sp]
            momap.update(orbitalPopMatch(set1, set2, Eweight=Eweight, diagBias=diagBias))
        return momap 
    # simple, single-spin case
    dPmat, dEmat, MOs1, MOs2 = AOpopdiffmats(df1, df2)
    # count the MOs in each orbital set
    norb1 = len(MOs1)
    norb2 = len(MOs2)
    nmo = min(norb1, norb2)
    # use unsigned energy differences
    diffmat = dPmat + Eweight * np.fabs(dEmat)
    # install the bias toward perserving the existing numbering
    # Note: Gaussian prints the populations only to 0.01 precision
    for i in range(norb1):
        imo = MOs1[i]
        try:
            j = MOs2.index(imo)
            diffmat[i, j] -= diagBias
        except:
            # probably an orbital missing from set 2
            pass
    # find closest distance for each row
    rowmin = diffmat.min(axis=1)
    # sort by increasing distance (i.e., best matches first)
    rowlist = rowmin.argsort()
    # truncate to smallest dimension
    rowlist = rowlist[0 : nmo]
    claimed = []  # list of orbitals in set2 as they are paired
    pairing = {}  # mapping between orbital indices (not MO numbers/labels)
    for iorb in rowlist:
        # loop over matrix rows, starting with those with best available matches
        for jorb in diffmat[iorb, :].argsort():
            # loop over columns, starting with best match
            if jorb in claimed:
                # this orbital already paired
                continue
            # this is a pairing
            claimed.append(jorb)
            pairing[iorb] = jorb
            break  # done with this first-set MO
    # convert into a mapping of MO numbers
    for i in pairing.keys():
        imo = MOs1[i]  # MO number from first set
        j = pairing[i]
        jmo = MOs2[j]  # MO number from second set
        if imo != jmo:
            # report only non-identity mappings
            momap[jmo] = imo  # key is the MO number in the 2nd set
    return momap 
##
def relabelOrbitals(df, momap):
    # re-label MOs based upon a mapping provided by 'orbitalPopMatch()'
    # Return value: the DataFrame with orbitals re-labeled
    #
    # loop once through the rows, changing MO labels
    for idx in df.index:
        imo = df.loc[idx, 'MO'] 
        if imo in momap.keys():
            # change this MO label
            df.loc[idx, 'MO'] = momap[imo]
    return df
##
def readXmol(fh, units='angstrom', handle=False):
    # Read an XYZ file (handle) and return (Geometry object, #atoms, comment)
    #   if 'handle' is True, expect a file handle instead of a file name
    if not handle:
        fh = open(fh, 'r')
    try:
        natom = int( fh.readline() )
        comment = fh.readline().rstrip()
        df = pd.read_csv(fh, names=['El', 'X', 'Y', 'Z'], delim_whitespace=True)
        # check the number of atoms
        if natom != df.shape[0]:
            print('Expected {:d} atoms but found {:d}!'.format(natom, df.shape[0]))
            return None
    except:
        print('Unable to read XMol file')
        return None
    if not handle:
        fh.close()
    return Geometry(df, intype='DataFrame', units=units), natom, comment
##
def r0_ref( elem1, elem2 ):
    # return single-bonded distances between elements (Angstrom)
    # from b3lyp/6-31g* calculations on molecules specified (3/2/10)
    #   added covalent radii 3/21/2012
    if ( elem1 > elem2 ):
        # put the elements in ascending lexical order
        t = elem1
        elem1 = elem2
        elem2 = t
    if elem1 == 'C':
        if elem2 == 'C':
            # C-C bond from C2H6
            return 1.5306
        if elem2 == 'H':
            # C-H bond from CH4
            return 1.0936
        if elem2 == 'N':
            # C-N bond from CH3NH2
            return 1.4658
        if elem2 == 'O':
            # C-O bond from CH3OH
            return 1.4192
    if elem1 == 'H':
        if elem2 == 'H':
            # H-H bond from H2
            return 0.743
        if elem2 == 'N':
            # N-H bond from CH3NH2
            return 1.0189
        if elem2 == 'O':
            # O-H bond from CH3OH
            return 0.9691
    if elem1 == 'N':
        if elem2 == 'N':
            # N-N bond from N2H4
            return 1.4374
        if elem2 == 'O':
            # N-O bond from NH2OH
            return 1.4481
    if elem1 == 'O':
        if elem2 == 'O':
            # O-O bond from HOOH
            return 1.456
    # unknown case; estimate from rough covalent radii
    z1 = elz( elem1 )
    z2 = elz( elem2 )
    r1 = atomic_radius( z1 )
    r2 = atomic_radius( z2 )
    rsum = r1 + r2
    return rsum
##
def atomic_radius( iz ):
    # return covalent atomic radius given Z (3/21/2012) (Angstrom)
    # values are from Wikipedia (attributed to Slater 1964);
    #   I filled blanks with a guess (e.g., Z-1 value)
    r = [ 0, 0.25, 0.25, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.50,
             1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.00,
             2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35,
             1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.15,
             2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40,
             1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, 1.40,
             2.60, 2.15,
             1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.80, 1.75,
             1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
             1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35,
             1.35, 1.50, 1.90, 1.80, 1.60, 1.90, 1.90, 1.90,
             2.80, 2.15,
             1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
             1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
             1.75, 1.75, 1.75, 1.75, 1.75, 1.75 ]
    if type(iz) == int:
        return r[ iz ]
    else:
        # convert symbol to nuclear charge
        z = elz( iz )
        return r[z]
##
def from_ltriangle(vec):
    # given a 1D numpy array that is a flattened lower-triangle,
    #   return the corresponding symmetric, square numpy array
    n = len(vec)
    dim = int(round(0.5 * (-1 + np.sqrt(1+8*n))))  # dimension of the square matrix
    idx = np.tril_indices(dim)
    mat = np.zeros((dim, dim))
    mat[idx] = vec
    # symmetrize
    mat = mat + np.triu(mat.T, 1)
    return mat
##
def inertia_tensor(masses, xyz):
    # moment-of-inertia tensor of point-masses
    #  m is a list of masses, xyz is a numpy array of Cartesian triples
    inertia = np.zeros((3,3))
    n = len(masses)
    if n != len(xyz):
        print('Argument inconsistency in inertia_tensor(): {:d} masses but {:d} positions'.format(n, len(xyz)))
        return None
    for i in range(n):
        m = masses[i]
        (x, y, z) = tuple(xyz[i])
        inertia[0][0] += m * (y*y + z*z)
        inertia[1][1] += m * (x*x + z*z)
        inertia[2][2] += m * (y*y + x*x)
        inertia[0][1] -= m * x * y
        inertia[0][2] -= m * x * z
        inertia[1][2] -= m * y * z
    inertia[1][0] = inertia[0][1]
    inertia[2][0] = inertia[0][2]
    inertia[2][1] = inertia[1][2]
    return inertia
##
def orthogonalize_rows(M, norm=0):
    # orthogonalize rows of numpy 2D array M
    #   normalize each row to length 'norm' if norm > 0
    for i in range(M.shape[0]-1):
        # project row 'i' from all later rows
        v = M[i] / np.linalg.norm(M[i])
        for j in range(i+1, M.shape[0]):
            p = np.dot(v, M[j])
            M[j] -= p * v
    if norm > 0:
        # normalize each row to specified length
        nrm = np.linalg.norm(M, axis=1)
        M = np.divide(M.T, nrm).T
    return M
##
def vib_harmonic(fc, mass, sayvetz=False, xyz=[]):
        # given numpy arrays of cartesian force constants and atomic masses,
        # return harmonic frequencies (cm^-1) and mode vectors
        # This function does not do Sayvetz projection unless requested
        #   the projection requires atomic coordinates (as flattened list)
        # Following Joe Ochterski's description 
        mwt = []   # mass-weighting vector
        for m in mass:
            mwt.extend( [1/np.sqrt(m)] * 3 )  #  same mass for (x,y,z) of an atom
        wmat = np.outer(mwt, mwt) # mass-weighting matrix
        # apply the mass-weighting matrix to the force constants
        wfc = np.multiply(fc, wmat)
        wfc /= AMU_AU  # mass-weighted force constant matrix in atomic units
        eigval, eigvec = np.linalg.eigh(wfc)
        esign = np.sign(eigval)   # save the sign of each eigenvalue
        eigval = np.fabs(eigval)      # all values are now positive
        eigval = np.sqrt(eigval)
        eigval = np.multiply(esign, eigval)      # imaginary frequencies are "negative"
        eigval *= AU_WAVENUMBER
        if not sayvetz:
            # no projections; return eigenvectors as rows
            return eigval, eigvec.T
        else:
            # Use Sayvetz conditions to project out external coordinates
            print('WARNING: SAYVETZ PROJECTION IS NOT WORKING!')
            natom = len(mass)
            dimen = 3 * natom
            if len(xyz) != dimen:
                print('Unable to do Sayvetz projects: {:d} masses but {:d} coordinates'.format(natom, len(xyz)))
                return eigval, eigvec.T
            # project out the translations and rotations
            xyz = xyz.reshape(-1, 3)  # each row of 'xyz' is now for one atom
            com = np.zeros(3)  # center of mass
            mtot = 0  # total mass
            for i in range(natom):
                mtot += mass[i]
                com += mass[i] * xyz[i]
            com /= mtot
            print('total mass = {:.3f}'.format(mtot))
            print('center of mass:', com)
            # translate COM to the origin
            for i in range(natom):
                xyz[i] -= com
            # get principal axes
            inert = inertia_tensor(mass, xyz)
            print('inertial tensor:\n', inert)
            inert_val, inert_vec = np.linalg.eigh(inert)
            print('inert_val:', inert_val)
            print('inert_vec:\n', inert_vec)
            # translation S vectors (called D1, D2, D3 by Ochterski)
            for i in range(natom):
                mat = np.eye(3) * np.sqrt(mass[i])
                try:
                    S = np.concatenate((S, mat), axis=1)
                except:
                    # probably haven't created S yet
                    S = mat.copy()
            # rotation S vectors (Ochterski's D4, D5, D6)
            if False:
                # following Ochterski
                print('*** Following Ochterski\'s white paper')
                for n in range(natom):
                    mat = np.zeros((3,3))
                    for i in [0, 1, 2]:
                        j = (i+1) % 3
                        k = (j+1) % 3
                        mat[i]  = np.dot(xyz[n], inert_vec[j]) * inert_vec[k]
                        mat[i] -= np.dot(xyz[n], inert_vec[k]) * inert_vec[j]
                        mat[i] /= np.sqrt(mass[n])
                    try:
                        Sr = np.concatenate((Sr, mat), axis=1)
                    except:
                        # probably haven't created Sr yet
                        Sr = mat.copy()
                S = np.concatenate((S, Sr), axis=0)
            else:
                # following G03 source code:  routine TRVect() in utilnz.F 
                print('*** Following G03 source code')
                for n in range(natom):
                    mat = np.zeros((3,3))
                    CP = np.dot(inert_vec, xyz[n])
                    mat[0,0] = CP[1]*inert_vec[2,0] - CP[2]*inert_vec[1,0]
                    mat[0,1] = CP[1]*inert_vec[2,1] - CP[2]*inert_vec[1,1]
                    mat[0,2] = CP[1]*inert_vec[2,2] - CP[2]*inert_vec[1,2]
                    mat[1,0] = CP[2]*inert_vec[0,0] - CP[0]*inert_vec[2,0]
                    mat[1,1] = CP[2]*inert_vec[0,1] - CP[0]*inert_vec[2,1]
                    mat[1,2] = CP[2]*inert_vec[0,2] - CP[0]*inert_vec[2,2]
                    mat[2,0] = CP[0]*inert_vec[1,0] - CP[1]*inert_vec[0,0]
                    mat[2,1] = CP[0]*inert_vec[1,1] - CP[1]*inert_vec[0,1]
                    mat[2,2] = CP[0]*inert_vec[1,2] - CP[1]*inert_vec[0,2]
                    mat *= np.sqrt(mass[n])
                    try:
                        Sr = np.concatenate((Sr, mat), axis=1)
                    except:
                        # probably haven't created Sr yet
                        Sr = mat.copy()
                S = np.concatenate((S, Sr), axis=0)
            print('combined S:\n', S)
            # remove any zero-vector rows
            nrm = np.linalg.norm(S, axis=1)
            print('nrm(S) =', nrm)
            for i in range(5, -1, -1):
                # loop over rows of S
                if nrm[i] < 1.0e-03:  # I picked this threshold arbitrarily! 
                    S = np.delete(S, (i), axis=0)
                    print('*** deleting row {:d} of S ***'.format(i))
                else:
                    S[i] /= nrm[i]  # normalize the row
            # orthogonalize rows and re-normalize (only needed when following Ochterski)
            S = orthogonalize_rows(S, norm=1)
            print('normalized S:\n', S)
            print('S dot S:\n', np.dot(S, S.T))
            # Start from a mass-weighted unit matrix and project out the rows of S
            #   also project out previous rows of growing D matrix
            D = np.eye(dimen, dimen)  # initialize D to the identity matrix
            for n in range(natom):
                for i in range(3*n, 3*n+3):
                    # apply mass-weighting
                    D[i] *= np.sqrt(mass[n])
            print('D before any projection:\n', D)
            for i in range(S.shape[0]):
                # project out each row of S from D
                p = np.dot(S[i], D.T)
                D -= np.outer(p, S[i])
                nrm = np.linalg.norm(D, axis=1)
            print('D after projecting out S:\n', D)
            # now orthogonalize the remaining basis vectors
            D = orthogonalize_rows(D, norm=0)  # do not renormalize after orthogonalization
            print('D after orthogonalization:\n', D)
            nrm = np.linalg.norm(D, axis=1)
            print('norm of D rows:\n', nrm)
            # Delete the zero rows
            zrow = np.where(nrm < 0.001)[0]  # I picked this threshold arbitrarily! 
            zrow = tuple(zrow)  # convert to tuple
            print('zrow =', zrow)
            if len(zrow) != S.shape[0]:
                # something is wrong
                print('*** Error: There are {:d} external coordinates but {:d} have been eliminated ***'.format(S.shape[0], len(zrow)))
                print('...continuing anyway!...')
            D = np.delete(D, zrow, axis=0)
            # re-normalize the rows of D
            nrm = np.linalg.norm(D, axis=1)
            print('shape of D =', D.shape)
            print('norm of D rows:\n', nrm)
            D = np.divide(D.T, nrm).T
            print('D after normalization:\n', D)
            # adjoin S to D
            D = np.concatenate((D, S), axis=0)
            print('new shape of D =', D.shape)
            nrm = np.linalg.norm(D, axis=1)
            print('norm of D rows:\n', nrm)
            # change basis for force constants
            fcint = np.dot(D, np.dot(fc, D.T))
            print('internal-coordinate force constants:\n', fcint)
            print('Frequencies before projection:\n', eigval)
            igval, igvec = np.linalg.eigh(fcint)
            esign = np.sign(igval)   # save the sign of each eigenvalue
            igval = np.fabs(igval)      # all values are now positive
            igval = np.sqrt(igval)
            igval = np.multiply(esign, igval)      # imaginary frequencies are "negative"
            igval *= AU_WAVENUMBER
            print('Frequencies after projection:\n', igval)
            print('Ratios:\n', np.divide(igval, eigval))
            return eigval, eigvec.T
##
def filename_root(filename):
    # remove any file suffix
    m = re.match(r'(.+)\.\w+$', filename)
    if m:
        return m.group(1)
    else:
        # no suffix
        return filename
##
def rotation_mat_angle(v, a, unit='radian'):
    # return a matrix that will rotation by angle a around axis v
    # method is from StackExchange.com
    if unit == 'degree':
        # convert to radians for trig functions
        a = np.deg2rad(a)
    # normalize vector
    u = v / np.linalg.norm(v)
    [x, y, z] = u.tolist()
    s = np.sin(a)
    s2 = np.sin(a/2)
    W = np.array([ [0.,-z,y], [z,0.,-x], [-y,x,0.] ])
    R = np.identity(3) + s*W + 2*s2*s2*np.dot(W,W)
    return R
##
def rotation_mat_align(A, B, scale=False):
    # given two numpy vectors (in R3), return the matrix that rotates A into B
    # method is from StackExchange.com
    # if scale is True, then also scale the magnitude to match
    if (len(A) != 3) or (len(B) != 3):
        print('**** must be vectors in R3! ****')
        return np.zeros((3,3))
    # normalize
    a = A / np.linalg.norm(A)
    b = B / np.linalg.norm(B)
    c = np.dot(a, b)  # angle cosine
    if np.isclose(c, 1.):
        # no rotation needed
        R = np.identity(3)
    elif np.isclose(c, -1.):
        # antiparallel; rotate by pi about a perpendicular axis
        p = np.cross(a, 1. - a)
        R = rotation_mat_angle(p, np.pi)
    else:
        # general case
        v = np.cross(a, b)
        [v1, v2, v3] = v.tolist()
        vx = np.array([ [0.,-v3,v2], [v3,0.,-v1], [-v2,v1,0] ])
        R = np.identity(3) + vx + np.dot(vx,vx)/(1+c)
    if scale:
        s = np.linalg.norm(B) / np.linalg.norm(A)  # scaling factor
        R *= s
    return R
##
def normalize(v, length=1.0):
    # given a vector, return it scaled to desired length
    try:
        n = np.linalg.norm(v)
        if n == 0:
            return np.zeros_like(v)
        else:
            return np.array(v) * length / n
    except:
        print('*** failure computing length in normalize()')
        print('typeof(v) = ', type(v))
        print('v = ', v)
        sys.exit(1)
##
def to_radian(angle, reverse=False):
    # given an angle in degrees, convert it to radians (or the reverse)
    if reverse:
        # convert from radians to degrees
        return angle * 180. / np.pi
    else:
        # convert from degrees to radians
        return angle * np.pi / 180.
##
def angular_momentum(m, r, v):
    # given atomic masses, positions, and velocities,
    #   return the total angular momentum
    rxv = np.cross(r,v)
    L = (rxv.T * m).T.sum(axis=0)
    return L
##
def angle_canon(a, unit='radian'):
    # given an angle (or numpy array of them), return the equivalent
    #   value in the interval (-pi, pi]
    if unit == 'degree':
        c = (-a + 180.) % 360. - 180.
    else:
        c = (-a + np.pi) % (2 * np.pi) - np.pi
    return -c
##
def in_bounds(x, target, tolerance):
    # is 'x' in the open interval 'target' +- 'tolerance' ?
    tolerance = np.abs(tolerance)
    return ( (x < target+tolerance) and (x > target-tolerance) )
##
def smoothing(x, y, x2, style='gau', width=-1, normalize=True):
    # return smoothed y values for (x,y) data series (numpy arrays)
	#   ouput is over the smoothed range defined by x2 (a numpy array)
    # no sorting necessary
    # styles: 'exp' for exponential; 'gau' for gaussian
    # width parameter (sigma) defaults to 1% of x-range
    if len(x) != len(y):
        # bad input data
        return None
    xlo = min(x)
    xhi = max(x)
    if (width <= 0):
        width = (xhi - xlo) * 0.01
    y2 = np.zeros_like(x2)
    for i in range(len(y)):
        dx = x2 - x[i]
        if style == 'gau':
            dx = (dx/width)**2
            t = np.exp(-dx)
        if style == 'exp':
            dx = abs(dx/width)
            t = np.exp(-dx)
        if normalize:
            t = t / t.sum()
        y2 = y2 + t * y[i]
    return y2
##
def joinGeometries(Glist):
    # Given a list of Geometry objects, return a single Geometry
    #   that includes all their atoms
    # if charges are specified, sum them
    atomlist = []
    q = 0
    for G in Glist:
        atomlist += G.atom
        try:
            q += G.charge
        except:
            q = None
    Gtot = Geometry(atomlist, intype='atlist')
    Gtot.charge = q
    return Gtot
##
def same_connectivity(Struct1, Struct2, tol=1.3):
    # compare connectivity tables 
    # return True if same, else False
    conn1 = Struct1.connection_table(tol)
    conn2 = Struct2.connection_table(tol)
    return np.array_equal(conn1, conn2)
##
def RMSD_align(Geom, refGeom, use_masses=False):
    # translate and rotate Geometry object 'Geom' to minimize RMSD with 'refGeom'
    # return a new Geometry object
    G = Geom.copy()  # avoid damaging the input geometries
    refG = refGeom.copy()
    if not use_masses:
        # Use unit mass for every atom
        mvec = np.ones(G.natom())
        G.set_masses(mvec)
        refG.set_masses(mvec)
    transl = refG.COM()
    #print('::: initial RMSD = ', RMSD(G, refG), end='')
    G.center(use_masses=use_masses)
    refG.center(use_masses=use_masses)
    U = Kabsch(G, refG, use_masses=use_masses)
    G.rotate(U)
    #print('   after align = ', RMSD(G, refG))
    G.translate(transl)
    return G
##
def RMSD(Geom1, Geom2):
    # return the RMSD between two Geometry objects (no weights)
    v1 = Geom1.toVector()
    v2 = Geom2.toVector()
    if len(v1) != len(v2):
        print_err('', 'Inconsistent atom counts: {:d} for Geom1 and {:d} for Geom2'.format(natom, Geom2.natom()))
    natom = len(v1) // 3
    rmsd = distance(v1, v2) / np.sqrt(natom)
    return rmsd
##
def Kabsch(Geom1, Geom2, use_masses=False):
    # return the rotation matrix that mimizes the unweighted RMSD (Wikipedia: "Kabsch algorithm")
    #   (tranform G1 toward G2)
    G1 = Geom1.copy()   # avoid damaging the input Geometry objects
    G2 = Geom2.copy()
    natom = G1.natom()
    if natom != G2.natom():
        print_err('', 'Inconsistent atom counts: {:d} for Geom1 and {:d} for Geom2'.format(natom, G2.natom()))
    # translate barycenters to origin
    if not use_masses:
        # Use unit mass for every atom
        mvec = np.ones(natom)
        G1.set_masses(mvec)
        G2.set_masses(mvec)
    G1.center(use_masses=use_masses)
    G2.center(use_masses=use_masses)
    elem, P = G2.separateXYZ()  # the reference
    elem, Q = G1.separateXYZ()
    A = np.dot(P.T, Q)
    V, s, W = np.linalg.svd(A)
    d = np.sign(np.linalg.det(np.dot(V,W)))
    D = np.diag([1., 1., d])
    U = np.dot(V, np.dot(D,W))
    return U
##
def average_structure(Struct1, Struct2, weight1=0.5, weight2=0.5):
    # given two compatible structures, return a similar structure
    #   with coordinates that are the weighted average of the
    #   input structures
    if (Struct1.coordtype != Struct2.coordtype) or (Struct1.natom() != Struct2.natom()):
        # structures are not compatible
        return None
    v1 = Struct1.toVector()
    v2 = Struct2.toVector()
    try:
        v3 = (weight1 * v1 + weight2 * v2) / (weight1 + weight2)
    except:
        # probably weights sum to zero
        return np.nan
    Result = Struct1.copy()
    unitS = Struct1.unitX()
    Result.fromVector(v3, unitS)
    return Result
##
def print_dict(nestdict, indent=0, space='    '):
    # nice printing of nested dict
    if not isinstance(nestdict, dict):
        print_err('', 'non-dict argument', halt=False)
    spacing = space * indent
    for k in sorted(nestdict):
        if isinstance(nestdict[k], dict):
            print(spacing+k+':')
            print_dict(nestdict[k], indent+1)
        else:
            print(spacing+k+':', nestdict[k])
##
def dict_delkey(d, key):
    # delete one or more keys from a dict, if present
    # return the number of keys deleted
    # 'key' can be a list or a scalar (including string)
    ndel = 0
    if isinstance(key, list):
        for k in key:
            if dict_delkey(d, k):
                ndel += 1
    else:
        # simple key
        if key in d:
            del d[key]
            ndel = 1
    return ndel
##
def backfill_dict(defaults, userinput):
    # recursively install any missing entries in dict 'userinput',
    # based upon default values in dict 'defaults'
    # return False on non-dict arguments, else True
    if not (isinstance(defaults, dict) and isinstance(userinput, dict)):
        # this routine does not apply
        return False
    for key in defaults:
        if key in userinput:
            try:
                backfill_dict(defaults[key], userinput[key])
            except:
                # probably at the bottom of the structure
                pass
        else:
            userinput[key] = defaults[key]
    return True
##
def conv_list_scalar(name, ret_type=None):
    # convert a scalar to a list of scalar, OR
    # return the first element of the list
    if ret_type == 'list':
        # return a list
        if type(name) is list:
            return name
        else:
            return [name]
    elif ret_type == 'string':
        if type(name) is list:
            return name[0]
        else:
            return name
    elif ret_type is None:
        # toggle: return the other thing
        if type(name) is list:
            return name[0]
        else:
            return [name]
    else:
        print_err('', 'unrecognized ret_type = {}'.format(ret_type))
##
getframe_expr = 'sys._getframe({}).f_code.co_name'
def print_err(errtype, details='', halt=True):
    # print a line about the error, with the name of the function
    if errtype ==   'code':
        msg = '*** Unrecognized quantum chemistry code "{:s}"'.format(details)
    elif errtype == 'io':
        msg = '*** Unrecognized I/O code "{:s}"'.format(details)
    elif errtype == 'write_fail':
        msg = '*** Failure writing file "{:s}"'.format(details)
    elif errtype == 'open_fail':
        msg = '*** Failure opening file "{:s}"'.format(details)
    elif errtype == 'autodetect':
        msg = '*** Autodection failure'
        print(msg)
        1/0
    elif errtype == 'task':
        msg = '*** Unrecognized task "{:s}"'.format(details)
    elif errtype == 'atom_order':
        msg = '*** Inconsistent atom ordering ({:s})'.format(details)
    elif errtype == 'coordtype':
        msg = '*** Unrecognized type of coordinates "{:s}"'.format(details)
    elif errtype == 'maxiter':
        msg = '*** Maximum number of iterations ({:d}) exceeded'.format(details)
    elif errtype == 'option':
        msg = '*** Unrecognized option: {:s}'.format(details)
    elif errtype == 'units':
        msg = '*** Unrecognized or inappropriate units: {:s}'.format(details)
    elif errtype == 'need_int':
        msg = '*** Integer required: {:s}'.format(details)
    elif errtype == 'zmatrix':
        msg = '*** Zmatrix input problem: {:s}'.format(details)
    elif errtype == 'undone':
        msg = '*** Feature not yet implemented: {:s}'.format(details)
    else:
        # generic message
        if halt:
            msg = '*** Fatal error: "{:s}"'.format(details)
        else:
            msg = '*** Error: "{:s}"'.format(details)
    # add name of calling routine
    caller = eval(getframe_expr.format(2))
    msg += ' in {:s}()'.format(caller)
    if halt:
        # print the message and exit
        #   may cause trouble with 'multiprocessing' module
        sys.exit(msg)
    else:
        # just print the message, then return
        print(msg)
    return
##
