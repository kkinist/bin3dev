#!/usr/bin/python3
# Analysis of trajectory files from IPS
#   Detect reactions and then:
#       - optimize reaction products
#       - create reaction-path "strings"
# Karl Irikura, NIST 2017
#
import pdb, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, glob, multiprocessing, time, re
from chem_subs import *
from qm_subs import *
from ips_subs import *
import scipy.optimize as sciopt
import scipy.signal as scignal
##
""" Analyze results from an IPS run.
    Start by comparing connectivity matrices.
"""
##
class BeadString(object):
    # A sequence of Geometry objects, with additional information
    #   (e.g., "energy" gradients)
    def __init__(self, Geom, Xmat=None, Evec=None, Gmat=None):
        # 'Geom'    : Geometry() object
        # 'Xmat'    : array of coordinate vectors that define the structures (one per row)
        # 'Evec'    : array of energies (may include pseudo-energies)
        # 'Gmat'    : array of gradient vectors (may include pseudo-forces)
        if Geom.coordtype != 'Geometry':
            # convert to Geometry() and attempt to proceed
            print('** Warning: converting wrong structure type into Geometry() **')
            self.Geom = Geom.toGeometry()
        else:
            self.Geom = Geom.copy()
        natom = self.natom()
        ndof = natom * 3
        if Xmat is None:
            # there is only one bead
            self.Xmat = Geom.toVector()
            nb = 1
        else:
            # check number of atoms for consistency
            try:
                self.Xmat = Xmat.reshape((-1, ndof)).copy()
                nb = len(self.Xmat)
            except:
                print('** Dimensions of Xmat are not compatible with Geom **')
                nb = len(Xmat.flatten()) // ndof
                if nb > 0:
                    print('** Truncating Xmat **')
                    self.Xmat = Xmat.flatten()[:nb*ndof].reshape((-1, ndof)).copy()
                else:
                    # not enough coordinates; use default
                    print('** Discarding Xmat **')
                    self.Xmat = Geom.toVector()
                    nb = 1
        # similar procedure for gradient
        if Gmat is None:
            self.Gmat = None
        else:
            # dimensions of Gmat must match those of Xmat
            if len(self.Xmat.flatten()) != len(Gmat.flatten()):
                print('** Dimensions of Gmat are incommensurate with Xmat **')
                print('** Discarding Gmat **')
                self.Gmat = None
            else:
                self.Gmat = Gmat.reshape(self.Xmat.shape).copy()
        # install NaN in Gmat if needed
        if self.Gmat is None:
            self.Gmat = np.ones(self.Xmat.shape) * np.nan
        # check number of energies
        if Evec is None:
            self.Evec = None
        else:
            if len(Evec) != nb:
                print('** Length of Evec is inconsistent with number of beads **')
                print('** Discarding Evec **')
                self.Evec = None
            else:
                self.Evec = Evec.copy()
        # install NaN in Evec is needed
        if self.Evec is None:
            self.Evec = np.ones(nb) * np.nan               
    def natom(self):
        # return the number of atoms (in each structure)
        return self.Geom.natom()
    def nbead(self):
        # return the number of structures
        return len(self.Xmat)
    def unitX(self):
        # return the units of the coordinates (as a 1-tuple)
        return self.Geom.unitX()
    def Erel(self, scale=1.):
        # return a vector of scaled relative energies
        #   zero is the minimum bead energy
        Emin = self.Evec.min()
        Erel = (self.Evec - Emin) * scale
        return Erel
    def printXYZ(self, filename):
        # print all structures to a single XYZ file
        Gbuf = self.Geom.copy()
        E0 = self.Evec[0]   # reactant energy
        with open(filename, 'w') as fxyz:
            for ibead in range(self.nbead()):
                Gbuf.fromVector(self.Xmat[ibead,:], self.unitX())
                Eabs = self.Evec[ibead]
                Erel = au_kjmol * (Eabs - E0)
                Gbuf.printXYZ(fxyz, 'Bead {:d} with relative energy = {:.1f} kJ/mol (Eabs = {:.5f})'.format(ibead, Erel, Eabs), handle=True)
    def beadGeom(self, ibead):
        # return a Geometry() for the specified bead
        if (ibead < 0) or (ibead > self.nbead()):
            # bad value of ibead
            return None
        Gbuf = self.Geom.copy()
        Gbuf.fromVector(self.Xmat[ibead,:], self.unitX())
        return Gbuf
    def setGeom(self, ibead):
        # install coordinates for bead #ibead into self.Geom
        unitS = self.unitX()
        self.Geom.fromVector(self.Xmat[ibead,:], unitS)
        return
    def beadDistance(self, ibead, jbead):
        # return the cartesian distance between two beads
        d = np.linalg.norm(self.Xmat[ibead,:] - self.Xmat[jbead,:])
        return d
    def beadSpacings(self):
        # return a vector of the distances between adjacent beads
        #   bead #i to bead #(i+1)
        nbead = self.nbead()
        d2next = np.ones(nbead-1) * np.nan
        for i in range(nbead-1):
            d2next[i] = self.beadDistance(i, i+1)
        return d2next
    def length(self):
        # return length of BeadString as sum of bead spacings
        length = self.beadSpacings().sum()
        return length
    def deleteBeads(self, idel):
        # delete beads; 'idel' is (single or list-like) bead number(s)
        self.Xmat = np.delete(self.Xmat, idel, axis=0)
        self.Evec = np.delete(self.Evec, idel)
        self.Gmat = np.delete(self.Gmat, idel, axis=0)
        return
    def insertInterpolated(self, iprev, wt1=0.5, wt2=0.5):
        # add a bead after position 'iprev'
        #   use linear interpolation for the coordinates,
        #   but install NaN's for energy and gradient
        # 'wt1' and 'wt2' are the weights to use for points
        #   #(iprev) and #(iprev+1)
        wtot = wt1 + wt2
        Xnew = wt1 * self.Xmat[iprev,:] + wt2 * self.Xmat[iprev+1,:]
        Xnew /= wtot
        nanvec = np.ones_like(Xnew) * np.nan
        self.Xmat = np.insert(self.Xmat, iprev+1, Xnew, axis=0)
        self.Gmat = np.insert(self.Gmat, iprev+1, nanvec, axis=0)
        self.Evec = np.insert(self.Evec, iprev+1, np.nan)
        return
    def adjust_count(self, Nmin=10, Ravg=0.5):
        # Adjust the number of beads up or down
        # args: Nmin = minimum number of beads
        #       Ravg = desired average inter-bead distance
        nbead = self.nbead()
        d2next = self.beadSpacings()  # distance from one bead to the following bead
        length = d2next.sum()
        Navg = int(length // Ravg)  # desired number of beads, based upon length
        # 'dspan' is the indirect distance from bead (i-1) to bead (i+1)
        #   it's used to find beads that can be deleted
        #   terminal beads should never be deleted
        dspan = [np.inf] + [d2next[i-1]+d2next[i] for i in range(1, nbead-1)] + [np.inf]
        dspan = np.array(dspan)
        print('BeadString length = {:.1f}, Ravg = {:.1f}, so Navg = {:d}'.format(length, Ravg, Navg))
        #print('d2next = ', ', '.join(['{:.2f}'.format(d) for d in d2next]))
        #print('dspan = ', ', '.join(['{:.2f}'.format(d) for d in dspan]))
        excess = nbead - max(Navg, Nmin)
        if excess > 0:
            # remove some beads
            idel = np.argsort(dspan)[:excess]
            print('removing {:d} excess beads: '.format(excess), idel)
            self.deleteBeads(idel)
        elif excess < 0:
            # add some beads by interpolation; set energies and gradients to NaN
            print('adding {:d} beads by linear interpolation'.format(-excess))
            for ix in range(-excess):
                # keep bisecting the largest interval until there are enough points
                interv = d2next.argmax()
                self.insertInterpolated(interv)
                # patch up d2next[]
                newd = d2next[interv] / 2
                d2next[interv] = newd
                d2next = np.insert(d2next, interv, newd)
        ravg = self.length()/self.nbead()  # new average inter-bead distance
        return ravg
    def oneQCgradient(self, ips_input, ibead):
        # Use a quantum chemistry program to compute the energy and gradient
        #   for the specified bead.
        self.setGeom(ibead)
        ips_input['coordtype'] = 'cartesian'
        ips_input['cartesian'] = self.Geom
        froot = '{:s}_bead_{:d}_grad'.format(ips_input['molecule'], ibead)
        unitR = self.Geom.unitX()[0]
        E, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR=unitR, ID=ibead, fileroot=froot)
        return E, grad, ibead
    def QCgradient(self, ips_input):
        # Using a quantum chemistry program, compute the energy and gradient
        #   for all beads.  Install them in the BeadString.
        # Return: total energy (sum over beads) and gradient
        nbead = self.nbead()
        ndof = 3 * self.natom()
        # initialize with NaN
        Energy = np.ones(nbead) * np.nan
        Gradient = np.ones((nbead, ndof)) * np.nan
        if __name__ == '__main__':
            tasks = [(ips_input, i) for i in range(nbead)]
            #print('PPPP set up {:d} tasks'.format(len(tasks)))
            pool = multiprocessing.Pool(ips_input['nprocs'])
            results = [pool.apply_async(self.oneQCgradient, t) for t in tasks]
            pool.close()
            pool.join()
            for result in results:
                (E, G, ibead) = result.get()
                Energy[ibead] = E
                Gradient[ibead,:] = G
        else:
            print('__name__ = {:s} in BeadString.QCgradient()'.format(__name__))
        self.Evec = Energy.copy()
        self.Gmat = Gradient.copy()
        return Energy.sum(), Gradient
    def springForce(self, kforce, Req=1.):
        # Return the spring energy and its gradient
        #   'kforce' is the spring constant (hartree/ang**2)
        #   'Req' is equilibrium spring length, as a multiple
        #       of the mean interbead distance
        # There is one spring between adjacent beads
        # Terminal beads experience no force
        # Higher-energy bead feels less force than lower-energy partner
        d2next = self.beadSpacings()
        # multiply Req by the average interbead distance
        Re = Req * d2next.sum() / (self.nbead() - 1)
        dR = d2next - Re
        # E = 0.5 * k * (R-Re)**2 for each spring
        Espring = 0.5 * kforce * np.dot(dR, dR)
        Gspring = np.zeros_like(self.Gmat)
        # G = k * (R-Re) * <v12>, divided between two beads (except termini)
        nbead = self.nbead()
        Erel = self.Erel()
        for ibead in range(nbead-1):
            v12 = self.Xmat[ibead+1,:] - self.Xmat[ibead,:]
            G = kforce * dR[ibead] * normalize(v12)
            if ibead == 0:
                # reactant bead; does not get force
                Gspring[ibead+1] -= G
            elif ibead == nbead-2:
                # don't share force with product bead
                Gspring[ibead] += G
            else:
                # put more force on the lower-energy bead
                G /= Erel[ibead]**2 + Erel[ibead+1]**2
                Gspring[ibead] += G * Erel[ibead+1]**2
                Gspring[ibead+1] -= G * Erel[ibead]**2
        return Espring, Gspring
    def effectivePotential(self, X, ips_input, kforce, Req=1.):
        # Real energy plus spring energy
        # args:
        #   ips_input : the parameter dictionary for IPS generally
        #   X         : array of bead cartesian coordinates
        #   kforce    : force constant for the artificial springs (hartee/angstrom**2)
        #   Req       : equilibrium length of springs as fraction of mean interbead distance
        # Return: energy, energy gradient
        #   where 'energy' is the real (quantum chemistry) energy + the spring energy
        if X is None:
            # use the current coordinates
            pass
        else:
            try:
                X = X.reshape(self.Xmat.shape)
            except:
                print_err('', 'Incompatible coordinate dimensions: self.Xmat = {:s}, X = {:s}'.format(str(self.Xmat.shape), str(X.shape)))
            self.Xmat = X
        Ereal, Greal = self.QCgradient(ips_input)
        Espring, Gspring = self.springForce(kforce=kforce, Req=Req)
        Esudo = Ereal + Espring
        Gsudo = Greal + Gspring
        return Esudo, Gsudo.flatten()
    def minimizeEP(self, ips_input, kforce, Req=1., tol=1e-2, maxiter=20):
        # minimize the pseudo-energy of the BeadString
        # Uses scipy.optimize.minimize (Polak-Ribiere algorithm)
        # uses gradients
        X0 = self.Xmat.flatten()
        result = sciopt.minimize(self.effectivePotential, X0, args=(ips_input, kforce, Req), 
            method='CG', jac=True, tol=tol, options={'maxiter':maxiter, 'disp':True})
        print('XXXX result from scipy:\n', result)
        print('XXXX norm(jac) = {:.3f}'.format(np.linalg.norm(result['jac'])))
        niter = result['nit']
        if result['success']:
            print('String convergence after {:d} iterations'.format(niter))
        else:
            print('String not converged after {:d} interations'.format(niter))
        # install optimized coordinates
        self.Xmat = result['x'].reshape(self.Xmat.shape)
##
def guess_spinmult(ips_input, newgeom):
    # Count the connections (not bonds) broken to form 
    #   'newgeom' and guess a good spin multplicity.
    #   Also count the fragments.  Assume each fragment
    #   is either singlet or doublet.
    #   Return a spinmult integer (like '3' for triplet)
    geom0 = ips_input['geom0']
    tol = 1.3
    nconn0 = int(round(geom0.connection_table(tol=tol).sum() / 2))
    nconn = int(round(newgeom.connection_table(tol=tol).sum() / 2))
    nbroken = nconn0 - nconn  # the number of connections broken
    nfrag0 = len(geom0.separateNonbonded(tol=tol))
    nfrag = len(newgeom.separateNonbonded(tol=tol))
    dfrag = nfrag - nfrag0  # the change in number of fragments
    mult0 = ips_input['spinmult']
    if nbroken == 0:
        mult = mult0
    elif nbroken == 1:
        if mult0 == 1:
            # expect closed-shell to form two radicals
            mult = 3
        else:
            # expect a radical to form "olefin" + radical 
            mult = mult0
    elif nbroken == 2:
        if dfrag == 1:
            # a piece broke off; assume as a singlet
            mult = mult0
        elif dfrag == 2:
            # two pieces broke off; assume high-spin coupled
            mult = mult0 + 4
        else:
            s = '*** Puzzling situation: nbroken = {:d} but dfrag = {:d}; see file puzzle.xyz'.format(nbroken, dfrag)
            print(s)
            newgeom.printXYZ('puzzle.xyz', comment=s)
            mult = mult0 + 4
    elif nbroken == -1:
        # reduce spin if possible
        mult = max((mult0+1) % 2 + 1, mult0 - 2)
    else:
        s = '*** Unhandled situation: nbroken = {:d} and dfrag = {:d}; see file unhandled.xyz'.format(nbroken, dfrag)
        newgeom.printXYZ('unhandled.xyz', comment=s)
        mult = mult0
    print(',,,,, from nconn0 = {:d}, nbroken = {:d};  nfrag0 = {:d}, dfrag = {:d};  return mult = {:d}'.format(nconn0, nbroken, nfrag0, dfrag, mult))
    return mult
##
def optimize_one(ips_input, newgeom, label, froot, frag_sep=0):
    # run a single geometry optimization
    # return raw energy, structure object, and the label
    #   (the label is needed for asynchronous parallel operation)
    # If there are multiple, nonbonded fragments, translate them
    #   away from each other by the distance 'frag_sep'
    if frag_sep > 0:
        # separate any fragments
        if ips_input['coordtype'] != 'cartesian':
            # convert to Geometry() object for fragment manipulation
            newG = newgeom.toGeometry()
        else:
            # it's already a Geometry()
            newG = newgeom.copy()
        nfrag = newG.spread_fragments(dist=frag_sep)  # might need to activate 'tol' parameter in the future
        ips_input['coordtype'] = 'cartesian'
        ips_input['cartesian'] = newG
        if (nfrag > 1) and ips_input['pbc']['use_pbc']:
            edge = ips_input['pbc']['edge']['value']
            newedge = edge + frag_sep
            ips_input['pbc']['edge']['value'] = newedge
            if ips_input['verbose']:
                print('Temporarily increasing cell size to ' + 
                    '{:.2f} to allow for frag_sep = {:.2f}'.format(newedge, frag_sep))
    else:
        # no fragment detection
        ips_input[ips_input['coordtype']] = newgeom
    #ips_input['spinmult'] = guess_spinmult(ips_input, newgeom)
    E, Struct, IDx = qm_function(ips_input, 'minimize', ID=label, fileroot=froot)
    Erel = au_kjmol * (E - ips_input['E0'])
    if np.isnan(E):
        print('Geometry optimization failed for label = ', label)
    else:
        print('Erel = {:.1f} kJ/mol after geometry optimization for label = '.format(Erel), label)
    if (nfrag > 1) and ips_input['pbc']['use_pbc'] and (frag_sep > 0):
        ips_input['pbc']['edge']['value'] = edge
        print('Cell size returned to {:.2f}'.format(edge))
    return E, Erel, Struct, label
##
def optimize_parallel(ips_input, labels, geoms, frag_sep=0):
    # run geometry optimizations in parallel
    #   'labels' is a list of (walker, step) tuples
    #   'geoms' is a list of structure objects
    # return results as a pandas DataFrame
    coordtype = ips_input['coordtype']
    geom = ips_input[coordtype].copy()
    molec = ips_input['molecule']
    nprocs = ips_input['nprocs']
    partasks = []
    if len(labels) != len(geoms):
        print_err('', 'len(labels) = {:d} is not consistent with len(geoms) = {:d}'.format(len(labels), len(geoms)))
    for i in range(len(labels)):
        #ips_input[coordtype] = geoms[i]
        (iwalker, istep) = labels[i][0]  # label from the 'first' time the structure was found
        fileroot = '{:s}_minim_{:d}_{:d}'.format(molec, iwalker, istep)
        partasks.append( (ips_input.copy(), geoms[i], labels[i][0], fileroot, frag_sep) )
    pool = multiprocessing.Pool(nprocs)
    print('Optimizing {:d} structures using {:d} processes.'.format(len(labels), min(len(labels), nprocs)))
    #print('\tApparent fragments are spread apart by distance {:.1f} before minimization.'.format(frag_sep))
    results = [pool.apply_async(optimize_one, t) for t in partasks]
    pool.close()
    pool.join()
    # optimize_one() will return a relative energy, a structure object, and the original label
    # construct a DataFrame
    dfminim = pd.DataFrame(columns=('Struct', 'Eabs', 'Erel', 'walker', 'step', 'success'))
    irow = 0
    for result in results:
        (Eabs, E, Struct, lbl) = result.get()
        if np.isnan(E):
            # geometry optimization failed
            dfminim.loc[irow] = [Struct, Eabs, E, lbl[0], lbl[1], False]
        else:
            dfminim.loc[irow] = [Struct, Eabs, E, lbl[0], lbl[1], True]
        irow += 1
    ips_input[coordtype] = geom   # restore original structure (maybe unnecessary)
    return dfminim
##
def walker_number(fname):
    # extract walker number from CSV filename
    m = re.search('.*_(\d+)_exg.csv', fname)
    if m:
        iwalker = int(m.group(1))
    else:
        # no idea
        iwalker = float('nan')
    return iwalker
##
def find_conn(conn, connList):
    # return the index in connList[] that matches the connecitivity
    #   matrix 'conn'; else return -1
    idx = 0
    for old in connList:
        if np.array_equal(conn, old):
            return idx
        idx += 1
    return -1
##
def TSnature(Geom, thr_small, thr_large):
    # Determine 'transition-state nature' of a structure as
    #   the difference in the number of connections when 
    #   using two threshold values
    # 'Geom' is a Geometry() object.
    conn1 = Geom.connection_table(thr_small)
    conn2 = Geom.connection_table(thr_large)
    dc = conn2 - conn1
    return(np.fabs(dc).sum())
##
def coulmat_compare(Geom0, Geom1, select=0, bontol=1.3):
    # Given two structure objects, return the algebraic
    #   difference between their Coulomb matrices.
    # When select > 0, the difference is restricted to 
    #   atom pairs distant by 'select' bonds.
    cm0 = Geom0.Coulomb_mat(select, bondtol)
    cm1 = Geom1.Coulomb_mat(select, bondtol)
    return cm1 - cm0
##
def find_step_list(dfuniq, iwalker, istep):
    # return list of step numbers that make sense for a bead-string
    changeSteps = []  # list of steps where connectivity changed for walker 'iwalker'
    for isom, row in dfuniq.iterrows():
        for pair in row['Found']:
            if pair[0] == iwalker:
                changeSteps.append(pair[1])
    # find the two changes that bracket 'istep'
    llim = 0
    ulim = max(changeSteps)
    for s in changeSteps:
        if (s < istep) and (s > llim):
            llim = s
        if (s > istep) and (s < ulim):
            ulim = s
    # range only should go up to ulim-1
    return list(range(llim, ulim))
##
def printXYZ_string(fname, dfBeads):
    # create an XYZ file with all the beads' structures
    with open(fname, 'w') as fxyz:
        for i, row in dfBeads.iterrows():
            row['Geom'].printXYZ(fxyz, 'Bead {:d} with Erel = {:.1f}'.format(i, row['Erel']), handle=True)
    return
##
def relax_string_old(ips_input, fexg, stepList, iprod, Product, Eprod, Eabs):
    # return a DataFrame of relaxed bead points
    #   work in cartesian coordinates
    # 'fexg' is the EXG (walker) file from which to extract points
    # 'stepList' lists the step numbers to included in the beadstring
    # 'Product' is the minimized product Geometry()
    # 'Eprod' is product relative energy (kJ/mol)
    # 'Eabs' is product energy (hartree)
    dXconvg = 0.01  # rms bead displacement defining convergence
    dEconvg = 1.    # kJ/mol for mean energy change
    maxIter = 20  # patience limit
    #
    dfexg = pd.read_csv(fexg)  # the trajectory for one walker
    E0 = ips_input['E0']
    coordtype = ips_input['coordtype']
    Struct = ips_input[coordtype].copy() # reactant
    ndof = Struct.nDOF()
    unitS = ('angstrom', 'radian')  # assumption
    # initialize 'dfBeads'
    columns = ['Step', 'Geom', 'Erel', 'Eabs', 'grad']
    nbead = len(stepList) + 2
    dfBeads = pd.DataFrame(index=range(nbead), columns=columns)
    dfBeads['Step'] = [-1] + stepList + [9999]
    if coordtype != 'cartesian':
        Reactant = ips_input['geom0'].toGeometry()
    else:
        Reactant = ips_input['geom0'].copy()
    # copy data from the trajectory into 'dfBeads'
    Gprev = None
    for istep in stepList:
        row = dfexg.loc[istep]
        Erel = row[0]
        X = row[1 : ndof+1]
        Struct.fromVector(X, unitS)
        if coordtype != 'cartesian':
            Geom = Struct.toGeometry()
            grad = np.ones_like(X) * np.nan   # don't attempt coordinate transformation
        else:
            # cartesian coordinates
            Geom = Struct.copy()
            grad = np.array(row[ndof+1 : 2*ndof+1 ])
        jrow = dfBeads[dfBeads['Step'] == istep].index.tolist()[0]
        # set elements one at a time to avoid 'ValueError' problem
        if Gprev is not None:
            # align with previous structure
            Geom = RMSD_align(Geom, Gprev)
        dfBeads.loc[jrow, 'Geom'] = Geom
        dfBeads.loc[jrow, 'Erel'] = Erel
        dfBeads.loc[jrow, 'Eabs'] = np.nan
        dfBeads.set_value(jrow, 'grad', grad)
        Gprev = Geom.copy()
        if istep == stepList[0]:
            # first regular bead; save structure
            G1 = Geom.copy()
    # first bead is the reactant, last bead is the product
    # align reactant with first regular bead
    Reactant = RMSD_align(Reactant, G1)
    dfBeads.loc[0] = [0, Reactant, 0., E0, np.ones(ndof)*np.nan]
    # align product with last regular structure
    Product = RMSD_align(Product, Gprev)
    dfBeads.loc[nbead-1] = [nbead-1, Product, Eprod, Eabs, np.zeros(ndof)]
    printXYZ_string('dfbeads.xyz', dfBeads)
    print('LLLL printed initial dfBeads structures to file dfbeads.xyz')
    # move the beadstring downhill, up to 'maxIter' times
    prevE = np.nan
    for i in range(maxIter):
        if __name__ == '__main__':
            # let each bead take one step
            Ravg = tidy_string_old(dfBeads) 
            print('RRRR Ravg = {:.2f}'.format(Ravg))
            Req = 0.8 * Ravg
            kforce = 1.5
            dX, dfBeads = step_string_old(ips_input, dfBeads, Req=Req, kforce=kforce)
            printXYZ_string('dfbeads{:d}.xyz'.format(i), dfBeads)
            print('LLLL after step, dfBeads structures to file dfbeads{:d}.xyz'.format(i))
        meanE = dfBeads['Erel'].mean()
        dE = prevE - meanE
        prevE = meanE
        print('"""""  iteration {:d} gives meanE = {:.1f} (dE = {:.1f}) kJ/mol with dX = {:.3f}'.format(i, meanE, -dE, dX))
        if (dX < dXconvg) and (dE < dEconvg):
            # convergence on step length and on energy change
            print('String relaxed to step length {:.3f} (dE = {:.1f} kJ/mol) after {:d} moves.'.format(dXconvg, dE, i))
            break
    else:
        print('*** String relaxation exceeded iteration limit of {:d}: dX = {:.3f}'.format(maxIter, dX))
    return dX, dfBeads
##
def relax_dfstring(ips_input, dfbeads, kforce, Req=1., etol=5, ends=False, curved=False):
    # Minimize the pseudo-energy of the DataFrame bead string 'dfbeads'
    #   The spring force is kept up-to-date for each bead individually
    # 'etol' is energy tolerance in kJ/mol
    # 'Req' is the equilibrium spring length, in absolute units
    # If 'ends' is False, eliminate the terminal springs
    # If 'curved' is True, use curved springs instead of straight
    maxiter = 20  # iterations of string
    print('Relaxing bead string: energy tol = {:g} kJ/mol'.format(etol))
    nbeads = dfbeads.shape[0]
    unitS = ips_input[ips_input['coordtype']].unitX()
    eprev = sprev = np.inf
    E0 = dfbeads.iloc[0]['Eabs']  # reactant energy
    for i in range(maxiter):
        # minimize each bead along its pseudo-energy gradient 
        # run them in parallel 
        print('IIIII iter = {:d} in relax_dfstring()'.format(i))
        #pdb.set_trace()
        smean, fmean = load_springs(dfbeads, kforce, Req, (ends,ends), curved)  # spring-only energy/gradient
        emean = dfbeads['Eabs'].mean()
        schange = (smean - sprev) * au_kjmol
        echange = (emean - eprev) * au_kjmol
        esudo = smean + emean
        if i > 0:
            print('RRRR <Espring> = {:.5f} Eh,  <E> = {:.5f} Eh, <Erel> = {:.1f} kJ/mol'.format(smean, emean, au_kjmol*(emean-E0)))
            if i > 1:
                print('RRRR <Espring> change = {:.1f} kJ/mol, <E> change = {:.1f} kJ/mol'.format(schange, echange))
        if abs(echange + schange) < etol:
            # mean string energy has converged
            break
        if __name__ == '__main__':
            tasks = []
            for ibead in range(1, nbeads-1):
                beads3 = dfbeads.iloc[ibead-1:ibead+2].copy()
                end_springs = (True, True)
                if not ends:
                    # possibly turn off a terminal spring
                    if ibead == 1:
                        end_springs = (False, True)
                    if ibead == nbeads-2:
                        end_springs = (True, False)
                tasks.append( (ips_input.copy(), beads3, kforce, Req, end_springs, curved, etol/au_kjmol, maxiter, ibead) )
            pool = multiprocessing.Pool(ips_input['nprocs'])
            results = [pool.apply_async(relax1beadG, t) for t in tasks]
            pool.close()
            pool.join()
            failure = False
            for result in results:
                (Emin, Xmin, ibead) = result.get()
                failure = failure or (Emin is np.nan)
                if failure:
                    raise ValueError('Failure to minimize string energy')
                # update the DataFrame
                lbl = dfbeads.index.values[ibead]  # probably lbl == ibead anyway
                dfbeads.loc[lbl, 'Eabs'] = Emin
                dfbeads.loc[lbl, 'Geom'].fromVector(Xmin, unitS=unitS)
                dfbeads.loc[lbl, 'Erel'] = au_kjmol * (Emin - ips_input['E0'])
        # re-align successive structures to minimize RMSD
        lbl = dfbeads.index.values
        for ibead in range(1, nbeads):
            dfbeads.loc[lbl[ibead], 'Geom'] = RMSD_align(dfbeads.loc[lbl[ibead], 'Geom'], dfbeads.loc[lbl[ibead-1], 'Geom'])  # reactant
        eprev = emean
        sprev = smean
    return i
##
def relax_dfstring0(ips_input, dfbeads, kforce, Req=1.):
    # Minimize the pseudo-energy of the DataFrame bead string 'dfbeads'
    #   The spring force is only computed at major iterations
    maxiter = 20
    ecrit = 1.  # conv. crit. for mean bead relative electronic energy (kJ/mol)
    gcrit = 1.0e-5  # conv. crit. for mean pseudo-force
    nbeads = dfbeads.shape[0]
    unitS = ips_input[ips_input['coordtype']].unitX()
    eprev = sprev = np.inf
    for i in range(maxiter):
        # minimize each bead along its pseudo-energy gradient 
        # run them in parallel 
        print('IIIII iter = {:d} in relax_dfstring0()'.format(i))
        smean, fmean = load_springs(dfbeads, kforce, Req)  # spring-only energy/force
        emean = dfbeads['Eabs'].mean()
        schange = (smean - sprev) * au_kjmol
        echange = (emean - eprev) * au_kjmol
        esudo = smean + emean
        print('RRRR <Espring> = {:.5f},  <E> = {:.5f}'.format(smean, emean))
        print('RRRR <Espring> change = {:.1f} kJ/mol, <E> change = {:.1f} kJ/mol'.format(schange, echange))
        if abs(echange + schange) < ecrit:
            # mean string energy has converged
            break
        if __name__ == '__main__':
            tasks = []
            for ibead in range(1, nbeads-1):
                tasks.append( (ips_input.copy(), dfbeads.iloc[[ibead]].squeeze(), ecrit/au_kjmol, gcrit, maxiter, ibead) )
            pool = multiprocessing.Pool(ips_input['nprocs'])
            results = [pool.apply_async(relax1bead0, t) for t in tasks]
            pool.close()
            pool.join()
            for result in results:
                (Emin, Xmin, ibead) = result.get()
                # update the DataFrame
                lbl = dfbeads.index.values[ibead]  # probably lbl == ibead
                dfbeads.loc[lbl, 'Eabs'] = Emin
                dfbeads.loc[lbl, 'Geom'].fromVector(Xmin, unitS=unitS)
                dfbeads.loc[lbl, 'Erel'] = au_kjmol * (Emin - ips_input['E0'])
        eprev = emean
        sprev = smean
    return i
##
def qm_grad(ips_input, X, unitR, ID):
    # return QM energy and gradient at cartesian coordinate vector X
    tmp_input = ips_input.copy()
    if tmp_input['coordtype'] != 'cartesian':
        # convert to cartesians
        tmp_input['cartesian'] = tmp_input[tmp_input['coordtype']].toGeometry()
        tmp_input['coordtype'] = 'cartesian'
    # install 'X' as the coordinates
    tmp_input['cartesian'].fromVector(X, unitS=unitR)
    E, G, ident = qm_function(tmp_input, 'gradient', option='flatten', unitR=unitR, ID=ID)
    return E, G
##
def qm_energy(X, ips_input, unitR, ID):
    # return QM energy at cartesian coordinate vector X
    tmp_input = ips_input.copy()
    if tmp_input['coordtype'] != 'cartesian':
        # convert to cartesians
        tmp_input['cartesian'] = tmp_input[tmp_input['coordtype']].toGeometry()
        tmp_input['coordtype'] = 'cartesian'
    # install 'X' as the coordinates
    tmp_input['cartesian'].fromVector(X, unitS=unitR)
    E, fout, ident = qm_function(tmp_input, 'energy', option='flatten', unitR=unitR, ID=ID)
    #print('QQQQ ID =', ID, ' E =', E, 'fout =', fout)
    return E
##
def relax1beadG(ips_input, beads3, kforce, Req, end_springs, curved, tol, maxiter, ID):
    # args: ips_input dict; pandas Series for one bead
    #   Move the bead to minimize its pseudo-energy,
    #   which is its real energy plus the spring force.
    #   Uses gradients.
    # 'beads3' is a DataFrame containing the active bead
    #   and its two neighbors. 
    # 'end_springs' should be a tuple
    A = beads3.iloc[1]['Geom'].toVector()  # initial position
    fileroot = '{:s}_relaxbead_{:s}'.format(ips_input['molecule'], str(ID))
    method = 'BFGS'
    options = {'maxiter': 20, 'disp': False}
    args = (ips_input, beads3, kforce, Req, end_springs, curved, ID, fileroot)
    try:
        result = sciopt.minimize(sudo_grad, A, args=args, method=method, jac=True, tol=tol, options=options)
    except:
        print('** failure minimizing bead', ID)
        return np.nan, A, ID
    #print('RRRR\n', result)
    Emin = result['fun']
    Xmin = result['x']
    return Emin, Xmin, ID
##
def relax1bead(ips_input, beads3, kforce, Req, tol, gtol, maxiter, ID):
    # args: ips_input dict; pandas Series for one bead
    #   Move the bead to minimize its pseudo-energy,
    #   which is its real energy plus the spring force,
    #   along the initial gradient of the pseudo-energy.
    #   The spring force/gradient is kept current.
    # 'beads3' is a DataFrame containing the active bead
    #   and its two neighbors.
    veclen = 0.1  # length of initial search vector
    Geom = beads3.iloc[1]['Geom']
    unitR = Geom.unitX()
    A = Geom.toVector()  # initial position
    E, G = qm_grad(ips_input, A, unitR=unitR, ID=ID)
    Espring, fspring = load_springs(beads3, kforce, Req)  # return values not needed
    Gsudo = G + beads3.iloc[1]['Gspring']
    gnorm = np.linalg.norm(Gsudo)
    if gnorm < gtol:
        # gradient is already small; do nothing
        print('RRRR bead ID = {:d} already has small gradient norm ({:.5f})'.format(ID, gnorm))
        return E, A, ID
    B = A - normalize(Gsudo, veclen)  # subtraction to move downhill
    Efunc = lambda X: qm_energy(X, ips_input, unitR=unitR, ID=ID) + spring_energy(X, beads3, kforce, Req)
    Xmin, Emin = mnbrent(Efunc, A, B, tol=tol, maxiter=maxiter)
    #print('RRRR  bead ID = {:d} has Emin = {:.5f} and gnorm = {:.5f}'.format(ID, Emin, gnorm))
    return Emin, Xmin, ID
##
def spring_energy(X, beads3, kforce, Req):
    # Install cartesian coordinates 'X' into the middle bead of the 
    #   DataFrame 'beads3'.
    # Return the spring energy
    unitS = beads3.iloc[1]['Geom'].unitX()
    beads3.iloc[1]['Geom'].fromVector(X, unitS=unitS)
    E, fmean = load_springs(beads3, kforce, Req)
    return E
##
def qmgrad_wrap(X, ips_input, ID, fileroot):
    # Wrapper for QM gradient calculation
    # X is a flattened vector of cartesian atomic coordinates
    #
    # insert X into the molecular structure
    unitS = ips_input['cartesian'].unitX()
    ips_input['cartesian'].fromVector(X, unitS=unitS)
    E, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', ID=ID, fileroot=fileroot)
    return E, grad
##
def sudo_grad(X, ips_input, beads3, kforce, Req, ends, curved, ID, fileroot):
    # sum of QM and spring energy/gradient
    # 'ends' should be a tuple
    Eq, Gq = qmgrad_wrap(X, ips_input, ID, fileroot)
    #
    unitS = beads3.iloc[1]['Geom'].unitX()
    beads3.iloc[1]['Geom'].fromVector(X, unitS=unitS)
    Es, fs = load_springs(beads3, kforce, Req, ends, curved)
    Gs = beads3.iloc[1]['Gspring']
    E = Eq + Es
    G = Gq + Gs
    return E, G
##
def relax1bead0(ips_input, beadSer, tol, gtol, maxiter, ID):
    # args: ips_input dict; pandas Series for one bead
    #   Move the bead to minimize its pseudo-energy,
    #   which is its real energy plus the spring force,
    #   along the initial gradient of the pseudo-energy.
    #   The spring force/gradient is static here.
    veclen = 0.1  # length of initial search vector
    Geom = beadSer['Geom']
    unitR = Geom.unitX()
    A = Geom.toVector()  # initial position
    E, G = qm_grad(ips_input, A, unitR=unitR, ID=ID)
    Gsudo = G + beadSer['Gspring']
    gnorm = np.linalg.norm(Gsudo)
    if gnorm < gtol:
        # gradient is small; do nothing
        print('RRRR bead ID = {:d} has small gradient'.format(ID))
        return E, A, ID
    B = A - normalize(Gsudo, veclen)  # subtraction to move downhill
    Efunc = lambda X: qm_energy(X, ips_input, unitR=unitR, ID=ID)
    Xmin, Emin = mnbrent(Efunc, A, B, tol=tol, maxiter=maxiter)
    print('RRRR  bead ID = {:d} has Emin = {:.5f} and gnorm = {:.5f}'.format(ID, Emin, gnorm))
    return Emin, Xmin, ID
##
def mnbrent(func, A, B, tol=1.0e-6, maxiter=30):
    # minimize multidimensional function along the A-B direction
    # Wrapper for scipy.optimize.brent()
    # does not use derivatives
    # args:
    #   'func': multidimensional function to investigate
    #   'A': the starting point
    #   'B': second point, also defines search direction
    #   'tol': convergence criterion on function value
    #   'maxiter': iteration cap
    # return (X, func(X)) at the minimum
    V = normalize(B-A)  # search direction
    fscalar = lambda x: func(A + x*V)
    b = distance(A, B)
    x, fx, niter, neval = sciopt.brent(fscalar, brack=(0., b), tol=tol, full_output=True, maxiter=maxiter)
    #print('MMMMM sciopt.brent() returned x, fx, niter, neval:', x, fx, niter, neval)
    X = A + x*V
    return X, fx
##
def fswitch(x, a):
    # switching function, used by load_springs() once upon a time
    #   'a' is a scale parameter
    if a == 0:
        # ignore the argument
        w = 0.5
    else:
        # keep s within [0, 1] for cosine
        s = max(0, x/a)
        s = min(s, 1)
        w = 0.5 * np.cos(np.pi * s)
    return w
##
def load_springs(dfbeads, kforce, Req, ends=(True,True), curved=False):
    # Install spring energies and gradients into 'dfbeads'
    # Termini are fixed
    # Forces are divided equally between sprung pairs of beads
    # return values:
    #   (1) mean spring energy
    #   (2) mean of spring force magnitudes
    # If 'ends' is False, eliminate terminal spring
    nbeads = dfbeads.shape[0]
    xnext = None
    xprev = dfbeads.iloc[0]['Geom'].toVector()  # reactant
    emean = 0   # mean spring energy
    fmean = 0   # mean spring force (norm)
    for ibead in range(1, nbeads-1):
        # do not move the terminal beads
        # spring force F = -k(x-Req) pointed toward neighbor
        # spring energy Es = 0.5*k*(x-Req)**2
        if xnext is None:
            x = dfbeads.iloc[ibead]['Geom'].toVector()  # current bead
        else:
            x = xnext # save a little time
        xnext = dfbeads.iloc[ibead+1]['Geom'].toVector()  # next bead in string
        if not curved:
            # linear springs
            # spring connected to previous bead
            d = distance(x, xprev)
            vspring = xprev - x  # force vector
            xeq = Req
            if (ibead == 1) and (not ends[0]):
                # no spring connected to reactant
                xeq = d
            force = kforce * (d - xeq)  # magnitude; can be negative
            Es = 0.5 * force * (d - xeq)  # non-negative energy
            # force on this bead
            forcing = force * normalize(vspring) / 2
            eforce = Es / 2
            # spring connected to following bead
            d = distance(x, xnext)
            vspring = xnext - x
            xeq = Req
            if (ibead == nbeads-1) and (not ends[1]):
                # no spring connected to product
                xeq = d
            force = kforce * (d - xeq)
            Es = 0.5 * force * (d - xeq) 
            forcing += force * normalize(vspring) / 2
            eforce += Es / 2
        else:
            # curved springs (but using linear distances)
            dprev = distance(x, xprev)
            vprev = normalize(xprev - x)
            dnext = distance(x, xnext)
            vnext = normalize(x - xnext)
            # force vector is same (but sign) for both springs (weighted average)
            vspring = (dnext * vprev + dprev * vnext) / (dprev + dnext)
            vspring = normalize(vspring)
            xeq = Req
            # spring to previous bead
            if (ibead == 1) and (not ends[0]):
                # no spring connected to reactant
                xeq = dprev
            force = kforce * (dprev - xeq)  # magnitude; can be negative
            Es = 0.5 * force * (dprev - xeq)  # non-negative energy
            # half force of this spring goes on this bead
            forcing = force * vspring / 2
            eforce = Es / 2
            # spring connected to following bead
            vspring = -vspring
            xeq = Req
            if (ibead == nbeads-1) and (not ends[1]):
                # no spring connected to product
                xeq = dnext
            force = kforce * (dnext - xeq)
            Es = 0.5 * force * (dnext - xeq) 
            # half force of this spring goes on this bead
            forcing += force * vspring / 2
            eforce += Es / 2
        lbl = dfbeads.index.values[ibead]  # get row label to permit cell assignment
        dfbeads.at[lbl, 'Espring'] = eforce
        dfbeads.at[lbl, 'Gspring'] = -forcing  # gradient is negative of force
        emean += eforce
        fmean += np.linalg.norm(forcing)
        xprev = x
    emean /= nbeads - 2  # no spring force or energy on termini
    fmean /= nbeads - 2
    return emean, fmean
##
def relax_string(ips_input, fexg, stepList, iprod, Product, Eprod, Eabs):
    # return a DataFrame of relaxed bead points
    #   work in cartesian coordinates
    # 'fexg' is the EXG (walker) file from which to extract points
    # 'stepList' lists the step numbers to included in the beadstring
    # 'Product' is the minimized product Geometry()
    dXconvg = 0.01  # rms bead displacement defining convergence
    dEconvg = 1.    # kJ/mol for mean energy change
    maxIter = 20  # patience limit
    #
    dfexg = pd.read_csv(fexg)  # the trajectory for one walker
    E0 = ips_input['E0']
    coordtype = ips_input['coordtype']
    Struct = ips_input[coordtype].copy()  # geometry template
    ndof = Struct.nDOF()
    unitS = ('angstrom', 'radian')  # assumption
    # initialize 'dfBeads'
    columns = ['Step', 'Geom', 'Erel', 'Eabs', 'Egrad', 'Espring', 'Gspring']
    nbead = len(stepList) + 2   # first bead is reactant, last is product
    dfBeads = pd.DataFrame(index=range(nbead), columns=columns)
    dfBeads['Step'] = [-9999] + stepList + [9999]
    if coordtype != 'cartesian':
        Reactant = ips_input['geom0'].toGeometry()
    else:
        Reactant = ips_input['geom0'].copy()
    # copy data from the trajectory into 'dfBeads'
    Gprev = None
    for istep in stepList:
        row = dfexg.loc[istep]
        Erel = row[0]
        X = row[1 : ndof+1]
        Struct.fromVector(X, unitS)
        if coordtype != 'cartesian':
            Geom = Struct.toGeometry()
            grad = np.ones_like(X) * np.nan   # don't attempt coordinate transformation
        else:
            # cartesian coordinates
            Geom = Struct.copy()
            grad = np.array(row[ndof+1 : 2*ndof+1 ])
        jrow = dfBeads[dfBeads['Step'] == istep].index.tolist()[0]
        # set elements one at a time to avoid 'ValueError' problem
        if Gprev is not None:
            # align with previous structure
            Geom = RMSD_align(Geom, Gprev)
        dfBeads.loc[jrow, 'Geom'] = Geom
        dfBeads.loc[jrow, 'Erel'] = Erel
        dfBeads.loc[jrow, 'Eabs'] = np.nan
        dfBeads.set_value(jrow, 'Egrad', grad)
        Gprev = Geom.copy()
        if istep == stepList[0]:
            # first regular bead; save structure
            G1 = Geom.copy()
    # first bead is the reactant, last bead is the product
    # align reactant with first regular bead
    Reactant = RMSD_align(Reactant, G1)
    dfBeads.loc[0] = [-9999, Reactant, 0., E0, np.zeros(ndof), 0., 0.]
    # align product with last regular structure
    Product = RMSD_align(Product, Gprev)
    dfBeads.loc[nbead-1] = [9999, Product, Eprod, Eabs, np.zeros(ndof), 0., 0.]
    printXYZ_string('dfbeads.xyz', dfBeads)
    print('LLLL printed initial dfBeads structures to file dfbeads.xyz')
    # move the beadstring downhill, up to 'maxIter' times
    kforce = 1.5
    Req = 0.8 # as multiple of Ravg
    prevE = np.nan
    for i in range(maxIter):
        Ravg = tidy_string_old(dfBeads, Nmin=10, Rmin=0.5) 
        print('RRRR Ravg = {:.2f} in dfBeads:\n'.format(Ravg), dfBeads)
        if __name__ == '__main__':
            dX, dfBeads = step_string(ips_input, dfBeads, Req=Req*Ravg, kforce=kforce)
        printXYZ_string('dfbeads{:d}.xyz'.format(i), dfBeads)
        print('LLLL after step, dfBeads structures to file dfbeads{:d}.xyz'.format(i))
        #sys.exit('stop beads')
    meanE = dfBeads['Erel'].mean()
    dE = prevE - meanE
    prevE = meanE
    print('"""""  iteration {:d} gives meanE = {:.1f} (dE = {:.1f}) kJ/mol with dX = {:.3f}'.format(i, meanE, -dE, dX))
    if (dX < dXconvg) and (dE < dEconvg):
        # convergence on step length and on energy change
        print('String relaxed to step length {:.3f} (dE = {:.1f} kJ/mol) after {:d} moves.'.format(dXconvg, dE, i))
        #break
    else:
        print('*** String relaxation exceeded iteration limit of {:d}: dX = {:.3f}'.format(maxIter, dX))
    return dX, dfBeads
##
def thin_string(dfBeads, Nmin=10, Ravg=0.5, ends=False):
    # change DataFrame dfBeads to adjust the number of beads
    # args: Nmin = minimum number of beads
    #       Ravg = desired average inter-bead distance
    # return: string length; end-end distance; number of beads
    # If 'ends' is False, don't include the end beads (reactant and
    #   product) when computing the distances.
    print('Nmin = {:d}, Ravg = {:.2f}'.format(Nmin, Ravg))
    Nbead = dfBeads.shape[0]
    print('There are {:d} beads before "thinning"'.format(Nbead))
    # compute length of string
    d2next = np.zeros(Nbead-1)  # distance of one bead to the following bead
    dspan = np.zeros(Nbead-1)  # indirect distance from (i-1)th bead to (i+1)th bead
    dspan[0] = np.inf  # protect the first bead from removal
    for i in range(Nbead-1):
        d = structure_distance(dfBeads.iloc[i]['Geom'], dfBeads.iloc[i+1]['Geom'])
        d2next[i] = d
        if i > 0:
            dspan[i] = d2next[i-1] + d
    print('DDDD initial d2next = [{:s}]'.format(' '.join(['{:.2f}'.format(d2) for d2 in d2next])))
    if ends:
        length = d2next.sum()
        lstraight = structure_distance(dfBeads.iloc[0]['Geom'], dfBeads.iloc[-1]['Geom'])  # end-end distance
    else:
        # exclude termini
        length = d2next[1:-1].sum()
        lstraight = structure_distance(dfBeads.iloc[1]['Geom'], dfBeads.iloc[-2]['Geom'])
        dspan[1] = dspan[-1] = np.inf  # protect the near-terminal beads from removal
        # don't allow interpolation adjacent to the reactant
        d2next[0] = -np.inf
        # don't allow interpolation adjacent to the product
        d2next[-1] = -np.inf
    Navg = int(length // Ravg)  # distance-based number of beads
    excess = Nbead - max(Navg, Nmin)
    if excess > 0:
        # remove some beads
        print('removing {:d} excess beads'.format(excess))
        irow = np.argsort(dspan)
        # delete some beads
        dfBeads.drop([irow[i] for i in range(excess)], inplace=True)
    elif excess < 0:
        # add some beads by interpolation; set energies and gradients to NaN
        nanvec = np.ones(dfBeads.iloc[0]['Geom'].nDOF()) * np.nan
        for ix in range(-excess):
            # keep bisecting the largest interval until there are enough points
            interv = d2next.argmax()
            newG = average_structure(dfBeads.iloc[interv]['Geom'], dfBeads.iloc[interv+1]['Geom'])
            stepno = (dfBeads.iloc[interv]['Step'] + dfBeads.iloc[interv+1]['Step']) / 2
            # patch up d2next[]
            newd = d2next[interv] / 2
            d2next[interv] = newd
            d2next = np.insert(d2next, interv, [newd])
            # add a new row to the DataFrame
            dfBeads.loc[-1-ix] = [stepno, newG, np.nan, np.nan, np.nan, nanvec]
            # order the beads along the reaction coordinate
            dfBeads.sort_values('Step', inplace=True)
            dfBeads.reset_index(drop=True, inplace=True)
        pass
    print('DDDD final d2next = [{:s}]'.format(' '.join(['{:.2f}'.format(d2) for d2 in d2next])))
    print('Final string has {:d} beads'.format(dfBeads.shape[0]))
    return length, lstraight, dfBeads.shape[0]
##
def no_reaction(dfbeads, bondtol):
    # Compare the connectivity of the first and last beads. 
    # Return True if they are the same, else False.
    # THIS IGNORES CONFORMATIONAL DIFFERENCES
    conn = same_connectivity(dfbeads.iloc[0]['Geom'], dfbeads.iloc[-1]['Geom'], bondtol)
    return conn
##
def step_string(ips_input, dfBeads, Req, kforce=0.01):
    # One step of string "energy" minimization
    #   terminal beads do not move
    #   internal beads are connected by "springs" to their two neighbors
    # Calculate energy and gradient at the new points (in parallel), 
    #   update 'dfBeads' and return it
    # Called by 'relax_string()'.
    # last two arguments:
    #   'Req'   : equilib. distance for springs
    #   'kforce': spring force constant in hartree/angstrom
    tmp_input = ips_input.copy()
    tmp_input['coordtype'] = 'cartesian'
    molec = tmp_input['molecule']
    tasks = []
    nbeads = dfBeads.shape[0]
    #imax = dfBeads['Erel'].argmax() 
    xprev = dfBeads.iloc[0]['Geom'].toVector()  # reactant
    xnext = None
    bigE = 0.  # sum of bead energies (including springs)
    bigX = []  # cartesians for all active beads together
    bigG = []  # gradient+forcing for all active beads together
    for ibead in range(1, nbeads-1):
        # do not move the terminal beads
        # spring force F = -k(x-Req) pointed toward neighbor
        # spring energy Es = 0.5*k*(x-Req)**2
        if xnext is None:
            x = dfBeads.iloc[ibead]['Geom'].toVector()  # current bead
        else:
            x = xnext.copy() # save a little time
        bigX.append(x)
        xnext = dfBeads.iloc[ibead+1]['Geom'].toVector()  # next bead in string
        # spring connected to previous bead
        d = distance(x, xprev)
        vspring = xprev - x  # vector
        force = kforce * (d - Req)  # can be negative
        Es = 0.5 * force * (d - Req)  # non-negative
        forcing = force * normalize(vspring)
        # spring connected to following bead
        d = distance(x, xnext)
        vspring = xnext - x
        force = kforce * (d - Req)
        forcing += force * normalize(vspring)
        Es += 0.5 * force * (d - Req) 
        grad = dfBeads.iloc[ibead]['Egrad']
        bigE += Es + dfBeads.iloc[ibead]['Eabs']
        bigG.append(grad - forcing)  # gradient = -force
        tmp_input['cartesian'] = dfBeads.iloc[ibead]['Geom'].copy()
        istep = dfBeads.iloc[ibead]['Step']
        Erel = dfBeads.iloc[ibead]['Erel']
        # energy/gradient computations run in parallel
        froot = '{:s}_bead_{:d}'.format(molec, istep)
        tasks.append( (tmp_input.copy(), 'gradient', False, '', 'angstrom', istep, froot) )
        xprev = x.copy()
    # run the tasks in parallel
    pool = multiprocessing.Pool(ips_input['nprocs'])
    results = [pool.apply_async(qm_function, t) for t in tasks]
    #results = [pool.apply_async(move_bead, t) for t in tasks]
    pool.close()
    pool.join()
    dX = []  # list of bead displacements 
    for result in results:
        (istep, Eabs, Erel, geom, grad, dx) = result.get()
        # set elements one at a time to avoid 'ValueError' problem
        jrow = dfBeads[dfBeads['Step'] == istep].index.tolist()[0]
        dfBeads.loc[jrow, 'Geom'] = geom
        dfBeads.loc[jrow, 'Erel'] = Erel
        dfBeads.loc[jrow, 'Eabs'] = Eabs
        dfBeads.set_value(jrow, 'Egrad', grad)
        dX.append(dx)
    # return the RMS displacement and the modified dataframe
    dX = np.linalg.norm(dX) / np.sqrt(nbeads)  # replace list with its RMS
    return dX, dfBeads
##
def tidy_string_old(dfBeads, Nmin=10, Rmin=1.):
    # change DataFrame dfBeads to redistribute beads, adjust their number, etc.
    # args: Nmin = minimum number of beads
    #       Rmin = minimum average inter-bead distance
    # return the average distance between beads, after adjustments
    print('Nmin = {:d}, Rmin = {:.2f}'.format(Nmin, Rmin))
    Nbead = dfBeads.shape[0]
    print('there are {:d} beads'.format(Nbead))
    # compute length of string
    length = 0
    d2next = np.zeros(Nbead-1)  # distance of one bead to the following bead
    dspan = np.zeros(Nbead-1)  # distance from (i-1)th bead to (i+1)th bead
    dspan[0] = np.inf  # the first bead must not be deleted
    for i in range(Nbead-1):
        d = structure_distance(dfBeads.iloc[i]['Geom'], dfBeads.iloc[i+1]['Geom'])
        d2next[i] = d
        if i > 0:
            dspan[i] = d2next[i-1] + d
    length = d2next.sum()
    Nmax = int(length // Rmin)  # maximum number of beads
    print('string length = {:.3f}, so Nmax = {:d}'.format(length, Nmax))
    print('d2next = ', ', '.join(['{:.2f}'.format(d) for d in d2next]))
    print('dspan = ', ', '.join(['{:.2f}'.format(d) for d in dspan]))
    excess = Nbead - max(Nmax, Nmin)
    irow = np.argsort(dspan)
    print('QQQ irow = ', irow)
    # delete some beads
    dfBeads.drop([irow[i] for i in range(excess)], inplace=True)
    ravg = length/dfBeads.shape[0]  # new average inter-bead distance
    return ravg
##
def step_string_old(ips_input, dfBeads, Req, kforce=0.08):
    # Use 'multiprocessing' to move each bead one step downhill
    #   terminal beads do not move
    #   internal beads are connected by "springs" to their two neighbors
    # Calculate energy and gradient at the new points, update 'dfBeads' and 
    #   return it
    # Called by 'relax_string_old()'.
    # last two arguments:
    #   'Req'   : equilib. distance for springs
    #   'kforce': spring force constant in hartree/angstrom
    tmp_input = ips_input.copy()
    tmp_input['coordtype'] = 'cartesian'
    tasks = []
    nbeads = dfBeads.shape[0]
    # Force constant; F = -k(x-Ravg) 
    #imax = dfBeads['Erel'].argmax() 
    xprev = dfBeads.iloc[0]['Geom'].toVector()  # reactant
    xnext = None
    for ibead in range(1, nbeads-1):
        # exclude terminal beads
        if xnext is None:
            x = dfBeads.iloc[ibead]['Geom'].toVector()  # current bead
        else:
            x = xnext.copy()
        xnext = dfBeads.iloc[ibead+1]['Geom'].toVector()  # next bead in string
        # spring connected to previous bead
        d = distance(x, xprev)
        vspring = xprev - x  # vector
        force = kforce * (d - Req)
        forcing = force * normalize(vspring)
        # spring connected to following bead
        d = distance(x, xnext)
        vspring = xnext - x
        force = kforce * (d - Req)
        forcing += force * normalize(vspring)
        grad = dfBeads.iloc[ibead]['grad'].copy()
        tmp_input['cartesian'] = dfBeads.iloc[ibead]['Geom'].copy()
        istep = dfBeads.iloc[ibead]['Step']
        Erel = dfBeads.iloc[ibead]['Erel']
        #tasks.append( (tmp_input.copy(), istep, grad, vperp, Erel) )  # for move_bead1()
        tasks.append( (tmp_input.copy(), istep, forcing, grad, Erel) )
        xprev = x.copy()
    # run the tasks in parallel
    pool = multiprocessing.Pool(ips_input['nprocs'])
    #results = [pool.apply_async(move_bead1, t) for t in tasks]
    results = [pool.apply_async(move_bead, t) for t in tasks]
    pool.close()
    pool.join()
    dX = []  # list of bead displacements 
    for result in results:
        (istep, Eabs, Erel, geom, grad, dx) = result.get()
        # set elements one at a time to avoid 'ValueError' problem
        jrow = dfBeads[dfBeads['Step'] == istep].index.tolist()[0]
        dfBeads.loc[jrow, 'Geom'] = geom
        dfBeads.loc[jrow, 'Erel'] = Erel
        dfBeads.loc[jrow, 'Eabs'] = Eabs
        dfBeads.set_value(jrow, 'grad', grad)
        dX.append(dx)
    # return the RMS displacement and the modified dataframe
    dX = np.linalg.norm(dX) / np.sqrt(nbeads)  # replace list with its RMS
    return dX, dfBeads
##
def move_bead(ips_input, istep, forcing, grad, Erel):
    # move one bead (called by step_string_old() in parallel)
    # this algorithm has an artificial force to pull it toward its neighbors
    # return many things
    maxStep = 0.2
    stepScale = 0.5  # for multiplying the force
    molec = ips_input['molecule']
    froot = '{:s}_bead_{:d}'.format(molec, istep)
    if pd.isnull(grad).any():
        # must compute the initial forces
        Eabs, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR='angstrom', ID=istep, fileroot=froot)
        Erel = (Eabs - ips_input['E0']) * au_kjmol
    # change sign to point downhill
    grad *= -1
    # add spring effects
    grad = grad + forcing
    if False:
        # merely scale, to get a displacement vector
        dx = stepScale
    else:
        # energy-derived step size
        eDown = (Erel + ips_input['tolerance'])
        eDown /= au_kjmol  # convert to hartree
        dx = eDown / np.linalg.norm(grad)
    vStep = dx * grad
    dx = np.linalg.norm(vStep)
    #print('MMM dx = {:.3f} for istep = {:d}'.format(dx, istep))
    if dx > maxStep:
        # proposed step is too large; shrink to 'maxStep'
        print('......  dx = {:f} is too big in move_bead()'.format(dx))
        vStep = normalize(vStep, maxStep)
        dx = maxStep
    # displace the point
    Geom = ips_input['cartesian']
    Geom.fromVector(vStep, unitS=('angstrom',), add=True)
    ips_input['cartesian'] = Geom
    # compute energy and gradient at the new position
    Eabs, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR='angstrom', ID=istep, fileroot=froot)
    Erel = (Eabs - ips_input['E0']) * au_kjmol
    return istep, Eabs, Erel, Geom, grad, dx
##
def move_bead1(ips_input, istep, grad, vPerp, Erel):
    # move one bead (called by step_string_old() in parallel)
    # this algorithm will only move perpendicular to 'vPerp'
    # return many things
    maxStep = 0.1
    molec = ips_input['molecule']
    froot = '{:s}_bead_{:d}'.format(molec, istep)
    if pd.isnull(grad).any():
        # must compute the initial forces
        Eabs, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR='angstrom', ID=istep, fileroot=froot)
        Erel = (Eabs - ips_input['E0']) * au_kjmol
    # change sign to point downhill
    grad *= -1
    # project out any component of 'grad' that lies along 'vPerp'
    vPerp = normalize(vPerp)
    grad = grad - np.dot(grad, vPerp) * vPerp
    # get a displacement vector
    eDown = (Erel + ips_input['tolerance']) # arithmetic mean
    eDown /= au_kjmol  # hartree
    dx = eDown / np.linalg.norm(grad)
    vStep = dx * grad
    dx = np.linalg.norm(vStep)
    if dx > maxStep:
        # proposed step is too large; shrink to 'maxStep'
        #print('......  dx = {:f} is too big in move_bead()'.format(dx))
        vStep = normalize(vStep, maxStep)
        dx = maxStep
    # displace the point
    Geom = ips_input['cartesian']
    Geom.fromVector(vStep, unitS=('angstrom',), add=True)
    ips_input['cartesian'] = Geom
    # compute energy and gradient at the new position
    Eabs, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR='angstrom', ID=istep, fileroot=froot)
    Erel = (Eabs - ips_input['E0']) * au_kjmol
    return istep, Eabs, Erel, Geom, grad, dx
##
def get_reference_structure():
    # using globals
    # was the molecule optimized at the beginning of the IPS calculation?
    minim = False
    if coordtype == 'cartesian':
        # look for optimized geometry XYZ file
        fmin = '{:s}_E0_geometry.xyz'.format(molec)
        if os.path.isfile(fmin):
            # found it; use the optimized geometry as the origin of coordinates
            print('Reading optimized geometry from file {:s}'.format(fmin))
            initCoord, natom, comment = readXmol(fmin)
            minim = True
    elif coordtype == 'zmatrix':
        # read optimized geometry from quantum chemistry output file
        fmin = supply_qm_filename_suffix(code, '{:s}_E0_geometry'.format(molec), 'output')
        if os.path.isfile(fmin):
            print('Reading optimized geometry from file {:s}'.format(fmin))
            initCoord = read_qm_ZMatrix(code, fmin)[-1]  # want the last geometry
            minim = True
    if minim:
        initConn = initCoord.connection_table(tol=bondtol)
        # check for the unlikely event that the connectivity changed during the geometry optimization
        if not np.array_equal(inputConn, initConn):
            print('*** Surprise!  The connectivity changed during the geometry optimization! ***')
            print('The input structure will be ignored during this analysis.')
    else:
        # no geometry optimization; use the input geometry as the origin
        initCoord = inputCoord.copy()
        initConn = inputConn
    if coordtype == 'zmatrix':
        initCoord.toRadian()
    return minim, initCoord, initConn 
##
def build_string(ips_input, fexg, stepList, Reactant, Product, Eprod, Eabs):
    # return a DataFrame of bead points
    #   work in cartesian coordinates
    # 'fexg' is the EXG (walker) file from which to extract points
    # 'stepList' lists the step numbers to included in the beadstring
    # 'Reactant' will be the first bead in the string
    # 'Product' is the minimized product Geometry() (last bead)
    #
    dfexg = pd.read_csv(fexg)  # the trajectory for one walker
    E0 = ips_input['E0']
    coordtype = ips_input['coordtype']
    Struct = ips_input[coordtype].copy()  # geometry template
    ndof = Struct.nDOF()
    nulvec = np.zeros(ndof)
    nanvec = nulvec + np.nan
    unitS = ('angstrom', 'radian')  # assumption
    # initialize 'dfBeads'
    columns = ['Step', 'Geom', 'Erel', 'Eabs', 'Espring', 'Gspring']
    nbead = len(stepList) + 2   # first bead is reactant, last is product
    dfBeads = pd.DataFrame(index=range(nbead), columns=columns)
    dfBeads['Step'] = [-9999.] + stepList + [9999.]  # make float in case of interpolation
    if Reactant.coordtype != 'Geometry':
        Reactant = Reactant.toGeometry()
    # copy data from the trajectory into 'dfBeads'
    Gprev = None
    for istep in stepList:
        row = dfexg.loc[istep]
        Erel = row[0]
        X = row[1 : ndof+1]
        # ignore the energy gradient
        Struct.fromVector(X, unitS)
        if coordtype != 'cartesian':
            Geom = Struct.toGeometry()
        else:
            # cartesian coordinates
            Geom = Struct.copy()
        jrow = dfBeads[dfBeads['Step'] == istep].index.tolist()[0]
        # set elements one at a time to avoid 'ValueError' problem
        if Gprev is not None:
            # align with previous structure
            Geom = RMSD_align(Geom, Gprev)
        dfBeads.at[jrow, 'Geom'] = Geom
        dfBeads.at[jrow, 'Erel'] = Erel
        dfBeads.at[jrow, 'Eabs'] = Erel/au_kjmol + E0
        dfBeads.at[jrow, 'Gspring'] = nanvec
        Gprev = Geom.copy()
        if istep == stepList[0]:
            # first regular bead; save structure
            G1 = Geom.copy()
    # first bead is the reactant, last bead is the product
    # align reactant with first regular bead
    Reactant = RMSD_align(Reactant, G1)
    dfBeads.loc[0] = [-9999., Reactant, 0., E0, 0., nulvec]
    # align product with last regular structure
    Prod = RMSD_align(Product, Gprev)
    dfBeads.loc[nbead-1] = [9999., Prod, Eprod, Eabs, 0., nulvec]
    printXYZ_string('dfbeads.xyz', dfBeads)
    print('LLLL printed initial dfBeads structures to file dfbeads.xyz')
    return dfBeads
##
def create_BeadString(ips_input, fexg, stepList, Product, Eabs):
    # return a BeadString() object
    # 'fexg' is the EXG (walker) file from which to extract points
    # 'stepList' lists the step numbers to included in the beadstring
    # 'Product' is the minimized product Geometry()
    # Structures are re-oriented to minimize RMSD
    # Input may be in ZMatrix() form
    #
    dfexg = pd.read_csv(fexg)  # the trajectory for one walker
    E0 = ips_input['E0']
    coordtype = ips_input['coordtype']
    Struct = ips_input[coordtype].copy()  # geometry template
    ndof = Struct.nDOF()
    natom = Struct.natom()
    unitS = Struct.unitX()  # assumption about EXG file
    nbead = len(stepList) + 2   # first bead is reactant, last is product
    # initialize E, X, G with NaN
    Xmat = np.ones((nbead, natom*3)) * np.nan 
    Evec = np.ones(nbead) * np.nan
    Evec[ 0] = E0
    Evec[-1] = Eabs
    Gmat = np.ones((nbead, natom*3)) * np.nan
    Gmat[ 0,:] = np.zeros(natom*3)  # first bead will never be moved
    Gmat[-1,:] = np.zeros(natom*3)  # last bead will never be moved
    # Reactant (first bead)
    if coordtype != 'cartesian':
        Reactant = ips_input['geom0'].toGeometry()
    else:
        Reactant = ips_input['geom0'].copy()
    Geomlist = [np.nan] * nbead  # geometry buffer, used for RMSD alignment
    # copy data from the trajectory into arrays
    Gprev = None
    ibead = 1
    for istep in stepList:
        row = dfexg.loc[istep]
        Erel = row[0]
        Eabs = Erel/au_kjmol + E0
        X = row[1 : ndof+1]
        Evec[ibead] = Eabs
        Struct.fromVector(X, unitS)
        if coordtype != 'cartesian':
            # don't attempt coordinate transformation on gradient
            Geom = Struct.toGeometry()
        else:
            # cartesian coordinates
            Geom = Struct.copy()
            grad = np.array(row[ndof+1 : 2*ndof+1 ])
            Gmat[ibead,:] = grad.copy()
        if Gprev is not None:
            # align with previous structure
            Geom = RMSD_align(Geom, Gprev)
        Geomlist[ibead] = Geom
        Gprev = Geom.copy()
        if istep == stepList[0]:
            # first regular bead; save structure for later alignment
            G1 = Geom.copy()
        ibead += 1
    # first bead is the reactant, last bead is the product
    # align reactant with first regular bead
    Reactant = RMSD_align(Reactant, G1)
    # align product with last regular structure
    Product = RMSD_align(Product, Gprev)
    Geomlist[0] = Reactant
    Geomlist[-1] = Product
    # install coordinates in Xmat
    for igeom in range(nbead):
        Xmat[igeom,:] = Geomlist[igeom].toVector()
    String = BeadString(Product, Xmat, Evec, Gmat)
    return String 
##
def ldihediff(Geom0, Geom1, bondtol=1.3, unit='radian', norm=True, methyl=False):
    # Compute differences in dihedral angles
    # The structures should be the same molecule, with
    #   the atoms numbered the same way.
    # If 'norm' == False, then return a list, where
    #   each list element is a tuple:
    #       (i, j, k, l), difference
    # If 'norm' == True, return a single scalar,
    #   the sqrt of sum of squares of differences.
    # If 'methyl' is False, then exclude all methyl
    #   torsions from consideration.
    a0 = Geom0.simple_dihedrals(bondtol, unit)
    a1 = Geom1.simple_dihedrals(bondtol, unit)
    dlist = []
    vals = []
    methylist = []
    if not methyl:
        methylist = Geom0.find_methyls(bondtol)
    for dihe in a0:
        ijkl = dihe[0]
        ang = dihe[1]
        ismethyl = False
        for dih1 in a1:
            if dih1[0] == ijkl:
                # same dihedral
                if not methyl:
                    # is this a methyl torsion?
                    jk = ijkl[1:3]
                    for abcd in methylist:
                        # 'abcd' is tuple of atom numbers: (C, H, H, H)
                        c = abcd[0]
                        if c in jk:
                            # yes, one of the axis atoms is a methyl carbon
                            ismethyl = True
                            break  # from 'abcd' loop
                    if ismethyl:
                        # ignore this dihedral
                        break  # from 'dih1' loop
                # take difference
                dang = dih1[1] - ang
                dlist.append( (ijkl, dang) )
                vals.append(dang)
                break
    if norm:
        return np.linalg.norm(vals)
    else:
        return dlist
##
def minmax_bondtol(Struct, conn, initol=1.3, resol=0.02, direction='min'):
    # return the the maximum (or minimum) bond tolerance that preserves the 
    #   connectivity matrix
    maxlim = 10.  # do not exceed
    minlim = 0.8  # do not go below
    retval = initol
    if direction == 'max':
        lim = maxlim
        step = resol
    else:
        lim = minlim
        step = -resol
    for btol in np.arange(initol+step, lim+step, step):
        newconn = Struct.connection_table(tol=btol)
        if np.array_equal(conn, newconn):
            retval = btol
        else:
            break
    return retval
##
def find_precursor(iwalker, stepList, dfuniq, dfminim, bondtol, nfrag0):
    # Return the stable structure that most closely precedes
    #   the steps 'stepList' taken by walker 'iwalker'. 
    # Ignore stable structures that are in more fragments than the 
    #   initial structure
    # If none, then return None
    step1 = min(stepList)
    # build a dataframe for this walker's preceding unoptimized structures
    dfpre = pd.DataFrame(columns=['step', 'Found'])
    for iuniq, row in dfuniq.iterrows():
        for pair in row['Found']:
            # pair[0] is walker# and pair[2] is step# where this connectivity was found
            if pair[0] != iwalker:
                continue
            # this is the right walker
            if (pair[1] <= step1):
                # add this structure to 'dfpre'
                dfpre.loc[dfpre.shape[0]] = [pair[1], row['Found']]
                Found = row['Found']  # list of (walker, step) pairs for this structure
    if dfpre.shape[0] == 0:
        # nothing was found
        return None
    # sort by decreasing step number (for the present walker)
    dfpre.sort_values(by='step', ascending=False, inplace=True)
    print('FFFF dfpre:\n', dfpre)
    # Find minima in 'dfminim' that correspond to these structures
    #   they may be labeled using other walkers
    for iuniq, uniq in dfpre.iterrows():
        Found = uniq['Found']
        for irow, row in dfminim.iterrows():
            pair = (row['walker'], row['step'])
            if pair in Found:
                # this is a match; check number of fragments
                print('FFFF found match for iuniq =', iuniq, 'with pair =', pair)
                nfrag = len(row['Struct'].find_fragments(bondtol))
                if nfrag == nfrag0:
                    # same number of fragments as starting structure
                    return row['Struct']
        print('FFFF found no matches for iuniq =', iuniq, ' Last pair =', pair)
    # if we got here, nothing was found
    return None
##
def gauss_func(x, A, mu, sigma, B):
    # used for peak-finding in extract_string_peak()
    y = (x - mu)/sigma
    return A * np.exp(-y*y/2) + B
##
def extract_string_peak(dfbeads, nsigma=4):
    # Fit the energy profile to a single Gaussian 
    # Discard the string beads that are far from the peak
    # Return the shortened DataFrame
    nbead0 = dfbeads.shape[0]
    x = list(range(nbead0))
    evec = dfbeads['Erel'].values.astype('float64')
    # intial guess for fitting parameters
    mu = np.argmax(evec)
    A = evec[mu]
    sigma = len(evec)/5
    B = 0
    # fit
    try:
        [A, mu, sigma, B], pcov = sciopt.curve_fit(gauss_func, x, evec, p0=[A,mu,sigma,B])
    except:
        # give up, just return the input DataFrame
        return dfbeads
    # select a range, not less that 20% of the orginal range
    llim = int(max(0, mu-nsigma*sigma))
    ulim = int(min(nbead0-1, mu+nsigma*sigma))
    # don't discard more than 80% of the initial range
    shortage = (nbead0 // 5) - (ulim - llim + 1)
    if shortage > 0:
        ulim += shortage // 2
        llim -= shortage // 2
        if ulim > nbead0-1:
            # move range to the left
            xs = ulim - (nbead0-1)
            ulim = nbead0-1
            llim -= xs
        if llim < 0:
            # move range to the right
            xs = -llim
            llim = 0
            ulim = min(nbead0-1, ulim + xs)
    return dfbeads.iloc[llim:ulim+1]
##
## MAIN PROGRAM
##
try:
    # read the IPS input file
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        # maybe user did not include the file suffix
        input_file += '.yml'
    print('Reading input file "{:s}"'.format(input_file))
    ips_input = read_yinput(input_file, DEFAULT_CONFIG_FILE)
    ips_input = parse_yinput(ips_input)
    print_dict(ips_input)
except Exception as ex:
    print(ex)
    sys.exit('Usage: ips_trajanal.py <name of IPS input file> <commands>')
# parameter for deciding whether atoms are 'bonded'
bondtol = ips_input['bondtol']
#
molec = ips_input['molecule']
coordtype = ips_input['coordtype']
code = ips_input['code']
inputCoord = ips_input[coordtype].copy()
inputConn = inputCoord.connection_table(tol=bondtol)
# was the molecule optimized at the beginning of the IPS calculation?
minim, initCoord, initConn = get_reference_structure()
# save the reference geometry in the 'ips_input' dictionary
ips_input['geom0'] = initCoord
nfrag0 = initCoord.find_fragments(bondtol)
nfrag0 = len(nfrag0)  # the number of fragments in the initial structure
# read the reference energy from the text file
with open('{:s}_E0.txt'.format(molec), 'r') as f0:
    t = f0.readline().split()
ips_input['E0'] = float(t[0])
##
# make lists of trajectory files inside a DataFrame,
#   then scan trajectories for changes in connectivity
fpkl = '{:s}_dfuniq.pkl'.format(molec)
fisom = '{:s}_isom.xyz'.format(molec)
print()
try:
    # have unique structures been identified in an earlier run?
    dfuniq = pd.read_pickle(fpkl)
    niso = dfuniq.shape[0] - 1
    print('{:d} unique structures (besides reactant) were read from file {:s}'.format(niso, fpkl))
    if not ( os.path.isfile(fisom) and (os.path.getsize(fisom) > 0) ):
        # the structures file is missing; re-create it
        print('--re-creating structure file {:s}'.format(fisom))
        #print(dfuniq)
        with open(fisom, 'w') as FOUT:
            for irow, row in dfuniq.iterrows():
                if irow == 0:
                    # skip the reactant; no structure is present anyway
                    continue
                (iwalker, istep) = row['Found'][0]
                Erel = row['Erel']
                #FOUT.write(row['Struct'].XmolXYZ('Isomer {:d} found by walker {:d} at step {:d}; Erel = {:.1f} kJ/mol'.format(irow, iwalker, istep, Erel)))
                FOUT.write(row['Struct'].XmolXYZ('Isomer {:d} found by walker {:d} at step {:d}'.format(irow, iwalker, istep)))
    print('\tsee file {:s} for their structures.'.format(fisom))
except:
    # identify unique structures (create 'dfuniq')
    dfuniq = pd.DataFrame(columns=('Conn', 'X', 'G', 'Found', 'Erel', 'Struct', 'bondtol'))
    # first row is the initial structure, included temporarily in order to
    #   have its connectivity matrix in the comparisons
    X = initCoord.toVector()
    dfuniq.loc[0] = [initConn.copy(), X, np.zeros_like(X), [(-1,-1)], 0., Geometry(), bondtol]
    exglist = glob.glob('{:s}_*exg.csv'.format(molec))
    print('Found {:d} EXG files to examine.'.format(len(exglist)))
    ndof = initCoord.nDOF()
    unitS = initCoord.unitX()
    FOUT = open(fisom, 'w')   # for XYZ structures of uniquely-connected points from the trajectories
    if False:
        # write reactant geometry to isomers XYZ file (why??)
        if minim:
            FOUT.write(initCoord.XmolXYZ('Optimized geometry of {:s}'.format(molec)))
        else:
            FOUT.write(initCoord.XmolXYZ('Input structure for {:s}'.format(molec)))
    #
    # read each EXG in turn
    #
    nconn0 = initConn.sum()  # number of connections in the reactant
    ilabel = 1
    for f in sorted(exglist):
        dfexg = pd.read_csv(f)
        print('\t{:d} points from file {:s}'.format(dfexg.shape[0], f))
        iwalker = walker_number(f)
        # for each point, construct connectivity matrices and compare them
        istep = 0
        for row in dfexg.values.tolist():
            Erel = row[0]
            X = row[1 : ndof+1]
            G = row[ndof+1 : 2*ndof+1]
            newCoord = initCoord.copy()
            newCoord.fromVector(X, unitS)
            conn = newCoord.connection_table(tol=bondtol)
            # check whether this is a new connectivity pattern 
            iconn = find_conn(conn, dfuniq['Conn'].tolist())
            if iconn == -1:
                # a new structure; add it to 'dfuniq'
                dfuniq.loc[ilabel] = [conn, X, G, [(iwalker,istep)], Erel, newCoord.copy(), bondtol]
                # print its cartesians to the XYZ output file
                FOUT.write(newCoord.XmolXYZ('Isomer {:d} found by walker {:d} at step {:d}'.format(ilabel, iwalker, istep)))
                ilabel += 1
            else:
                # a known structure; first time found by this walker?
                first = True
                for pair in dfuniq.loc[iconn]['Found']:
                    if iwalker == pair[0]:
                        # already found by this walker; don't record it again
                        first = False
                if first:
                    # record that this walker also found this structure 
                    dfuniq.loc[iconn]['Found'].append( (iwalker,istep) )
                # Is this a "better" structure for this isomer? 
                nconn = conn.sum()
                oldtol = dfuniq.loc[iconn, 'bondtol']
                update = False
                if nconn < nconn0:
                    # connections have been severed; find max bondtol that preserves connectivity
                    newtol = minmax_bondtol(newCoord, conn, oldtol, 0.02, 'max')
                    if newtol > oldtol:
                        # yes; update this row of 'dfuniq'
                        update = True
                else:
                    # connections have been added or no net change; find min bondtol
                    newtol = minmax_bondtol(newCoord, conn, oldtol, 0.02, 'min')
                    if newtol < oldtol:
                        # yes; update this row of 'dfuniq'
                        update = True
                if update:
                    foundlist = dfuniq.at[iconn, 'Found']
                    dfuniq.loc[iconn] = [conn, X, G, foundlist, Erel, newCoord.copy(), newtol]
            istep += 1
    FOUT.close()
    niso = dfuniq.shape[0] - 1
    print('Found {:d} unique connection tables, excluding reactant.'.format(niso))
    print('The corresponding geometries are written to {:s}'.format(fisom))
    dfuniq.to_pickle(fpkl)
if niso < 1:
    sys.exit('Nothing interesting!')
print('Isomer\t#walkers\tLabel (walker, step)')
for i in range(1, niso+1):
    lbl = dfuniq.loc[i]['Found']
    print('{:6d}\t{:6d}\t\t{:s}'.format(i, len(lbl), str(lbl[0])))
#
# when the molecule appears to be dissociating, optimize the fragments
#
fpkl = '{:s}_dffrag.pkl'.format(molec)
print()
try:
    dffrag = pd.read_pickle(fpkl)
    print('Optimized fragments read from file {:s}'.format(fpkl))
    print('See files "{:s}_<walker>_<step>_frag<n>.xyz" for their structures.'.format(molec))
except:
    print('For incipient fragmentation reactions, optimizing separated fragments.')
    #spread_dist = ips_input['spread_dist']['value']
    # make DataFrame of fragments and corresponding supermolecules
    dffrag = pd.DataFrame(columns=('Isomer', 'Label', 'Erel', 'Eabs', 'Fragments', 'Supermol'))
    # parallel computation
    if __name__ == '__main__':
        partasks = []
        labels = [None]*(niso+1)
        for i in range(1, niso+1):
            lbl = dfuniq.loc[i]['Found'][0]  # (walker, step) tuple
            labels[i] = lbl
            frag_input = ips_input.copy()
            frag_input[coordtype] = dfuniq.loc[i]['Struct']
            froot = '{:s}_{:d}_{:d}'.format(ips_input['molecule'], lbl[0], lbl[1])
            partasks.append( (frag_input.copy(), froot, i) )
        pool = multiprocessing.Pool(ips_input['nprocs'])
        results = [pool.apply_async(spread_opt, t) for t in partasks]
        pool.close()
        pool.join()
        ifrag = 0
        for result in results:
            (Efrag, Gfrag, superGeom, isomer) = result.get()
            if Efrag is None:
                # no fragmentation
                continue
            Erel = au_kjmol * (Efrag.sum() - ips_input['E0'])
            dffrag.loc[ifrag] = [isomer, labels[isomer], Erel, Efrag, Gfrag, superGeom]
            ifrag += 1
    dffrag.to_pickle(fpkl)
if dffrag.shape[0] > 0:
    # there was some fragmentation
    print('Isomer\tLabel\t\tErel (kJ/mol)\tCharges\tFormulas')
    for i, row in dffrag.iterrows():
        # save fragment structures to XYZ files; assemble charges and formulas
        ffrag = []
        qfrag = []
        ifrag = 0
        for G in row['Fragments']:
            fname = '{:s}_{:d}_{:d}_frag{:d}.xyz'.format(ips_input['molecule'], row['Label'][0], row['Label'][1], ifrag)
            ffrag.append(G.stoichiometry())
            qfrag.append(G.charge)
            G.printXYZ(fname, 'Fragment {:d} of isomer #{:d} {:s}'.format(ifrag, row['Isomer'], str(row['Label'])))
            ifrag += 1
        print('{:d}\t{:15s}\t{:.1f}\t\t{:s}\t{:s}'.format(row['Isomer'], str(row['Label']), row['Erel'], str(qfrag), str(ffrag)))
else:
    print('No fragmentation was observed.')
##
#
# Do geometry optimization on possible products
#    Some structures may be changed from ZMatrix() to Geometry()
#
fpkl = '{:s}_dfminim.pkl'.format(molec)
try:
    dfminim = pd.read_pickle(fpkl)
    nminim = dfminim.shape[0]
    print('{:d} optimized isomers read from file {:s}'.format(nminim, fpkl))
    print('See files "{:s}_minim_<walker>_<step>.xyz" for their structures.'.format(molec))
except:
    if ips_input['verbose']:
        print('Refining geometries of apparent reaction products')
    frag_sep = 5.  # to inhibit undesired rebonding during geometry optimization
    if __name__ == '__main__':
        dfminim = optimize_parallel(ips_input, dfuniq['Found'].tolist()[1:].copy(), dfuniq['Struct'].tolist()[1:].copy(), frag_sep)
    # write optimized structures to XYZ files; add filenames to DataFrame
    nminim = 0
    xyzfiles = {}
    fnames = []
    for i, row in dfminim.iterrows():
        if not row['success']:
            # geometry optimization failed
            continue
        fname = '{:s}_minim_{:.0f}_{:.0f}.xyz'.format(molec, row['walker'], row['step'])
        fnames.append(fname)
        txt = 'Minimized isomer from {:s}; Erel = {:.1f} kJ/mol; label = ({:d}, {:d})'.format(molec,
            row['Erel'], row['walker'], row['step'])
        xyzfiles[i] = fname  # use dict because index has gaps for failed geometry optimizations
        FXYZ = open(fname, 'w')
        FXYZ.write(row['Struct'].XmolXYZ(txt))
        FXYZ.close()
        nminim += 1
    dfminim['xyzfile'] = pd.Series(fnames, index=dfminim.index)
    dfminim.to_pickle(fpkl)
    print('{:d} Minimized structures written to XYZ files'.format(nminim))
print('SSSSSSSSS dfminim:\n', dfminim[['Erel', 'walker', 'step', 'xyzfile']])
#
# identify unique, energy-minimized structures ('dfminim')
#
# drop the reactant from DataFrame 'dfuniq'
dfuniq.drop(dfuniq.index[0], inplace=True)
# After minimization, most of the structures will probably revert to the reactant.
#   Identify those that are differently bonded.
dfproduct = pd.DataFrame(columns=('product', 'Eabs', 'Erel', 'isomer', 'Found', 'Times-found', 'xyzfile', 'connectivity'))
# In 'dfproduct', the 'Found' field will be copied from 'dfuniq', and 'isomer' will be the index
#   for that structure (before geometry optimization) in 'dfuniq'
iproduct = 1  # product number (numbers starting with 1)
connList = [initConn]   # reactant is first in list
for i, row in dfminim.iterrows():
    if not row['success']:
        # geometry optimization failed
        continue
    else:
        # successful geometry optimization
        conn = row['Struct'].connection_table(tol=bondtol)
    matchrow = find_conn(conn, connList)
    if (len(conn) > 0):
        if matchrow == -1:
            # a unique connection table--add to 'connList' and to 'dfproduct'
            connList.append(conn)
            if False:
                ## results in some confusing (although equivalent) choices of xyz files
                # find the isomer number and the 'Found' list in 'dfuniq'
                lbl = (row['walker'], row['step'])
                for isomer, isorow in dfuniq.iterrows():
                    Found = isorow['Found']
                    if lbl == Found[0]:
                        # this is a match
                        dfproduct.loc[iproduct-1] = [iproduct, row['Eabs'], row['Erel'], int(isomer), Found, len(Found), row['xyzfile'], conn.copy() ]
                        iproduct += 1
                        break
            else:
                # isomer number is (index+1) in 'dfminim'
                lbl = (row['walker'], row['step'])
                Found = dfuniq.loc[i+1, 'Found']
                nfound = len(Found)
                dfproduct.loc[iproduct-1] = [iproduct, row['Eabs'], row['Erel'], i+1, Found.copy(), nfound, row['xyzfile'], conn ]
                iproduct += 1
        elif matchrow == 0:
            # a possible conformation of the reactant
            if row['Erel'] < 0:
                print('More-stable reactant conformation found with Erel = {:.1f} ({:s})'.format(row['Erel'], row['xyzfile']))
        else:
            print('Structure {:s} is the same as {:s}'.format(row['xyzfile'], dfproduct.loc[matchrow-1]['xyzfile']))
print()
print('{:d} uniquely connected reaction products were found.'.format(iproduct - 1))
print('SSSSSSSSS dfproduct:\n', dfproduct[['product', 'Erel', 'isomer', 'Times-found', 'xyzfile', 'Found']])
#
# sort products by energy
dfproduct.sort_values(by=['Eabs'], inplace=True)
if iproduct > 1:
    # some products were found; print summary information
    print('Product\tIsomer\tErel/kJ\tTimes-found\tXYZfile')
    for irow, row in dfproduct.iterrows():
        ntimes = len(row['Found'])
        print('{:6d}\t{:6d}\t{:7.1f}\t{:6d}\t\t{:s}'.format(row['product'], row['isomer'], row['Erel'], ntimes, row['xyzfile']))
    # write all product structures to a single XYZ file, to facilitate review
    fprodxyz = '{:s}_product.xyz'.format(molec)
    regx = re.compile('Erel = .*')
    with open(fprodxyz, 'w') as fxyz:
        for irow, row in dfproduct.iterrows():
            G, n, c = readXmol(row['xyzfile'])
            # modify comment to include isomer number and product number
            m = regx.search(c)
            if m:
                c = '{:s} product {:d} (isomer {:d}); {:s}'.format(molec, row['product'], row['isomer'], m.group(0))
            else:
                # comment not as expected; don't change it
                pass
            fxyz.write( G.XmolXYZ(c) )
    print('Concatenated product structures written to file {:s}'.format(fprodxyz))
#
# Find approximate reaction paths: string-of-beads algorithm
#
f_pkllist = '{:s}_stringlist.pkl'.format(molec)
if os.path.isfile(f_pkllist):
    df_stringlist = pd.read_pickle(f_pkllist)
    print('List of strings read from file {:s}:'.format(f_pkllist))
else:
    # 'stringalgo' specifies the string minimization algorithm.  None of
    #   the options is very good. 
    #   'single*'    : one bead steps downhill one step on each string iteration
    #   'global'    : optimize whole string using scipy.optimize.minimize() CG method
    stringalgo = 'single2'
    kforce = 0.2  # spring force constant, hartree/ang**2
    # Spring length, Req, is specified in relative units.  The unit is either
    #   the average inter-bead spacing (for Req > 0) or the end-end distance
    #   divided by the number of beads (Req < 0). 
    Req = 0.7
    #Req = -0.9
    Nmin = 15     # required minimum number of beads in each string
    Ravg = 0.5    # desired average inter-bead spacing (absolute distance)
    include_ends = True  # include the terminal springs and allow interpolating points at the string ends
    curved = True   # use curved springs instead of straight
    df_stringlist = pd.DataFrame(columns=['product', 'walker', 'step', 'steplist', 'init_XYZfile', 'relaxed_XYZfile', 'pickle_file'],
        dtype=object)
    print('DFDF df_stringlist\n', df_stringlist)
    maxtimes = dfproduct['Times-found'].max()  # max number of times any single product was found
    string_num = -1
    for itime in range(maxtimes):
        for iprod, prod in dfproduct.iterrows():
            # find one path for each product before seeking additional paths from other walkers ('itime')
            # work on one string at a time
            finder = prod['Found']
            if len(finder) <= itime:
                # there are no more paths for this product
                continue
            prodno = prod['product']
            string_num += 1
            (iwalker, istep) = finder[itime]
            # find the sequence of steps (for this walker) that are relevant to this product
            stepList = find_step_list(dfuniq, iwalker, istep)
            print('^^^^^ iprod = {:d}, prodno = {:d}:  for ({:d}, {:d}), stepList = '.format(iprod, prodno, iwalker, istep), stepList)
            ftxyz = '{:s}_prod{:d}_string_w{:d}_{:d}-{:d}_thinned.xyz'.format(molec, prodno, iwalker, stepList[0], stepList[-1])
            df_stringlist.loc[string_num] = [iprod, iwalker, istep, 'nul', ftxyz, '', '']  # including stepList inside this list gives error
            df_stringlist.loc[string_num, 'steplist'] = stepList 
            fpkl = '{:s}_prod{:d}_string_w{:d}_{:d}-{:d}.pkl'.format(molec, prodno, iwalker, stepList[0], stepList[-1])
            fxyz = fpkl.replace('.pkl', '.xyz')
            if os.path.isfile(fpkl) and os.path.isfile(fxyz):
                # processed in an earlier job--skip it now
                print('String information found on disk for product {:d} found by walker {:d}, steps {:d}-{:d}'.format(prodno, iwalker, stepList[0], stepList[-1]))
                df_stringlist.loc[string_num] = [iprod, iwalker, istep, 'nul', ftxyz, fxyz, fpkl]
                df_stringlist.loc[string_num, 'steplist'] = stepList 
                continue
            fexg = '{:s}_{:d}_exg.csv'.format(molec, iwalker)
            # include the product structure as the last bead in the string
            isomer = prod['isomer']
            isorow = dfminim.loc[isomer-1]
            Product = isorow['Struct']
            # identify the appropriate reactant, to install as the first bead in the string
            Reactant = find_precursor(iwalker, stepList, dfuniq, dfminim, bondtol, nfrag0)
            if Reactant is None:
                # use the initial structure as the reactant
                Reactant = ips_input['geom0'].copy()
            print('JJJJ prodno = {:d}, isomer = {:d}, isorow = \n'.format(prodno, isomer), isorow)
            if stringalgo == 'single2':
                # this is the current method
                dfbeads = build_string(ips_input, fexg, stepList, Reactant, Product, isorow['Erel'], isorow['Eabs'])
                length, lstraight, nbeads = thin_string(dfbeads, Nmin, Ravg, include_ends)
                printXYZ_string(ftxyz, dfbeads)
                print('LLLL printed "thinned" dfBeads structures to file {:s}'.format(ftxyz))
                print('LLLL strength length = {:.2f}, end-end distance = {:.2f}'.format(length, lstraight))
                if no_reaction(dfbeads, bondtol):
                    print('** For this string, the reaction product is the same as the reactant.  Move on.')
                    df_stringlist.loc[string_num, 'relaxed_XYZfile'] = 'no net reaction'
                    continue
                # determine the equilibrium length of the springs in absolute units, 'xeq'
                if Req > 0:
                    xeq = Req * (length/nbeads)
                else:
                    xeq = -Req * (lstraight/nbeads)
                print('EEEE Spring length = {:.2f}, based upon Req = {:.1f}'.format(xeq, Req))
                etol_list = [20, 6]
                for itol in range(len(etol_list)):
                    etol = etol_list[itol]
                    print('TTTT etol = {:g}, time ='.format(etol), time.asctime( time.localtime(time.time()) ))
                    try:
                        niter = relax_dfstring(ips_input, dfbeads, kforce, xeq, etol, include_ends, curved)
                        printXYZ_string(fxyz, dfbeads)
                        df_stringlist.loc[string_num, 'relaxed_XYZfile'] = fxyz
                        print('LLLL printed {:d} relaxed bead structures to file {:s}'.format(dfbeads.shape[0], fxyz))
                    except ValueError:
                        # probably SCF failure
                        print('** string optimization failed for product {:d} from walker {:d} steps {:d}-{:d}'.format(prodno, iwalker, stepList[0], stepList[-1]))
                        df_stringlist.loc[string_num, 'relaxed_XYZfile'] = 'relaxation failed'
                        print(traceback.format_exc())
                        break
                    # locate the biggest peak
                    if itol == 0:
                        # after the first refinement, extract the biggest energy peak and create a denser string
                        nbead0 = dfbeads.shape[0]
                        dfbeads = extract_string_peak(dfbeads, 3)
                        nbeads = dfbeads.shape[0]
                        if nbead0 > nbeads:
                            # number of beads changed; re-interpolate or truncate as needed
                            # also increase bead density
                            print('A shorter string of {:d} beads was extracted from the crudely relaxed string of {:d}.'.format(nbeads, nbead0))
                            length, lstraight, nbeads = thin_string(dfbeads, Nmin, Ravg/2, include_ends)
                            print('After padding, string has {:d} beads, length {:.2f} and end-end distance {:.2f}'.format(nbeads, length, lstraight))
                            # use a different file name for the XYZ file
                            fxyz = fxyz.replace('.xyz', '_excerpt.xyz')
                print('TTTT time now:', time.asctime( time.localtime(time.time()) ))
            if stringalgo == 'single1':
                # XXX does not excecute
                dX, dfbeads = relax_string(ips_input, fexg, stepList, iprod, Product, isorow['Erel'], isorow['Eabs'])
                fxyz = '{:s}_prod{:d}_string_w{:d}_{:d}-{:d}.xyz'.format(molec, prodno, iwalker, stepList[0], stepList[-1])
                with open(fxyz, 'w') as fout:
                    for irow, row in dfbeads.iterrows():
                        fout.write(row['Geom'].XmolXYZ(comment='product {:d}: walker {:d}, bead {:d}, Erel = {:.1f} kJ/mol'.format(prodno, iwalker, irow, row['Erel'])))
            if stringalgo == 'single0':
                dX, dfbeads = relax_string_old(ips_input, fexg, stepList, iprod, Product, isorow['Erel'], isorow['Eabs'])
                fxyz = '{:s}_prod{:d}_string_w{:d}_{:d}-{:d}.xyz'.format(molec, prodno, iwalker, stepList[0], stepList[-1])
                with open(fxyz, 'w') as fout:
                    for irow, row in dfbeads.iterrows():
                        fout.write(row['Geom'].XmolXYZ(comment='product {:d}: walker {:d}, bead {:d}, Erel = {:.1f} kJ/mol'.format(prodno, iwalker, irow, row['Erel'])))
            if stringalgo == 'global':
                String = create_BeadString(ips_input, fexg, stepList, Product, isorow['Eabs'])
                String.printXYZ('dfString.xyz')
                print('NNNN printed String structures to file dfString.xyz')
                String.adjust_count(Nmin=10, Ravg=0.5)  
                String.printXYZ('dfString_adj.xyz')
                print('NNNN after adjustment, file dfString_adj.xyz and BeadString length = {:.1f}'.format(String.length()))
                Esudo, Gsudo = String.effectivePotential(None, ips_input, kforce=kforce, Req=Req)
                print('PPPP Esudo = {:.5f}'.format(Esudo))
                fxyz = '{:s}_prod{:d}_string_{:d}_{:d}-{:d}_iter{:d}.xyz'.format(molec, prodno, iwalker, stepList[0], stepList[-1], 0)
                String.printXYZ(fxyz)
                print('Unoptimized String ({:d}, {:d}) for product {:d} written to file {:s} (length = {:.2f})'.format(iwalker, istep, prodno, fxyz, String.length()))
                print('QQQQ shapes: Xmat = ', String.Xmat.shape, ' Gmat = ', String.Gmat.shape, '; natom = {:d}'.format(String.natom()))
                print('Optimizing string using kforce = {:g} and Req = {:g}'.format(kforce, Req))
                String.minimizeEP(ips_input, kforce=kforce, Req=Req)
                fxyz = '{:s}_prod{:d}_string_{:d}_{:d}-{:d}_opt_k{:d}_r{:d}.xyz'.format(molec, prodno, iwalker, stepList[0], stepList[-1], int(kforce*10), int(Req*10))
                String.printXYZ(fxyz)
                print('String written to file {:s} (length = {:.2f})'.format(fxyz, String.length()))
                print('norm(Gmat) = {:.3f}'.format(np.linalg.norm(String.Gmat)))
            dfbeads.to_pickle(fpkl)
            df_stringlist.loc[string_num, 'pickle_file'] = fpkl
            print(df_stringlist)
            #sys.exit('stop string')
    df_stringlist.to_pickle(f_pkllist)
    print('ZZZZ String list saved to pickle file {:s}'.format(f_pkllist))
print(df_stringlist)
#
# Read string pickle files, analyze, generate transition states
#
for istring, row in df_stringlist.iterrows():
    fpkl = row['pickle_file']
    fxyz = row['relaxed_XYZfile']
    if (fpkl == '') or (fxyz == 'relaxation failed'):
        continue
    print('pickle file {:s}'.format(fpkl))
    dfbeads = pd.read_pickle(fpkl)
    print('DDDD dfbeads:\n', dfbeads[['Step', 'Erel', 'Eabs', 'Espring']])
    # Plot the energy profiles
    nbeads = len(dfbeads)
    xvals = np.linspace(0, 1, nbeads)
    evals = dfbeads['Erel'].values
    plt.plot(xvals, evals, '-')  # the relative energies 
    # now add the "transition-like" nature of the geometry, defined
    #   as the difference in atomic connectivity when using different
    #   threshold values
    beadgeoms = dfbeads['Geom'].values
    tsness = [TSnature(bG, 1.1, 1.5) for bG in beadgeoms]
    tsness = (tsness - tsness[0]) * 100  # translate and scale for visibility
    plt.plot(xvals, tsness, 'x', label='TSness')
    #cmat14sum = np.array([(np.fabs(coulmat_compare(beadgeoms[i], beadgeoms[i+1], select=4)).sum()) for i in range(nbeads-1)] + [0])
    #cmat14sum = cmat14sum * 10 # scale for visibility
    #plt.plot(xvals, cmat14sum, 'o', markerfacecolor='None', markeredgecolor='g', label='14sum')
    #dihenorm = np.array([ldihediff(beadgeoms[i], beadgeoms[i+1], methyl=False) for i in range(nbeads-1)] + [0])
    #dihenorm *= 10
    #plt.plot(xvals, dihenorm, '^', markerfacecolor='None', markeredgecolor='c', label='dihe')
    plt.title(row['relaxed_XYZfile'])
    plt.legend(loc='best')
    plt.draw()
    plt.show()
    # NEED TO WRITE THE CODE TO PREPARE FOR QST3 #
    #centers = string_connection_change(dfbeads)
    #print('DDDD centers at ', centers)
