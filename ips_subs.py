# Functions to support isopotential searching (python3)
# KK Irikura, NIST 2017
#
import re, sys, copy, glob, os, pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import yaml
from chem_subs import *
from qm_subs import *
from subprocess import call
#
KNOWN_CODES = ['gaussian09', 'quantum-espresso']
##
class Walker(object):
    # includes history of energy, position, gradient
    # coordinates may be Geometry() or ZMatrix()
    # includes any constraints
    def __init__(self, ID, Coord, historyE, historyX, historyG, active=True, historyConn=[], connexChange=[], constraint={}, tol=1.3, remember=4, steps=None):
        # 'ID': integer label unique to this Walker
        # 'Coord': current coordinates as Geometry() or ZMatrix() object
        #    this identifies the meaning of X and G vectors
        # 'historyE': list of energies
        # 'historyX': list of positions, as simple vectors (i.e., numpy 1D arrays)
        # 'historyG': list of gradients, as simple vectors
        # 'active': flag to indicate that walker is still walking
        # 'historyConn': list of connection tables (explicit or [] to denote no change)
        # 'connexchange': list of boolean (True for a new connectivity pattern)
        # 'constraint': dict of constraints
        # 'tol': bond-distance tolerance for calling 'bonded'
        # 'remember': the number of recent structures to keep in the lists
        # 'steps': the step numbers for the elements in the lists
        self.ID = ID
        self.Coord = Coord
        self.historyE = historyE
        self.historyX = historyX
        self.historyG = historyG
        self.active = active
        self.historyConn = historyConn
        self.connexChange = connexChange
        self.constraint= constraint
        self.remember = remember
        if steps is None:
            # assume this is starting from scratch
            steps = list(range(len(historyE)))
        self.steps = steps
        ctype = typeCoord(Coord)  # either 'Geometry' or 'ZMatrix'
        if ctype == 'Geometry':
            self.coordtype = 'cartesian'
        elif ctype == 'ZMatrix':
            self.coordtype = 'zmatrix'
            # be sure angles are in radians
            self.Coord.toRadian()
        else:
            self.coordtype = 'unknown'
        # update connection information
        self.update_connections(tol=tol)
    def make_active(self):
        self.active = True
    def make_inactive(self):
        self.active = False
    def deltaX(self, istep):
        # return vector for specified step (displacement)
        try:
            idx = self.steps.index(istep)
            return self.historyX[idx] - self.historyX[idx-1]
        except:
            print_err(name='request for unavailable step {:d}'.format(istep))
    def step_length(self, istep):
        # return norm of vector for specified step (displacement)
        return np.linalg.norm(self.deltaX(istep))
    def update_connections(self, tol=1.3):
        # update connectivity info
        for i in range(len(self.historyConn), len(self.historyX)):
            # loop is over the missing indices in self.historyConn[]
            position = self.Coord.copy()
            position.fromVector(self.historyX[i], unitS=position.unitX())
            connex = position.connection_table(tol=tol)
            self.historyConn.append(connex.copy())
        # find where the connectivity has changed
        for i in range(len(self.connexChange), len(self.historyConn)):
            if self.steps[i] == 0:
                self.connexChange = [False]
            else:
                self.connexChange.append(not np.array_equal(self.historyConn[i-1], self.historyConn[i]))
        # keep length of lists to 'remember'
        if len(self.historyConn) > self.remember:
            self.historyConn = self.historyConn[-self.remember:]
        if len(self.connexChange) > self.remember:
            self.connexChange = self.connexChange[-self.remember:]
        return
    def copy(self):
        return copy.deepcopy(self)
    def last_connections(self):
        # return the most recent connection table
        try:
            return self.historyConn[np.argwhere(self.connexChange)[-1]]
        except:
            # no change within recent memory
            return self.historyConn[-1]
    def add_point(self, energy, position, gradient, tol=1.3):
        # add one (E, X, G) triple to respective lists
        #   'X' is a structural object; 'G' is a simple vector
        self.historyE.append(energy)
        self.Coord = position.copy()
        self.historyX.append(position.toVector())
        self.historyG.append(gradient.copy())
        # increment 'steps'
        self.steps.append(self.steps[-1] + 1)
        # keep within requested buffer lengths
        if len(self.historyE) > self.remember:
            # assume these lists are all the same length
            self.historyE = self.historyE[-self.remember:]
            self.historyX = self.historyX[-self.remember:]
            self.historyG = self.historyG[-self.remember:]
            self.steps = self.steps[-self.remember:]
        # also add connectivity info
        self.update_connections(tol=tol)
        return
    def update_traj(self, fileroot, step, comment=''):
        # add the current point to the appropriate 
        #   XYZ and EXG trajectory files
        # first the XYZ file (Xmol format)
        trajname = '{:s}_{:d}_traj.xyz'.format(fileroot, self.ID)
        txt = 'Walker {:d}, step {:d} '.format(self.ID, step)
        txt += comment
        xyzstring = self.Coord.XmolXYZ(txt)
        fxyz = open(trajname, 'a')
        fxyz.write(xyzstring)
        fxyz.close()
        # now the EXG file (comma-delimited)
        exgname = '{:s}_{:d}_exg.csv'.format(fileroot, self.ID)
        add_data = os.path.exists(exgname) 
        fexg = open(exgname, 'a')
        if not add_data:
            # create new CSV file
            # include headers to identify each variable within (E, X, G)
            varnames = self.Coord.varlist()
            hdr = ['Erel'] + varnames + ['G_' + v for v in varnames]
            hdr.append('isom')  # for changes in connectivity
            fexg.write(','.join(hdr) + '\n')
        # construct the list of fields
        exgstring = [ '{:.1f}'.format(self.historyE[-1]) ]
        for x in self.historyX[-1]:
            exgstring.append('{:.5f}'.format(x))
        for g in self.historyG[-1]:
            exgstring.append('{:.5f}'.format(g))
        exgstring.append(str(self.connexChange[-1]))
        fexg.write(','.join(exgstring) + '\n')
        fexg.close()
        return
    def npoints(self):
        # return the number of EXG points, historically
        return self.steps[-1]
    def write_exg(self, molec, style='text', X0=''):
        # OBSOLETE
        # write to file
        if style == 'text':
            # step number, Erel (kJ/mol), position vector (radians), gradient vector
            with open('{:s}_{:d}.txt'.format(molec, self.ID), 'w') as fexg:
                for i in range(self.npoints()):
                    fexg.write('{:d}\n{:.2f}\n'.format(i, self.historyE[i]))  # step and Erel (kJ/mol)
                    fexg.write('[{:s}]\n'.format(','.join(['{:.5f}'.format(x) for x in self.historyX[i]])))
                    fexg.write('[{:s}]\n'.format(','.join(['{:.6f}'.format(x) for x in self.historyG[i]])))
        elif style == 'norms':
            # step number, Erel (kJ/mol), distance from origin (radians etc.), gradient norm
            with open('{:s}_{:d}.nrm'.format(molec, self.ID), 'w') as fexg:
                for i in range(self.npoints()):
                    Erel = self.historyE[i]
                    dist = self.historyX[i] - X0
                    if self.coordtype == 'zmatrix':
                        # move dihedral differences into the interval (-pi, pi]
                        dist = self.Coord.adjust_dTau(dist)
                    dist = np.linalg.norm(dist)
                    gnorm = np.linalg.norm(self.historyG[i])
                    fexg.write('{:d} {:.2f} {:.5f} {:.7f}\n'.format(i, Erel, dist, gnorm))
        return
##
def concat_xyz(filename, start, end):
    # create a file, 'filename_traj.xyz', that is simply
    # the concatenation of all available files 'filename_[n].xyz',
    # where 'n' is a natural number
    with open('{:s}_traj.xyz'.format(filename), 'w') as outfile:
        for i in range(start, end+1):
            infile = '{:s}_{:d}.xyz'.format(filename, i)
            with open(infile, 'r') as f:
                outfile.write(f.read())
    return
##
def walker_net_forces(walkers, weight='1/R'):
    """ Return a list of total pseudo-forces on all walkers,
    as computed using the specified weight. """
    npt = len(walkers)
    ndof = walkers[0].Coord.nDOF()  # number of coordinates, same for each walker 
    fpseudo = [np.zeros(ndof)] * npt
    for i in range(npt):
        for j in range(i):
            # add the force vectors applied to walker[i] by the other walkers
            F = walker_force(walkers[i].Coord, walkers[j].Coord, weight=weight)
            fpseudo[i] = fpseudo[i] + F
            # assume that the force function is anticommutative
            fpseudo[j] = fpseudo[j] - F
    return fpseudo
##
def walker_force(geom1, geom2, weight='1/R'):
    """ The pseudo-force vector exerted on geom1 by geom2,
    using the specified weight. """
    # 'weight' usually defines a pseudo-force
    # Assume that the geometries are of the same type; skip checking
    #   to save time in walker_net_forces() loop
    #geomtype = typeCoord(geom1)
    #if not ((geomtype in ['Geometry', 'ZMatrix']) and (geomtype == typeCoord(geom2))):
    #    print_err('', 'Illegal or mismatched geometry types: {:s} and {:s}'.format(geomtype, typeCoord(geom2)))
    v1 = geom1.toVector()
    v2 = geom2.toVector()
    diff = v1 - v2  # directed from geom2 toward geom1
    if weight == 'L2':
        # cartesian distance, R
        wt = np.linalg.norm(diff)
    elif weight == '1/R':
        # 1/R
        wt = 1 / np.linalg.norm(diff)
    elif weight == 'L1':
        # Manhattan distance
        wt = np.fabs(diff).sum()
    else:
        print_err('', 'Unknown weight = {:s}'.format(weight))
    # normalize the force vector (using L2), to the length 'wt'
    F = normalize(diff, wt)
    return F
##
def step_many_repulsive(ips_input, walkers, maxiter=20):
    """ Use a pairwise pseudo-force between walkers to define
    the direction in which each will move. Then find a new
    point for each walker. """
    stepl = ips_input['stepl']
    threshold = ips_input['dissociated']
    # compute the list of pseudo-forces on each walker
    Flist = walker_net_forces(walkers, weight='1/R')
    nprocs = ips_input['nprocs']
    coordtype = ips_input['coordtype']
    unitS = walkers[0].Coord.unitX()
    pool = mp.Pool(nprocs)
    #print('\tstep_many_repulsive() with {:d} process{plural}'.format(nprocs, plural='es' if nprocs > 0 else ''))
    tasks = []
    dX = []
    nActive = 0
    for ID in range(len(walkers)):
        if not walkers[ID].active:
            # this walker is no longer walking
            #print('walker {:d} is inactive'.format(ID))
            dX.append([])
            continue
        if threshold > 0:
            # check for dissociation
            sep, ijNearest = walkers[ID].Coord.fragment_distances(loc='nearest', tol=ips_input['bondtol'])
            if (sep > threshold).any():
                # at least one fragment has departed
                nfrag = sep.shape[0]  # number of fragments (not necessarily distant)
                frags0 = ips_input[coordtype].find_fragments(tol=ips_input['bondtol'])
                nfrag0 = len(frags0)
                if nfrag > nfrag0:
                    # the number of fragments has increased
                    print('\tWalker {:d} has dissociated and will be retired.'.format(ID))
                    walkers[ID].active = False
                    dX.append([])
                    continue
        nActive += 1
        # move the walker according to the pseudo-force
        F = Flist[ID]
        # project out the energy gradient
        G_prev = walkers[ID].historyG[-1]
        gnorm = normalize(G_prev)
        F -= gnorm * (np.dot(gnorm, F))
        if ips_input['suppress_rotation'] and coordtype == 'cartesian':
            # iteratively suppress rotation and translation
            F = walkers[ID].Coord.suppress_translation(F)
            F = walkers[ID].Coord.suppress_rotation(F)
        # normalize the displacement to desired length
        F *= stepl / np.linalg.norm(F)
        dX.append(F)
        # displace the coordinates
        walkers[ID].Coord.fromVector(F, unitS, add=True)
        # install the new position in ips_input[]
        ips_input[coordtype] = walkers[ID].Coord
        tasks.append( (copy.deepcopy(ips_input), ID, dX[ID]) )
    # run the quantum calculations in parallel
    results = [pool.apply_async(seek_qm_energy, t) for t in tasks]
    pool.close()
    pool.join()
    for result in results:
        (E, X, G, niter, iwalker) = result.get()
        # update the appropriate walker
        W = walkers[iwalker]
        if np.isnan(E):
            # failure; mark this walker as inactive
            print('\tFailed to hit energy target for walker {:d}'.format(iwalker))
            #print(X, G, niter, iwalker)
            W.make_inactive()
            nActive -= 1
            continue
        Erel = au_kjmol * (E - ips_input['E0'])
        if coordtype == 'zmatrix':
            # canonicalize the dihedral angles
            X.canonical_angles()
        W.add_point(Erel, X, G)
    # return the updated walkers and the number that were updated
    return walkers, nActive
##
def step_specular(ips_input, walker, maxiter=20, trustR=0.5):
    """ Use the two previous position vectors and the 
    previous gradient vector to find the next point. 
    The step length cannot exceed 'trustR'.  """
    X_prev = walker.historyX[-1].copy()
    dX = X_prev - walker.historyX[-2]
    coordtype = ips_input['coordtype']
    if coordtype == 'zmatrix':
        # avoid artificially large changes in dihedral variables
        dX = ips_input[coordtype].adjust_dTau(dX)
    if np.linalg.norm(dX) > trustR:
        # shorten the step
        print('##### decrease step length from {:.3f} to {:.3f}'.format(np.linalg.norm(dX), trustR))
        dX = normalize(dX, length=trustR)
    G_prev = walker.historyG[-1]
    if np.isnan(G_prev).any():
        print('*** NaN values in initial gradient vector G_prev!\n', G_prev)
    unitX = ips_input[coordtype].unitX()  # remember the initial units (tuple)
    gnorm = normalize(G_prev)
    # now adjust to the target energy
    for istep in range(maxiter):
        curve = 2 * np.dot(gnorm, dX) * gnorm
        changeX = dX - curve
        if 'active' in ips_input:
            # zero out any inactive coordinates
            changeX = changeX * ips_input['active_vec']
        if ips_input['suppress_rotation'] and coordtype == 'cartesian':
            # iteratively suppress rotation and translation
            changeX = walker.Coord.suppress_translation(changeX)
            changeX = walker.Coord.suppress_rotation(changeX)
        Xnew = X_prev + changeX
        #print('##### istep = {:d}, Xnew = '.format(istep), Xnew)
        ips_input[coordtype].fromVector(Xnew, unitX)  # install the new coordinates
        # use the quantum chemistry program to find the isopotential
        E, Struct, grad, niter, ID = seek_qm_energy(ips_input)
        if np.isnan(E):
            # failure; try a smaller step by half
            stepl = np.linalg.norm(dX)
            if stepl < 0.001:
                sys.exit('Step length already very short: {:f}'.format(stepl))
            print('#####  reducing step length by half (new norm(dX) = {:.3f})'.format(stepl))
            dX /= 2
        else:
            Erel = au_kjmol * (E - ips_input['E0'])
            return Erel, Struct, grad, niter
    else:
        # failure
        sys.exit('*** Maxiter = {:d} exceeded in step_specular()! ***'.format(maxiter))
##
def seed_points(ips_input, maxiter=20):
    # generate all needed seed points
    # return list of Walker objects
    algorithm = ips_input['ips_method']['name']
    coordtype = ips_input['coordtype']
    if algorithm.upper() == 'SRIPS':
        # specular reflection algorithm
        nseed = 2
    elif algorithm.upper() == 'MRWIPS':
        # many repulsive walkers
        nseed = ips_input['ips_method']['walkers']  # one seed for each walker
    else:
        sys.exit('Unrecognized algorithm in seed_points(): {:s}'.format(algorithm))
    print('Seeking {:d} seed points for algorithm {:s}'.format(nseed, algorithm.upper()))
    Seed = []  # geometries
    gradient = []  # gradients
    Eseed = [] # energies
    # choose random directions
    Origin = ips_input[coordtype].copy()  # E0 geometry--usually a minimum
    ndof = Origin.nDOF()  # number of coordinates (including inactive)
    while len(Eseed) < nseed:
        nneed = nseed - len(Eseed)
        # Choose directions at random.  Alternatives may be offered in the future. 
        direction = np.random.uniform(-1., 1., (nneed, ndof))
        if coordtype == 'cartesian':
            # suppress rotation and translation in the seed directions
            install_unit_masses(Origin)  # give ALL atoms unit mass; this is IPS, not dynamics
            for i in range(direction.shape[0]):
                direction[i,:] = Origin.suppress_translation(direction[i])
                direction[i,:] = Origin.suppress_rotation(direction[i])
        elif 'active' in ips_input:
            # zmatrix with only certain coordinates active (initially)
            # zero out the components in inactive directions
            direction = direction * ips_input['active_vec']
        ## choose orthonormal directions
        #direction = orthogonalize_rows(direction, norm=1)
        ## don't allow any single component to be too large
        #toolarge = 0.1   # angstrom or radian
        #dbig = np.abs(direction) > toolarge
        #direction[dbig] = toolarge
        # run the seed searches in parallel
        nprocs = ips_input['nprocs']
        tasks = []
        for i in range(nneed):
            tasks.append( (ips_input, direction[i], maxiter, i) )
        pool = mp.Pool(nprocs)
        print('\tusing Pool with {:d} process{plural}'.format(nprocs, plural='es' if nprocs > 1 else ''))
        results = [pool.apply_async(find_ips_seed, t) for t in tasks]
        pool.close()
        pool.join()
        for result in results:
            # the ordering of the seed points does not matter
            (E, sGeom, sGrad, nit) = result.get()
            if np.isnan(E):
                # this seed point failed
                continue
            Eseed.append(E)
            if coordtype == 'zmatrix':
                # make sure all angles are in the interval (-pi, pi]
                sGeom.canonical_angles()
            Seed.append(sGeom.copy())
            gradient.append(sGrad.copy())
    # All seeds generated; now create Walkers
    if ips_input['verbose']:
        print('vv {:d} seed points generated'.format(len(Eseed)))
    walkers = [] # list of Walkers
    if algorithm == 'srips':
        # two seeds needed for one Walker
        histX = []
        for Coord in Seed:
            histX.append(Coord.toVector())
        # order them by decreasing magnitude of gradient
        Gmag = [np.linalg.norm(g) for g in gradient]
        idx = np.argsort(Gmag)
        idx = idx[[1,0]]  # in case there are more than two (??)
        W = Walker( 0, Seed[idx[1]], 
            [Eseed[i] for i in idx],
            [histX[i] for i in idx],
            [gradient[i] for i in idx] )
        walkers = [W]
    if algorithm == 'mrwips':
        # one seed per Walker
        for i in range(len(Eseed)):
            W = Walker(i, Seed[i], [Eseed[i]], [Seed[i].toVector()], [gradient[i]])
            walkers.append(W.copy())
    # initialize trajectory and EXG files
    for W in walkers:
        ns = W.npoints()  # number of seed points to write
        for ipt in range(ns):
            W.update_traj(ips_input['molecule'], ipt-ns, comment='Erel = {:.1f} kJ/mol'.format(W.historyE[ipt]))
    return walkers
##
def find_ips_seed(ips_input, direction, maxiter=20, ID=0):
    # find a geometry at the target energy
    # look in 'direction' (a 1-D vector) relative to the origin
    # return an energy (in hartree), a structure Object, 
    #   the energy gradient (vector) in the same units, 
    #   and the iteration count
    coordtype = ips_input['coordtype']  # 'cartesian' or 'zmatrix'
    frac = 5.0e-4 * ips_input['energy']['value']  # guess for initial displacements
    # scale 'direction' to L2 length of 'frac' (units may be mixed radian/angstrom!)
    # this is assumed to be an uphill direction
    dvec = normalize(direction, length=frac)
    # Seek a geometry with the target energy with 'direction' as
    #   the search direction (and initial displacement)
    E, Scoord, grad, niter = seek_qm_energy_line(ips_input, dvec, maxiter=maxiter, ID=ID)
    # convert E from (hartree absolute) to (kJ/mol relative)
    Ekj = au_kjmol * (E - ips_input['E0'])
    # 'Ekj' will be NaN if seek_qm_energy_line() failed
    return Ekj, Scoord, grad, niter
##
def seek_qm_energy_line(ips_input, dvec, maxiter=20, ID=0):
    """ Follow the specified direction to find a geometry that
    has the target energy, within the specified tolerance. 
    Used only for generating seed points. """
    # use repeated energy/gradient calculations to find
    #   a molecular structure that has the energy ips_input['energy']['value'], 
    #   with a tolerance of ips_input['tolerance']
    # The search is restricted to the line defined by 'dvec'
    #   'dvec' is assumed to be a reasonable step length and uphill
    # Return energy, distance along 'dvec', coordinates, and gradient of an acceptable point
    #   also return number of iterations
    # The coordinates and gradient are in the same distance units as the input structure
    # 'ID' is a walker identifier, used to avoid filename collisions.
    #
    trustR = 0.5  # maximum permitted step length
    # convert from (kJ/mol relative) to (hartree absolute)
    E0 = ips_input['E0']
    etarget = E0 + convert_unit(ips_input['energy'], 'hartree')['value']
    etol = convert_unit(ips_input['tolerance'], 'hartree')['value']
    coordtype = ips_input['coordtype']
    Geom = ips_input['E0_geom'].copy()
    if ips_input['verbose'] and ID == 0:
        print('\tvv E0 = {:.5f}, Etarget = {:.5f} +/- {:.5f}'.format(E0, etarget, etol))
    unitX = Geom.unitX()  # initial units (tuple)
    # is the origin within the tolerance?
    if in_bounds(E0, etarget, etol):
        # yes!  but have no gradient information
        print('>>>>  Origin point good (??) with E0 = {:f} and etarget = {:f} (etol = {:f})'.format(E0, etarget, etol))
        return E0, Geom, [], 0
    # keep looking
    if coordtype == 'zmatrix':
        Geom.toRadian()  # put angles in radians
    unitS = Geom.unitX()  # working units (tuple)
    # convert initial coordinate values to a vector
    X0 = Geom.toVector()
    # 'S' is the step length along the search vector
    S = np.linalg.norm(dvec)  # initial step length
    dvec = normalize(dvec)   # now a unit vector
    # line search; keep track of upper/lower bounds
    # is the initial point too low or too high?
    if E0 < etarget:
        lowS = 0. # S = 0 gives too-low energy
        lowE = E0
        highE = np.inf
    else:
        # this should not happen
        highS = 0. # S = 0 gives too-high energy
        highE = E0
        S *= -1.  
        lowE = -np.inf
    # find the target energy
    debug = True
    if debug:
        print('dddd  maxiter = {:d}'.format(maxiter))
    Svec = []
    Evec = []
    for itry in range(maxiter):
        # compute energy and gradient for current value of S
        # S is relative to the origin
        X = X0 + S * dvec
        Geom.fromVector(X, unitS)
        ips_input[coordtype] = Geom
        E, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR=unitX[0], ID=ID)
        if E is None:
            # calculation failed; quit
            break
        Svec.append(S)
        Evec.append(E)
        # are we done?
        erel = (E - E0) * AU_KJMOL 
        if in_bounds(E, etarget, etol):
            # yes!
            if ips_input['verbose']:
                print('\tvv iter={:d}, ID={}: seed point converged with Erel = {:.1f}'.format(itry+1, ID, erel))
            if 'degree' in unitX:
                # convert angles back to degrees (but not in the gradient)
                Geom.toUnits(unitX)
            return E, Geom, grad, itry+1  # show the user first iteration as no. 1
        # keep looking
        # keep track of bounds 
        # too high or too low?
        if ips_input['verbose']:
            print('\tvv iter={:d}, ID={}:  S = {:.2f}, E = {:.5f}; erel = {:.1f}'.format(itry+1, ID, S, E, erel), end='')
        if E < etarget:
            # too low
            if ips_input['verbose']:
                print(' is too low')
            if E > lowE:
                # a new lower bound
                lowE = E
                lowS = S
        else:
            # too high, E > etarget
            if ips_input['verbose']:
                print(' is too high')
            if E < highE:
                # a new upper bound
                highE = E
                highS = S
        #print(',,,,, itry = {:d} gives E = {:.4f} for S = {:.4f}\tlowE = {:.4f}, highE = {:.4f}'.format(itry, E, S, lowE, highE))
        # project 'grad' along 'dvec'
        gradS = np.dot(grad, dvec)  # component of gradient along the search direction
        # 'Sproj' is proposed next step, based only upon the gradient
        Sproj = (etarget - E) / gradS
        # don't exceed trust radius
        if abs(Sproj) > trustR:
            if ips_input['verbose']:
                print('\tvv\tstep of {:.2f} exceeds trustR; scaling back'.format(Sproj))
            Sproj *= trustR / abs(Sproj)
        deltaS = Sproj
        try:
            # 'Slinterp' is a linear interpolation between the high and low points
            slope = (highE - lowE) / (highS - lowS)
            intcpt = highE - slope * highS
            Slinterp = (etarget - intcpt) / slope
            # sometimes include the interpolated value, to damp oscillations
            if np.random.random() > 0.6:
                deltaS = (Sproj + Slinterp) / 2
            if debug:
                print('dddd Sproj = {:.3f}, Slinterp = {:.3f}, deltaS = {:.3f}'.format(Sproj, Slinterp, deltaS))
        except:
            # probably variables are not yet defined
            pass
        S += deltaS
        # don't exceed known limits
        # note that 'highS' may be smaller than 'lowS' (because name is based on energy)
        if abs(lowE * highE) != np.inf:
            # upper and lower bounds have been found
            (minS, maxS) = (min(lowS, highS), max(lowS, highS))
            # the target must be in the interval (minS, maxS) 
            if (S < minS) or (S > maxS):
                # this is outside the known limits; use linear interpolation only
                S = Slinterp
                if ips_input['verbose']:
                    print('\tvv out-of-bounds step replaced by linear interpolation')
                    if debug:
                        print('S({})\tE({})'.format(ID, ID))
                        for i in range(len(Svec)):
                            print('{:.3f}\t{:.5f}'.format(Svec[i], Evec[i]))
                # choose 3/4 position between the bounds, in the direction of the gradient
                #if S < minS:
                #    S = (3*lowS + highS) / 4
                #else:
                #    S = (lowS + 3*highS) / 4
                    slope = (highE - lowE) / (highS - lowS)
                    intcpt = highE - slope * highS
                    Slinterp = (etarget - intcpt) / slope
    #else:
    #    print_err('maxiter', maxiter, halt=False)
    # Arrive here only upon failure
    if ips_input['verbose']:
        if itry >= maxiter:
            print('vv Iteration limit exceeded for ID={}'.format(ID))
            if debug:
                print('S\tE')
                for i in range(len(Svec)):
                    print('{:.3f}\t{:.5f}'.format(Svec[i], Evec[i]))
        if E is None:
            print('vv Gradient computation failed for ID={}'.format(ID))
    return np.nan, [], [], 0
##
def seek_qm_energy(ips_input, ID=0, dX0=None, maxiter=20, trustR=0.5):
    """ Follow the energy gradient to adjust the geometry to have
    the target energy, within the specified tolerance. 
    Only change coordinates that are active. """
    # use repeated energy/gradient calculations to find
    #   a molecular structure that has the energy ips_input['energy']['value'], 
    #   with a tolerance of ips_input['tolerance']
    # dX0 (if any) is the step that was applied to create the input geometry
    # Return energy, coordinates, and gradient of an acceptable point
    #   also return number of iterations and ID
    # The coordinates and gradient are in the same distance units as the input structure
    # 'trustR' is maximum permitted step length

    # convert to hartree
    etarget = ips_input['E0'] + convert_unit(ips_input['energy'], 'hartree')['value']
    etol = convert_unit(ips_input['tolerance'], 'hartree')['value']
    algorithm = ips_input['ips_method']['name']
    if ips_input['verbose']:
        print('\tvv Etarget = {:5f} for walker {:d}'.format(etarget, ID))
    coordtype = ips_input['coordtype']
    Geom = ips_input[coordtype].copy()
    unitX = Geom.unitX()  # initial units (tuple)
    unitS = unitX
    if coordtype == 'zmatrix':
        Geom.toRadian()  # put angles in radians
        unitS = (unitX[0], 'radian')  # temporary working units
    # convert initial coordinate values to a vector
    X = Geom.toVector()
    if np.isnan(X).any():
        print('*** NaN values in initial coordinate vector X!\n', X)
        #Geom.print()
        #ips_input[coordtype].print()
        qt = 1/0  # do this because of possible parallel jobs--avoid zombies
        sys.exit(1)
    S = 1.  # step scaling factor (a scalar)
    grad = np.zeros_like(X)  # for initial point (no displacement)
    # seek the target energy
    for itry in range(maxiter):
        # compute energy and gradient for current value of S
        # S is relative to the previous position
        if itry % 3 == 2:
            # reduce the step scaling factor every 3 iterations
            S /= GOLD
        dX = S * grad
        X = X + dX
        Geom.fromVector(X, unitS)
        if coordtype == 'zmatrix':
            # force all bond angles into the interval (0, pi)
            Geom.cap_angles()
        ips_input[coordtype] = Geom
        # call the quantum chemistry program
        E, grad, IDx = qm_function(ips_input, 'gradient', option='flatten', unitR=unitX[0], ID=ID)
        if E is None:
            # gradient calculation failed.  Maybe because of atomic crowding
            if (itry == 0) and (algorithm == 'srips'):
                # The first step; return failure
                if ips_input['verbose']:
                    print('\tvv failure in first step for walker {}'.format(ID))
                return np.nan, 'failure in first step', 0, 0, ID
            else:
                if itry == 0:
                    # first point failed; return to the previous successful point
                    X -= dX0
                    if ips_input['verbose']:
                        print('\tvv retreat to previous successful point for ID = {:d}, iter = {:d}: '.format(ID, itry+1))
                else:
                    # try a smaller step
                    X -= dX  # retreat by the full step
                    S /= 2.   # smaller step
                    if ips_input['verbose']:
                        print('\tvv retry using a smaller step for ID = {:d}, iter = {:d}: '.format(ID, itry+1))
                grad = np.zeros_like(X)  # replace grad = [] from failed QM calculation
                continue
        erel = (E - ips_input['E0']) * AU_KJMOL
        eratio = erel / ips_input['energy']['value']  # relative E / target E
        erratio = (erel - ips_input['energy']['value'])/ips_input['tolerance']['value']  # error / tolerance
        # are we done?
        if in_bounds(E, etarget, etol):
            # yes!
            if ips_input['verbose']:
                print('\tvv walker {:d} converged to Erel = {:.1f} in {:d} iterations'.format(ID, erel, itry+1))
            if 'degree' in unitX: 
                # convert angles back to degrees (but not in the gradient)
                Geom.toUnits(unitX)
                # make sure all dihedrals are in the interval (-pi, pi]
                Geom.canonical_angles()
            return E, Geom, grad, itry, ID
        # keep looking
        if ips_input['verbose']:
            print('\tvv itry = {:d} for ID = {:d} gives Erel = {:.1f} (eratio = {:.1f}, erratio = {:.1f})'.format(itry, ID, erel, eratio, erratio))
        # return to top of loop unless special situations below
        if (eratio > 2) and (erratio > 10):
            # landed much too high (use double test in case of very low target energy)
            if algorithm in ['srips']:
                if itry == 0:
                    # first step; return failure and message
                    if ips_input['verbose']:
                        print('\tvv eratio too high for walker {}'.format(ID))
                    return np.nan, 'eratio too high', 0, 0, ID
            elif algorithm in ['mrwips']:
                X = X - dX0  # retreat toward the progenitor point
                S /= GOLD
                X = X + dX0 * S  # advance in the same direction but more timidly
                grad = np.zeros_like(X)
                if ips_input['verbose']:
                    print('\tvv energy much too high; retreat to S({:d}) = {:.3f} for ID = {:d}'.format(itry+1, S, ID))
                continue
        else:
            if 'active' in ips_input:
                # remove components along inactive coordinates
                grad = grad * ips_input['active_vec']
            # Here is the normal stepping procedure
            gradl = np.linalg.norm(grad)  # length of gradient vector
            S = (etarget - E) / gradl  # positive to raise, negative to lower the energy
            grad = normalize(grad)
        # don't exceed trust radius
        if abs(S) > trustR:
            if ips_input['verbose']:
                print('\tvv S = {:.2f} exceeds trustR; scaling back to {:.2f}'.format(S, trustR))
            S *= trustR / abs(S)
    else:
        # iteration limit reached
        if algorithm in ['mrwips']:
            # multi-walker method; only kill this walker
            if ips_input['verbose']:
                print('\tvv exceeded maxiter = {:d} for walker {}'.format(maxiter, ID))
            return np.nan, 'exceeded maxiter = {:d}'.format(maxiter), 0, 0, ID
        else:
            # fatal error
            print_err('maxiter', maxiter)
    return 
##
def read_qm_energy(code, theory, qmout):
    # return a list of the energies (specified by 'theory')
    #   from the quantum chemistry output file 'qmout'
    if 'theory' in ['mp2', 'mp3', 'mp4', 'ccsd']:
        # post-HF energy requires different parser (NOT YET WRITTEN)
        print_err('', 'post-HF theories not yet handled: {:s}'.format(theory))
        Elist = read_qm_E_postHF(code, qmout)
    else:
        # SCF method; use appropriate parser
        Elist = read_qm_E_scf(code, qmout)
    return Elist
##
def qm_function(ips_input, task, verbose=False, option='', unitR='', ID='qm_function_noname', fileroot=''):
    # use quantum chemistry program to perform the specified task
    # output may be modified by 'option'
    # angular forces returned in hartree/radian
    # distance forces returned in hartree/unitR
    # 'ID' is an identifier (any dtype) to avoid filename collisions in parallel computations
    #   'ID' is used only if 'fileroot' is blank
    if not task in ['minimize', 'gradient', 'force', 'energy']:
        # the requested task is not implemented
        print_err('task', task)
    qminp = write_qm_input_ips(ips_input, task, ID, fileroot=fileroot)  #  'qminp' is name of QM input file
    code = ips_input['code']
    qmout = run_qm_job(code, qminp)
    # did it succeed?
    success = qm_calculation_success(code, task, qmout[0])
    if not success:
        if task == 'minimize':
            # geometry optimization failed: try harder 
            problem = qm_diagnose(code, task, qmout[0])
            if verbose:
                print('\tInitial geometry optimization failed: see {:s}'.format(qmout[0]))
            # regardless of the problem, just try again using the 
            #   last (cartesian) coordinates in the failed optimization
            #   This may change ZMatrix() to Geometry()
            if ips_input['coordtype'] == 'zmatrix':
                # change to Geometry()
                ips_input['coordtype'] = 'cartesian'
                ips_input['cartesian'] = ips_input['zmatrix'].toGeometry()
            ips_input['cartesian'] = read_qm_Geometry(code, qmout[0])[-1]
            E_last = read_qm_E_scf(code, qmout[0])[-1]
            if verbose:
                print('\t--trying again starting from E = {:.5f}'.format(E_last))
            mv_file_failed(qmout[0])  # rename the old output file
            qminp = write_qm_input_ips(ips_input, task, ID, fileroot=fileroot)  #  'qminp' is name of QM input file
            qmout = run_qm_job(code, qminp)
            success = qm_calculation_success(code, task, qmout[0])
        else:
            # non-minimization QM calculation failed; write error log and return failure
            qmerrlog = 'qm_err.log'
            with open(qmerrlog, 'a') as ferr, open(qmout[0], 'r') as badout:
                ferr.write('QM failure for file {:s}'.format(qmout[0]))
                for line in badout:
                    ferr.write(line)
            if verbose:
                print('*** QM calculation failed at task "{:s}"'.format(task) +
                    '; see file "{:s}"'.format(qmerrlog))
            # return energy as None and name of output file instead of geometry
            return None, qmout[0], ID
    # get the energy
    Elist = read_qm_energy(ips_input['code'], ips_input['theory'], qmout[0])
    if task == 'energy':
        return Elist[-1], qmout, ID
    coordtype = ips_input['coordtype']
    if task == 'minimize':
        if verbose:
            print('Initial E = {:.6f}; minimized E = {:.6f}'.format(Elist[0], Elist[-1]))
        # also retrieve coordinates
        if coordtype == 'cartesian':
            # retrieve a Geometry object
            optCoord = read_qm_Geometry(code, qmout[0])
        elif coordtype == 'zmatrix':
            # retrieve a ZMatrix object
            optCoord = read_qm_ZMatrix(code, qmout[0])
        # return the last energy and Geometry, and the success flag
        return Elist[-1], optCoord[-1], ID
    if task in ['gradient', 'force']:
        # also retrive force/gradient
        # ******************************************************************** #
        # **  Note that QM gradient units are:  hartree/radian & hartree/bohr  ** #
        # ******************************************************************** #
        if coordtype == 'cartesian':
            # return a numpy array of [x,y,z] force/gradient triples
            gradCoord = read_qm_gradient(code, qmout[0])
            # choose only the last one
            gradCoord = gradCoord[-1]
            if unitR == 'angstrom':
                gradCoord /= BOHR  # change the units from bohr to angstrom
            if task == 'force':
                # change the sign 
                gradCoord *= -1
            if option == 'flatten':
                # return a simple array 
                gradCoord = gradCoord.flatten()
        elif coordtype == 'zmatrix':
            # return a dict of z-matrix variables and their
            #   forces/gradients
            ZMref = ips_input['zmatrix']
            gradCoord = read_qm_ZMgradient(code, qmout[0], ZMref)
            # use only the last one
            gradCoord = gradCoord[-1]
            if unitR == 'angstrom':
                # change the (inverse) distance units from bohr to angstrom
                for zvar in gradCoord:
                    if ZMref.vtype[zvar] == 'distance':
                        gradCoord[zvar] /= BOHR
            if task == 'force':
                # change the sign
                for zvar in gradCoord:
                    gradCoord[zvar] *= -1 
            if option == 'flatten':
                # return a simple array, without labels
                vec = [gradCoord[zvar] for zvar in sorted(ZMref.val)]
                gradCoord = np.array(vec)
        else:
            print_err('coordtype', coordtype)
        return Elist[-1], gradCoord, ID
##
def read_yinput(user_input_file, default_config_file):
    # read the user's YAML input file, *.yml
    # also read the default configuration (i.e., input) file
    # return a dict
    with open(user_input_file) as fyaml:
        inp_user = yaml.safe_load(fyaml)
    molecule = filename_root(user_input_file)
    inp_user['molecule'] = molecule
    # read defaults
    with open(default_config_file) as fdef:
        inp_def = yaml.safe_load(fdef)
    # recursively install anything present in the defaults
    # that is missing from the user input
    backfill_dict(inp_def, inp_user)
    return inp_user
##
def parse_yinput(input_raw):
    # convert units, simple input checking
    #
    # convert all top-level keywords to lowercase
    parsed = dict( (k.lower(), v) for k,v in input_raw.items() )
    known_ips = ['mrwips', 'srips']
    # below, the lists of units and conversion factors must correspond
    keyw_measure = {'energy'  : ['energy', 'tolerance', 'ramp'],
                    'distance': ['stepl', 'stepl_atom', 
                                 'max_stepl', 'random_kick',
                                 'max_stepl_atom']}
    known_units  = {'energy'  : ['kcal/mol', 'ev', 'hartree', 'kj/mol'],
                    'distance': ['bohr', 'ang']}
    pref_unit    = {'energy'  : 'kj/mol',
                    'distance': 'angstrom'}
    #unit_conver  = {'energy'  : [KCAL_KJ, EV_KJMOL, au_kjmol, 1.0],
    #                'distance': [bohr, 1.0]}
    # some theories do not have an explicit basis set
    no_basis = ['am1', 'pm3', 'pm6']
    # some keywords require integer values
    keyw_int = ['ramp', 'steps', 'pixel', 'charge', 'ramp',
                'random_kick', 'nprocs', 'save_freq', 'nstep']
    parsed['coordtype'] = None
    # check for both cartesian and z-matrix coordinate blocks
    if ('cartesian' in parsed) and ('zmatrix' in parsed):
        print_err('cartesian and zmatrix blocks are both present')
    # check that max atomic step is larger than desired atomic step
    try:
        s = parsed['stepl_atom']['value']
        smax = parsed['max_stepl_atom']['value']
    except:
        s = 0
        smax = 1
    if s > smax:
        print_err('', 'max_stepl_atom ({}) must be '.format(smax) +
            'larger than stepl_atom ({})'.format(s))
    to_delete = []  # list of keywords to delete
    # loop through the input dict
    for key in parsed:
        try:
            field = parsed[key].copy()
        except:
            # not an object
            field = parsed[key]
        # known quantum code?
        if key == 'code':
            parsed[key] = parsed[key].lower()
            if parsed[key] not in KNOWN_CODES:
                print_err('unknown quantum code {:s}'.format(parsed[key]))
        # known IPS algorithm?
        if key == 'ips_method':
            ips = parsed[key]['name'].lower()
            if ips not in known_ips:
                print_err('', 'unknown IPS algorithm: {:s}'.format(ips))
            parsed[key]['name'] = ips
        # the user must specify an energy target for IPS
        if key == 'energy' and 'value' not in parsed[key]:
            print_err('', 'energy target value must be specified')
        if key == 'pbc':
            # periodic boundary conditions 
            if not parsed['pbc']['use_pbc']:
                # remove the keyword
                to_delete.append('pbc')
        # unit checking and conversions
        for quantity in keyw_measure:
            # quantity = 'energy' or 'distance'
            if key in keyw_measure[quantity]:
                # expect a value and an appropriate unit
                if ('value' not in parsed[key]) and ('how_much' not in parsed[key]):
                    # it's just a unit with no value; flag it for deletion
                    to_delete.append(key)
                    break
                field = convert_unit(field, pref_unit[quantity])
                if False:
                    uinp = field['unit'].lower()
                    for iunit in range(len(known_units[quantity])):
                        if known_units[quantity][iunit] in uinp:
                            # convert value to preferred unit
                            try:
                                field['value'] *= unit_conver[quantity][iunit]
                            except:
                                # 'how_much' instead of 'value'
                                field['how_much'] *= unit_conver[quantity][iunit]
                            field['unit'] = pref_unit[quantity]
                            break
                    else:
                        # unknown unit 
                        print_err('units', '{:s} for {:s}'.format(uinp, key))
        if key == 'spinmult':
            # convert from string (chosen for clarity) and integer (2S+1)
            field = spinname(field)
        if key in keyw_int:
            # require some parameter to be an integer
            try:
                if type(field['how_often']) is not int:
                    print_err('need_int', key)
            except:
                # simple scalar, or no 'howmuch' value
                if type(field) is not int:
                    print_err('need_int', key)
        if key == 'normal_mode':
            # replace range strings by list of int
            modes = []  # list to grow
            for word in field:
                # convert to int; possible hyphenated range
                modes.extend(range_to_list(word))
            field = modes.copy()
        if key == 'cartesian':
            # Cartesian coordinates
            parsed['coordtype'] = 'cartesian'
            # convert to a Geometry object
            crd = [line.split() for line in parsed['cartesian'].splitlines()]
            field = Geometry(crd, intype='1list', units=parsed['coord_unit'])
            # ensure coordinates unit is angstrom
            field.toAngstrom()
        if key == 'zmatrix':
            # molecule specified using Z-matrix
            parsed['coordtype'] = 'zmatrix'
            # convert to a ZMatrix object
            # first strip any leading atom numbers
            regI = re.compile(r'^\s*\d+\s*')  # check for leading atom index
            zmt = parsed['zmatrix'].splitlines()
            for i in range(len(zmt)):
                zmt[i] = regI.sub('', zmt[i])
            # input angles are assumed to be in degrees
            field = parse_ZMatrix(zmt)
            # check zmatrix for missing variable values
            if field.checkVals(verbose=True):
                sys.exit('You need to fix the Z-matrix.')
            # ensure distances are in angstrom
            field.toAngstrom()
        if key == 'constraint':
            field = parse_constraint_input(field)
        # replace the original keyword values
        parsed[key] = field
    # done with loop over keys
    #
    # delete useless keys
    dict_delkey(parsed, to_delete)
    if (parsed['code'] in ['quantum-espresso']) and \
        ('pbc' not in parsed):
        print_err('', 'code ' + parsed['code'] + 
            ' requires periodic boundary conditions (pbc)')
    # in the returned dict, 'code_options', 'theory', and 'basis'
    # should refer only to the selected quantum code
    options_default = parsed['code_options'][parsed['code']]
    line_options = ['theory', 'basis']
    for key in options_default:
        # specify 'theory' and 'basis' if user did not
        if key in line_options:
            if key not in parsed:
                parsed[key] = options_default[key]
        elif key not in parsed['code_options']:
            # user did not specify this code-specific option
            parsed['code_options'][key] = options_default[key]
    cleanup_code_options(parsed['code'], parsed['code_options'])
    # give ALL atoms unit mass; this is IPS, not dynamics
    install_unit_masses(parsed[parsed['coordtype']])
    #
    if ('zmatrix' in parsed) and ('active' in parsed):
        # list of always-active z-matrix variables (usually free torsions)
        #   expand any wildcards
        # also boolean vector (for variables in alphabetical order)
        parsed['active'], parsed['active_vec'] = \
            parse_active_coords(parsed['active'], parsed['zmatrix'])
    #
    if ('dflood' in parsed) and (parsed['coordtype'] != 'zmatrix'):
        print_err('', 'dimensional flooding requires a symbolic Z-matrix')
    if 'ips_method' not in parsed:
        print_err('', 'no IPS algorithm \("ips_method"\) was specified')
    # process step length information
    if 'stepl' in parsed:
        # don't need the lower-priority 'stepl_atom' parameter
        dict_delkey(parsed, 'stepl_atom')
    ips_opts = parsed['ips_method']
    apply_steplatom = False
    if ips_opts['name'] == 'mrwips':
        # MRWIPS algorithm
        # resolve multiple specifications of step length
        # priority:  stepl > step_angle > stepl_atom
        if 'stepl' not in parsed:
            if 'step_angle' not in ips_opts:
                if 'stepl_atom' in parsed:
                    # use stepl_atom value and units
                    apply_steplatom = True
                else:
                    print_err('', 'step length must be specified in some way')
            else:
                # will apply step_angle after the seed points are available
                #   to define the radius from the origin
                if ips_opts['step_angle'] > np.pi/2:
                    # user meant degrees?
                    print_err('', 'step angle of {:.2f} radian is unreasonably large')
                # don't need the lower-priority 'stepl_atom' key
                dict_delkey(parsed, 'stepl_atom')
        else:
            # won't need the lower-priority step-angle specifications
            dict_delkey(ips_opts, 'step_angle')
        if 'walkers' not in ips_opts:
            print_err('', 'number of walkers must be specified for MRWIPS')
    elif ips_opts['name'] == 'srips':
        # SRIPS algorithm
        # priority: stepl > stepl_atom
        if 'stepl' not in parsed:
            if 'stepl_atom' in parsed:
                # use stepl_atom
                apply_steplatom = True
            else:
                print_err('', 'step length must be specified in some way')
    if apply_steplatom:
        # compute 'stepl' value
        print('---- apply stepl_atom')
        parsed['stepl'] = {
            'value': parsed['stepl_atom']['value'] * \
                     parsed[parsed['coordtype']].natom(),
            'unit' : parsed['stepl_atom']['unit'] }
    # check that max steps are larger than desired steps
    try:
        s = parsed['stepl']['value']
        smax = parsed['max_stepl']['value']
    except:
        s = 0
        smax = 1
    if s > smax:
        print_err('', 'max_stepl ({}) must be larger than stepl ()'.format(smax,s))
    # remove the unneeded code_options from the default config file
    n = dict_delkey(parsed['code_options'], KNOWN_CODES)
    # if restarting, remove the 'silent_delete' key (in default config file)
    if parsed['continuation']:
        dict_delkey(parsed, 'silent_delete')
    # delete 'basis' key if not needed
    if parsed['theory'] in no_basis:
        dict_delkey(parsed, 'basis')
    return parsed
##
def cleanup_code_options(code, code_options):
    # given the dict parsed['code_options'], make it refer only to the specified code
    # return the modified dict
    to_delete = []
    if code == 'gaussian09':
        if 'nstep' in code_options:
            # rename this 'maxcyc'
            if 'maxcyc' not in code_options:
                code_options['maxcyc'] = code_options['nstep']
                to_delete.append('nstep')
    elif code == 'quantum-espresso':
        block_opts = {
            'control'  : ['calculation', 'restart_mode', 'nstep',
                'outdir', 'pseudo_dir', 'tprnfor'],
            'system'   : ['ecutwfc', 'ecutrho', 'occupations', 'degauss',
                'ntyp', 'nat'],
            'electrons': ['conv_thr', 'mixing_beta'],
            'ions'     : ['']}
        block_names = list(block_opts.keys())
        # move unassigned code_options inside the appropriate block
        # user may not specify 'title' or 'prefix'
        dict_delkey(code_options, ['title', 'prefix'])
        for block in block_opts:
            # create a separate list for each block
            if block not in code_options:
                code_options[block] = {}
        if 'maxcyc' in code_options:
            # rename this 'nstep'
            if 'nstep' not in code_options:
                code_options['nstep'] = code_options['maxcyc']
                to_delete.append('maxcyc')
        allowed_cmd = [x for y in block_opts for x in block_opts[y]]
        allowed_cmd += ['k_points']
        for opt in code_options:
            if opt not in (allowed_cmd + block_names + KNOWN_CODES):
                print_err('', 'unknown code option {}'.format(opt))
            for block in block_opts:
                if opt in block_opts[block]:
                    code_options[block][opt] = code_options[opt]
                    to_delete.append(opt)
    else:
        print_err('code', code)
    dict_delkey(code_options, to_delete)
    return
##
def read_input(input_file):
    # read the user's input file, *.ips
    # return a dict
    # keywords are all changed to lowercase
    retval = {}
    regxComment = re.compile(r'^\s*#')  # comment lines have '#' as first non-blank character
    regxKey = re.compile(r'^\s*[A-Za-z_]\w*\s*{')  # keyword begins line and followed by '{'
    regxOneLiner = re.compile(r'{\s*(\S+.*)\s*}')  # for one-line data lists
    regxBlank = re.compile(r'^\s*$')
    regxSplitLeft = re.compile(r'[{\s]')
    regxSplitRight = re.compile(r'[}\s]')
    with open(input_file, 'r') as inp:
        inField = False
        for line in inp:
            if regxComment.match(line):
                # ignore comment lines
                continue
            if regxBlank.match(line):
                # ignore blank lines
                continue
            words = regxSplitLeft.split(line)
            if not inField:
                # expect a new keyword
                if not regxKey.match(line):
                    # bad keyword
                    print('Invalid keyword specification on line:\n\t{:s}'.format(line.rstrip()))
                    continue
                keyword = words[0].lower()  # convert keyword to lower case
                m = regxOneLiner.search(line)
                if m:
                    # read all keyword data from this one line
                    retval[keyword] = m.group(1).split()
                else:
                    # a multi-line keyword section; each line will be kept together
                    inField = True
                    buf = line.strip().split('{')
                    if buf[-1] != '':
                        # use the remainder of the keyword line
                        values = [buf[-1].strip()]  # string to the right of the (last) '{' character
                    else:
                        values = []
            else:
                # continue to read data for present keyword
                if not '}' in line:
                    values.append(line.strip())  # don't break into space-delimited words
                else:
                    # don't include the (first) '}' character or anything after
                    buf = line.strip().split('}')
                    if buf[0] != '':
                        values.append(buf[0].strip())
                    inField = False  # done reading data for keyword
                    retval[keyword] = values.copy()
    molecule = filename_root(input_file)
    retval['molecule'] = molecule
    return retval
##
def parse_input(input_raw):
    # convert keyword arguments to appropriate types
    parsed = input_raw.copy()
    parsed['coordtype'] = 'none'  # should end up either 'cartesian' or 'zmatrix'
    for key in parsed:
        try:
            # object
            field = parsed[key].copy()
        except:
            # scalar
            field = parsed[key]
        if key in ['energy', 'tolerance', 'ramp']:
            #
            # these keywords may include explicit energy units as the last word
            #
            if re.search('kcal', field[-1], re.IGNORECASE):
                # convert from kcal to kJ
                field[-2] = float(field[-2]) * 4.184
                del field[-1]
            elif re.search('ev', field[-1], re.IGNORECASE):
                # convert from eV to kJ/mol
                field[-2] = float(field[-2]) * 96.485
                del field[-1]
            elif re.search('hartree', field[-1], re.IGNORECASE):
                # convert from hartree to kJ/mol
                field[-2] = float(field[-2]) * 2625.50
                del field[-1]
            elif re.search('kj', field[-1], re.IGNORECASE):
                # no unit conversion, but remove explicit units and convert string to float
                field[-2] = float(field[-2])
                del field[-1]
            elif re.search(r'[^\d\.-]', field[-1]):
                print('*** Ignoring unrecognized energy unit for keyword {:s}: {:s}'.format(keyword, field[-1]))
                field[-2] = float(field[-2])
                del field[-1]
            else:
                # no energy unit was specified
                field[-1] = float(field[-1])
        if key in ['ramp', 'steps', 'pixel', 'charge', 'ramp', 'random_kick', 'nprocs', 'save_freq']:
            #
            # convert first keyword argument to int
            #
            field[0] = int(field[0])
        if key in ['method']:
            #
            # convert all except first argument to int
            #
            for i in range(1, len(field)):
                field[i] = int(field[i])
        if key in ['stepl', 'stepl_atom', 'random_kick', 'step_angle', 'dflood', 'dissociated', 'bondtol']:
            #
            # convert last keyword argument to float
            #
            field[-1] = float(field[-1])
        if key == 'normal_mode':
            #
            # replace range strings by list of int
            #
            modes = []  # list to grow
            for word in field:
                # convert to int; possible hyphenated range
                modes.extend(range_to_list(word))
            field = modes.copy()
        if key in ['suppress_rotation', 'minimize', 'continuation']:
            #
            # convert first argument to boolean
            #
            w = field[0].lower()
            field[0] = ( (w == 'true') or (w == 'yes') )
        if key == 'spinmult':
            # convert from string (chosen for clarity) and integer (2S+1)
            field = [spinname(field[0])]
        if key == 'cartesian':
            # Cartesian coordinates
            parsed['coordtype'] = 'cartesian'
            # convert to a Geometry object
            elem = []
            xyz = []  # list of lists [x,y,z]
            natom = 0  # counter
            for row in field:
                triple = []
                line = row.split()
                natom += 1
                for a in line[0:4]:
                    # ignore columns past the fourth
                    try:
                        triple.append(float(a))
                    except:
                        # must be an element symbol
                        elem.append(a)
                if len(elem) != natom:
                    print('*** natom = {:d} but there are {:d} element symbols in "parse_input()"'.format(natom, len(elem)))
                elif len(triple) != 3:
                    print('*** Expected 3 coordinates for atom but found:', triple)
                else:
                    xyz.append(triple)
                    if len(xyz) != natom:
                        print('*** natom = {:d} but there are {:d} coordinate triples in "parse_input()"'.format(natom, len(xyz)))
                    else:
                        # all is well
                        field = Geometry(elem, xyz, intype='2lists')
        if key == 'zmatrix':
            # molecule specified using Z-matrix
            parsed['coordtype'] = 'zmatrix'
            # convert to a ZMatrix object
            # first strip any leading atom numbers
            regI = re.compile(r'^\s*\d+\s*')  # check for leading atom index
            for i in range(len(field)):
                field[i] = regI.sub('', field[i])
            # input angles are assumed to be in degrees
            field = parse_ZMatrix(field)
            # check zmatrix for missing variable values
            if field.checkVals(verbose=True):
                sys.exit('You need to fix the Z-matrix.')
        if key == 'constraint':
            field = parse_constraint_input(field)
        if key == 'code_options':
            # replace with a dict
            # insert another element to indicate which part of the QM input file
            #    this options belongs in
            field = parse_code_options(input_raw['code'][0], field)
        if key in ['check_input', 'continuation', 'energy', 'tolerance', 'steps', 'stepl', 'minimize', 'pixel', 'charge', 
            'spinmult', 'theory', 'code', 'basis', 'comment', 'nprocs', 'step_angle', 'suppress_rotation', 'dflood',
            'stepl_atom', 'dissociated', 'bondtol']:
            #
            # instead of a list of length one, return the element as a scalar
            #
            field = field[0]
        # replace the original keyword values
        parsed[key] = field
    ##
    # check for both cartesian and z-matrix coordinate blocks
    if ('cartesian' in parsed) and ('zmatrix' in parsed):
        print('*** Error: cartesian and zmatrix blocks are both present.')
        sys.exit('Delete one from your input file and try again.')
    if ('zmatrix' in parsed) and ('active' in parsed):
        # list of always-active z-matrix variables (usually free torsions)
        #   expand any wildcards
        # also boolean vector (for variables in alphabetical order)
        parsed['active'], parsed['active_vec'] = parse_active_coords(parsed['active'], parsed['zmatrix'])
    # give ALL atoms unit mass; this is IPS, not dynamics
    install_unit_masses(parsed[parsed['coordtype']])
    #
    # check for conflicts
    #
    if ('dflood' in parsed) and (not parsed['coordtype'] == 'zmatrix'):
        sys.exit('*** Conflict: dimensional flooding requires a symbolic Z-matrix.')
    if ('step_angle' in parsed) and (not parsed['method'][0] in ['mrwips']):
        # 'step_angle' only applies for MRWIPS algorithm; remove the keyword
        del parsed['step_angle']
    #
    # Install default values for missing keywords
    #
    if not 'nprocs' in parsed:
        # number of (parallel) processes
        parsed['nprocs'] = 1
    # things that should default to True
    for boolean_keyw in ['continuation', 'suppress_rotation', 'minimize']:
        if not (boolean_keyw in parsed):
            parsed[boolean_keyw] = True
    if not 'code_options' in parsed:
        parsed['code_options'] = []
    if not 'dissociated' in parsed:
        # distance threshold for judging dissociation
        parsed['dissociated'] = 0.  # default is to ignore
    if not 'bondtol' in parsed:
        parsed['bondtol'] = 1.3
    if parsed['method'][0] in ['mrwips']:
        # resolve multiple specifications of step length
        # priority:  stepl > step_angle > stepl_atom
        if not ('stepl' in parsed):
            if not ('step_angle' in parsed):
                if 'stepl_atom' in parsed:
                    # use stepl_atom
                    parsed['stepl'] = parsed['stepl_atom'] * parsed[parsed['coordtype']].natom()
                else:
                    print('*** Using default initial stepl = 0.1')
                    parsed['stepl'] = 0.1
            else:
                # will apply step_angle after the seed points are available
                pass
    else:
        # SRIPS algorithm
        # priority: stepl > stepl_atom
        if not ('stepl' in parsed):
            if 'stepl_atom' in parsed:
                # use stepl_atom
                parsed['stepl'] = parsed['stepl_atom'] * parsed[parsed['coordtype']].natom()
            else:
                print('*** Using default initial stepl = 0.1')
                parsed['stepl'] = 0.1           
    #
    return parsed
##
def parse_active_coords(crdlist, ZM):
    # parse any wildcard characters to return a fully explicit
    #   list of z-matrix coordinates
    # also return a boolean vector indicating which z-matrix
    #   variables are active
    xlist = []
    regxwild = re.compile('[*?]')
    for crd in crdlist:
        if regxwild.search(crd):
            # this string contains at least one '*' wildcard
            rstr = crd.replace('?', '.')
            rstr = rstr.replace('*', '.*')
            rstr += '$'  # must match the whole name
            regx = re.compile(rstr)
            for varname in ZM.val:
                if regx.match(varname):
                    xlist.append(varname)
        else:
            # simple coordinate name
            xlist.append(crd)
    # build the boolean vector
    # ordering is alphabetical by variable name to match ZMatrix.toVector()
    varnames = sorted(ZM.val)
    nvar = len(varnames)
    boolvec = np.full(nvar, False, dtype='bool')
    for icrd in range(nvar):
        if varnames[icrd] in xlist:
            boolvec[icrd] = True
    return xlist, boolvec
##
def parse_code_options(code, optlist):
    # return a dict of code options, including the part of the QM
    #   input file in which the option belongs (header, command, or trailer),
    #   as needed by write_qm_input() in qm_subs.py
    # expected syntax:  option=value  ('=' as the delimiter)
    # output 'value' is a list, where 0-th element is file-part
    options = {}
    regex = re.compile(r'(.*?)=(.*)')
    if code == 'gaussian09':
        for elem in optlist:
            m = regex.match(elem)
            if m:
                (opt, val) = (m.group(1).lower(), m.group(2))
            else:
                # this option must not take any value; assign ''
                (opt, val) = (elem.lower(), '')
            # assign file-part; default is 'command'
            part = 'command'
            if opt in ['chk', 'mem', 'nprocs']:
                # belongs in header ("Link0" command)
                part = 'header'
            elif opt in []:
                # I can't think of any examples for file trailer
                part = 'trailer'
            options[opt] = [part, val]
    else:
        print_err('code', code)
    return options
##
def parse_constraint_input(inp_list):
    # return a dict of constraint keywords and their arguments
    parsed = {}
    for field in inp_list:
        words = field.split()
        keyword = words.pop(0)
        if keyword in ['contraction']:
            # no arguments for these keywords
            parsed[keyword] = ''
        elif keyword in ['sphere', 'sticky']:
            # convert argument to float
            parsed[keyword] = float(words[0])
        elif keyword in ['frozen', 'phi', 'r', 'theta']:
            # convert to list of integers
            args = []
            for w in words:
                args.extend(range_to_list(w))
            parsed[keyword] = args.copy()
    return parsed
##
def range_to_list(hyphenated_range):
    # convert one hyphenated range of int to a list of int
    #   a simple argument will return a list with one element
    retval = []
    w = re.split(r'(?<=\d)-(?=[-\d])', hyphenated_range)  # termini may be negative
    if len(w) == 1:
        # just add this number to the list
        retval.append(int(w[0]))
    elif len(w) == 2:
        # add the numbers in this range (including both ends)
        w = [int(x) for x in w]
        incr = 1
        if w[1] < w[0]:
            incr = -1
        retval.extend(list(range(w[0], w[1]+incr, incr)))
    else:
        print('*** Unrecognized integer range in range_to_list():', hyphenated_range)
    return retval
##
def format_g09_input_ips(ips_input):
    # prepare to write an input file for Gaussian09
    #    based upon 'ips_input'
    #
    # '%' commands go in the header
    cmd_header = ['chk', 'mem', 'nprocs']
    # some are specified with the 'scf' keyword
    cmd_scf = ['xqc']
    # some are specified with the 'geom' keyword
    cmd_geom = ['nocrowd']
    #
    coordtype = ips_input['coordtype']
    contents = ips_input.copy()
    # prepare the header and the command line
    hdr = {}
    scf_cmd = []
    geom_cmd = []
    cmd = '# ' + ips_input['theory']
    if 'basis' in ips_input:
        cmd += '/' + ips_input['basis']
    code_options = ips_input['code_options']
    for opt in cmd_header:
        if opt in code_options:
            hdr[opt] = code_options[opt]
    for opt in cmd_scf:
        if opt in code_options:
            scf_cmd.append(opt)
    for opt in cmd_geom:
        if opt in code_options:
            geom_cmd.append(opt)
    if len(scf_cmd) > 0:
        cmd += ' scf=(' + ','.join(scf_cmd) + ')'
    if len(geom_cmd) > 0:
        cmd += ' geom=(' + ','.join(geom_cmd) + ')'
    if 'command' in code_options:
        # add this string to the command line without change
        cmd += ' ' + ' '.join(code_options['command'])
    contents['command'] = cmd
    if len(hdr) > 0:
        contents['header'] = hdr
    # prepare the coordinates (Cartesian); replace object
    coordstr = format_qm_coordinates('gaussian09', ips_input[coordtype])
    contents['coordinates'] = coordstr
    # descriptive comment
    contents['comment'] = 'IPS calculation for {:s}'.format(ips_input['molecule'])
    return contents
##
def write_qm_input_ips(ips_input, task='', ID=0, fileroot=''):
    # just call the routine tailored to the quantum code
    code = ips_input['code']
    if code == 'gaussian09':
        fname = write_g09_input_ips(ips_input, task=task,
                ID=ID, fileroot=fileroot)
    elif code == 'quantum-espresso':
        fname = write_qe_input_ips(ips_input, task=task,
                ID=ID, fileroot=fileroot)
    else:
        print_err('code', code)
    return fname
##
def format_qe_input_ips(ips_input, task):
    # prepare to write an input file for Quantum Espresso
    #    based upon 'ips_input'
    # this can modify ips_input
    amp_cmd = {
        'control'  : ['calculation', 'restart_mode', 'nstep',
            'outdir', 'pseudo_dir', 'tprnfor'],
        'system'   : ['ecutwfc', 'ecutrho', 'occupations', 'degauss',
            'ntyp', 'nat'],
        'electrons': ['conv_thr', 'mixing_beta'],
        'ions'     : ['']}
    allowed_cmd = [x for y in amp_cmd for x in amp_cmd[y]]
    allowed_cmd += ['k_points']
    # require that blocks appear in a specific order
    block_names = ['control', 'system', 'electrons', 'ions']
    code_options = ips_input['code_options']
    if task == 'gradient':
        # remove the IONS section
        del amp_cmd['ions']
        block_names.remove('ions')
    if False:
        # move unassigned 'code_options' inside the appropriate section
        # user may not specify 'title' or 'prefix'
        dict_delkey(code_options, ['title', 'prefix'])
        for block in block_names:
            # create a separate lists for each &section
            if block not in code_options:
                code_options[block] = {}
        to_delete = []
        for opt in code_options:
            if opt not in (allowed_cmd + block_names):
                print_err('', 'unknown code option {}'.format(opt))
            for block in block_names:
                if opt in amp_cmd[block]:
                    code_options[block][opt] = code_options[opt]
                    to_delete.append(opt)
        dict_delkey(code_options, to_delete)
    # get PBC parameters
    pbc_opts = ips_input['pbc']
    if 'ang' in pbc_opts['edge']['unit']:
        # QE requires celldm in bohr; convert
        pbc_opts['edge']['value'] /= BOHR
        pbc_opts['edge']['unit'] = 'bohr'
    code_options['system']['celldm(1)'] = pbc_opts['edge']['value']
    # compute 'ntyp' (no. of atom types) and 'natom' (no. of atoms)
    coordtype = ips_input['coordtype']
    stoich = ips_input[coordtype].stoichiometry(as_dict=True)
    code_options['system']['ntyp'] = len(stoich)
    code_options['system']['nat'] = ips_input[coordtype].natom()
    contents = ips_input.copy()
    # YAML does not handle scientific notation adequately
    regex_sci = re.compile(r'[-]?\d*\.\d+[DdEe][-+]?\d+')
    # sections starting with &<BLOCK NAME>
    amp_str = ''  
    for block in block_names:
        amp_str += ' &{:s}\n'.format(block.upper())
        blk = code_options[block]
        for opt in blk:
            # formatting depends upon type of value
            if type(blk[opt]) is str:
                if regex_sci.match(blk[opt]):
                    # scientific notation, not string
                    s = '{:s} ,'.format(blk[opt])
                else:
                    # add single quotes around string 
                    s = "'{:s}' ,".format(blk[opt])
            elif type(blk[opt]) is bool:
                # fortran-style .true. or .false.
                s = '.{:s}. ,'.format(str(blk[opt]).lower())
            else:
                # probably numeric
                s = '{} ,'.format(blk[opt])
            amp_str += '    {:s} = {:s}\n'.format(opt, s)
        # add closing line
        amp_str += ' /\n'
    # remaining sections, without & prefix
    no_amp = 'ATOMIC_SPECIES\n'
    # chemical elements, with mass and pseudopotential
    for e in stoich:
        e = e.upper()
        s = '   {:s}  {:8.4f}  {:s}.{:s}\n'.format(e, 
            atomic_weight(e), e, ips_input['basis'])
        no_amp += s
    # atomic coordinates
    no_amp += 'ATOMIC_POSITIONS {:s}\n'.format(ips_input['coord_unit'])
    coordstr = format_qm_coordinates('quantum-espresso', ips_input[coordtype])
    no_amp += coordstr
    # k points
    no_amp += 'K_POINTS {:s}\n'.format(code_options['k_points'])
    contents['amp'] = amp_str
    contents['noamp'] = no_amp
    return contents
##
def write_qe_input_ips(ips_input, task='', ID=0, fileroot=''):
    # write an input file for Quantum Espresso based upon the ips_input
    # 'task' may indicate one additional action:
    #   'minimize', 'gradient', 'freq'
    # 'ID' is an identifier (any dtype) to avoid filename collisions in parallel jobs
    #   it's only used if 'fileroot' is blank
    # 'fileroot' will be used to construct the filenames unless blank
    #
    code = 'quantum-espresso'
    # act upon 'task'
    if task == 'minimize':
        ips_input['code_options']['control']['calculation'] = 'relax'
    if task == 'gradient':
        # this is the default configuration
        ips_input['code_options']['control']['tprnfor'] = True
        ips_input['code_options']['control']['calculation'] = 'scf'
    if task == 'freq':
        # this is not yet implemented: requires ph.x instead of pw.x
        print_err('undone', 'phonon calculation using {:s}'.format(code))
    # create a filename and 'prefix'
    if len(fileroot) == 0:
        # create a filename
        froot = '{:s}_{:s}'.format(ips_input['molecule'], str(ID))  # root of quantum chemistry input filename
    else:
        # use the supplied name
        froot = fileroot
    ips_input['code_options']['control']['prefix'] = froot
    ips_input['code_options']['control']['title'] = \
        'IPS calculation for {:s}'.format(ips_input['molecule'])
    filename = supply_qm_filename_suffix(code, froot, 'input')
    # prepare the contents of the QE input file
    contents = format_qe_input_ips(ips_input, task)
    write_qm_input(filename, code, contents)
    return filename
##
def write_g09_input_ips(ips_input, task='', ID=0, fileroot=''):
    # write an input file for Gaussian09 based upon the ips_input
    # 'task' may indicate one additional action:
    #   'minimize', 'gradient', 'freq'
    # 'ID' is an identifier (any dtype) to avoid filename collisions in parallel jobs
    #   it's only used if 'fileroot' is blank
    # 'fileroot' will be used to construct the filenames unless blank
    contents = format_g09_input_ips(ips_input)
    code = 'gaussian09'
    # check for some possible problems
    # see if 'guess=check' is requested but the checkpoint file 
    #   does not yet exist 
    if 'header' in contents:
        for key in contents['header']:
            if key == 'chk':
                chkfile = contents['header'][key]
                # does it exist?
                if not os.path.isfile(chkfile):
                    # No. Remove any 'guess=check' command
                    m = re.search(r'\s(guess=check)', contents['command'], re.IGNORECASE)
                    if m:
                        print('*** Suppressing "guess=check" because checkpoint file {:s} does not exist.'.format(chkfile))
                        contents['command'] = re.sub(m.group(1), '', contents['command'])
    # add any additional action to the command
    action = {'minimize': ' opt', 'gradient': ' force',
        'freq': ' freq'}
    if task in action:
        contents['command'] += action[task]
    if len(fileroot) == 0:
        # create a filename
        froot = '{:s}_{:s}'.format(ips_input['molecule'], str(ID))  # root of quantum chemistry input filename
    else:
        # use the supplied name
        froot = fileroot
    filename = supply_qm_filename_suffix(code, froot, 'input')
    write_qm_input(filename, code, contents)
    return filename
##
def write_exg(molec, walkers, style='text', X0=''):
    # write E, X, G data to files 
    for i in range(len(walkers)):
        walkers[i].write_exg(molec, style=style, X0=X0)
    return
##
def confirm_newstart(ips_input):
    # prompt user (or not) to delete existing files for this molecule
    molec = ips_input['molecule']
    files = glob.glob('{:s}_*.*'.format(molec))
    nfiles = len(files)
    if nfiles == 0:
        # nothing to do
        return
    if (not ips_input['continuation']) and (not ips_input['silent_delete']):
        # prompt user
        print('Do you really want to delete all {:d} earlier files for "{:s}"? '.format(nfiles, molec), end='')
        u = input().lower()
        if 'y' in u:
            # delete the files
            for file in files:
                #print('\t{:s} '.format(file))
                os.remove(file)
            print('{:d} files deleted.'.format(nfiles))
        else:
            # do nothing
            print('OK, nevermind.')
    return
##
def store_E0(ips_input, E0):
    # install reference energy in appropriate places
    ips_input['E0'] = E0
    # save E0 to a text file
    f0 = open('{:s}_E0.txt'.format(ips_input['molecule']), 'w')
    f0.write('{:6f}\n'.format(E0))
    f0.close()
    return 
##
def recall_E0(ips_input):
    # retrieve reference energy
    f0 = open('{:s}_E0.txt'.format(ips_input['molecule']), 'r')
    E0 = f0.readline().rstrip()
    f0.close()
    ips_input['E0'] = float(E0)
    return 
##
def user_interrupt(molec):
    # check for existence of a file named '{molec}_ips.stop'
    return os.path.exists('{:s}_ips.stop'.format(molec))
##
def save_pickle(ips_input, walkers):
    # save the pickle file
    with open('{:s}.pkl'.format(ips_input['molecule']), 'wb') as fpkl:
        pickle.dump(walkers, fpkl)
        pickle.dump(ips_input, fpkl)
    return
##
def restore_pickle(ips_input):
    # restore data from earlier calculations
    # return walkers[] list
    with open('{:s}.pkl'.format(ips_input['molecule']), 'rb') as fpkl:
        walkers = pickle.load(fpkl)
        old_input = pickle.load(fpkl)
    # merge the new into the old input commands
    for keyw in old_input:
        if keyw in ['code_options', 'dflood', 'stepl', 'stepl_atom', 'nprocs', 'ramp',
            'random_kick', 'tolerance', 'steps', 'suppress_rotation', 'continuation']:
            # overwrite with the new value
            try:
                old_input[keyw] = ips_input[keyw].copy()
            except:
                # maybe not an object
                try:
                    old_input[keyw] = ips_input[keyw]
                except:
                    # maybe missing from new input; do nothing
                    pass
    return walkers, old_input.copy()
##
def install_unit_masses(Struct):
    # given a Geometry() object, set all atomic masses = 1.
    natom = Struct.natom()
    try:
        Struct.set_masses( [1.]*natom )
    except:
        # probably a ZMatrix object; do nothing
        pass
    return
##
def mv_file_failed(filename):
    # rename a file {froot}.{suf} to {froot}_fail.{suf}
    froot, suf = os.path.splitext(filename)
    newname = '{:s}_fail{:s}'.format(froot, suf)
    print('\trenaming file {:s} to {:s}'.format(filename, newname))
    os.rename(filename, newname)
    return newname
##
def walker_string(ID):
    # provide a general label to simplify print statements
    if ID is None:
        wstring = ''
    else:
        wstring = ' for walker {:d}'.format(ID)
    return wstring
##
def energy_ramp(iteration, ips_input):
    # test whether it is time to change the target energy, and do so
    if not ('ramp' in ips_input):
        # no ramp was requested; do nothing
        return
    if (iteration > 0) and not (iteration % ips_input['ramp']['how_often']):
        # yes, change the target energy
        ips_input['energy']['value'] += ips_input['ramp']['how_much']
        print('Target energy ramped to {:.1f} kJ/mol relative.'.format(ips_input['energy']['value']))
    return
##
def dflood_activate(gradient, ips_input, ID=None):
    # Check whether any inactive Z-matrix coordinates should be activated.
    # Activate them and return a boolean numpy array showing which coordinates
    #   were turned on.
    if not ('dflood' in ips_input):
        # dimensional flooding was not requested; do nothing
        return None
    if ips_input['coordtype'] != 'zmatrix':
        # this is a redundant test
        return None
    # does the gradient exceed threshold for any variable?
    gtest = np.abs(gradient) > ips_input['dflood']
    # are any of these variables currently frozen?
    gtest = np.logical_and(gtest, np.logical_not(ips_input['active_vec']))
    if np.any(gtest):
        # yes, at least one gradient is above threshold
        # activate the corresponding coordinates
        addvars = np.array(ips_input['zmatrix'].varlist())[gtest].tolist()
        wstring = walker_string(ID)
        print('Activating coordinates{:s}: '.format(wstring), '  '.join(addvars))
        ips_input['active_vec'] = np.logical_or(ips_input['active_vec'], gtest)
        ips_input['active'].extend(addvars)
        print('Now {:d} coordinates are active{:s}.'.format(len(ips_input['active']), wstring))
    return gtest
##
def spread_opt(ips_input, fileroot, dist=10., IDno=None):
    # Piecewise optimization of a nominally dissociated geometry
    #   Operates in cartesian coordinates (Geometry() object)
    #   'dist':     the space (angstrom) to put between fragments in the supermolecule
    #   'fileroot': to be used in file names (fragment number will be appended)
    #   'IDno':  to identify process when running in parallel
    # Return:
        # list of fragment energies, 
        # list of optimized fragment structures, and
        # supermolecule structure
    # If there is only one fragment, do nothing.
    if ips_input['coordtype'] != 'cartesian':
        # convert to Geometry()
        ips_input['cartesian'] = ips_input[ips_input['coordtype']].toGeometry()
    originalGeom = ips_input['cartesian'].copy()
    newGeom = ips_input['cartesian']
    tol = ips_input['bondtol']
    #
    # move the fragments apart, to distance 'dist'
    nfrag = newGeom.spread_fragments(dist=dist, tol=tol)
    #print('found {:d} fragments'.format(nfrag))
    if nfrag < 2:
        # do nothing
        if IDno is None:
            return None, 0, 0
        else:
            return None, 0, 0, IDno
    #
    # Do single-point energy to get Mulliken charges
    ips_input['cartesian'] = newGeom  # install the new coordinates
    froot = '{:s}_spread_mulliken'.format(fileroot)
    Emull, qmout, IDx = qm_function(ips_input, 'energy', verbose=False, option='', unitR='', fileroot=froot)
    #print('Single-point energy after spreading = {:.6f} (see file {:s})'.format(Emull, qmout[0]))
    dfMulcharge = read_Mulliken_charges(qmout[0])
    mullik = dfMulcharge.iloc[-1]['Mulliken']
    haveSpin = 'SpinD' in list(mullik)
    mullQ = mullik['Charge'].values
    # sum charges (and any spin densities) within fragments
    fragments = newGeom.find_fragments(tol=tol)
    fragcharge = np.zeros(nfrag)
    for ifrag in range(nfrag):
        fragcharge[ifrag] = mullQ[fragments[ifrag]].sum()
    #print('QQQQ {:s} fragment mullQ = '.format(froot), fragcharge)
    if haveSpin:
        spinD = mullik['SpinD'].values
        fragspin = np.zeros(nfrag)
        for ifrag in range(nfrag):
            fragspin[ifrag] = spinD[fragments[ifrag]].sum()
        #print('QQQQ  {:s} fragment spinD = '.format(froot), fragspin)
    #
    # do geometry optimization on each fragment in isolation
    submols = newGeom.separateNonbonded(tol=tol)
    optfrag = [None] * nfrag
    Efrag = np.zeros(nfrag)
    for ifrag in range(nfrag):
        frag_input = ips_input.copy()
        frag_input['cartesian'] = submols[ifrag]
        frag_input['charge'] = np.around(fragcharge[ifrag]).astype(int)
        if haveSpin:
            frag_input['spinmult'] = np.around(fragspin[ifrag]).astype(int) + 1
        froot = '{:s}_frag{:d}'.format(fileroot, ifrag)
        Efrag[ifrag], optfrag[ifrag], IDx = qm_function(frag_input, 'minimize', verbose=False, fileroot=froot)
        # store charge and spin multiplicity in the Geometry() object
        optfrag[ifrag].charge = frag_input['charge']
        optfrag[ifrag].spinmult = frag_input['spinmult']
    # 
    # translate each fragment to be near its pre-optimized position
    for ifrag in range(nfrag):
        transl = submols[ifrag].COM(use_masses=False) - optfrag[ifrag].COM(use_masses=False)
        optfrag[ifrag].translate(transl)
        optfrag[ifrag] = RMSD_align(optfrag[ifrag], submols[ifrag], use_masses=False)
    #
    # re-assemble the fragments into a supermolecule
    superGeom = joinGeometries(optfrag)
    #
    # restore the original geometry to 'ips_input'
    ips_input['cartesian'] = originalGeom
    if IDno is None:
        return Efrag, optfrag, superGeom
    else:
        return Efrag, optfrag, superGeom, IDno
##
