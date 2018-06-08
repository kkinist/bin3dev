#!/usr/bin/python3
# Isopotential searching, take 3
# Karl Irikura, NIST 2017
#
import numpy as np
import pandas as pd
import sys, os
from chem_subs import *
from qm_subs import *
from ips_subs import *
#
try:
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        # maybe user did not include the file suffix
        input_file += '.ips'
    ips_input = read_input(input_file)
except IOError:
    print('*** Failure reading input file')
    sys.exit('Usage: ips.py <name of IPS input file>')
print('Reading input file "{:s}"'.format(input_file))
ips_input = parse_input(ips_input)
confirm_newstart(ips_input)
algorithm = ips_input['method'][0]
coordtype = ips_input['coordtype']
if not coordtype in ['cartesian', 'zmatrix']:
    print_err('coordtype', coordtype)
inputCoord = ips_input[coordtype].copy()   # save input geometry
inputUnits = inputCoord.unitX()   # save input units
code = ips_input['code']
molec = ips_input['molecule']
print_dict(ips_input)
if not ips_input['continuation']:
    # this is a fresh start
    # is pre-minimization requested?
    if ips_input['minimize']:
        print('Running requested geometry optimization.')
        Emin, optCoord, IDx = qm_function(ips_input, 'minimize', fileroot='{:s}_E0_geometry'.format(molec))
        if not (Emin and optCoord):
            # the geometry optimization failed
            sys.exit('You have to figure out why the geometry optimization failed.')
        E0 = Emin
        install_unit_masses(optCoord)
        # install the optimized coordinates in 'ips_input'
        ips_input[coordtype] = optCoord.copy()
        # save coordinates to a file called '{molec}_E0_geometry.xyz'
        optCoord.printXYZ('{:s}_E0_geometry.xyz'.format(molec), comment='Minimized structure for {:s}, E = {:.6f}'.format(molec, E0))
        if coordtype == 'zmatrix':
            # also save optimized z-matrix 
            with open('{:s}_E0_geometry.zmt'.format(molec), 'w') as zmf:
                zmf.write(optCoord.printstr())
        grad = np.zeros(optCoord.nDOF())  # the gradient should be zero
    else:
        # calculate initial energy and gradient
        E0, grad, IDx = qm_function(ips_input, 'gradient', unitR=inputUnits[0], option='flatten')
        if not (E0 and len(grad)):
            # the calculation failed
            sys.exit('The gradient was not returned.')
    # relative energies are calculated with respect to E0
    E0_geom = ips_input[coordtype].copy()
    ips_input['E0_geom'] = E0_geom
    store_E0(ips_input, E0)
    #
    print('Target electronic energy is E = {:.5f}'.format(E0 + ips_input['energy']/au_kjmol))
    # Find the necessary seed points at the target energy and create the Walkers
    # 'walkers' is a list of Walkers
else:
    # this is continuing from where a previous calculations stopped
    # read reference geometry and walkers[] from the pickle file
    walkers, ips_input = restore_pickle(ips_input)
    print('&&&&& input as restored from file:')
    print_dict(ips_input)
    E0 = ips_input['E0']
    E0_geom = ips_input['E0_geom']
    print('Read data for {:d} walkers covering most recent {:d} steps'.format(len(walkers), len(walkers[0].historyE)))
#
if (coordtype == 'zmatrix'):
    # express angles in radians during IPS
    ips_input[coordtype].toRadian()
    E0_geom.toRadian()  # for compatibility with trajectory
#
if not ips_input['continuation']:
    if __name__ == '__main__':
        # generate seed points
        walkers = seed_points(ips_input)
    if algorithm == 'mrwips':
        if not 'stepl' in ips_input:
            # User specified an angle but not a step length for MRWIPS
            # Set the initial step length to ips_input['step_angle'] multiplied
            #   by the mean distance of seed points from the origin.
            meanR = 0.
            vec0 = ips_input['E0_geom'].toVector()
            for W in walkers:
                vec = W.Coord.toVector()
                meanR += np.linalg.norm(vec - vec0)
            meanR /= len(walkers)
            ips_input['stepl'] = ips_input['step_angle'] * meanR
    print('Molecule contains {:d} atoms.'.format(E0_geom.natom()))
    print('Starting search using algorithm "{:s}"'.format(algorithm))
    print('Maximum step length = {:.2f}'.format(ips_input['stepl']))
    istep = 0
else:
    # resuming earlier calculations
    nactive = 0
    for W in walkers:
        if W.active:
            nactive += 1
            istep = W.steps[-1]
    print('Resuming calculations with {:d} active walkers.'.format(nactive))
    print('coordtype =', walkers[-1].coordtype)
    print('Search algorithm is "{:s}"'.format(algorithm))
    print('Last step was number {:d}'.format(istep-1))
#
if 'active' in ips_input:
    print('{:d} of {:d} z-matrix coordinates are active:'.format(len(ips_input['active']), E0_geom.nDOF()))
    print('\t', '  '.join(ips_input['active']))
#
if algorithm == 'srips':
    # specular reflection algorithm for isopotential searching (serial; one walker)
    while (istep < ips_input['steps']) and (not user_interrupt(molec)):
        # possibly change the target energy
        energy_ramp(istep, ips_input)
        E, X, G, nspec = step_specular(ips_input, walkers[0], trustR=ips_input['stepl'])
        print('step {:d}: E = {:.2f} after {:d} iterations'.format(istep, E, nspec+1))
        if coordtype == 'zmatrix':
            # canonicalize the dihedral angles
            X.canonical_angles()
            # possibly activate some coordinates
            dflood_activate(G, ips_input)
        walkers[0].add_point(E, X, G)
        if walkers[0].connexChange[-1]:
            print('--- change in connectivity at step {:d} ---'.format(istep))
        walkers[0].update_traj(molec, istep, 'Erel = {:.1f} kJ/mol'.format(E))
        save_pickle(ips_input, walkers)
        #X.printXYZ(fname='{:s}_{:d}.xyz'.format(molec, istep), comment='SRIPS search step {:d}'.format(istep))
        istep += 1
    if not user_interrupt(molec):
        print('Maximum number of steps reached.')
    else:
        print('Search halted by user request.')
elif algorithm == 'mrwips':
    # many walkers that repel each other
    while (istep < ips_input['steps']) and (not user_interrupt(molec)):
        # possibly change the target energy
        energy_ramp(istep, ips_input)
        if __name__ == '__main__':
            # have all walkers take one step, in parallel
            walkers, nActive = step_many_repulsive(ips_input, walkers)
        print('step {:d} succeeded for {:d} walkers'.format(istep, nActive))
        if nActive < 2:
            # no walkers left!
            print('Need at least two walkers--halting.')
            break
        # initialize array to record newly activated Z-matrix coordinates
        awakened = np.full(ips_input[coordtype].nDOF(), False, dtype='bool')
        for W in walkers:
            if W.active:
                W.update_traj(molec, istep, 'Erel = {:.1f} kJ/mol'.format(W.historyE[-1]))
                # possibly activate some coordinates
                getup = dflood_activate(W.historyG[-1], ips_input)
                if getup is not None:
                    awakened = np.logical_or(awakened, getup)
        save_pickle(ips_input, walkers)
        istep += 1
    else:
        # loop completed
        if not user_interrupt(molec):
            print('Maximum number of steps reached.')
        else:
            print('Search halted by user request.')
else:
    print('*** Unknown method requested:', ips_input['method'])
