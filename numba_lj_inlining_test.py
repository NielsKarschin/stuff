# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:58:47 2022

@author: niels
"""

import numpy as np
import numba
from IPython import get_ipython
ipython = get_ipython()


# %% original parallel

@numba.njit
def lj_numba_scalar_prange(r):
    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


@numba.njit
def distance_numba_scalar_prange(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx * dx + dy * dy + dz * dz) ** 0.5

    return r


def potential_numba_scalar_prange(cluster):
    energy = 0.0
    # numba.prange requires parallel=True flag to compile.
    # It causes the loop to run in parallel in multiple threads.
    for i in numba.prange(len(cluster)-1):
        for j in range(i + 1, len(cluster)):
            r = distance_numba_scalar_prange(cluster[i], cluster[j])
            e = lj_numba_scalar_prange(r)
            energy += e

    return energy


# %% merged functions

@numba.njit
def lj_numba_scalar_merged(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx * dx + dy * dy + dz * dz) ** 0.5

    sr6 = (1./r)**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


def potential_numba_scalar_merged(cluster):
    energy = 0.0
    for i in numba.prange(len(cluster)-1):
        for j in range(i + 1, len(cluster)):
            e = lj_numba_scalar_merged(cluster[i], cluster[j])
            energy += e

    return energy


# %% rearranged functions

@numba.njit
def lj_numba_scalar_rearranged(r_inv):
    sr6 = r_inv**6
    pot = 4.*(sr6*sr6 - sr6)
    return pot


@numba.njit
def distance_numba_scalar_rearranged(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx * dx + dy * dy + dz * dz) ** 0.5
    r_inv = (1./r)

    return r_inv


def potential_numba_scalar_rearranged(cluster):
    energy = 0.0
    for i in numba.prange(len(cluster)-1):
        for j in range(i + 1, len(cluster)):
            r_inv = distance_numba_scalar_rearranged(cluster[i], cluster[j])
            e = lj_numba_scalar_rearranged(r_inv)
            energy += e

    return energy


# %% run and timeit

np.random.seed(0)
v = np.random.randn(10_000, 3)


for b in (False, True):
    print(f'parallel {b}')
    for f_py in (potential_numba_scalar_prange, potential_numba_scalar_merged,
                 potential_numba_scalar_rearranged):
        print(f_py)
        f_nb = numba.njit(f_py, parallel=b)
        f_nb(v)
        ipython.run_line_magic('timeit', 'f_nb(v) ')
    print('')
