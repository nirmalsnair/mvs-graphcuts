# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:50:00 2020
@author: Nirmal

PyGMO fitness functions
"""

import numpy as np
from numba import jit
import pygmo as pg
import modules.cam as cam
import modules.func as func


@jit(nopython=True, nogil=True)
def _fitness_daisy(t, DSY, P, cam_list, d0, rpos, rdir, k_best):
    X = np.ones((4,1))
    X[0:3] = rpos + t * rdir

    _, n_rows, n_cols, _ = DSY.shape
    nft = len(cam_list)
    S = np.zeros((nft))

    for idx, val in enumerate(cam_list):
        xi = cam.cam_project(P[val], X)
        x = int(np.round(xi[0,0]))
        y = int(np.round(xi[1,0]))

        if (x<0) or (y<0) or (x>=n_cols) or (y>=n_rows):
            S[idx] = 9999                       # A large value to discourage
        else:
            di = DSY[val, y, x]
            S[idx] = np.linalg.norm(d0 - di)

    k = k_best
    S = np.sort(S)                              # Sort values in ascending order
    S = S[0:k]                                  # Consider only 'k' lowest values
    f = np.sum(S)
    return [f]



def estimate_depth_daisy_shm(P, lb, ub, ANG, idx, x, y, ang_bound, k_best):
    import modules.sharedmem as sm
    DSY = sm.shared_data

    class Search(object):
        def fitness(self, t):
            return _fitness_daisy(t, DSY, P, cam_list, d0, rpos, rdir, k_best)
        def get_bounds(self):
            return (lb, ub)

    cam_list = np.where((ANG > ang_bound[0]) & (ANG < ang_bound[1]))[0]
    point = np.array((x, y, 1)).reshape((3, 1))
    rpos, rdir = cam.camera_ray(P[idx], point)
    d0 = DSY[idx, y, x]

    nl = pg.nlopt('sbplx')                                  # !!!: sbplx, praxis, neldermead
    algo = pg.algorithm(nl)
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=1)                  # Population size = 1

    # Add initial solution (middle of bounds) to population
    init_val = (lb[0] + ub[0]) / 2
    init_val = np.array(init_val).reshape((1,))             # PyGMO requires (ndim,) array
    pop.push_back(init_val, f=None)                         # pop.set_x(0, init_val)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f, pop.problem.get_fevals()]


def estimate_depth_daisy(DSY, P, lb, ub, ANG, idx, x, y, ang_bound, k_best):
    class Search(object):
        def fitness(self, t):
            return _fitness_daisy(t, DSY, P, cam_list, d0, rpos, rdir, k_best)
        def get_bounds(self):
            return (lb, ub)

    cam_list = np.where((ANG > ang_bound[0]) & (ANG < ang_bound[1]))[0]
    point = np.array((x, y, 1)).reshape((3, 1))
    rpos, rdir = cam.camera_ray(P[idx], point)
    d0 = DSY[idx, y, x]

    nl = pg.nlopt('sbplx')                                  # !!!: sbplx, praxis, neldermead
    algo = pg.algorithm(nl)
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=1)                  # Population size = 1

    # Add initial solution (middle of bounds) to population
    init_val = (lb[0] + ub[0]) / 2
    init_val = np.array(init_val).reshape((1,))             # PyGMO requires (ndim,) array
    pop.push_back(init_val, f=None)                         # pop.set_x(0, init_val)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f, pop.problem.get_fevals()]


@jit(nopython=True, nogil=True)
def _fitness_zncc(t, I, P, cam_list, p0, rpos, rdir, psize, k_best):
    X = np.ones((4, 1))
    X[0:3] = rpos + t * rdir

    nft = len(cam_list)
    S = np.zeros((nft))

    for idx, val in enumerate(cam_list):
        xi = cam.cam_project(P[val], X)
        pi = func.get_patch(I[val], xi[0,0], xi[1,0], psize)
        S[idx] = func.zncc(p0, pi)

    k = k_best
    S = S[~np.isnan(S)]
    S = np.sort(S)[::-1]            # Sort values in descending order (ZNCC)
    S = S[0:k]                      # Consider only 'k' highest values
    f = np.nansum(S)
    return [-f]                     # For ZNCC, higher is better


def estimate_depth_zncc_shm(P, lb, ub, ANG, idx, x, y, psize, ang_bound, k_best):
    import modules.sharedmem as sm
    I = sm.shared_data

    class Search(object):
        def fitness(self, t):
            return _fitness_zncc(t, I, P, cam_list, p0, rpos, rdir, psize, k_best)
        def get_bounds(self):
            return (lb, ub)

    cam_list = np.where((ANG > ang_bound[0]) & (ANG < ang_bound[1]))[0]
    point = np.array((x, y, 1)).reshape((3, 1))
    rpos, rdir = cam.camera_ray(P[idx], point)
    p0 = func.get_patch(I[idx], x, y, psize)

    nl = pg.nlopt('sbplx')                                  # !!!: sbplx, praxis, neldermead
    algo = pg.algorithm(nl)
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=1)                  # Population size = 1

    # Add initial solution (upscaled_L2) to population
    init_val = (lb[0] + ub[0]) / 2                          # Middle of bounds (= upscaled_L2)
    init_val = np.array(init_val).reshape((1,))             # PyGMO requires (ndim,) array
    pop.push_back(init_val, f=None)                         # pop.set_x(0, init_val)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f, pop.problem.get_fevals()]


def estimate_depth_zncc(I, P, lb, ub, ANG, idx, x, y, psize, ang_bound, k_best):
    class Search(object):
        def fitness(self, t):
            return _fitness_zncc(t, I, P, cam_list, p0, rpos, rdir, psize, k_best)
        def get_bounds(self):
            return (lb, ub)

    cam_list = np.where((ANG > ang_bound[0]) & (ANG < ang_bound[1]))[0]
    point = np.array((x, y, 1)).reshape((3, 1))
    rpos, rdir = cam.camera_ray(P[idx], point)
    p0 = func.get_patch(I[idx], x, y, psize)

    nl = pg.nlopt('sbplx')                                  # !!!: sbplx, praxis, neldermead
    algo = pg.algorithm(nl)
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=1)                  # Population size = 1

    # Add initial solution (upscaled_L2) to population
    init_val = (lb[0] + ub[0]) / 2                          # Middle of bounds (= upscaled_L2)
    init_val = np.array(init_val).reshape((1,))             # PyGMO requires (ndim,) array
    pop.push_back(init_val, f=None)                         # pop.set_x(0, init_val)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f, pop.problem.get_fevals()]
