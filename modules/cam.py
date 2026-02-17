# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:09:00 2020
@author: Nirmal

Camera functions

Refer: http://ksimek.github.io/2012/08/14/decompose/
"""

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
from numba import jit


def cam_factor(P):
    K, R = spla.rq(P[:,0:3])
    S = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, S)
    R = np.dot(S, R)
    T = np.dot(spla.inv(K), P[:,3])
    T = T.reshape((3,1))
    return K, R, T


def cam_position(P):
    K, R, T = cam_factor(P)
    return -np.transpose(R).dot(T)


def cam_position_all(P):
    ncam = len(P)
    POS = np.zeros((ncam,3,1))
    for i in range(ncam):
        POS[i] = cam_position(P[i])
    return POS


def cam_orientation(P):
    o = P[-1,0:3].reshape((3,1))
    if npla.norm(o) != 0:
        o = o / npla.norm(o)
    return o


def cam_orientation_all(P):
    ncam = len(P)
    ORN = np.zeros((ncam,3,1))
    for i in range(ncam):
        ORN[i] = cam_orientation(P[i])
    return ORN


def camera_ray(P, x):
    o = cam_position(P)
    ox = np.dot(npla.pinv(P), x)

    flip = False
    if (ox[3] < 0):
        # print("Camera ray negative")
        flip = True
    if (np.isclose(ox[3], 0)):
        # print("Camera ray divide by 0")
        ox[3] = 1.0

    ox = ox / ox[3]
    ox = ox[:3] - o
    ox = ox / npla.norm(ox)

    if (flip):
        ox = ox * -1
    return o, ox


# -----------------------------------------------------------------------------
# Modified version of cam_position() for Numba
# Taken from: https://mathoverflow.net/a/68488/130043
# This function is ~13 times faster than cam_position() (6us vs. 80us)
@jit(nopython=True, nogil=True)
def cam_position_2(P):
    pos = npla.pinv(P[:,:3]).dot(P[:,-1]) * -1
    pos = pos.reshape((3,1))
    return pos


# Modified version of camera_ray() for Numba
# This function is ~15 times faster than camera_ray() (13us vs. 200us)
@jit(nopython=True, nogil=True)
def camera_ray_2(P, x):
    o = cam_position_2(P)
    x = x.astype(np.float64)        # For Numba
    ox = np.dot(npla.pinv(P), x)

    flip = False
    if (ox[3,0] < 0):
        # print("Camera ray negative")
        flip = True

    ox = ox / ox[3]
    ox = ox[:3] - o
    ox = ox / npla.norm(ox)

    if (flip):
        ox = ox * -1
    return o, ox
# -----------------------------------------------------------------------------


# Taken from https://stackoverflow.com/a/13849249/7046003
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
    """
    v1 = v1 / npla.norm(v1)
    v2 = v2 / npla.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def cam_layout(P):
    ncam = len(P)
    POS = cam_position_all(P)
    ORN = cam_orientation_all(P)
    ANG = np.zeros((ncam, ncam))
    for i in range(ncam):
        for j in range(ncam):
            v1 = ORN[i].reshape((3))
            v2 = ORN[j].reshape((3))
            angle = angle_between(v1, v2)
            angle = np.rad2deg(angle)
            ANG[i,j] = angle
    return POS, ORN, ANG


@jit(nopython=True, nogil=True)
def cam_project(P, X):
    x = np.dot(P, X)
    x = x / x[2]
    return x
