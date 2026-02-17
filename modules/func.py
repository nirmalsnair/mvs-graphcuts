# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:51:14 2020
@author: Nirmal

Generic functions
"""

import os
import gc
import sys
import numpy as np
import cv2
from numba import jit


# Taken from: https://stackoverflow.com/a/53705610/7046003
# Returns size of object in bytes
def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


# mode: 0 = Grayscale, 1 = Color, -1 = Unchanged
def load_images_from_folder(folder, mode=0, is_return_file_list=False):
    images = []
    file_list = []

    for file_name in sorted(os.listdir(folder)):
        file_list.append(file_name)
        img = cv2.imread(os.path.join(folder, file_name), mode)

        if img is not None:
            if img.ndim > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert BGR to RGB
            images.append(img)
    images = np.asarray(images)                                 # Convert to NumPy

    if (is_return_file_list):
        return images, file_list
    else:
        return images


# Taken from: https://stackoverflow.com/a/37882746/7046003
# Replace NaN values with interpolated values
def interp_missing(a, method):
    from scipy.interpolate import griddata
    x, y = np.indices(a.shape)
    interp = np.array(a)
    mask = np.isnan(a)
    interp[np.isnan(interp)] = griddata(
                                (x[~mask], y[~mask]),           # points we know
                                a[~mask],                       # values we know
                                (x[mask], y[mask]),             # points to interpolate
                                method = method)                # nearest/linear/cubic
    return interp


@jit(nopython=True, nogil=True)
def get_patch(im, x, y, psize):
    d = psize // 2
    x = np.int_(np.round(x))
    y = np.int_(np.round(y))
    xl = x - d
    xh = x + d + 1
    yl = y - d
    yh = y + d + 1

    r, c = im.shape
    if (xl<0) or (yl<0) or (xh>c) or (yh>r):
        return np.full((psize,psize), np.nan, dtype=np.float32)
    else:
        return im[yl:yh, xl:xh].astype(np.float32)


# Passing patches with 0 gradient will return np.nan since std is 0
# Passing patches with NaN values will return np.nan
@jit(nopython=True, nogil=True)
def zncc(im1, im2):
    r1, c1 = im1.shape
    r2, c2 = im2.shape

    if (r1==r2) and (c1==c2) and (r1>0) and (c1>0):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        std1 = np.std(im1)
        std2 = np.std(im2)
        result = (im1-mean1) * (im2-mean2) / (std1 * std2)
        return np.round(np.sum(result)/(r1*c1), 4)
    else:
        return -1


# Python implementation of np.gradient since Numba (0.45.1) doesnt yet support np.gradient
# Benchmark: 1.81s (Python) vs 4.7ms (Jit) vs 6.5ms (Numpy) | 1 vs 380 vs 280 times speedup
# Benchmarked on 640x480 image
# CAUTION: np.gradient returns gy, gx but this function returns gx, gy, gm.
@jit(nopython=True, nogil=True)
def gradient(im):
    r, c = im.shape
    im = np.asarray(im, dtype=np.float32)
    gx = np.zeros((r, c))
    gy = np.zeros((r, c))

    for i in range(r):
        gx[i,0] = im[i, 1] - im[i, 0]           # First column
        gx[i,c-1] = im[i, c-1] - im[i, c-2]     # Last column

        for j in range(1, c-1):                 # Except first and last columns
            gx[i,j] = (im[i, j+1] - im[i, j-1]) / 2

    for j in range(c):
        gy[0,j] = im[1, j] - im[0, j]
        gy[r-1,j] = im[r-1, j] - im[r-2, j]

        for i in range(1, r-1):
            gy[i,j] = (im[i+1, j] - im[i-1, j]) / 2

    gm = (gx**2 + gy**2) ** 0.5

    return gx, gy, gm                           # np.gradient returns gy, gx


# Calculate distance map for grayscale image
def distance_map(I, pre_median, canny_low, canny_high, dist_metric, g_sigma):
    import cv2
    from scipy.ndimage import median_filter
    from scipy.ndimage import gaussian_filter

    assert I.ndim==2, 'Only grayscale image is supported.'
    n_row, n_col = I.shape
    M = np.zeros_like(I, dtype=np.float32)

    im = median_filter(I, pre_median)
    edge_map = cv2.Canny(im, canny_low, canny_high)
    edge_map = (np.max(edge_map) - edge_map) / np.max(edge_map)         # Reverse and normalize

    M = cv2.distanceTransform(np.uint8(edge_map), dist_metric, 0)
    M = gaussian_filter(M, g_sigma)
    M = np.floor(M)
    return M


# Project 3D point to image and calculate its depth, use the lowest depth as final
import modules.cam as cam
@jit(nopython=True, nogil=True)
def calc_dmap(P, X, n_row, n_col):
    n_pts = X.shape[1]
    D = np.full((n_row, n_col), np.inf, dtype=np.float32)

    x = P.dot(X)
    x = x / x[-1,:]

    dummy = np.empty_like(x)            # For Numba
    x_int = np.round_(x, 0, dummy)
    x_int = x_int.astype(np.int32)

    for i in range(n_pts):
        u = x_int[0,i]
        v = x_int[1,i]

        if (u>=0) and (v>=0) and (u<n_col) and (v<n_row):
            xi = np.expand_dims(x_int[:,i], 1)
            rpos, rdir = cam.camera_ray_2(P, xi)
            depth = (rdir.T).dot(np.expand_dims(X[0:3,i],1) - rpos)
            depth = depth[0,0]

            if (depth < D[v,u]):
                D[v,u] = depth
    return D


def calc_dmap_shm(P, n_row, n_col):
    import modules.sharedmem as sm
    X = sm.shared_data

    out = calc_dmap(P, X, n_row, n_col)
    return out


# Taken from: https://matplotlib.org/gallery/animation/image_slices_viewer.html
class ImageScroller(object):
    def __init__(self, ax, X, vmin=None, vmax=None, cmap=None):
        # self.ax = ax
        self.X = X
        self.slices, _, _ = X.shape
        self.ind = 0
        if (vmin is None):
            vmin = X.min()
        if (vmax is None):
            vmax = X.max()
        if (cmap):
            self.im = ax.imshow(self.X[self.ind], vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            self.im = ax.imshow(self.X[self.ind], vmin=vmin, vmax=vmax)
        self.update()

    # `step` is positive for up-scroll and negative for down-scroll
    def onscroll(self, event):
        # print(event.button, event.step)
        self.ind = (self.ind + int(event.step)) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.im.axes.figure.suptitle('Slice {}/{}'.format(self.ind+1, self.slices))
        self.im.axes.figure.canvas.draw()


# Scroll through set of multiple line plots
# `data` should be in the format: n_sets x n_pts x n_lines
# Preserves the items in the initial plot passed to this function (eg. axvline)
class PlotScroller(object):
    def __init__(self, ax, data, labels=None, event_handling=True):
        self.ax = ax
        self.data = data
        self.n_sets, _, _ = data.shape
        self.ind = 0
        self.step_mul = 1

        # Remove initial lines, plot supplied data, append the removed lines to end of list
        lines = ax.lines.copy()
        ax.lines = []
        self.ax.plot(self.data[self.ind])
        ax.lines.extend(lines)

        # Plot legend
        if labels:
            self.ax.legend(labels)

        # Set title
        self.ax.figure.suptitle('Slice {}'.format(self.ind))

        # Add event handlers
        if (event_handling):
            ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
            ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
            ax.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

    # `step` is positive for up-scroll and negative for down-scroll
    def on_scroll(self, event):
        # print(event.button, event.step)
        self.ind = (self.ind + int(event.step * self.step_mul)) % self.n_sets
        self.update()

    def on_key_press(self, event):
        # print(event.key)
        if event.key == 'control':
            self.step_mul = 5
        elif event.key == 'shift':
            self.step_mul = 50
        elif event.key == 'alt':
            self.step_mul = 500
        elif event.key == ' ':
            self.ind = 0
            self.update()

    def on_key_release(self, event):
        self.step_mul = 1

    def update(self):
        for i in range(self.data.shape[2]):
            self.ax.lines[i].set_ydata(self.data[self.ind,:,i])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.figure.suptitle('Slice {}'.format(self.ind))
        self.ax.figure.canvas.draw_idle()           # draw_idle() is so much faster than draw()


# Plot 2D data as 3D
# Taken from: https://stackoverflow.com/a/11409882/7046003
def plot_3D(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x = range(data.shape[1])
    y = range(data.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, data)
    plt.show()
