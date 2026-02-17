# -*- coding: utf-8 -*-
"""
Created on Wed Jan 05 19:08:00 2022
@author: Nirmal

Depth map refinement using graph-cut.
"""

# -------------------------------------------- Notes ----------------------------------------------
# This modified version supports dividing images into `nseg_row` x `nseg_col` segments and merging
# back the results for faster processing of high-resolution images.
# -------------------------------------------------------------------------------------------------

import time
import numpy as np
from scipy.ndimage import median_filter
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import maxflow
import modules.func as func

# ------------------------------------------ Parameters -------------------------------------------
# No. of CPU processes to spawn: (1) for single core, (> 1) for parallel execution
NPROC = 4

# # Fitness landscape files (FL_1=L0, FL_2=Lx)
flscape_file_1 = Path("data/scan65-32_flscape.npz")
flscape_file_2 = Path("data/scan65-32_flscape.npz")

# Fitness offset (To make FL values positive)
fl_offset = 50

# Depth bounds (To clip)
vmin = 500          # DTU-scan65
vmax = 750
    
# Image directory
im_dir = Path("data/Dataset/scan65-32/images")

# Foreground masks directory
mask_dir = Path("data/Dataset/scan65-32/foreground_mask")

# # Whether distance or gradient map
map_type = 'gradient'

# # Median to apply on image before distance/gradient computation
pre_median = 1

# Distance map parameters (Ignored if map_type='gradient')
dist_metric = cv2.DIST_L2                   # Manhattan (L1), Euclidean (L2), Chess-King (C) etc
canny_low = 50                              # Canny low threshold
canny_high = 100                            # Canny high threshold
g_sigma = 5                                 # Gaussian sigma for post-filtering

# # Weight map parameters
m_min = 0
m_max = 10
w_min = 0.1
w_max = 1.0

# # Smoothness term parameters
f1_mul = 0.01

# L0/Lx depth map fusion parameters
# (Will use FL_1 if M[i,j] is less than lx_thresh, FL_2 otherwise)
lx_thresh = 0

# Add front nodes to account for differences in base depths
is_add_front_nodes = True

# Whether edge capacities are symmetric (bi-directional) or asymmetric (uni-directional).
symm_data = False
symm_smooth = True

# Number of segments (row, col) to divide the image-portion into
nseg_row = 2
nseg_col = 2

# Write results to disk as npz file
is_write = True
# -------------------------------------------------------------------------------------------------


def pbar_update(p):
    pbar.update()


def create_graph(volume, d_front, w, wx, wy, f1_mul):
    n_row, n_col, n_depth = volume.shape

    # Create graph nodes
    g = maxflow.Graph[float]()
    n_nodes = np.sum(~np.isnan(volume))
    nodes = g.add_grid_nodes(n_nodes)

    # Structure for connecting to next depth node
    struct_d = np.array(
        [[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
         [[0, 0, 0],
          [0, 0, 1],
          [0, 0, 0]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]]
    )
    # Structure for connecting to right node
    struct_r = np.array(
        [[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
          [[0, 0, 0],
          [0, 0, 0],
          [0, 1, 0]],
          [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]]
    )
    # Structure for connecting to bottom node
    struct_b = np.array(
        [[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
          [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
          [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]]
    )

    # Shape of offsetted volume
    n_offset_depth = n_depth + np.max(d_front)
    shape = (n_row, n_col, n_offset_depth)

    # Create mask of offsetted volume in order to create grid
    valid = ~np.isnan(volume)
    mask = np.zeros(shape, dtype=np.bool)
    for r in range(n_row):
        for c in range(n_col):
            d = d_front[r,c]
            mask[r, c, d:d+n_depth] = valid[r,c]

    # Grid (contains -1 in non-node locations)
    grid = np.full(mask.shape, -1)
    grid[mask] = nodes

    # Process volume column-by-column to limit memory consumption
    # Add data edge and smoothness-bottom edge
    for c in range(n_col):
        v = np.full((n_row, 1, n_offset_depth), np.nan, dtype=np.float64)
        for r in range(n_row):
            d = d_front[r,c]
            v[r, 0, d:d+n_depth] = volume[r,c]

        # Repeat to 3D
        wi = np.dstack([w[:,c,None]] * n_offset_depth)
        wyi = np.dstack([wy[:,c,None]] * n_offset_depth)

        # Data term
        v = v * (1-wi)
        # Replace 0s with very small value
        v[v==0] = 0.0001
        # Add bi-directional edge to next depth node
        g.add_grid_edges(grid[:,c,None,:], weights=v, structure=struct_d, symmetric=symm_data)

        # Smoothness-bottom term
        sy = wyi * f1_mul
        # Replace 0s with very small value
        sy[sy==0] = 0.0001
        # Add bi-directional edge to bottom node
        g.add_grid_edges(grid[:,c,None,:], weights=sy, structure=struct_b, symmetric=symm_smooth)

    # Add smoothness-right edge on 2-column slice
    for c in range(n_col-1):
        v = np.full((n_row, 2, n_offset_depth), np.nan, dtype=np.float64)
        for r in range(n_row):
            d_1 = d_front[r,c]
            d_2 = d_front[r,c+1]
            v[r, 0, d_1:d_1+n_depth] = volume[r,c]
            v[r, 1, d_2:d_2+n_depth] = volume[r,c+1]

        # Repeat to 3D
        wxi = np.dstack([wx[:,c:c+2]] * n_offset_depth)

        # Smoothness-right term
        sx = wxi * f1_mul
        # Replace 0s with very small value
        sx[sx==0] = 0.0001
        # Add bi-directional edge to right node
        g.add_grid_edges(grid[:,c:c+2,:], weights=sx, structure=struct_r, symmetric=symm_smooth)

    # Find first non-nan occurrence for each pixel (to connect to source)
    dx = np.argmax(mask, axis=2)
    # Connect source node to leftmost non-nan nodes
    rg, cg = np.mgrid[0:n_row, 0:n_col]
    g.add_grid_tedges(grid[rg,cg,dx], np.inf, 0)

    # Find last non-nan occurrence for each pixel (to connect to sink)
    dx = np.argmax(np.flip(mask, axis=2), axis=2)
    dx = n_offset_depth - dx - 1
    # Connect rightmost non-nan nodes to sink node
    rg, cg = np.mgrid[0:n_row, 0:n_col]
    g.add_grid_tedges(grid[rg,cg,dx], 0, np.inf)

    # Convert grid to bool to conserve memory (0=non-node, 1=node)
    grid = np.asarray(grid+1, np.bool)

    return grid, nodes, g


def process_image(imx, volume, w, wx, wy, d_low, fl_grain, fl_steps):
    # Only perfectly divisible numbers are allowed
    assert volume.shape[0] % nseg_row == 0
    assert volume.shape[1] % nseg_col == 0

    # Retain original data
    volume_org = volume.copy()
    w_org = w.copy()
    wx_org = wx.copy()
    wy_org = wy.copy()
    d_low_org = d_low.copy()

    D_GCUT = np.zeros_like(d_low)
    row_stride = volume.shape[0] // nseg_row
    col_stride = volume.shape[1] // nseg_col

    # Process each segment
    for i in range(nseg_row):
        for j in range(nseg_col):
            i1 = i * row_stride
            j1 = j * col_stride
            i2 = (i + 1) * row_stride
            j2 = (j + 1) * col_stride
            print(imx, i, j)
            print(i1, i2, j1, j2)

            # Slice required segment
            volume = volume_org[i1:i2, j1:j2]
            w = w_org[i1:i2, j1:j2]
            wx = wx_org[i1:i2, j1:j2]
            wy = wy_org[i1:i2, j1:j2]
            d_low = d_low_org[i1:i2, j1:j2]

            # Skip optimisation if all values are np.nan
            if (np.all(np.isnan(d_low))):
                D_GCUT[i1:i2, j1:j2] = fl_steps - 1
                print('All values are NaN, skipping this segment.\n')
                continue

            if (is_add_front_nodes):
                # Insert dummy nodes appropriately at the front & back end to account for different
                # D_low values. Zero or more nodes may be inserted as required at each pixel location.
                n_row, n_col, n_depth = volume.shape
                rmin, cmin = np.unravel_index(np.argmin(d_low), d_low.shape)
                rmax, cmax = np.unravel_index(np.argmax(d_low), d_low.shape)
                dlow_min = np.nanmin(d_low)
                d_front = np.int16(np.round((d_low - dlow_min) / fl_grain))     # np.nan is converted to 0
                shape = (n_row, n_col) + (n_depth + np.max(d_front),)
                print('\n*** Added offset nodes. ***')
                print('*** New graph shape = {} ***'.format(shape))
            else:
                d_front = np.zeros_like(d_low)

            # Create graph and compute maxflow
            grid, nodes, g = create_graph(volume, d_front, w, wx, wy, f1_mul)
            flow = g.maxflow()

            # Get segments
            segments = g.get_grid_segments(nodes)
            segments = np.array(segments, dtype=np.uint8)
            s = np.zeros(shape, dtype=np.uint8)
            idx = np.where(grid==1)
            s[idx] = segments
            s, segments = segments, s

            # Get surface
            D_gcut = np.argmax(segments==1, axis=2)

            # HACK: Move surface boundary
            D_gcut = D_gcut - 1
            print('\n*** HACK: Move surface boundary. ***\n')

            # Account for the added nodes at the front & back end
            if (is_add_front_nodes):
                D_gcut = D_gcut - d_front

            # Merge back result
            D_GCUT[i1:i2, j1:j2] = D_gcut

    return D_GCUT


def process_image_shm(imx, w, wx, wy, d_low, fl_grain, fl_steps):
    import modules.sharedmem as sm
    FL = sm.shared_data

    D_gcut = process_image(imx, FL[imx], w, wx, wy, d_low, fl_grain, fl_steps)
    return D_gcut


if __name__ == '__main__':
    # Print details
    print('')
    print('NPROC                        = {}'.format(NPROC))
    print('')
    print('Fitness-landscape file 1     = {}'.format(flscape_file_1))
    print('Fitness-landscape file 2     = {}'.format(flscape_file_2))
    print('Fitness offset               = {}'.format(fl_offset))
    print('')
    print('Image directory              = {}'.format(im_dir))
    print('Mask directory               = {}'.format(mask_dir))
    print('')
    print('Map type                     = {}'.format(map_type))
    print('Pre-median                   = {}'.format(pre_median))
    print('Distance metric              = {}'.format(dist_metric))
    print('Canny                        = {}, {}'.format(canny_low, canny_high))
    print('Gaussian sigma               = {}'.format(g_sigma))
    print('Dist/Grad. map thresholds    = {}, {}'.format(m_min, m_max))
    print('Weight map thresholds        = {:.4f}, {:.4f}'.format(w_min, w_max))
    print('Smoothness multiplier        = {}'.format(f1_mul))
    print('L0/Lx flscape threshold      = {}'.format(lx_thresh))
    print('')
    print('Add offset nodes?            = {}'.format(is_add_front_nodes))
    print('Symmetric data edges?        = {}'.format(symm_data))
    print('Symmetric smoothness edges?  = {}'.format(symm_smooth))
    print('')
    print('No. of segments (row, col)   = {}, {}'.format(nseg_row, nseg_col))
    print('')

    # Load first fitness-landscape file
    data = np.load(flscape_file_1)
    FL_1 = data['FL']
    fl_grain = data['fl_grain']
    fl_steps = data['fl_steps']
    D_low = data['D_low']

    # Load second fitness-landscape file
    data = np.load(flscape_file_2)
    FL_2 = data['FL']

    # fl_grain, fl_steps and D_low should be same for both flscapes
    assert FL_1.shape==FL_2.shape, 'FL shape mismatch!'
    assert data['fl_grain']==fl_grain, 'fl_grain mismatch!'
    assert data['fl_steps']==fl_steps, 'fl_steps mismatch!'
    assert np.all(data['D_low']==D_low), 'D_low mismatch!'

    # Fitness landscape shape
    n_images, n_rows, n_cols, n_flevels = FL_1.shape

    # Load images
    I = func.load_images_from_folder(im_dir, 0)
    assert I.shape[0]==n_images

    # Calculate distance/gradient maps
    M = np.zeros(I.shape, dtype=np.float32)
    if (map_type=='distance'):
        for i in range(n_images):
            M[i] = func.distance_map(I[i], pre_median, canny_low, canny_high, dist_metric, g_sigma)
        Mx = My = M
    elif (map_type=='gradient'):
        Mx = np.zeros(I.shape, dtype=np.float32)
        My = np.zeros(I.shape, dtype=np.float32)
        for i in range(n_images):
            im = median_filter(I[i], pre_median)
            Mx[i], My[i], M[i] = func.gradient(im)
    else:
        raise ValueError('Invalid map_type!')

    # Normalize distance/gradient maps to obtain weight maps (Low=Edge, High=Homogeneous)
    Mx[Mx > m_max] = m_max
    Mx[Mx < m_min] = m_min
    My[My > m_max] = m_max
    My[My < m_min] = m_min
    M[M > m_max] = m_max
    M[M < m_min] = m_min
    W = w_min + ((w_max - w_min) / (m_max - m_min)) * (M - m_min)
    Wx = w_min + ((w_max - w_min) / (m_max - m_min)) * (Mx - m_min)
    Wy = w_min + ((w_max - w_min) / (m_max - m_min)) * (My - m_min)

    # Invert weight map in case of gradient type
    if (map_type=='gradient'):
        W = W.max() - W + W.min()
        Wx = Wx.max() - Wx + Wx.min()
        Wy = Wy.max() - Wy + Wy.min()

    # Convert to float16 to conserve memory
    W = np.asarray(W, dtype=np.float16)
    Wx = np.asarray(Wx, dtype=np.float16)
    Wy = np.asarray(Wy, dtype=np.float16)

    # Fuse flscapes based on distance/gradient/weight map threshold
    if map_type=='distance':
        idx = np.where(M >= lx_thresh)
    elif map_type=='gradient':
        idx = np.where(M <= lx_thresh)
    FL = FL_1.copy()
    FL[idx] = FL_2[idx]

    # Delete original fitness-landscapes to conserve memory
    del FL_1, FL_2

    # Convert to float64, otherwise summing the fitness values will overflow and return inf.
    FL = FL.astype(np.float64)

    # Add offset to FL values to make them positive
    FL = FL + fl_offset

    # Load foreground masks
    mask = func.load_images_from_folder(mask_dir, 0)
    if (np.unique(mask)[0] == 0) and (np.unique(mask)[1] == 255):
        mask = mask.astype(bool)
    else:
        raise ValueError('Foreground mask contains values other than 0 and 255')

    # Initial solution
    D_init = D_low + fl_steps//2 * fl_grain

    # Apply foreground mask
    M[~mask] = 0
    W[~mask] = 0
    D_low[~mask] = np.nan
    D_init[~mask] = 0

    # Clip depth bounds to reduce offset depth
    D_low[D_low<vmin] = np.nan
    D_low[D_low>vmax] = np.nan
    D_init[D_init<vmin] = np.nan
    D_init[D_init>vmax] = np.nan

    # Replace non-mask pixels with NaN values
    FL[~mask] = np.nan

    # Images to process
    im_list = list(range(n_images))

    # Initialise multiprocessing and shared memory
    if (NPROC > 1):
        import multiprocessing as mp
        import ctypes
        import modules.sharedmem as sm

        shared_size = FL.size
        shared_shape = FL.shape
        shared_base = mp.Array(ctypes.c_float, shared_size)
        sm.init(shared_base, shared_shape)
        sm.shared_data[:] = FL[:]

        # Required for Windows systems
        mp.freeze_support()

        # Check if NPROC greater than available CPU cores
        if (NPROC > mp.cpu_count()):
            raise ValueError('NPROC greater than available CPU cores!')

        R = []
        pool = mp.Pool(processes = NPROC,
                       initializer = sm.init,
                       initargs = (shared_base, shared_shape))

    # Optimisation
    start_time = time.time()
    print('Processing {} images on {} threads...'.format(n_images, NPROC), flush=True)
    pbar = tqdm(total=n_images, unit=' images')
    GCUT = []

    for i,imx in enumerate(im_list):
        if (NPROC > 1):
            r = pool.apply_async(process_image_shm,
                                 args = (imx, W[imx], Wx[imx], Wy[imx], D_low[imx], fl_grain, fl_steps),
                                 callback = pbar_update)
            R.append(r)

        else:
            out = process_image(imx, FL[imx], W[imx], Wx[imx], Wy[imx], D_low[imx], fl_grain, fl_steps)
            pbar.update()
            GCUT.append(out)

    if (NPROC > 1):
        GCUT = [r.get() for r in R]
        pool.close()
        pool.join()

    pbar.close()
    end_time = time.time() - start_time
    print('Time elapsed             = {:.2f} seconds'.format(end_time))

    # Convert to Numpy
    GCUT = np.array(GCUT)

    # Depth maps
    D_gcut = np.zeros_like(D_low)
    for i,imx in enumerate(im_list):
        D_gcut[imx] = D_low[imx] + GCUT[i] * fl_grain

    # Save to disk as Numpy compressed (npz) file
    if (is_write):
        rand_int = np.random.randint(100000, 999999)
        file_name = 'gcut_' + str(rand_int) + '.npz'
        np.savez_compressed(file_name,
                            NPROC=NPROC,
                            flscape_file_1=flscape_file_1, flscape_file_2=flscape_file_2,
                            fl_offset=fl_offset,
                            pre_median=pre_median, dist_metric=dist_metric,
                            canny_low=canny_low, canny_high=canny_high, g_sigma=g_sigma,
                            m_min=m_min, m_max=m_max, M=M, Mx=Mx, My=My,
                            w_min=w_min, w_max=w_max, W=W, Wx=Wx, Wy=Wy,
                            f1_mul=f1_mul, lx_thresh=lx_thresh,
                            is_add_front_nodes=is_add_front_nodes,
                            symm_data=symm_data, symm_smooth=symm_smooth,
                            GCUT=GCUT, D_gcut=D_gcut)

    # Plot graph-cut optimised results
    fig, ax = plt.subplots(3, 3, constrained_layout=True)
    ax[0,0].set_title('M')
    ax[0,1].set_title('W')
    ax[0,2].set_title('Image')
    ax[1,0].set_title('Mx')
    ax[1,1].set_title('Wx')
    ax[1,2].set_title('Init')
    ax[2,0].set_title('My')
    ax[2,1].set_title('Wy')
    ax[2,2].set_title('GCut')
    W = np.float32(W)
    Wx = np.float32(Wx)
    Wy = np.float32(Wy)
    tracker_00 = func.ImageScroller(ax[0,0], M, vmin=M.min(), vmax=M.max())
    tracker_01 = func.ImageScroller(ax[0,1], W, vmin=W.min(), vmax=W.max())
    tracker_02 = func.ImageScroller(ax[0,2], I, vmin=0, vmax=255)
    tracker_10 = func.ImageScroller(ax[1,0], Mx, vmin=Mx.min(), vmax=Mx.max())
    tracker_11 = func.ImageScroller(ax[1,1], Wx, vmin=Wx.min(), vmax=Wx.max())
    tracker_12 = func.ImageScroller(ax[1,2], D_init, vmin=vmin, vmax=vmax)
    tracker_20 = func.ImageScroller(ax[2,0], My, vmin=My.min(), vmax=My.max())
    tracker_21 = func.ImageScroller(ax[2,1], Wy, vmin=Wy.min(), vmax=Wy.max())
    tracker_22 = func.ImageScroller(ax[2,2], D_gcut, vmin=vmin, vmax=vmax)
    fig.canvas.mpl_connect('scroll_event', tracker_00.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_01.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_02.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_10.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_11.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_12.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_20.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_21.onscroll)
    fig.canvas.mpl_connect('scroll_event', tracker_22.onscroll)
    plt.show()
    plt.get_current_fig_manager().window.showMaximized()
