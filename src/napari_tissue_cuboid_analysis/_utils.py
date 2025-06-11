import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy.interpolate import NearestNDInterpolator, interpn
from scipy.ndimage import binary_fill_holes, uniform_filter1d
from scipy.stats import norm
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from skimage.feature import canny, peak_local_max
from skimage.filters import threshold_multiotsu
from skimage.measure import block_reduce, marching_cubes
from skimage.morphology import (
    binary_closing,
    binary_opening,
    closing,
    disk,
    opening,
    remove_small_objects,
)
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.util import img_as_ubyte
from sklearn.mixture import GaussianMixture
from spam.label import (
    detectOverSegmentation,
    fixOversegmentation,
    setVoronoi,
    watershed,
)
from tqdm import tqdm


def bin_median(img: np.ndarray, kernel_size: int = 2, verbose: bool = False):
    """
    Reduces the resolution of a 3D image by applying a median filter over non-overlapping blocks.

    Args:
    - img (np.ndarray): The input 3D image as a NumPy array.
    - kernel_size (int): The size of the blocks for downsampling.

    Returns:
    - binned (np.ndarray): The binned 3D image with reduced resolution
    """

    binned = block_reduce(
        img, block_size=(kernel_size, kernel_size, kernel_size), func=np.median
    ).astype(np.uint16)

    if verbose:
        print(f"Original size: {img.shape}, binned size: {binned.shape}")

    return binned


def extract_pipette_2D(
    slice_2D: np.ndarray,
    canny_sigma: int = 2,
    canny_thresh: tuple = (5, 8),
    win_size: int = 7,
    plot: bool = False,
    verbose: bool = False,
) -> tuple[int, np.ndarray]:
    """
    Detects the inner radius and center of a pipette in a 2D image slice using edge detection and Hough transform.

    Args:
    - slice (np.ndarray): input 2D image slice
    - canny_sigma (int): standard deviation for the Gaussian filter used in Canny edge detection
    - canny_thresh (tuple): low and high thresholds for Canny edge detection
    - win_size (int): size of the window around the image center for Hough transform refinement

    Returns:
    - radius (int): inner radius of the pipette
    - center (np.ndarray): coordinates of the pipette center
    - plt_img (np.ndarray): display of canny edges and circle detection
    """
    sample_frame = img_as_ubyte(slice_2D)
    opened = opening(sample_frame, disk(3))  # tune if size of image changes
    closed = closing(opened, disk(3))

    edges = canny(
        closed,
        sigma=canny_sigma,
        low_threshold=canny_thresh[0],
        high_threshold=canny_thresh[1],
    )

    hough_radii = np.arange(
        sample_frame.shape[0] / 6, sample_frame.shape[0] / 2, 2
    )
    hough_res = hough_circle(edges, hough_radii)

    frame_center = np.around(np.array(sample_frame.shape) / 2).astype(
        int
    )  # tune if size of image changes
    hough_center_win = hough_res[
        :,
        frame_center[0] - win_size : frame_center[0] + win_size,
        frame_center[1] - win_size : frame_center[1] + win_size,
    ]
    offset = win_size - frame_center

    accums, cx, cy, radii = hough_circle_peaks(
        hough_center_win, hough_radii, total_num_peaks=20
    )
    cx = cx[0] - offset[0]
    cy = cy[0] - offset[1]

    gmm = GaussianMixture(2, covariance_type="full")
    gmm.fit(radii.reshape(-1, 1))

    estimated_radius = np.min(
        gmm.means_
    )  # select between inner and outer face of the pipette

    # finetuning of the radius for the first slice
    hough_radii_precise = np.arange(
        estimated_radius - 10, estimated_radius + 10, 0.2
    )
    hough_res_precise = hough_circle(edges, radius=hough_radii_precise)
    accums, cx, cy, radius = hough_circle_peaks(
        hough_res_precise, hough_radii_precise, total_num_peaks=1
    )
    radius = int(round(radius[0]))

    cx = int(round(cx[0]))
    cy = int(round(cy[0]))
    center = np.array([cx, cy])

    if verbose:
        print("Pipette inner radius slice found: ", radius)

    plt_img = None
    if plot:
        plt_img = gray2rgb(255 * np.array(edges, dtype=np.uint16))
        # image = sk.color.gray2rgb(np.array(closed, dtype=np.uint16))
        circx, circy = circle_perimeter(cy, cx, radius, shape=plt_img.shape)
        plt_img[circx, circy] = (220, 20, 20)

    return radius, center, plt_img


def pipette_mask_auto(
    img: np.ndarray,
    canny_sigma: int = 2,
    canny_thresh: tuple = (5, 8),
    win_size: int = 7,
    plot: bool = False,
    verbose: bool = False,
):
    """
    Creates a mask of the interior of the pipette using Hough circle transform

    Args:
    - img (np.ndarray): input 3D image
    - canny_sigma (int): standard deviation for the Gaussian filter used in Canny edge detection
    - canny_thresh (tuple): low and high thresholds for Canny edge detection
    - win_size (int): size of the window around the image center for Hough transform refinement
    - auto (bool): wether to use fully automatic pipette detection or not

    Returns:
    - pipette mask (np.ndarray): boolean mask of the interior of the pipette
    """

    pipette_mask = np.zeros(img.shape, dtype="bool")
    y, x = np.ogrid[: pipette_mask.shape[1], : pipette_mask.shape[2]]

    top_radius, top_center, top_img = extract_pipette_2D(
        img[0], canny_sigma, canny_thresh, win_size, plot=plot, verbose=verbose
    )
    bottom_radius, bottom_center, bottom_img = extract_pipette_2D(
        img[-1], canny_sigma, canny_thresh, plot=plot, verbose=verbose
    )

    for i in range(img.shape[0]):
        slice_radius = top_radius + i * (bottom_radius - top_radius) / (
            img.shape[0] - 1
        )
        slice_center = top_center + i * (bottom_center - top_center) / (
            img.shape[0] - 1
        )
        pipette_mask[i] = (slice_center[0] - x) ** 2 + (
            slice_center[1] - y
        ) ** 2 <= slice_radius**2

    if plot:
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.imshow(bottom_img)

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.imshow(top_img)

        plt.show()

    return pipette_mask


def circle_from_pts(points: np.ndarray):
    """
    Finds the circle passing through three points
    Args:
    - points (np.ndarray): three non colinear 2D points

    Returns:
    - radius (float): radius of the circle
    -center (np.ndarray): center of the circle
    """

    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    mid_ab = (p1 + p2) / 2
    mid_bc = (p2 + p3) / 2

    dir_ab = p2 - p1
    dir_bc = p3 - p2

    perp_ab = np.array([-dir_ab[1], dir_ab[0]])
    perp_bc = np.array([-dir_bc[1], dir_bc[0]])

    mat = np.vstack([perp_ab, -perp_bc]).T
    rhs = mid_bc - mid_ab

    t = np.linalg.solve(mat, rhs)
    center = mid_ab + t[0] * perp_ab
    radius = np.linalg.norm(center - p1)

    return radius, center


def pipette_mask_manual(img: np.ndarray, points: np.ndarray):
    """
    Creates a mask of the interior of the pipette using manually selected points

    Args:
    - img (np.ndarray): input 3D image
    - points (np.ndarray): points on the edge of the pipette on the first and last slice

    Returns:
    - pipette mask (np.ndarray): boolean mask of the interior of the pipette
    """

    pipette_mask = np.zeros(img.shape, dtype="bool")
    x, y = np.ogrid[: pipette_mask.shape[1], : pipette_mask.shape[2]]

    if points.shape[0] != 6:
        print("Manual pipette extraction expects exactly 6 points")
        return pipette_mask

    if np.any(points[0:3, 0] != 0):
        print("The three first points should be on the first slice")
        return pipette_mask

    if np.any(points[4:6, 0] != img.shape[0] - 1):
        print("The three last points should be on the last slice")
        return pipette_mask

    bottom_radius, bottom_center = circle_from_pts(points[:3, 1:])
    top_radius, top_center = circle_from_pts(points[3:, 1:])

    for i in range(img.shape[0]):
        slice_radius = bottom_radius + i * (top_radius - bottom_radius) / (
            img.shape[0] - 1
        )
        slice_center = bottom_center + i * (top_center - bottom_center) / (
            img.shape[0] - 1
        )
        pipette_mask[i] = (slice_center[0] - x) ** 2 + (
            slice_center[1] - y
        ) ** 2 <= slice_radius**2

    return pipette_mask


# global definition of heavy variables to reduce RAM load in multiprocessing
IMG = None
MASK = None


def window_gmm(
    win_id: int,
    pt: np.ndarray,
    win_size: float,
    min_std: float,
    global_thresh: float,
):
    """
    Applies the GMM thresholding algorithm to a single window. Used for parallel computation of the local threshold.

    Args:
    - win_id (int): id of the window
    - pt (np.ndarray): center point of the window in the image
    - win_size (int): size of the local window
    - min_std (float): minimum std of the window to be valid
    - global_thresh (float): reference global threshold for minima selection

    Returns:
    - id (int): id of the window
    - valid (bool): wether the window results in a valid threshold
    - threshold (float): threshold computed from the GMM
    """

    valid = False
    thresh = 0

    x_start = max(0, pt[0] - win_size)
    x_end = min(pt[0] + win_size - 1, IMG.shape[0] - 1)
    y_start = max(0, pt[1] - win_size)
    y_end = min(pt[1] + win_size - 1, IMG.shape[1] - 1)
    z_start = max(0, pt[2] - win_size)
    z_end = min(pt[2] + win_size - 1, IMG.shape[2] - 1)

    window_vals = IMG[x_start:x_end, y_start:y_end, z_start:z_end][
        MASK[x_start:x_end, y_start:y_end, z_start:z_end]
    ]

    hist, _ = np.histogram(window_vals, 1024)
    hist = uniform_filter1d(
        hist, 20
    )  # smooth histogram to remove high frequency peaks
    peaks = peak_local_max(hist, min_distance=80, num_peaks=3)

    n_comp = max(min(len(peaks), 4), 2)  # limited to [2,4]

    if (
        len(window_vals) < n_comp or np.std(window_vals) < min_std
    ):  # discard unstable edge cases and windows not containing any tissue
        return win_id, valid, thresh

    gmm = GaussianMixture(n_components=n_comp)
    gmm.fit(window_vals.reshape(-1, 1))

    # only accept threshold candidates between the first and last Gaussian means in the pdf
    thresh_candidates = np.arange(np.min(gmm.means_), np.max(gmm.means_), 1)
    mixture_pdf = np.zeros_like(thresh_candidates)
    for i in range(n_comp):
        mixture_pdf += (
            norm.pdf(
                thresh_candidates,
                float(gmm.means_[i, 0]),
                np.sqrt(float(gmm.covariances_[i, 0, 0])),
            )
            * gmm.weights_[i]
        )
    minimas = peak_local_max(-mixture_pdf)
    if len(minimas) > 0:
        minima = minimas[
            np.argmin(
                np.abs(thresh_candidates[minimas] - global_thresh)
            )  # chose the minima that is closest to the global reference
        ]
        thresh = thresh_candidates[minima]
        valid = True

    return win_id, valid, thresh


def window_gmm_wrapper(args):
    # wrapper function to unpack the parameters for multiprocessing
    return window_gmm(*args)


def local_threshold_gmm(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: int,
    win_size: float,
    min_std: float,
    n_processes: int,
):
    """
    Computes a grid of thresholds by fitting GMM to a grid of windows and interpolates this grid to get a full threshold map. Using multiprocessing to paralellise the gmm fitting process

    Args:
    - img (np.ndarray): image on which the local thresholding operation is performed
    - mask (np.ndarray): boolean mask to exlude voxels that lie outside the region of interest
    - spacing (int): spacing between the local windows
    - win_size (float): size of the windows, as a fraction of the spacing
    - min_std (float): minimum std of the window to be valid
    - n_processes (int): number of processes to be run in parallel

    Returns:
    - thresh_dense (np.ndarray): dense threshold map of the same dimensions as the input image
    - grid_filtered (np.ndarray): grid points that resulted in a valid window from which a local threshold was computed prior to interpolation
    """
    mask = mask.astype(bool)
    global IMG, MASK
    IMG = img
    MASK = mask

    x_grid = np.arange(0, img.shape[0] + 1, spacing - 1)
    y_grid = np.arange(0, img.shape[1] + 1, spacing - 1)
    z_grid = np.arange(0, img.shape[2] + 1, spacing - 1)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    img_center = np.array(img.shape) // 2
    grid_center = np.mean(grid_points, axis=0).astype(int)

    _, global_thresh = global_threshold_gmm(img, mask)

    grid_points -= grid_center - img_center

    input_data = []
    win_size = int(round(spacing * win_size))  # scale to spacing

    min_std *= np.std(img)  # absolute thresh from relative thresh

    for i, pt in enumerate(grid_points):
        input_data.append((i, pt, win_size, min_std, global_thresh))

    with Pool(processes=n_processes) as pool:
        # use imap to allow progress bar, unordered for faster processing
        results = list(
            tqdm(
                pool.imap_unordered(window_gmm_wrapper, input_data),
                total=len(input_data),
                desc="Local windows",
            )
        )
    # reorder results
    ordered_results = np.zeros(
        len(input_data), dtype=[("valid", bool), ("thresh", float)]
    )
    for i, valid, thresh in results:
        ordered_results[i] = (valid, thresh)

    valid_mask = ordered_results["valid"]
    grid_filtered = grid_points[valid_mask]  # discard non valid points
    thresh_sparse = ordered_results["thresh"][valid_mask]

    interpolator = NearestNDInterpolator(grid_filtered, thresh_sparse)
    thresh_grid = interpolator(grid_points)  # to get a regular grid

    thresh_grid = thresh_grid.reshape(
        x_grid.shape[0], y_grid.shape[0], z_grid.shape[0]
    )

    x_dense = np.arange(img.shape[0])
    y_dense = np.arange(img.shape[1])
    z_dense = np.arange(img.shape[2])

    X, Y, Z = np.meshgrid(x_dense, y_dense, z_dense, indexing="ij")

    thresh_dense = interpn(
        points=(x_grid, y_grid, z_grid),
        values=thresh_grid,
        xi=(X, Y, Z),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    thresh_dense = thresh_dense.reshape(
        x_dense.shape[0], y_dense.shape[0], z_dense.shape[0]
    )
    thresh_dense *= mask

    return thresh_dense, grid_filtered


def local_threshold_gmm_simple(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: int,
    win_size: float,
    min_std: float,
):
    """
    Computes a grid of thresholds by fitting GMM to a grid of windows and interpolates this grid to get a full threshold map.

    Args:
    - img (np.ndarray): image on which the local thresholding operation is performed
    - mask (np.ndarray): boolean mask to exlude voxels that lie outside the region of interest
    - spacing (int): spacing between the local windows
    - win_size (float): size of the windows, as a fraction of the spacing
    - min_std (float): minimum std of the window to be valid

    Returns:
    - thresh_dense (np.ndarray): dense threshold map of the same dimensions as the input image
    - grid_filtered (np.ndarray): grid points that resulted in a valid window from which a local threshold was computed prior to interpolation
    """

    mask = mask.astype(bool)
    x_grid = np.arange(0, img.shape[0] + 1, spacing - 1)
    y_grid = np.arange(0, img.shape[1] + 1, spacing - 1)
    z_grid = np.arange(0, img.shape[2] + 1, spacing - 1)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    img_center = np.array(img.shape) // 2
    grid_center = np.mean(grid_points, axis=0).astype(int)

    grid_points -= grid_center - img_center

    grid_filtered = []
    thresh_sparse = []

    win_size = int(round(spacing * win_size))  # scale to spacing

    _, global_thresh = global_threshold_gmm(img, mask)
    min_std *= np.std(img)  # absolute thresh from relative thresh

    for i, pt in enumerate(tqdm(grid_points, desc="Local windows")):
        x_start = max(0, pt[0] - win_size)
        x_end = min(pt[0] + win_size - 1, img.shape[0] - 1)
        y_start = max(0, pt[1] - win_size)
        y_end = min(pt[1] + win_size - 1, img.shape[1] - 1)
        z_start = max(0, pt[2] - win_size)
        z_end = min(pt[2] + win_size - 1, img.shape[2] - 1)

        window_vals = img[x_start:x_end, y_start:y_end, z_start:z_end][
            mask[x_start:x_end, y_start:y_end, z_start:z_end]
        ]

        hist, _ = np.histogram(
            window_vals, 1024
        )  # smooth histogram to remove high frequency peaks
        hist = uniform_filter1d(hist, 20)
        peaks = peak_local_max(hist, min_distance=80, num_peaks=3)

        n_comp = max(min(len(peaks), 4), 2)  # limited to [2,4]

        if (
            len(window_vals) < n_comp or np.std(window_vals) < min_std
        ):  # discard edge cases and windows not containing any tissue
            continue

        gmm = GaussianMixture(n_components=n_comp)
        gmm.fit(window_vals.reshape(-1, 1))
        # only accept threshold candidates between the first and last Gaussian means in the pdf
        thresh_candidates = np.arange(
            np.min(gmm.means_), np.max(gmm.means_), 1
        )
        mixture_pdf = np.zeros_like(thresh_candidates)
        for i in range(n_comp):
            mixture_pdf += (
                norm.pdf(
                    thresh_candidates,
                    float(gmm.means_[i, 0]),
                    np.sqrt(float(gmm.covariances_[i, 0, 0])),
                )
                * gmm.weights_[i]
            )
        minimas = peak_local_max(-mixture_pdf)
        if len(minimas) > 0:
            minima = minimas[
                np.argmin(
                    np.abs(thresh_candidates[minimas] - global_thresh)
                )  # only accept threshold candidates between the first and last Gaussian means in the pdf
            ]
            gmm_thresh = thresh_candidates[minima]
            grid_filtered.append(pt)
            thresh_sparse.append(gmm_thresh)

    grid_filtered = np.array(grid_filtered)
    thresh_sparse = np.array(thresh_sparse)

    print("Interpolation in progress...")
    interpolator = NearestNDInterpolator(grid_filtered, thresh_sparse)
    thresh_grid = interpolator(
        grid_points
    )  # to get a rectangular/regular grid

    thresh_grid = thresh_grid.reshape(
        x_grid.shape[0], y_grid.shape[0], z_grid.shape[0]
    )

    x_dense = np.arange(img.shape[0])
    y_dense = np.arange(img.shape[1])
    z_dense = np.arange(img.shape[2])

    X, Y, Z = np.meshgrid(x_dense, y_dense, z_dense, indexing="ij")

    thresh_dense = interpn(
        points=(x_grid, y_grid, z_grid),
        values=thresh_grid,
        xi=(X, Y, Z),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    thresh_dense = thresh_dense.reshape(
        x_dense.shape[0], y_dense.shape[0], z_dense.shape[0]
    )
    thresh_dense *= mask

    return thresh_dense, grid_filtered


def global_threshold_gmm(img, mask):
    """
    Computes a global thresholds by fitting a GMM to the intensity histogram of an image.

    Args:
    - img (np.ndarray): image on which the thresholding operation is performed
    - mask (np.ndarray): boolean mask to exlude voxels that lie outside the region of interest

    Returns:
    - binary (np.ndarray): binary image resulting from the thresolding operation
    """

    mask = mask.astype(bool)
    masked = img[mask]

    subsample = np.random.choice(masked.ravel(), size=int(1e6), replace=False)

    gmm = GaussianMixture(2, covariance_type="full")
    gmm.fit(subsample.reshape(-1, 1))

    thresh_candidates = np.arange(np.min(gmm.means_), np.max(gmm.means_), 1)
    mixture_pdf = np.zeros_like(thresh_candidates)
    for i in range(2):
        mixture_pdf += norm.pdf(
            thresh_candidates,
            float(gmm.means_[i, 0]),
            np.sqrt(float(gmm.covariances_[i, 0, 0])),
        )

    gmm_thresh = thresh_candidates[np.argmin(mixture_pdf)]
    binary = img > gmm_thresh

    binary &= mask
    binary = binary_fill_holes(binary)  # fill closed holes

    return binary, gmm_thresh


def local_threshold_multi_otsu(
    img: np.ndarray, mask: np.ndarray, spacing: int, win_size: float
):
    """
    Computes a grid of thresholds by applying multi-otsu thresholding to windows and interpolates this grid of thresholds to get a full threshold map.

    Args:
    - img (np.ndarray): image on which the local thresholding operation is performed
    - mask (np.ndarray): boolean mask to exlude voxels that lie outside the region of interest
    - spacing (int): spacing between the local windows
    - win_size (float): size of the windows, as a fraction of the spacing

    Returns:
    - thresh_dense (np.ndarray): dense threshold map of the same dimensions as the input image
    - grid_filtered (np.ndarray): grid points that resulted in a valid window from which a local threshold was computed prior to interpolation
    """

    mask = mask.astype(bool)

    x_grid = np.arange(0, img.shape[0] + 1, spacing - 1)
    y_grid = np.arange(0, img.shape[1] + 1, spacing - 1)
    z_grid = np.arange(0, img.shape[2] + 1, spacing - 1)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    img_center = np.array(img.shape) // 2
    grid_center = np.mean(grid_points, axis=0).astype(int)

    grid_points -= grid_center - img_center

    grid_filtered = []
    thresh_sparse = []

    win_size = int(round(spacing * win_size))  # scale to spacing

    global_otsu = threshold_multiotsu(img[mask], 3).min()

    for pt in tqdm(grid_points, desc="Local windows"):
        if not mask[int(pt[0]), int(pt[1]), int(pt[2])]:
            continue

        x_start = max(0, pt[0] - win_size)
        x_end = min(pt[0] + win_size - 1, img.shape[0] - 1)
        y_start = max(0, pt[1] - win_size)
        y_end = min(pt[1] + win_size - 1, img.shape[1] - 1)
        z_start = max(0, pt[2] - win_size)
        z_end = min(pt[2] + win_size - 1, img.shape[2] - 1)

        window_vals = img[x_start:x_end, y_start:y_end, z_start:z_end][
            mask[x_start:x_end, y_start:y_end, z_start:z_end]
        ]
        if len(window_vals) < 10 or np.std(window_vals) < 6000:
            continue

        hist, bins = np.histogram(window_vals, 1024)
        hist = uniform_filter1d(hist, 20)
        peaks = peak_local_max(hist, min_distance=80, num_peaks=3)

        if len(peaks) < 2:
            continue

        n_classes = min(len(peaks) + 1, 4)
        thresh = threshold_multiotsu(classes=n_classes, hist=(hist, bins))
        thresh = thresh[np.argmin(np.abs(thresh - global_otsu))]

    grid_filtered.append(pt)
    thresh_sparse.append(thresh)

    grid_filtered = np.array(grid_filtered)
    thresh_sparse = np.array(thresh_sparse)

    interpolator = NearestNDInterpolator(grid_filtered, thresh_sparse)
    thresh_grid = interpolator(grid_points)  # to get a regular grid

    thresh_grid = thresh_grid.reshape(
        x_grid.shape[0], y_grid.shape[0], z_grid.shape[0]
    )

    x_dense = np.arange(img.shape[0])
    y_dense = np.arange(img.shape[1])
    z_dense = np.arange(img.shape[2])

    X, Y, Z = np.meshgrid(x_dense, y_dense, z_dense, indexing="ij")

    thresh_dense = interpn(
        points=(x_grid, y_grid, z_grid),
        values=thresh_grid,
        xi=(X, Y, Z),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    thresh_dense = thresh_dense.reshape(
        x_dense.shape[0], y_dense.shape[0], z_dense.shape[0]
    )
    thresh_dense *= mask

    return thresh_dense, thresh_grid


def global_threshold_multi_otsu(img, mask):
    """
    Computes a global thresholds using multi-otsu thresholding to the intensity histogram of an image.

    Args:
    - img (np.ndarray): image on which the thresholding operation is performed
    - mask (np.ndarray): boolean mask to exlude voxels that lie outside the region of interest

    Returns:
    - binary (np.ndarray): binary image resulting from the thresolding operation
    """

    mask = mask.astype(bool)

    thresh = threshold_multiotsu(img[mask], classes=3).min()
    binary = img > thresh

    binary = binary_fill_holes(binary)
    binary &= mask
    return binary


def apply_threshold(img, thresh_dense, mask):
    """
    Applies the local threshold map to an image

    Args:
    - img (np.ndarray): image on which the thresholding operation is performed
    - thresh_dense (np.ndarray): dense threshold map of the same dimensions as the input image
    - mask (np.ndarray): boolean mask to exlude voxels that lie outside the region of interest

    Returns:
    - binary (np.ndarray): binary image resulting from the thresolding operation
    """
    mask = mask.astype(bool)
    binary = img > thresh_dense
    binary = binary_fill_holes(binary)
    binary &= mask
    return binary


def ball(d: int):
    """
    Constructs a spherical structuring element for morphology

    Args:
    - d (int): diameter of the ball, should be odd

    Returns:
    - b (np.ndarray): structuring element of the specified diameter
    """

    z, y, x = np.ogrid[:d, :d, :d]
    b = np.zeros((d, d, d))
    r = (d - 1) / 2
    b = (x - r) ** 2 + (y - r) ** 2 + (z - r) ** 2 <= r**2
    return b


def bin_opening(labels: np.ndarray, d: int, single: bool = None):
    """
    Applies binary morphological opening to the input image. If the image contains non-binary labels, opening is applied to each label individually.

    Args:
    - labels (np.ndarray): input labels to which opening is applied
    - d (int): diameter of the spherical structuring element
    - single (int): which label should opening be applied to. If None, opening is applied to all labels. Only valid for non-binary images.

    Returns:
    - opened (np.ndarray): opened image
    """

    labels = labels.astype(np.uint16)
    opened = np.zeros_like(labels)

    if labels.max() <= 1:
        return binary_opening(labels, ball(d))

    elif single is not None:
        binary, z0, z1, y0, y1, x0, x1 = cuboid_binary_tight(
            labels, single, pad_width=(d // 2 + 1)
        )
        binary = binary_opening(binary, ball(d))

        opened[z0:z1, y0:y1, x0:x1][binary] = single

    else:
        for label in tqdm(range(1, labels.max() + 1), desc="Opening"):
            binary, z0, z1, y0, y1, x0, x1 = cuboid_binary_tight(
                labels, label, pad_width=(d // 2 + 1)
            )
            binary = binary_opening(binary, ball(d))

            opened[z0:z1, y0:y1, x0:x1][binary] = label

    return opened


def bin_closing(labels, d, single=None):
    """
    Applies binary morphological closing to the input image. If the image contains non-binary labels, closing is applied to each label individually.

    Args:
    - labels (np.ndarray): input labels to which closing is applied
    - d (int): diameter of the spherical structuring element
    - single (int): which label should closing be applied to. If None, closing is applied to all labels. Only valid for non-binary images.

    Returns:
    - closed (np.ndarray): closed image
    """

    labels = labels.astype(np.uint16)
    closed = labels.copy()

    if closed.max() <= 1:
        return binary_closing(closed, ball(d))

    elif single is not None:
        binary, z0, z1, y0, y1, x0, x1 = cuboid_binary_tight(
            closed, single, pad_width=(d // 2 + 1)
        )
        binary = binary_closing(binary, ball(d))
        binary = binary_fill_holes(binary)

        closed[z0:z1, y0:y1, x0:x1][binary] = single

    else:
        for label in tqdm(range(1, closed.max() + 1), desc="Closing"):
            binary, z0, z1, y0, y1, x0, x1 = cuboid_binary_tight(
                closed, label, pad_width=(d // 2 + 1)
            )
            binary = binary_closing(binary, ball(d))
            binary = binary_fill_holes(binary)

            closed[z0:z1, y0:y1, x0:x1][binary] = label

    return closed


def watershed_auto_fix(
    binary: np.ndarray, watershed_lvl: int, overseg_threshold: float
):
    """
    Applies the watershed algorithm to a binary image to extract object labels. Oversegmentation is fixed automatically detected and fixed.

    Args:
    - binary (np.ndarray): input binary image
    - wateshed_lvl (int): threshold to merge shallow basins in the watershed algorithm (parameter of the watershed algorithm).
    - overseg_threshold (float): over-segmentation coefficient necessary to merge two regions

    Returns:
    - image iterations (list): list containing the original output watershed as well as intermediary and final result of over-segmentation fixing
    """
    print("Watershed in progress...")
    labelled = watershed(binary=binary, watershedLevel=watershed_lvl)

    max_iter = 20  # safeguard to avoid infinite loop
    image_iterations = [labelled]

    print("Fixing over-segmentation...")
    for i in range(max_iter):
        over_seg_coeff, touching_labels = detectOverSegmentation(
            image_iterations[i]
        )
        target_over = np.where(over_seg_coeff > overseg_threshold)[0]
        if target_over.size == 0:
            break
        image_iterations.append(
            fixOversegmentation(
                image_iterations[i], target_over, touching_labels
            )
        )

    return image_iterations


def merge_labels(labelled: np.ndarray, targets: list):
    """
    Merges N labels in an image and reorders the labels to avoid empty labels

    Args:
    - labelled (np.ndarray): input labelled image
    - targets (list): list of labels to be merged

    Returns:
    - merged (np.ndarray): new labelled image with merged target labels
    """
    merged = labelled.copy()
    mask = np.isin(labelled, targets[1:])
    merged[mask] = targets[0]

    for i in reversed(range(np.max(merged))):
        if not np.any(merged == i):
            mask = np.isin(merged, np.arange(i + 1, np.max(merged) + 1))
            merged[mask] -= 1

    return merged


def split_labels(
    labelled: np.ndarray,
    binary: np.array,
    targets: list,
    watershed_lvl: int,
):
    """
    Splits a set of labels in an image by reapplying watershed on a specific region of the image.

    Args:
    - labelled (np.ndarray): input labelled image
    - binary (np.ndarray): binary image before labelling
    - targets (list): list of labels to be merged
    - watershed_lvl: threshold to merge shallow basins in the watershed algorithm. Should be smaller than the one used to compute the original labels in order to split the target labels.

    Returns:
    - split (np.ndarray): new labelled image with split target labels
    """

    voronoi = setVoronoi(labelled, maxPoreRadius=4)
    target_mask = np.isin(voronoi, targets)

    target_binary = binary * target_mask
    local_split = watershed(target_binary, watershedLevel=watershed_lvl)

    new_labels = np.zeros(np.max(local_split) + 1)
    if len(new_labels) <= len(targets) + 1:
        new_labels[1:] = targets[0 : len(new_labels) - 1]
        print(f"Removed {len(targets)+ 1 -len(new_labels)} labels")
    else:
        new_labels[1 : len(targets) + 1] = targets
        new_labels[len(targets) + 1 :] = np.arange(
            np.max(labelled) + 1,
            np.max(labelled) + len(new_labels) - len(targets),
        )
        print(f"Added {len(new_labels) - len(targets) -1} labels")

    local_split = new_labels[local_split]
    split = local_split * target_mask + labelled * (1 - target_mask)
    split = split.astype(np.uint16)

    return split


def cuboid_binary_tight(labelled: np.ndarray, label: int, pad_width: int = 1):
    """
    Extracts the binary image of a target label with minimal dimensions.

    Args:
    - labelled (np.ndarray): input labelled image
    - label (int): target label
    - pad_width (int): number of layers of zero-padding around the cuboid in the binary output

    Returns:
    - binary (np.ndarray): tight binary image of the target label
    - z0, z1, y0, y1, x0, x1 (int): x, y, and z position of the binary image in the original labelled image
    """

    if not np.any(labelled == label):
        print(f"Label{label} doesn't appear in provided image")
        return None

    z, y, x = np.where(labelled == label)

    # padding is necessary for mesh generation, the cuboid can't touch the border of the binary image
    z0, z1 = z.min() - pad_width, z.max() + pad_width + 1
    y0, y1 = y.min() - pad_width, y.max() + pad_width + 1
    x0, x1 = x.min() - pad_width, x.max() + pad_width + 1

    z0 = max(z0, 0)
    z1 = min(z1, labelled.shape[0] + 1)
    y0 = max(y0, 0)
    y1 = min(y1, labelled.shape[1] + 1)
    x0 = max(x0, 0)
    x1 = min(x1, labelled.shape[2] + 1)

    tight = labelled[z0:z1, y0:y1, x0:x1]
    binary = (tight == label).astype(bool)

    return binary, z0, z1, y0, y1, x0, x1


def generate_single_cuboid(
    labelled: np.ndarray, label: int, vsize: float, smooth_iter: int
):
    """
    Constructs the surface mesh and metrics for a single cuboid

    Args:
    - labelled (np.ndarray): input labelled image
    - label (int): target label
    - vsize (int): voxel size in um^3 to compute the volume and surface area of the mesh
    - smooth_iter (int): number of taubin smoothing iterations applied to the mesh

    Returns:
    - vertices (np.ndarray): coordinates of the vertices of the mesh
    - faces (np.ndarray): triangular faces of the mesh. Each face is a list of three vertices that form a triangle.
    - metrics (np.ndarray): metrics computed fot the target label
    """

    binary_tight, *_ = cuboid_binary_tight(labelled, label)
    binary_tight = np.pad(
        binary_tight, pad_width=1, mode="constant", constant_values=0
    )

    cuboid = Cuboid(label=label, binary=binary_tight, vsize=vsize)

    # if mesh generation failed
    if cuboid.mesh is None:
        return None, None, None

    if not cuboid.mesh.is_watertight:
        return cuboid.mesh.vertices, cuboid.mesh.faces, None

    cuboid.smooth(iterations=smooth_iter)
    cuboid.align()

    metrics = cuboid.metrics()
    vertices = cuboid.mesh.vertices
    faces = cuboid.mesh.faces

    return vertices, faces, metrics


def generate_multiple_cuboids_simple(
    labelled: np.ndarray,
    vsize: float,
    smooth_iter: int,
    dir_path: str = None,
    metrics_only: bool = False,
):
    """
    Constructs the surface mesh and metrics for all labels in an image. Saves the meshes as .stl (optional) and metrics as .parquet and .csv
    This operation isn't paralellised but could easily be with multiprocessing

    Args:
    - labelled (np.ndarray): input labelled image
    - vsize (int): voxel size in um^3 to compute the volume and surface area of the mesh
    - smooth_iter (int): number of taubin smoothing iterations applied to the mesh
    - dir_path (str): output path for the saving of meshes and metrics
    - metrics_only (bool): controls wether meshes are saved or not
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    n = np.max(labelled)
    columns = ["volume", "compactness", "convexity", "IoU", "inertia_ratio"]
    df = pd.DataFrame(0.0, index=np.arange(1, n + 1), columns=columns)

    invalid_count = 0
    for label in tqdm(range(1, n + 1), desc="Generating"):
        binary_tight, *_ = cuboid_binary_tight(labelled, label)
        binary_tight = np.pad(
            binary_tight, pad_width=1, mode="constant", constant_values=0
        )

        cuboid = Cuboid(
            label=label, binary=binary_tight, vsize=vsize, dir_path=dir_path
        )

        if cuboid.mesh is None:
            invalid_count += 1
            df.drop(label, inplace=True)
            continue
        if not cuboid.mesh.is_watertight:
            invalid_count += 1
            df.drop(label, inplace=True)
            continue

        cuboid.smooth(iterations=smooth_iter)
        cuboid.align()

        if not metrics_only:
            cuboid.save()

        df.loc[label] = cuboid.metrics()

    print(
        f"Generated {n-invalid_count}/{n} cuboids successfuly\nInvalid cuboids are probably not watertight"
    )

    df.to_csv(dir_path + "/metrics.csv")
    df.to_parquet(dir_path + "/metrics.parquet")


class Cuboid:
    """
    Cuboid class containing the tools to contruct and modify the surface mesh of a cuboid, as well as compute and store shape metrics.
    """

    def __init__(self, label, dir_path=None, binary=None, vsize=6):
        """
        Initializes an instance of Cuboid

        Args:
        - label (np.ndarray): unique cuboid label
        - dir_path (str): directory to save or load cuboid data
        - vsize (int): voxel size in um^3 to compute the volume and surface area of the mesh
        - binary (np.ndarray): tight binary image of a cuboid to construct a surface mesh from. If None, the cuboid data is assumed to be existent and is loaded from dir_path.
        """
        self.label = label
        self.voxel_size = vsize * 1e-3
        self.dir_path = dir_path
        self.mesh = None

        if binary is not None:
            self.generate(binary)

        elif dir_path is not None:
            file_path = dir_path + f"/cuboid{label}.stl"
            if os.path.isfile(file_path):
                self.load(file_path)
            else:
                print(f"Cuboid{label}.stl does not exist")

        else:
            print(
                f"Cuboid{label} couldn't be generated\nBoth the directory path and the labelled image are invalid"
            )

    def generate(self, binary, verbose=False):
        """
        Generate a surface mesh from a tight binary image of a cuboid using Lewiner marching cubes. Use standard mesh fixing operation if the resulting mesh is not watertight.

        Args:
        - binary (np.ndarray): tight binary image
        """
        if binary.size == 0:
            print(f"Label {self.label} not found in image")
            return

        binary = binary.astype(bool)  # safety check

        binary = remove_small_objects(
            binary, min_size=100
        )  # remove noise to avoid watertighness issues
        try:
            verts, faces, _, _ = marching_cubes(binary)
        except RuntimeError:
            return
        self.mesh = trimesh.Trimesh(verts, faces)
        self.mesh.vertices -= self.mesh.center_mass
        trimesh.repair.fix_inversion(self.mesh)
        if not self.mesh.is_watertight:
            trimesh.repair.fill_holes(self.mesh)
            if not self.mesh.is_watertight and verbose:
                print(f"Mesh for cuboid {self.label} is not watertight")

    def load(self, file_path, verbose=False):
        with open(file_path, "rb") as f:
            self.mesh = trimesh.load_mesh(f, file_type="stl")
        if not self.mesh.is_watertight and verbose:
            print(f"Mesh for cuboid {self.label} is not watertight")

    def save(self):
        if self.mesh is not None:
            file_path = self.dir_path + f"/cuboid{self.label}.stl"
            self.mesh.export(file_path)

    def smooth(self, iterations=5):
        """
        Apply Taubin smoothing to the cuboid mesh.

        Args:
        - iterations (int): number of smoothing iterations applied to the mesh
        """
        # two iterations of trimesh taubin is one skrinkage pass and one restoration pass
        trimesh.smoothing.filter_taubin(
            self.mesh, lamb=0.5, nu=0.53, iterations=2 * iterations
        )  # using standard and stable lambda/nu parameters

    def decimate(self, decimation_percent):
        """
        Apply quadratic decimation to a surface mesh to simplify its representation

        Args:
        - decimation_percent (float)
        """
        self.simplified = self.mesh.simplify_quadric_decimation(
            percent=decimation_percent
        )

    def align(self):
        self.mesh.apply_transform(self.mesh.principal_inertia_transform)
        self.mesh.vertices -= self.mesh.center_mass

    def volume(self):
        return self.mesh.volume * self.voxel_size**3

    def surface_area(self):
        return self.mesh.area * self.voxel_size**2

    def compactness(self):
        compactness = (
            36 * np.pi * self.volume() ** 2 / self.surface_area() ** 3
        )  # normalized with compactness of sphere = 1
        return compactness

    def cube_IoU(self):
        """
        Creates a cube of similar volume and alignes the principal inertia vectors of the cuboid mesh with the faces of the cube
        """
        transform = self.mesh.principal_inertia_transform
        self.aligned_mesh = self.mesh.copy()
        self.aligned_mesh.apply_transform(transform)

        a = np.cbrt(self.aligned_mesh.volume)
        cube = trimesh.creation.box((a, a, a))

        IoU = (
            self.mesh.intersection(cube).volume / self.mesh.union(cube).volume
        )

        return IoU

    def convexity(self):
        convexity = self.volume() / (
            self.mesh.convex_hull.volume * self.voxel_size**3
        )
        return convexity

    def inertia_ratio(self):
        components = self.mesh.principal_inertia_components
        ratio = np.min(components) / np.max(components)
        return ratio

    def metrics(self):
        return (
            self.volume(),
            self.compactness(),
            self.convexity(),
            self.cube_IoU(),
            self.inertia_ratio(),
        )
