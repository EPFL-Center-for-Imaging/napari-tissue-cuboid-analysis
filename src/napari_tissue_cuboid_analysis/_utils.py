import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy.interpolate import NearestNDInterpolator, interpn
from scipy.signal import find_peaks
from scipy.stats import norm
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.measure import block_reduce, marching_cubes
from skimage.morphology import closing, disk, opening
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
    auto: bool = True,
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
    sample_frame = img_as_ubyte(slice)
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


IMG = None
MASK = None


def window_gmm(
    idx: int,
    pt: np.ndarray,
    win_size: float,
    min_std: float,
    n_comp: int,
    peak_height: float = -3e-5,
):
    # print(f'worker {id}', flush=True)
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
    if len(window_vals) < n_comp or np.std(window_vals) < min_std:
        valid = False
        return id, valid, thresh

    gmm = GaussianMixture(n_components=n_comp)
    gmm.fit(window_vals.reshape(-1, 1))

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

    minimas, _ = find_peaks(-mixture_pdf, height=peak_height)
    if len(minimas) > 0:
        valid = True
        thresh = thresh_candidates[np.min(minimas)]

    return id, valid, thresh


def window_gmm_wrapper(args):
    return window_gmm(*args)


def local_threshold(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: int,
    win_size: float,
    n_comp: int,
    min_std: float,
    n_processes: int,
):
    """
    Computes a grid of thresholds by applying GMM to windows and interpolates this grid of thresholds to get a full threshold image.
    """
    global IMG, MASK
    IMG = img
    MASK = mask.astype(bool)

    x_grid = np.arange(0, img.shape[0] + 1, spacing - 1)
    y_grid = np.arange(0, img.shape[1] + 1, spacing - 1)
    z_grid = np.arange(0, img.shape[2] + 1, spacing - 1)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    img_center = np.array(img.shape) // 2
    grid_center = np.mean(grid_points, axis=0).astype(int)

    grid_points -= grid_center - img_center

    input_data = []
    win_size = int(round(spacing * win_size))  # scale to spacing
    print(spacing, win_size, grid_points.shape)

    for i, pt in enumerate(grid_points):
        input_data.append((i, pt, win_size, min_std, n_comp))

    with Pool(processes=n_processes) as pool:
        # use imap to allow progress bar, unordered for faster processing
        results = list(
            tqdm(
                pool.imap_unordered(window_gmm_wrapper, input_data),
                total=len(input_data),
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

    binary = img > thresh_dense
    binary &= mask
    thresh_dense = np.zeros_like(img)
    binary = np.zeros_like(mask)

    return binary, thresh_dense


def local_threshold_simple(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: int,
    win_size: float,
    n_comp: int,
    min_std: float,
):
    """
    Computes a grid of thresholds by applying GMM to windows and interpolates this grid of thresholds to get a full threshold image.
    """

    mask = mask.astype(bool)
    print(img.dtype, img.shape)
    print(mask.dtype, mask.shape)
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

    for i, pt in enumerate(tqdm(grid_points)):
        x_start = max(0, pt[0] - win_size)
        x_end = min(pt[0] + win_size - 1, img.shape[0] - 1)
        y_start = max(0, pt[1] - win_size)
        y_end = min(pt[1] + win_size - 1, img.shape[1] - 1)
        z_start = max(0, pt[2] - win_size)
        z_end = min(pt[2] + win_size - 1, img.shape[2] - 1)

        window_vals = img[x_start:x_end, y_start:y_end, z_start:z_end][
            mask[x_start:x_end, y_start:y_end, z_start:z_end]
        ]
        if len(window_vals) < n_comp or np.std(window_vals) < min_std:
            continue

        gmm = GaussianMixture(n_components=n_comp)
        gmm.fit(window_vals.reshape(-1, 1))

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

        minimas, _ = find_peaks(-mixture_pdf, height=-3e-5)
        if len(minimas) > 0:
            thresh = thresh_candidates[np.min(minimas)]
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

    binary = img > thresh_dense
    binary &= mask
    thresh_dense = np.zeros_like(img)
    binary = np.zeros_like(mask)

    return binary, thresh_dense


def watershed_auto_fix(
    binary: np.ndarray, watershed_lvl: int, overseg_threshold: float
):

    labelled = watershed(binary=binary, watershedLevel=watershed_lvl)

    max_iter = 20  # safeguard to avoid infinite loop
    image_iterations = [labelled]
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


def merge_labels(labelled: np.ndarray, targets: np.ndarray):
    merged = labelled.copy()
    mask = np.isin(labelled, targets[1:])
    merged[mask] = targets[0]

    for i in range(np.max(merged)):
        if not np.any(merged == i):
            mask = np.isin(merged, np.arange(i + 1, np.max(merged) + 1))
            merged[mask] -= 1

    return merged


def split_labels(
    labelled: np.ndarray,
    binary: np.array,
    targets: np.ndarray,
    watershed_lvl: int,
):

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


def cuboid_binary_tight(labelled: np.ndarray, label: int):
    if not np.any(labelled == label):
        print(f"Label{label} doesn't appear in provided image")
        return None

    z, y, x = np.where(labelled == label)

    z0, z1 = z.min(), z.max() + 1
    y0, y1 = y.min(), y.max() + 1
    x0, x1 = x.min(), x.max() + 1

    tight = labelled[z0:z1, y0:y1, x0:x1]
    padded = np.pad(tight, pad_width=1, mode="constant", constant_values=0)
    binary = (padded == label).astype(bool)

    return binary


def generate_single_cuboid(
    labelled: np.ndarray, label: int, vsize: float, smooth_iter: int
):
    binary_tight = cuboid_binary_tight(labelled, label)

    cuboid = Cuboid(label=label, binary=binary_tight, vsize=vsize)
    cuboid.smooth(iterations=smooth_iter)
    cuboid.align()

    metrics = cuboid.metrics()
    vertices = cuboid.mesh.vertices
    faces = cuboid.mesh.faces

    return vertices, faces, metrics


def generate_multiple_cuboids_simple(
    labelled: np.ndarray, vsize: float, smooth_iter: int, dir_path: str = None
):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    n = np.max(labelled)
    columns = ["volume", "compactness", "convexity", "IoU", "intertia_ratio"]
    df = pd.DataFrame(0.0, index=np.arange(1, n + 1), columns=columns)

    for label in tqdm(range(1, n + 1), desc="Generating"):
        binary_tight = cuboid_binary_tight(labelled, label)

        cuboid = Cuboid(
            label=label, binary=binary_tight, vsize=vsize, dir_path=dir_path
        )
        cuboid.smooth(iterations=smooth_iter)
        cuboid.align()
        cuboid.save()

        df.loc[label] = cuboid.metrics()

    df.to_csv(dir_path + "/metrics.csv")
    df.to_parquet(dir_path + "/metrics.parquet")


class Cuboid:
    def __init__(self, label, dir_path=None, binary=None, vsize=6):
        self.label = label
        self.voxel_size = vsize * 1e-3
        self.dir_path = dir_path

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

    def generate(self, binary):
        if binary.size == 0:
            print(f"Label {self.label} not found in image")
            return

        verts, faces, _, _ = marching_cubes(binary)
        self.mesh = trimesh.Trimesh(verts, faces)
        self.mesh.vertices -= self.mesh.center_mass
        trimesh.repair.fix_inversion(self.mesh)
        if not self.mesh.is_watertight:
            print(f"Mesh for cuboid {self.label} is not watertight")

    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.mesh = trimesh.load_mesh(f, file_type="stl")
        if not self.mesh.is_watertight:
            print(f"Mesh for cuboid {self.label} is not watertight")

    def save(self):
        if self.mesh is not None:
            file_path = self.dir_path + f"/cuboid{self.label}.stl"
            self.mesh.export(file_path)

    def smooth(self, iterations=5):
        trimesh.smoothing.filter_taubin(self.mesh, iterations=iterations)

    def decimate(self, decimation_percent):
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
        ratio = np.max(components) / np.min(components)
        return ratio

    def metrics(self):
        return (
            self.volume(),
            self.compactness(),
            self.convexity(),
            self.cube_IoU(),
            self.inertia_ratio(),
        )
