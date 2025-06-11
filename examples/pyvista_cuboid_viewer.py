import math
import os
import sys

import numpy as np
import pandas as pd
import pyvista as pv

from napari_tissue_cuboid_analysis._utils import Cuboid

DIR_PATH = ""
max_files = None  # limit the number of cuboids rendered
grid_size = 2
plotter = pv.Plotter(shape=(grid_size, grid_size), border=True)
current_page = 0
plot_cubes = False
smooth_iter = 8


def set_dir_path(dir_path):
    global DIR_PATH
    DIR_PATH = dir_path


def load_cuboids(decimation_percent=0):
    cuboids = []
    df = pd.read_parquet(DIR_PATH + "/metrics.parquet")
    valid_labels = df.index.to_numpy()

    if max_files is not None:
        valid_labels = valid_labels[:max_files]

    for label in valid_labels:
        filename = f"cuboid{label}.stl"
        if os.path.exists(DIR_PATH + "/" + filename):
            cuboid = Cuboid(label=label, dir_path=DIR_PATH)
            cuboid.smooth(smooth_iter)
            cuboid.align()
            cuboids.append(cuboid)

    meshes = []
    metrics = np.zeros((len(cuboids), 6))
    df = pd.read_parquet(DIR_PATH + "/metrics.parquet")

    for i, cuboid in enumerate(cuboids):
        if decimation_percent > 0:
            cuboid.decimate(decimation_percent)
            mesh = cuboid.simplified
        else:
            mesh = cuboid.mesh
        faces = np.hstack(
            [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces]
        )
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        pv_mesh.compute_normals(
            auto_orient_normals=True,
            cell_normals=False,
            point_normals=True,
            inplace=True,
        )
        meshes.append(pv_mesh)
        metrics[i, 0] = cuboid.label
        metrics[i, 1:] = df.loc[cuboid.label]

    return meshes, metrics


def slice_meshes(page, meshes):
    per_page = grid_size**2
    start = page * per_page
    end = min(start + per_page, len(meshes))
    return meshes[start:end], metrics[start:end]


def plot_page(mesh_page, page_metrics):
    plotter.clear()
    for idx, mesh in enumerate(mesh_page):
        row, col = divmod(idx, grid_size)
        plotter.subplot(row, col)
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(light_type="headlight", intensity=1.0))
        plotter.add_mesh(
            mesh, smooth_shading=True, specular=0.5, ambient=0.3, diffuse=0.8
        )

        if plot_cubes:
            a = np.cbrt(mesh.volume)
            cube = pv.Box(bounds=[-a / 2, a / 2, -a / 2, a / 2, -a / 2, a / 2])
            plotter.add_mesh(cube, opacity=0.4)

        text = f"Cuboid{page_metrics[idx,0]:.0f}\nvolume:{page_metrics[idx,1]:.2e}\ncompactness: {page_metrics[idx,2]:.3f}\n"
        text += f"convexity: {page_metrics[idx,3]:.3f}\nIoU: {page_metrics[idx,4]:.3f}\ninertia ratio:{page_metrics[idx,5]:.3f}"
        plotter.add_text(text=text, position="upper_left", font_size=8)
        # feature_edges = mesh.extract_feature_edges(feature_angle=15)
        # plotter.add_mesh(feature_edges, color='black', line_width=2)

    plotter.link_views()
    plotter.reset_camera()
    plotter.render()


def update_plot(page):
    mesh_page, page_metrics = slice_meshes(page, meshes)
    plot_page(mesh_page, page_metrics)


def next_page():
    global current_page
    current_page += 1
    if current_page > math.ceil(len(meshes) / grid_size**2) - 1:
        current_page = 0
    elif current_page < 0:
        current_page = math.ceil(len(meshes) // grid_size**2) - 1
    update_plot(current_page)


def previous_page():
    global current_page
    current_page -= 1
    if current_page > math.ceil(len(meshes) / grid_size**2) - 1:
        current_page = 0
    elif current_page < 0:
        current_page = math.ceil(len(meshes) // grid_size**2) - 1
    update_plot(current_page)


def toggle_cubes():
    global plot_cubes
    plot_cubes = 1 - plot_cubes
    update_plot(current_page)


plotter.add_key_event("Right", next_page)
plotter.add_key_event("Left", previous_page)
plotter.add_key_event("c", toggle_cubes)


if __name__ == "__main__":
    set_dir_path(sys.argv[1])
    meshes, metrics = load_cuboids(decimation_percent=0.5)
    meshes_page, page_metrics = slice_meshes(0, meshes)
    plot_page(meshes_page, page_metrics)
    plotter.show()
