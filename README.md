# napari-tissue-cuboid-analysis

[![License BSD-3](https://img.shields.io/pypi/l/napari-tissue-cuboid-analysis.svg?color=green)](https://github.com/EPFL-Center-for-Imaging/napari-tissue-cuboid-analysis/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-tissue-cuboid-analysis.svg?color=green)](https://pypi.org/project/napari-tissue-cuboid-analysis)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-tissue-cuboid-analysis.svg?color=green)](https://python.org)
[![tests](https://github.com/EPFL-Center-for-Imaging/napari-tissue-cuboid-analysis/workflows/tests/badge.svg)](https://github.com/EPFL-Center-for-Imaging/napari-tissue-cuboid-analysis/actions)
[![codecov](https://codecov.io/gh/EPFL-Center-for-Imaging/napari-tissue-cuboid-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/EPFL-Center-for-Imaging/napari-tissue-cuboid-analysis)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-tissue-cuboid-analysis)](https://napari-hub.org/plugins/napari-tissue-cuboid-analysis)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

analysis of the shape of tissue explants in CT images

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-tissue-cuboid-analysis` via [pip]:

    pip install napari-tissue-cuboid-analysis

## License

Distributed under the terms of the [BSD-3] license,
"napari-tissue-cuboid-analysis" is free and open source software

## Usage
Open the plugin automatically when launching [napari]:

    napari --with napari-tissue-cuboid-analysis

### 1: Median binning
Reduces the size of the image by applying an isotropic median kernel. This step is not mandatory and only helps accelerating the rest of the pipeline and removing some noise in the image.

- **Input type:**     Graylevel image
- **Bin kernel**      Size of the median kernel

### 2: Pipette extraction
Extracts a binary mask of the pipette that will be used to discard useless areas in further steps. Computed using 6 manually annotated points on the boundary of the pipette. You can do this by creating a point layer to napari and using the dedicated tool to add points. The three first points must be on the first slice, and the three last points on the last slice.

- **Input type:**     Graylevel image
- **Points**          Points layer for the manual method

### 3: Thresholding
GMM based thresholding to separate tissue from the background. The algorithm can either chose a global threshold for the whole image or compute a continuous map of local thresholds. The local method is significantly slower but helps with small reconstruction artifacts in the images. It works by fitting GMMs on a grid of windows to produce a sparse grid of thresholds. The threshold map is then computed by linear interpolation of the sparse grid.

- **Input type:**    Graylevel image
- **Mask:**          Pipette mask computed in **2: Pipette extraction**
- **Spacing:**       Spacing of the GMM windows (local only)
- **Win. size**:     Size of the GMM windows, as a fraction of the spacing - local thresholding only.\
                     Window size of 0.5 results in non-overlapping but contiguous windows.\
                     Higher values result in overlapping windows.\
                     The window size must be bigger than the largest cuboid to avoid windows only containing tissue.
- **Min. std**       Criteria to discard windows that do not contain tissues in the threshold map - local thresholding only.\
                     Expressed as a fraction of the standard deviation of the whole image.\
                     Increase if empty areas are noisy in the binary result of the thresholding
- **Plot thresh.**   Wether to plot the threshold map the the centers of the all the valid local windows along with the resulting binary image.\
                     Useful to finetune parameters.

**Advice:** Always try global thresholding first. It is quicker and more stable. Only use local thresholding if some regional features in the image are not captured by the global threshold and result in noise in the binary image.

### 4: Morphology
Morphological binary opening and closing with a spherical structuring element on binary images, or on each individual label separately in a labelled image (see **5: Labelling**).

- **Input types:**   Binary or labels
- **Diameter:**      Diameter of the structuring element. Should be odd for consistency
- **Single:**        Wether to apply to a single label or to all - label morphology only

**Advice:** Only apply strictly necessary operations on the binary image for the smooth operation of **5: Labelling**. Then refine the quality of the segmentation and fill tissue porosity by applying those operation to the labels individually.

### 5: Labelling
Object labelling using the watershed algorithm. The results of watershed is usually either under-segmentated or over-segmentated. Over-segmentation is automatically fixed by merging labels based on their surface of contact and their characteristic length.

- **Input type:** Binary
- **Watershed lvl.:** Threshold to merge shallow basins in the watershed algorithm. Decreasing this parameter increases the number of labels generated before the over-segmentation correction. Should be chosen so that the result of the watershed algorithm does not feature any under-segmentation.
- **Merge thresh.:** Over-segmentation coefficient necessary to merge two regions. Decreasing results in more merging.
- **Plot interm.:** Wether to plot the intermediary images of the automatic over-segmentation fix. Iteration zero is the original output of the watershed algorithm. Useful to finetune parameters.

**Advice:** Prefer the *Merge thresh.* param. for refining your results and only change *watershed lvl.* if there is under-segmentation in the result or if you need to drastically reduce the number of labels.

#### Optional manual operations:
- **Input type:** Labels
- **Merge:** Merge a set of target labels selected using their integer id
- **Split:** Split a set of target labels selected using their integer id. The watershed process is re-applied to those specific region. The *watershed lvl.* must be reduced in order to split the target label(s) into more regions.


### 6: Mesh and metrics generation
Saves a mesh for each cuboid at the specified location in `.stl` format, as well as the metrics computed for all the cuboids in both `.parquet` and `.csv` formats. A PyVista viewer is made available to scroll through meshes along with their metrics.

- **Input type:** Labels
- **Directory:** Path to the output directory to store meshes and metrics. Relative to where napari was lauched from, or absolute.
- **Voxel size:** Size of the voxels in microns to normalize the volumes. Don't forget that binning modifies the resolution in the image.
- **Smoothing iterations:** Number of taubin smoothing iterations applied to the meshes. Tread carefully, has a significant impact on compactness and convexity. Other metrics are mostly undisturbed.
- **Single label:** Wether to only construct the mesh and metrics for a single mesh. If true, the mesh is displayed in napari and the metrics printed in the terminal (no saving).



## Shape metrics
- **Convexity**:
  Ratio between the volume of an explant and the volume of its convex hull.

  $$
  \frac{V}{V_{\text{convexhull}}}
  $$

- **Compactness**:
  Dimensionless and normalized quantity that relates the volume of an object to its surface area. The maximum value corresponds to a perfect sphere. A decreasing value indicates elongation or increased surface irregularity.

  $$
  \frac{1}{36\pi}\frac{V^2}{A^3}
  $$

- **Elongation**:
  Ratio between the smallest and largest eigenvalues of the inertia tensor. These eigenvalues describe how mass is distributed along the principal axes, so the metric reflects the overall elongation of the object. Values close to 1 indicate isotropic shapes, while lower values suggest elongated geometries.

  $$
  \frac{\lambda_{\text{min}}}{\lambda_{\text{max}}}
  $$

- **Cube similarity**:
  Intersection over Union (IoU) between the explant and an ideal cube of equal volume. The cube is oriented such that its faces are aligned with the explant’s principal inertia axes. This metric quantifies how closely the shape of the explant resembles a cube.

  $$
  \frac{V_{\text{intersect}}}{V_{\text{union}}}
  $$


## PyVista viewer
Allows you to browse through the cuboids. Each page displays four cuboids, you can scroll through pages using the &larr; and &rarr; keys. Metrics for each cuboid are displayed in the top left corner. You can display the *similar cubes* (same volume and similar orientation) using the `c` key.

### Usage:
[Download the script](https://github.com/EPFL-Center-for-Imaging/napari-tissue-cuboid-analysis/tree/main/examples) from the package examples and run:

    python pyvista_cuboid_viewer.py path_to_cuboids_data






## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
