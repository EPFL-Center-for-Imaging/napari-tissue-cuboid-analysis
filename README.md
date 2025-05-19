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
Extracts a binary mask of the pipette that will be used to discard useless areas in further steps. Can either be computed automatically or manually. Default method is automatic.

**Automatic:** Requires tuning of the parameters of a canny edge detector and the window size parameter. If tuned correctly, the pipette is detected automatically. The automatic detection is rather sensitive to the choice of parameters and takes a few second to compute. Intermediary steps of the algorithm can be displayed to help tune the parameters.

**Manual:** Requires selecting three points laying on the inner surface of the pipette on both the first and last slice. This is done by creating a point layer to napari and using the dedicated tool to add points. The three first points must be on the first slice, and the three last points on the last slice. The parameters are not used for this methods.

- **Input type:**     Graylevel image
- **Points**          Points layer for the manual method
- **Sigma**           Standard deviation of the Gaussian filter used by the Canny filter (auto only)
- **Low thr.**        Low threshold of the Canny filter (auto only)
- **High thr.**       High threshold of the Canny filter (auto only)
- **Win. size**       Size of the 2D window in which the algorithm looks for the center of the pipette (auto only)
                      The window is centered on the image
### 3: Thresholding
GMM based thresholding to separate tissue from the background. The algorithm can either chose a global threshold for the whole image or compute a continuous map of local thresholds. The local method is significantly slower but helps with artifacts in the images. It works by fitting GMMs on a grid of windows to produce a sparse grid of thresholds. The threshold map is then computed by linear interpolation of the sparse grid.

- **Input type:**     Graylevel image
- **Mask:**           Pipette mask computed in step 2
- **Spacing:**        Spacing of the GMM windows (local only)
- **Win. size**       Size of the GMM windows, as a ratio of the spacing (local only)
                      Window size of 0.5 results in non-overlapping but contiguous windows
                      Higher values result in overlapping windows
- **Components**      Number of components of the GMMs (local only)
                      Start with 2 components
                      Add up to 3 components if there is variation between the brightness of different object which results in missclassification of tissue
- **Min. std**        Criteria to discard windows that do not contain tissues in the threshold map (local only)
                      Increase if empty areas are noisy in the binary result of the thresholding
- **Processes**


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
