import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff

from napari_tissue_cuboid_analysis._utils import (
    apply_threshold,
    bin_closing,
    bin_opening,
    generate_multiple_cuboids_simple,
    watershed_auto_fix,
)


def compute_metrics(binned, thresh_map, mask, offsets, dir_path, morph=True):
    for offset in offsets:
        print(f"\nOffset {offset}\n\n")
        results_path = dir_path + f"/offset{offset}"
        if morph:
            results_path += "_morph"

        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        binary = apply_threshold(binned, thresh_map + offset, mask)
        # binary = bin_closing(binary, 3)
        binary = bin_opening(binary, 3)

        tiff.imwrite(results_path + "/Binary.tif", binary)
        *_, labels = watershed_auto_fix(
            binary, watershed_lvl=3, overseg_threshold=0.3
        )
        if morph:
            labels = bin_closing(labels, d=11)

        tiff.imwrite(results_path + "/Labels.tif", labels)

        generate_multiple_cuboids_simple(
            labelled=labels,
            vsize=6,
            smooth_iter=20,
            dir_path=results_path,
            metrics_only=True,
        )


def plot_analysis(dir_path, offsets, cut_size):
    vol_dist_df = pd.DataFrame(
        dtype=object,
        index=np.arange(0, 2 * len(offsets)),
        columns=["offset", "volume", "morph"],
    )

    for i in range(2 * len(offsets)):
        if i % 2 == 0:
            path = dir_path + f"/offset{offsets[i//2]}/metrics.parquet"
            morph = False
        else:
            path = dir_path + f"/offset{offsets[i//2]}_morph/metrics.parquet"
            morph = True

        metrics_df = pd.read_parquet(path)
        volumes = metrics_df["volume"]
        vol_dist_df.loc[i] = [offsets[i // 2], volumes, morph]

    plot_df = vol_dist_df.explode("volume").astype({"volume": float})

    # plot_df['volume'] /= (cut_size*1e-3)**3 #normalize with theoretical volume in mm³
    plot_df["offset"] /= 300
    plt.figure(dpi=300)
    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(
        data=plot_df,
        x="offset",
        y="volume",
        split=True,
        hue="morph",
        inner="quart",
        cut=0,
        density_norm="width",
    )

    ax.set(xlabel="Offset [%]", ylabel="Volume [mm²]")
    plt.axhline((cut_size * 1e-3) ** 3, linestyle="--", color="#55AB68")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tissue = "Lung"
    cut_size = 500
    working_dir = f"../Data/{tissue}{cut_size}/SensitivityAnalysis"

    binned_filepath = working_dir + "/Binned.tif"
    mask_filepath = working_dir + "/Mask.tif"
    thresh_map_filepath = working_dir + "/ThreshMap.tif"

    binned = tiff.imread(binned_filepath)
    mask = tiff.imread(mask_filepath)
    thresh_map = tiff.imread(thresh_map_filepath)
    # thresh_map = 15033 #for global thresholding

    offsets = [-1200, -600, 0, 600, 1200]
    compute_metrics(
        binned, thresh_map, mask, offsets, working_dir, morph=False
    )
    compute_metrics(binned, thresh_map, mask, offsets, working_dir, morph=True)
    plot_analysis(working_dir, offsets, cut_size)
