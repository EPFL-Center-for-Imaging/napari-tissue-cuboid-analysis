import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tifffile as tiff

from napari_tissue_cuboid_analysis._utils import (
    generate_multiple_cuboids_simple,
)


def compute_metrics(smooth_iters, dir_path):
    for iterations in smooth_iters:
        print(f"\nSmoothing iterations: {iterations}\n\n")
        results_path = dir_path + f"/iter{iterations}"

        if not os.path.isdir(results_path):
            os.mkdir(results_path)

        labels = tiff.imread(dir_path + "/Labels.tif")

        generate_multiple_cuboids_simple(
            labelled=labels,
            vsize=6,
            smooth_iter=iterations,
            dir_path=results_path,
            metrics_only=True,
        )


def plot_analysis(dir_path, smooth_iters):
    metrics = ["volume", "compactness", "convexity", "IoU", "inertia_ratio"]
    frames = []

    for n_iter in smooth_iters:
        df = (
            pd.read_parquet(
                f"{dir_path}/iter{n_iter}/metrics.parquet", columns=metrics
            )
            .reset_index()
            .rename(columns={"index": "id"})
        )
        df["iters"] = n_iter
        frames.append(df)

    full_raw = pd.concat(frames, ignore_index=True)
    full_raw = full_raw.set_index(["iters", "id"])

    base = full_raw.xs(0, level="iters")  # the “slice” where iters == 0
    percent = 100 * (full_raw.sub(base) / base)

    mean_percent = percent.groupby("iters").mean().reset_index()

    long_df = mean_percent.melt(
        "iters", var_name="metric", value_name="percent_change"
    )

    long_df["iters"] = long_df["iters"].astype("str")

    sns.set_theme(style="whitegrid")
    ax = sns.swarmplot(
        data=long_df,
        x="iters",
        y="percent_change",
        hue="metric",
        dodge=0.1,
        palette="tab10",
        size=6,
    )

    ax.set(
        xlabel="Smoothing iterations",
        ylabel="% change vs iter 0",
        title="Sensitivity of metrics to smoothing iterations",
    )

    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tissue = "Lung"
    cut_size = 200
    working_dir = f"../Data/{tissue}{cut_size}/SmoothSensitivityAnalysis"

    smooth_iters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    compute_metrics(smooth_iters, working_dir)
    plot_analysis(working_dir, smooth_iters)
