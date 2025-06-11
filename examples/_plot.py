import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plt_volume_dist(tissue_types: list, cut_size: int, working_dir: str):
    plot_df = pd.DataFrame(
        {"IoU": pd.Series(dtype="float"), "tissue": pd.Series(dtype="str")}
    )

    for tissue in tissue_types:
        metrics_path = (
            working_dir + f"/{tissue}{cut_size}/CuboidData/metrics.parquet"
        )

        metrics = pd.read_parquet(metrics_path)

        metrics["tissue"] = tissue
        metrics = metrics[["IoU", "tissue"]]

        plot_df = pd.concat([plot_df, metrics], ignore_index=True)

    plt.figure(figsize=(8, 7), dpi=300)

    ax = sns.ecdfplot(data=plot_df, x="IoU", hue="tissue")
    # plt.axvline(1, color='black', linestyle='--')

    legend = ax.get_legend()
    legend.get_title().set_text("")
    for text in legend.get_texts():
        text.set_fontsize(14)

    plt.xlabel("IoU", fontsize=14)
    plt.ylabel("cdf", fontsize=14)
    plt.title(f"Empirical CDF - cut size = {cut_size}\u03bcm", fontsize=16)
    plt.grid(True)
    plt.show()


def plt_metrics_correlation(filepath):
    df = pd.read_parquet(filepath)
    metrics = ["volume", "compactness", "convexity", "IoU", "inertia_ratio"]
    data = df[metrics]

    corr_matrix = data.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        yticklabels=metrics,
        annot_kws={"fontsize": 12},
    )
    plt.yticks(rotation=0, fontsize=14)
    plt.xticks(rotation=0, fontsize=14)
    plt.title("Metrics correlation matrix", fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    working_dir = "../Data"
    tissue_types = ["Liver", "Lung", "Kidney"]

    # plt_volume_dist(tissue_types, 200, working_dir)
    plt_metrics_correlation(
        working_dir + "/Liver500/CuboidData/metrics.parquet"
    )
