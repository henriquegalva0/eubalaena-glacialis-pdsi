import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .prepare import load_partition
from .preprocessing import BAND_MASK, FRAME_SAMPLES, FREQUENCIES, HOP_SAMPLES, SAMPLE_RATE, SPECTROGRAM_SHAPE


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BAND_FREQUENCIES = FREQUENCIES[BAND_MASK]
FIRST_FRAME_EDGE = (FRAME_SAMPLES - HOP_SAMPLES) / (2 * SAMPLE_RATE)
HALF_BIN = (FREQUENCIES[1] - FREQUENCIES[0]) / 2
SPECTROGRAM_EXTENT = (
    FIRST_FRAME_EDGE, FIRST_FRAME_EDGE + SPECTROGRAM_SHAPE[1] * HOP_SAMPLES / SAMPLE_RATE,
    BAND_FREQUENCIES[0] - HALF_BIN, BAND_FREQUENCIES[-1] + HALF_BIN,
)


def summarize(prepared, output, splits):
    output.mkdir(parents=True, exist_ok=True)
    tables, means = [], []
    for split in splits:
        spectra, clips = load_partition(prepared, split, "spectrograms")
        tables.append(clips.assign(split=split))
        for (dataset, label), indices in clips.groupby(["dataset", "label"]).groups.items():
            means.append((split, dataset, label, spectra[indices].mean(axis=0)))
    clips = pd.concat(tables, ignore_index=True)
    stats = clips.groupby(["split", "dataset", "label"])["snr"].agg(
        clipes="size", snr_disponivel="count", snr_mediana_db="median", snr_min_db="min",
        snr_max_db="max",
    )
    stats.to_csv(output / "contagens_snr.csv", float_format="%.3f")

    datasets = list(clips[["split", "dataset"]].drop_duplicates().itertuples(index=False, name=None))
    hist, hist_axes = plt.subplots(len(datasets), 1, figsize=(7, 2.5 * len(datasets)), squeeze=False)
    grid, grid_axes = plt.subplots(
        len(datasets), 2, figsize=(9, 2.5 * len(datasets)), squeeze=False,
        sharex=True, sharey=True, layout="constrained",
    )
    limit = max(1.0, max(float(np.percentile(np.abs(matrix), 99)) for _, _, _, matrix in means))
    plot_matrix = {(split, dataset, label): matrix for split, dataset, label, matrix in means}
    for index, (split, dataset) in enumerate(datasets):
        subset = clips.loc[(clips["split"] == split) & (clips["dataset"] == dataset)]
        bins = np.histogram_bin_edges(subset["snr"].dropna(), bins=24)
        axis = hist_axes[index, 0]
        for label, name in ((0, "Sem upcall"), (1, "Com upcall")):
            snr = subset.loc[subset["label"] == label, "snr"].dropna()
            axis.hist(snr, bins=bins, alpha=0.65, label=f"{name} · n={len(snr)}")
            panel = grid_axes[index, label]
            image = panel.imshow(plot_matrix[(split, dataset, label)], extent=SPECTROGRAM_EXTENT,
                                 aspect="auto", origin="lower", interpolation="nearest",
                                 cmap="RdBu_r", vmin=-limit, vmax=limit)
            panel.set_title(f"{dataset} · {split} · {name}")
        axis.set(title=f"{dataset} · {split}", xlabel="SNR (dB)", ylabel="Clipes")
        axis.legend(fontsize="small")
    hist.tight_layout()
    grid.supxlabel("Tempo (s)")
    grid.supylabel("Frequência (Hz)")
    grid.colorbar(image, ax=list(grid_axes.flat), label="dB após a subtração da mediana")
    hist.savefig(output / "distribuicao_snr.png", dpi=155, bbox_inches="tight")
    grid.savefig(output / "espectrogramas_medios.png", dpi=155, bbox_inches="tight")
    plt.close(hist)
    plt.close(grid)


def show_errors(prepared, cnn, output, split):
    output.mkdir(parents=True, exist_ok=True)
    runs = json.loads((cnn / "frozen.json").read_text(encoding="utf-8"))
    selected = max(runs, key=lambda run: (run["val_auc"], -run["seed"]))
    spectra, clips = load_partition(prepared, split, "spectrograms")
    indices = pd.Series(np.arange(len(clips)), index=clips["clip_id"])
    scores = pd.read_csv(cnn / f"{split}_predictions.csv")
    scores = scores.loc[scores["seed"] == selected["seed"]].copy()
    positive = scores["predicted_label"] == 1
    false_positives = scores.loc[positive & (scores["label"] == 0)].nlargest(20, "score")
    false_negatives = scores.loc[~positive & (scores["label"] == 1)].nsmallest(20, "score")
    for name, table in (("falsos_positivos", false_positives), ("falsos_negativos", false_negatives)):
        table.to_csv(output / f"{name}.csv", index=False)
        if table.empty:
            continue
        figure, axes = plt.subplots(int(np.ceil(len(table) / 5)), 5, figsize=(13, 8), squeeze=False)
        for axis in axes.flat:
            axis.set_visible(False)
        for axis, clip in zip(axes.flat, table.itertuples(index=False)):
            axis.set_visible(True)
            axis.imshow(spectra[indices[clip.clip_id]], extent=SPECTROGRAM_EXTENT,
                        aspect="auto", origin="lower", cmap="magma", vmin=-6, vmax=14)
            axis.set_title(f"{clip.clip_name} · score {clip.score:.2f} · {clip.snr:.1f} dB", fontsize=8)
            axis.set(xticks=(0, 1, 2), yticks=(50, 200, 350))
            axis.tick_params(labelsize=7)
        figure.supxlabel("Tempo (s)")
        figure.supylabel("Frequência (Hz)")
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Gera figuras exploratórias e painéis de erros.")
    parser.add_argument("stage", choices=("eda", "errors"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--cnn", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnostico"))
    parser.add_argument("--split", choices=("train", "val", "test"), nargs="+")
    args = parser.parse_args()
    if args.stage == "eda":
        summarize(args.prepared, args.output, args.split or ("train", "val"))
    else:
        for split in args.split or ("test",):
            show_errors(args.prepared, args.cnn, args.output / split, split)


if __name__ == "__main__":
    main()
