import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


SEEDS = (7, 23, 42)
METRICS = ("roc_auc", "average_precision", "precision", "recall", "f1")


def best_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, np.finfo(float).eps)
    # Thresholds are in ascending order. Keep the lower threshold when F1 ties.
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))[0]])


def prediction_table(clips, scores, seed, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(clips),) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per clip")
    return clips.assign(seed=seed, score=scores, threshold=threshold,
                        predicted_label=(scores >= threshold).astype(np.int8))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def metrics(table):
    y = table["label"].to_numpy()
    scores = table["score"].to_numpy()
    predicted = table["predicted_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    has_both_classes = 0 < y.sum() < len(y)
    return {
        "clips": len(y), "positives": int(y.sum()),
        "roc_auc": roc_auc_score(y, scores) if has_both_classes else np.nan,
        "average_precision": average_precision_score(y, scores) if has_both_classes else np.nan,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0) if y.sum() else np.nan,
        "f1": f1_score(y, predicted, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def export_comparison(predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    snr_rows = []
    source_rows = []
    n_splits = predictions["split"].nunique()
    figure, axes = plt.subplots(n_splits, 2, squeeze=False, figsize=(10, 4 * n_splits))
    for axis, (split, split_table) in zip(axes, predictions.groupby("split", sort=False)):
        for (model, seed), table in split_table.groupby(["model", "seed"]):
            identity = {"model": model, "split": split, "seed": seed}
            label = f"{model} · {seed}"
            rows.append({**identity, "dataset": "total", **metrics(table)})
            for dataset, group in table.groupby("dataset"):
                rows.append({**identity, "dataset": dataset, **metrics(group)})

            snr_band = pd.cut(table["snr"], bins=[-np.inf, 0, 5, 10, 15, np.inf], right=False)
            for band, group in table.groupby(snr_band, observed=True):
                snr_rows.append({**identity, "snr_db": str(band), **metrics(group)})

            if "source_filename" in table:
                for source, group in table.dropna(subset=["source_filename"]).groupby("source_filename"):
                    source_rows.append({**identity, "source_filename": source, **metrics(group)})

            fpr, tpr, _ = roc_curve(table["label"], table["score"])
            precision, recall, _ = precision_recall_curve(table["label"], table["score"])
            axis[0].plot(fpr, tpr, label=label)
            axis[1].plot(recall, precision, label=label)
        axis[0].plot([0, 1], [0, 1], ":", color="0.5")
        axis[0].set(title=f"{split} · ROC", xlabel="Taxa de falsos positivos", ylabel="Recall")
        axis[1].set(title=f"{split} · Precisão–recall", xlabel="Recall", ylabel="Precisão")
        for panel in axis:
            panel.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output / "curvas_roc_pr.png", dpi=160)
    plt.close(figure)

    individual = pd.DataFrame(rows)
    individual.to_csv(output / "metricas_por_seed.csv", index=False)
    summary = individual.groupby(["split", "dataset", "model"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "comparacao.csv", float_format="%.6f")
    pd.DataFrame(snr_rows).to_csv(output / "metricas_por_snr.csv", index=False)
    if source_rows:
        pd.DataFrame(source_rows).to_csv(output / "metricas_por_origem_c.csv", index=False)
    print(summary.loc[(slice(None), "total", slice(None)), :].round(4).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compara predições da baseline e da CNN.")
    parser.add_argument("--runs", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/avaliacao"))
    parser.add_argument("--split", choices=("val", "test"), nargs="+", default=["val"])
    args = parser.parse_args()
    tables = [
        pd.read_csv(args.runs / model / f"{split}_predictions.csv", float_precision="round_trip")
        .assign(model=model, split=split)
        for split in args.split
        for model in ("baseline", "cnn")
    ]
    export_comparison(pd.concat(tables, ignore_index=True), args.output)


if __name__ == "__main__":
    main()
