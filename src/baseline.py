import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluate import SEEDS, best_threshold, metrics, prediction_table, save_json
from .prepare import load_partition


def train(prepared, output, seeds):
    x_train, train_clips = load_partition(prepared, "train", "features")
    x_val, val_clips = load_partition(prepared, "val", "features")
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    manifest = []
    with threadpool_limits(limits=2, user_api="blas"):
        for seed in seeds:
            estimator = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=200,
                    early_stopping=True, n_iter_no_change=15, random_state=seed,
                ),
            )
            estimator.fit(x_train, train_clips["label"])
            scores = estimator.predict_proba(x_val)[:, 1]
            threshold = best_threshold(val_clips["label"], scores)
            table = prediction_table(val_clips, scores, seed, threshold)
            auc = float(metrics(table)["roc_auc"])
            checkpoint = f"seed_{seed}.joblib"
            joblib.dump(estimator, output / checkpoint)
            manifest.append({"seed": seed, "checkpoint": checkpoint, "threshold": threshold, "val_auc": auc})
            tables.append(table)
            print(f"baseline seed={seed}: validação ROC-AUC={auc:.4f}, limiar={threshold:.5f}", flush=True)
    pd.concat(tables, ignore_index=True).to_csv(output / "val_predictions.csv", index=False)
    save_json(output / "frozen.json", manifest)


def score_partition(prepared, output, split):
    values, clips = load_partition(prepared, split, "features")
    config = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
    tables = []
    for run in config:
        estimator = joblib.load(output / run["checkpoint"])
        scores = estimator.predict_proba(values)[:, 1]
        tables.append(prediction_table(clips, scores, run["seed"], run["threshold"]))
    pd.concat(tables, ignore_index=True).to_csv(output / f"{split}_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Treina ou aplica a MLP de referência.")
    parser.add_argument("stage", choices=("train", "val", "test"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if args.stage == "train":
        train(args.prepared, args.output, args.seeds)
    else:
        score_partition(args.prepared, args.output, args.stage)


if __name__ == "__main__":
    main()
