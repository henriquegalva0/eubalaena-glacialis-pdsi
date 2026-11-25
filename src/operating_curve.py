import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .continuous import MATCH_TOLERANCE, STEP, group_events, match_events
from .preprocessing import CLIP_SAMPLES, SAMPLE_RATE


matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = {
    "cnn_original": ("CNN original", False, 1),
    "cnn_media_5": ("CNN · média de 5 scores", True, 1),
    "cnn_duas_janelas": ("CNN · ≥2 janelas", False, 2),
    "molde_2d": ("Molde 2D", False, 1),
}
BUDGETS = (5, 10, 25, 50, 100, 200)


def moving_average(scores):
    """Five adjacent 0.5-s scores, with symmetric alignment and edge replication."""
    return np.convolve(np.pad(scores, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")


def aggregate_at_threshold(score_arrays, annotations, threshold, minimum_windows, hours):
    counts = {"tp": 0, "fp": 0, "fn": 0, "ignored_uncertain": 0}
    errors = []
    for filename, scores in score_arrays.items():
        found = group_events(scores, threshold)
        if minimum_windows > 1:
            found = [event for event in found if event["end_s"] - event["start_s"] >= minimum_windows * STEP]
        truth = annotations.get(filename)
        matches = match_events(found, truth.loc[~truth["uncertain"]])
        uncertain = truth.loc[truth["uncertain"], "timestamp"].to_numpy()
        for event, (timestamp, _) in zip(found, matches):
            if timestamp is not None:
                errors.append(abs(event["peak_s"] - timestamp))
            elif len(uncertain) and np.any(
                (uncertain >= event["start_s"] - MATCH_TOLERANCE)
                & (uncertain <= event["end_s"] + MATCH_TOLERANCE)
            ):
                counts["ignored_uncertain"] += 1
            else:
                counts["fp"] += 1
        counts["fn"] += int((~truth["uncertain"]).sum()) - sum(timestamp is not None for timestamp, _ in matches)
    counts["tp"] = len(errors)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "false_alarms_per_hour": fp / hours,
        "timestamp_abs_median_s": float(np.median(errors)) if errors else np.nan,
        "timestamp_abs_p95_s": float(np.percentile(errors, 95)) if errors else np.nan,
        "timestamps_over_1s": sum(error > 1 for error in errors),
    }


def load_window_scores(path, recordings):
    table = pd.read_csv(path)
    scores = {}
    for recording in recordings.itertuples(index=False):
        selected = table.loc[table["filename"] == recording.filename].sort_values("center_s")
        centers = CLIP_SAMPLES / (2 * SAMPLE_RATE) + np.arange(recording.windows) * STEP
        if len(selected) != len(centers) or not np.allclose(selected["center_s"], centers):
            raise ValueError(f"Invalid window grid in {recording.filename}: {path}")
        scores[recording.filename] = selected["score"].to_numpy(dtype=np.float32)
    return scores


def build_curve(scores_by_model, truth, hours, reference_threshold, template_threshold):
    rows = []
    for variant, (_, smooth, consecutive) in VARIANTS.items():
        model = "molde" if variant == "molde_2d" else "cnn"
        if model not in scores_by_model:
            continue
        values = scores_by_model[model]
        if smooth:
            values = {filename: moving_average(scores) for filename, scores in values.items()}
        reference = template_threshold if model == "molde" else reference_threshold
        thresholds = sorted(set(np.r_[np.arange(0.01, 1, 0.01), 1.01, reference]))
        for threshold in thresholds:
            rows.append({
                "variant": variant, "threshold": threshold, "clip_threshold": threshold == reference,
                **aggregate_at_threshold(values, truth, threshold, consecutive, hours),
            })
        print(f"Curva calculada: {variant}", flush=True)
    return pd.DataFrame(rows)


def operating_points(curve):
    selected = []
    for variant, group in curve.groupby("variant", sort=False):
        for budget in BUDGETS:
            candidates = group.loc[group["false_alarms_per_hour"] <= budget]
            best = candidates.sort_values(["recall", "false_alarms_per_hour", "threshold"],
                                          ascending=[False, True, False]).iloc[0]
            selected.append({"budget_fp_h": budget, **best.to_dict()})
    return pd.DataFrame(selected)


def plot_curve(curve, output, title):
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, group in curve.groupby("variant", sort=False):
        by_budget = group.groupby("false_alarms_per_hour")["recall"].max().sort_index().cummax()
        axis.step(by_budget.index, by_budget, where="post", linewidth=1.5, label=VARIANTS[name][0])
    axis.set(title=title, xlabel="Orçamento de falsos alarmes por hora", ylabel="Melhor recall observado")
    axis.set_xscale("symlog", linthresh=1)
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize="small")
    figure.savefig(output, dpi=175)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Compara regras no contínuo com scores já calculados.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, default=Path("artifacts/continuo/por_gravacao.csv"))
    parser.add_argument("--cnn-scores", type=Path, default=Path("artifacts/continuo/janelas.csv"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/continuo/resumo.json"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/novos_testes/continuo_exploratorio"))
    args = parser.parse_args()
    recordings = pd.read_csv(args.recordings)
    annotations = pd.read_csv(args.annotations, sep=";")
    if "uncertain" not in annotations:
        annotations["uncertain"] = False
    truth = {
        filename: annotations.loc[annotations["filename"] == filename]
        for filename in recordings["filename"]
    }
    scores = {"cnn": load_window_scores(args.cnn_scores, recordings)}
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["threshold"]
    template_threshold = None
    if args.template:
        scores["molde"] = load_window_scores(args.template / "janelas.csv", recordings)
        template_threshold = json.loads((args.template / "frozen.json").read_text(encoding="utf-8"))["threshold"]
    hours = recordings["duration_s"].sum() / 3600
    curve = build_curve(scores, truth, hours, reference, template_threshold)
    points = operating_points(curve)
    args.output.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output / "curva.csv", index=False)
    points.to_csv(args.output / "pontos_por_orcamento.csv", index=False)
    curve[curve["clip_threshold"]].to_csv(args.output / "limiares_originais.csv", index=False)
    plot_curve(curve, args.output / "recall_por_falsos_alarmes.png", "Recall por orçamento de falsos alarmes")
    print(points.loc[points["budget_fp_h"].isin([25, 50, 100]),
                     ["variant", "budget_fp_h", "false_alarms_per_hour", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
