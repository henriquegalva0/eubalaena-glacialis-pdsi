import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from .cnn import load_checkpoint, predict
from .evaluate import save_json
from .preprocessing import CLIP_SAMPLES, SAMPLE_RATE, denoised_spectrogram, mono_float, resample, stft_power


STEP = 0.5
MATCH_TOLERANCE = 1.0
EVENT_COLUMNS = ("start_s", "end_s", "peak_s", "score")


def window_scores(model, audio, batch_size):
    step_samples = int(STEP * SAMPLE_RATE)
    if len(audio) < CLIP_SAMPLES:
        raise ValueError("A continuous recording must contain at least one full 3-s window")
    windows = np.lib.stride_tricks.sliding_window_view(audio, CLIP_SAMPLES)[::step_samples]
    return np.concatenate([
        predict(model, denoised_spectrogram(stft_power(windows[start:start + batch_size])), batch_size)
        for start in range(0, len(windows), batch_size)
    ])


def group_events(scores, threshold):
    hits = np.flatnonzero(scores >= threshold)
    if len(hits) == 0:
        return []
    runs = np.split(hits, np.flatnonzero(np.diff(hits) != 1) + 1)
    center_offset = CLIP_SAMPLES / (2 * SAMPLE_RATE)
    return [
        {
            "start_s": float(center_offset + run[0] * STEP - STEP / 2),
            "end_s": float(center_offset + run[-1] * STEP + STEP / 2),
            "peak_s": float(center_offset + run[np.argmax(scores[run])] * STEP),
            "score": float(scores[run].max()),
        }
        for run in runs
    ]


def match_events(events, annotations, tolerance=MATCH_TOLERANCE):
    """Match each grouped event to at most one annotated center in temporal order."""
    annotated = annotations.sort_values("timestamp").reset_index(drop=True)
    timestamps = annotated["timestamp"].to_numpy()
    matches = []
    remaining = 0
    for event in events:
        start, end = event["start_s"] - tolerance, event["end_s"] + tolerance
        remaining = max(remaining, int(np.searchsorted(timestamps, start, side="left")))
        if remaining < len(timestamps) and timestamps[remaining] <= end:
            matches.append((float(timestamps[remaining]), float(annotated.at[remaining, "snr"])))
            remaining += 1
        else:
            matches.append((None, None))
    return matches


def evaluate_continuous(data_root, cnn, output, batch_size):
    folder = data_root / "data" / "continuous" / "dataset_B"
    annotations = pd.read_csv(folder / "annotations_B_cont.csv", sep=";")
    files = sorted((folder / "audio").glob("*.wav"))
    missing = set(annotations["filename"]) - {path.name for path in files}
    if len(files) != 50 or missing:
        raise ValueError(f"Expected 50 official WAVs, found {len(files)}; missing annotations' WAVs: {sorted(missing)}")

    runs = json.loads((cnn / "frozen.json").read_text(encoding="utf-8"))
    selected = max(runs, key=lambda run: (run["val_auc"], -run["seed"]))
    model = load_checkpoint(cnn / selected["checkpoint"])
    threshold = selected["threshold"]
    events, recordings, windows = [], [], []
    for index, path in enumerate(files, start=1):
        rate, raw = wavfile.read(path, mmap=True)
        duration = len(raw) / rate
        audio = resample(mono_float(raw), rate)
        scores = window_scores(model, audio, batch_size)
        detected = group_events(scores, threshold)
        truth = annotations.loc[annotations["filename"] == path.name]
        matches = match_events(detected, truth)
        tp = sum(timestamp is not None for timestamp, _ in matches)
        for event, (timestamp, snr) in zip(detected, matches):
            events.append({"filename": path.name, **event, "annotation_s": timestamp, "snr_db": snr})
        windows.append(pd.DataFrame({
            "filename": path.name,
            "center_s": CLIP_SAMPLES / (2 * SAMPLE_RATE) + np.arange(len(scores)) * STEP,
            "score": scores,
        }))
        recordings.append({
            "filename": path.name, "original_hz": rate, "duration_s": duration, "windows": len(scores),
            "annotations": len(truth), "detected_events": len(detected), "tp": tp,
            "fp": len(detected) - tp, "fn": len(truth) - tp,
        })
        print(f"{index:02d}/50 {path.name}: {tp}/{len(truth)} chamadas, {len(detected) - tp} falsos alarmes",
              flush=True)

    by_file = pd.DataFrame(recordings)
    hours = float(by_file["duration_s"].sum() / 3600)
    tp, fp, fn = (int(by_file[column].sum()) for column in ("tp", "fp", "fn"))
    summary = {
        "seed": selected["seed"], "threshold": threshold, "window_s": CLIP_SAMPLES / SAMPLE_RATE,
        "step_s": STEP, "match_tolerance_s": MATCH_TOLERANCE, "recordings": len(files), "hours": hours,
        "annotations": len(annotations), "windows": int(by_file["windows"].sum()),
        "tp": tp, "fp": fp, "fn": fn, "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
        "false_alarms_per_hour": fp / hours,
    }
    output.mkdir(parents=True, exist_ok=True)
    by_file.to_csv(output / "por_gravacao.csv", index=False)
    columns = ("filename", *EVENT_COLUMNS, "annotation_s", "snr_db")
    pd.DataFrame(events, columns=columns).to_csv(output / "eventos.csv", index=False)
    pd.concat(windows, ignore_index=True).to_csv(output / "janelas.csv", index=False)
    save_json(output / "resumo.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Detecta upcalls nos 50 WAVs contínuos oficiais.")
    parser.add_argument("--data", type=Path, required=True, help="Pasta narw-data")
    parser.add_argument("--cnn", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/continuo"))
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate_continuous(args.data, args.cnn, args.output, args.batch_size)


if __name__ == "__main__":
    main()
