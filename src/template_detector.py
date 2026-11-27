import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

from .continuous import STEP
from .evaluate import best_threshold, metrics, prediction_table, save_json
from .prepare import load_partition
from .preprocessing import (
    BAND_MASK, CLIP_SAMPLES, FRAME_SAMPLES, FREQUENCIES, HOP_SAMPLES, SAMPLE_RATE,
    denoised_spectrogram, mono_float, resample, stft_power,
)


SPECTROGRAM_FREQUENCIES = FREQUENCIES[BAND_MASK]
FREQUENCY_STEP = SAMPLE_RATE / FRAME_SAMPLES
SEARCH_BAND = (SPECTROGRAM_FREQUENCIES >= 60) & (SPECTROGRAM_FREQUENCIES <= 240)
TEMPLATE_BINS = round(140 / FREQUENCY_STEP) + 1
TEMPLATE_FRAMES = round(SAMPLE_RATE / HOP_SAMPLES)
relative_frequency = np.arange(TEMPLATE_BINS) * FREQUENCY_STEP
ridge = np.linspace(20, 120, TEMPLATE_FRAMES)
template = np.exp(-0.5 * ((relative_frequency[:, None] - ridge) / 8) ** 2)
template -= template.mean()
TEMPLATE = (template / np.linalg.norm(template)).astype(np.float32)


def patch_sums(values, height, width):
    cumulative = np.pad(values, ((0, 0), (1, 0), (1, 0)))
    cumulative = cumulative.cumsum(axis=1, dtype=np.float64).cumsum(axis=2)
    return (cumulative[:, height:, width:] - cumulative[:, :-height, width:]
            - cumulative[:, height:, :-width] + cumulative[:, :-height, :-width])


def correlation_maps(spectrograms):
    """Return zero-mean, locally normalized correlation for each valid placement."""
    patch = spectrograms[:, SEARCH_BAND, :]
    height, width = TEMPLATE.shape
    product = signal.fftconvolve(patch, TEMPLATE[None, ::-1, ::-1], mode="valid", axes=(1, 2))
    total = patch_sums(patch, height, width)
    square = patch_sums(patch.astype(np.float64) ** 2, height, width)
    energy = np.maximum(square - total**2 / TEMPLATE.size, 0)
    return np.divide(product, np.sqrt(energy), out=np.zeros_like(product), where=energy > 1e-8)


def score_spectrograms(spectrograms, batch_size):
    return np.concatenate([
        np.clip(correlation_maps(spectrograms[start:start + batch_size]).max(axis=(1, 2)), 0, 1)
        for start in range(0, len(spectrograms), batch_size)
    ])


def evaluate_clips(prepared, output, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    val, clips = load_partition(prepared, "val", "spectrograms")
    scores = score_spectrograms(val, batch_size)
    threshold = best_threshold(clips["label"], scores)
    save_json(output / "frozen.json", {"threshold": threshold, "selected_on": "A+B val"})
    validation = prediction_table(clips, scores, 0, threshold)
    test, clips = load_partition(prepared, "test", "spectrograms")
    holdout = prediction_table(clips, score_spectrograms(test, batch_size), 0, threshold)
    results = []
    for split, table in (("val", validation), ("test", holdout)):
        table.to_csv(output / f"{split}_predictions.csv", index=False)
        results.append({"split": split, "dataset": "total", **metrics(table)})
        for dataset, group in table.groupby("dataset"):
            results.append({"split": split, "dataset": dataset, **metrics(group)})
    results = pd.DataFrame(results)
    results.to_csv(output / "metricas_clipes.csv", index=False)
    print(results[results["dataset"] == "total"].to_string(index=False), flush=True)


def evaluate_recordings(data_root, recordings, output, batch_size):
    audio_dir = data_root / "data" / "continuous" / "dataset_B" / "audio"
    step_samples = round(STEP * SAMPLE_RATE)
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "janelas.tmp.csv"
    with pending.open("w", encoding="utf-8", newline="") as stream:
        for index, record in enumerate(pd.read_csv(recordings).itertuples(index=False)):
            rate, raw = wavfile.read(audio_dir / record.filename, mmap=True)
            waveform = resample(mono_float(raw), rate)
            windows = np.lib.stride_tricks.sliding_window_view(waveform, CLIP_SAMPLES)[::step_samples]
            if len(windows) != record.windows:
                raise ValueError(f"Window grid differs from the reference in {record.filename}")
            scores = np.concatenate([
                score_spectrograms(denoised_spectrogram(stft_power(windows[start:start + batch_size])), batch_size)
                for start in range(0, len(windows), batch_size)
            ])
            pd.DataFrame({
                "filename": record.filename,
                "center_s": CLIP_SAMPLES / (2 * SAMPLE_RATE) + np.arange(len(scores)) * STEP,
                "score": scores,
            }).to_csv(stream, index=False, header=index == 0)
            print(f"Molde, gravação {index + 1:02d}: {record.filename}", flush=True)
    pending.replace(output / "janelas.csv")


def main():
    parser = argparse.ArgumentParser(description="Detector de referência por correlação 2D.")
    parser.add_argument("stage", choices=("clips", "continuous"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--data", type=Path, help="Pasta narw-data para o contínuo")
    parser.add_argument("--recordings", type=Path, default=Path("artifacts/continuo/por_gravacao.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/novos_testes/molde"))
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.stage == "clips":
        evaluate_clips(args.prepared, args.output, args.batch_size)
    else:
        if args.data is None:
            parser.error("--data é obrigatório para o estágio continuous")
        evaluate_recordings(args.data, args.recordings, args.output, args.batch_size)


if __name__ == "__main__":
    main()
