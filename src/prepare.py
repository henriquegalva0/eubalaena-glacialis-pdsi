import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from .preprocessing import (
    CLIP_SAMPLES, SAMPLE_RATE, SPECTROGRAM_SHAPE, baseline_features,
    denoised_spectrogram, mono_float, stft_power,
)


PARTITIONS = {
    "train": (("A", "train"), ("B", "train")),
    "val": (("A", "val"), ("B", "val")),
    "test": (("C", "train"),),
}


def read_annotations(data_root, subset, part):
    folder = data_root / "data" / "clips" / f"dataset_{subset}" / part
    annotations = pd.read_csv(folder / f"annotations_{subset}_{part}.csv", sep=";")
    if annotations["clip_name"].duplicated().any() or not annotations["label"].isin([0, 1]).all():
        raise ValueError(f"Invalid clips or labels in {folder}")
    annotations["dataset"] = subset
    annotations["partition"] = part
    annotations["clip_id"] = subset + "/" + part + "/" + annotations["clip_name"]
    annotations["audio_path"] = annotations["clip_name"].map(lambda name: folder / "audio" / name)
    return annotations


def read_clip(path):
    rate, audio = wavfile.read(path)
    if rate != SAMPLE_RATE or len(audio) not in (CLIP_SAMPLES, CLIP_SAMPLES + 1):
        raise ValueError(f"{path}: expected a 3-s clip at 1 kHz, got {len(audio)} samples at {rate} Hz")
    # Some official clips include the endpoint at exactly 3.000 s.
    return mono_float(audio[:CLIP_SAMPLES])


def prepare(data_root, output, split, limit=None, *, bandpass=True, remove_median=True):
    frames = [read_annotations(data_root, subset, part) for subset, part in PARTITIONS[split]]
    if limit is not None:
        frames = [frame.groupby("label", group_keys=False).head(limit) for frame in frames]
    clips = pd.concat(frames, ignore_index=True)
    if not len(clips):
        raise ValueError("No annotated clips found")

    specs = np.empty((len(clips), *SPECTROGRAM_SHAPE), dtype=np.float32)
    features = np.empty((len(clips), 129), dtype=np.float32)
    for start in range(0, len(clips), 64):
        end = min(start + 64, len(clips))
        windows = np.stack([read_clip(path) for path in clips["audio_path"].iloc[start:end]])
        power = stft_power(windows, bandpass=bandpass)
        specs[start:end] = denoised_spectrogram(power, remove_median=remove_median)
        features[start:end] = baseline_features(power, specs[start:end])
    if not np.isfinite(specs).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite model inputs")

    folder = output / split
    folder.mkdir(parents=True, exist_ok=True)
    for name, array in (("spectrograms", specs), ("features", features)):
        pending = folder / f"{name}.tmp.npy"
        np.save(pending, array, allow_pickle=False)
        pending.replace(folder / f"{name}.npy")
    table = folder / "clips.tmp.csv"
    clips.drop(columns="audio_path").to_csv(table, index=False)
    table.replace(folder / "clips.csv")
    print(f"{split}: {len(clips)} clipes, {clips['label'].sum()} upcalls, {specs.shape}", flush=True)


def load_partition(prepared, split, kind):
    folder = prepared / split
    clips = pd.read_csv(folder / "clips.csv")
    values = np.load(folder / f"{kind}.npy", allow_pickle=False)
    if len(clips) != len(values):
        raise ValueError(f"Mismatched arrays and annotations in {folder}")
    return values, clips


def main():
    parser = argparse.ArgumentParser(description="Prepara clipes do dataset oficial.")
    parser.add_argument("--data", type=Path, required=True, help="Pasta narw-data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--split", choices=PARTITIONS, nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, help="Clipes por classe e conjunto para teste rápido")
    ablation = parser.add_mutually_exclusive_group()
    ablation.add_argument("--without-bandpass", action="store_true", help="Omite apenas o filtro Butterworth")
    ablation.add_argument("--without-median", action="store_true", help="Omite apenas a subtração da mediana")
    args = parser.parse_args()
    for split in args.split:
        prepare(args.data, args.output, split, args.limit,
                bandpass=not args.without_bandpass, remove_median=not args.without_median)


if __name__ == "__main__":
    main()
