import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import fft, ndimage, signal

from .continuous import STEP, group_events
from .diagnostics import SPECTROGRAM_EXTENT
from .evaluate import save_json
from .operating_curve import moving_average
from .prepare import load_partition
from .preprocessing import (
    BAND_MASK, CLIP_SAMPLES, FRAME_SAMPLES, FREQUENCIES, HOP_SAMPLES, SAMPLE_RATE, SPECTROGRAM_SHAPE, WINDOW,
    denoised_spectrogram, resample, stft_power,
)
from .template_detector import SEARCH_BAND, TEMPLATE, correlation_maps


matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPEC_FREQUENCIES = FREQUENCIES[BAND_MASK]
BIN_HZ = SAMPLE_RATE / FRAME_SAMPLES
FRAME_HOP_S = HOP_SAMPLES / SAMPLE_RATE
FRAME_CENTERS = FRAME_SAMPLES / (2 * SAMPLE_RATE) + np.arange(SPECTROGRAM_SHAPE[1]) * FRAME_HOP_S


def tones_after_resampling():
    input_rate = 32768
    original_time = np.arange(4 * input_rate) / input_rate
    original = (np.sin(2 * np.pi * 150 * original_time) + np.sin(2 * np.pi * 700 * original_time)).astype(np.float32)
    polyphase = resample(original, input_rate)
    times = np.arange(len(polyphase)) / SAMPLE_RATE
    # This interpolation shows the alias when no low-pass is used before sampling.
    naive = np.interp(times, original_time, original)
    return polyphase[500:3500], naive[500:3500]


def make_signal(upcall=True):
    """The 1–2 s interval contains a 80–200 Hz sweep or a 140 Hz control tone."""
    rng = np.random.default_rng(2026)
    audio = rng.normal(0, 0.06, size=CLIP_SAMPLES).astype(np.float32)
    local_time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    frequency = (80, 200) if upcall else (140, 140)
    tone = signal.chirp(local_time, f0=frequency[0], t1=1, f1=frequency[1])
    audio[SAMPLE_RATE:2 * SAMPLE_RATE] += tone * signal.windows.tukey(SAMPLE_RATE, alpha=0.25)
    return audio


def sampling_demo(output):
    polyphase, naive = tones_after_resampling()
    f = fft.rfftfreq(len(polyphase), 1 / SAMPLE_RATE)
    figure, axis = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    for name, waveform in (("Interpolação sem passa-baixa", naive), ("Reamostragem polifásica", polyphase)):
        magnitude = 2 * np.abs(fft.rfft(waveform)) / len(waveform)
        axis.plot(f, 20 * np.log10(np.maximum(magnitude, 1e-5)), label=name)
    axis.set(title="Antialiasing de 32.768 Hz para 1.000 Hz", xlabel="Frequência (Hz)",
             ylabel="Magnitude (dB re 1)", xlim=(0, 500), ylim=(-100, 6))
    axis.annotate("Tom preservado\n150 Hz", (150, 0), xytext=(57, -28), arrowprops={"arrowstyle": "->"})
    axis.annotate("Alias de 700 Hz\nem 300 Hz", (300, 0), xytext=(335, -24), arrowprops={"arrowstyle": "->"})
    axis.legend(loc="lower left", fontsize="small")
    figure.savefig(output / "le02_antialiasing.png", dpi=175)
    plt.close(figure)
    magnitude = 2 * np.abs(fft.rfft(polyphase)) / len(polyphase)
    wanted, alias = magnitude[[int(150 * len(polyphase) / SAMPLE_RATE), int(300 * len(polyphase) / SAMPLE_RATE)]]
    return {"amplitude_150_hz": float(wanted), "alias_300_relative_db": float(20 * np.log10(alias / wanted))}


def convolution_demo(prepared, original, output):
    scores = pd.read_csv(original / "janelas.csv")
    fname = "B_cont_13.wav"
    scores = scores[scores["filename"] == fname]
    record = pd.read_csv(original / "por_gravacao.csv").set_index("filename").loc[fname]
    if record.annotations != 0:
        raise ValueError("The LE03 example requires a recording with no annotated upcalls")
    values = scores["score"].to_numpy()
    centers = scores["center_s"].to_numpy()
    threshold = json.loads((original / "resumo.json").read_text(encoding="utf-8"))["threshold"]
    singles = [event for event in group_events(values, threshold) if event["end_s"] - event["start_s"] == STEP]
    smooth = moving_average(values)
    suppressed = [event for event in singles if np.all(smooth[np.abs(centers - event["peak_s"]) <= 2] < threshold)]
    event = max(suppressed, key=lambda item: item["score"])
    idx = np.flatnonzero(centers == event["peak_s"])[0]
    near = np.abs(centers - event["peak_s"]) <= 6
    figure, axis = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    axis.plot(centers[near], values[near], marker=".", label="Score da CNN")
    axis.plot(centers[near], smooth[near], marker=".", label="Média de cinco scores")
    axis.axhline(threshold, color="0.4", linestyle="--", label="Limiar congelado")
    axis.set(title=f"Convolução 1D sobre os scores · {fname}", xlabel="Tempo (s)", ylabel="Score")
    axis.legend(fontsize="small")
    figure.savefig(output / "le03_media_movel.png", dpi=175)
    plt.close(figure)

    specs, clips = load_partition(prepared, "val", "spectrograms")
    index = clips.loc[(clips["dataset"] == "A") & (clips["label"] == 1), "snr"].idxmax()
    spec = specs[index]
    corr = correlation_maps(spec[None])[0]
    row, col = np.unravel_index(corr.argmax(), corr.shape)
    figure, axes = plt.subplots(1, 3, figsize=(11.7, 3.6), width_ratios=(1, 1.15, 1.15), layout="constrained")
    axes[0].imshow(TEMPLATE, origin="lower", aspect="auto", cmap="coolwarm",
                   extent=(0, FRAME_HOP_S * TEMPLATE.shape[1], 0, BIN_HZ * (TEMPLATE.shape[0] - 1)))
    axes[0].set(title="Molde de rampa", xlabel="Tempo (s)", ylabel="Frequência relativa (Hz)")
    axes[1].imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=SPECTROGRAM_EXTENT)
    axes[1].set(title=clips.at[index, "clip_name"], xlabel="Tempo (s)", ylabel="Frequência (Hz)")
    image = axes[2].imshow(corr, origin="lower", aspect="auto", cmap="viridis")
    axes[2].scatter(col, row, s=44, facecolors="none", edgecolors="white")
    axes[2].set(title="Correlação normalizada", xlabel="Deslocamento temporal (frames)",
                ylabel="Deslocamento em frequência (bins)")
    figure.colorbar(image, ax=axes[2], label="Correlação")
    figure.savefig(output / "le03_correlacao_2d.png", dpi=175)
    plt.close(figure)
    return {"recording": fname, "single_window_time_s": event["peak_s"], "original_score": event["score"],
            "smoothed_score": float(smooth[idx]), "threshold": threshold,
            "clip": clips.at[index, "clip_name"], "correlation_peak": float(corr[row, col]),
            "matched_start_hz": float(SPEC_FREQUENCIES[SEARCH_BAND][row]),
            "matched_start_s": float(FRAME_CENTERS[col])}


def structure_tensor(spec, tracks):
    dx = ndimage.gaussian_filter(spec, sigma=1.0, order=(0, 1))
    dy = ndimage.gaussian_filter(spec, sigma=1.0, order=(1, 0))
    xx, xy, yy = [ndimage.gaussian_filter(value, sigma=1.5) for value in (dx * dx, dx * dy, dy * dy)]
    columns = np.flatnonzero((FRAME_CENTERS >= 1.2) & (FRAME_CENTERS <= 1.8))
    rows = np.abs(SPEC_FREQUENCIES[:, None] - tracks[columns]).argmin(axis=0)
    # The largest eigenvector points across the ridge, so use the tangent.
    theta = 0.5 * np.arctan2(2 * xy[rows, columns], xx[rows, columns] - yy[rows, columns]) + np.pi / 2
    slope = np.tan(theta) * BIN_HZ / FRAME_HOP_S
    return columns, rows, theta, float(np.median(slope))


def spectral_demos(output):
    sweep, tone = make_signal(), make_signal(upcall=False)
    spec_sweep, spec_tone = denoised_spectrogram(stft_power(np.stack([sweep, tone])))
    frame = sweep[1280:1280 + FRAME_SAMPLES] * WINDOW
    n = np.arange(FRAME_SAMPLES)
    matrix = np.exp(-2j * np.pi * np.outer(n[:FRAME_SAMPLES // 2 + 1], n) / FRAME_SAMPLES)
    expected, observed = matrix @ frame, fft.rfft(frame)
    error = float(np.max(np.abs(expected - observed)) / np.max(np.abs(expected)))
    central = (FRAME_CENTERS >= 1.15) & (FRAME_CENTERS <= 1.85)
    ridge = SPEC_FREQUENCIES[spec_sweep.argmax(axis=0)]
    slope = float(np.polyfit(FRAME_CENTERS[central], ridge[central], 1)[0])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
    axes[0].plot(FREQUENCIES, np.abs(observed), label="rFFT")
    axes[0].scatter(FREQUENCIES[::3], np.abs(expected)[::3], s=9, color="#d95f02", label="DFT direta", zorder=3)
    axes[0].set(title="Uma janela de 256 amostras", xlabel="Frequência (Hz)", ylabel="Magnitude")
    axes[0].legend(fontsize="small")
    axes[1].imshow(spec_sweep, origin="lower", aspect="auto", cmap="magma", extent=SPECTROGRAM_EXTENT)
    axes[1].plot(FRAME_CENTERS[central], ridge[central], color="cyan", linewidth=1, label="Pico por janela")
    axes[1].set(title="STFT do chirp", xlabel="Tempo (s)", ylabel="Frequência (Hz)")
    axes[1].legend(fontsize="small")
    figure.savefig(output / "le06_dft_stft.png", dpi=175)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharex=True, sharey=True)
    slopes = []
    for axis, spec, track, title in zip(
        axes, (spec_sweep, spec_tone), (80 + 120 * (FRAME_CENTERS - 1), np.full_like(FRAME_CENTERS, 140)),
        ("Chirp de 80 a 200 Hz", "Tom estacionário de 140 Hz"),
    ):
        columns, rows, theta, orientation = structure_tensor(spec, track)
        slopes.append(orientation)
        image = axis.imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=SPECTROGRAM_EXTENT,
                            vmin=-4, vmax=24)
        time, frequency = FRAME_CENTERS[columns[::3]], SPEC_FREQUENCIES[rows[::3]]
        dx = 0.11 * np.cos(theta[::3])
        dy = 0.11 * BIN_HZ / FRAME_HOP_S * np.sin(theta[::3])
        for t, f, x, y in zip(time, frequency, dx, dy):
            axis.plot([t - x / 2, t + x / 2], [f - y / 2, f + y / 2], color="cyan", linewidth=1.7)
        axis.set(title=title, xlabel="Tempo (s)", xlim=(0.65, 2.35), ylim=(65, 225))
    axes[0].set_ylabel("Frequência (Hz)")
    figure.colorbar(image, ax=axes, label="dB após subtração da mediana")
    figure.savefig(output / "le11_tensor_estrutura.png", dpi=175)
    plt.close(figure)
    return ({"bin_hz": BIN_HZ, "hop_ms": FRAME_HOP_S * 1000, "relative_dft_error": error,
             "chirp_ridge_hz_per_s": slope},
            {"chirp_tangent_hz_per_s": slopes[0], "stationary_tangent_hz_per_s": slopes[1]})


def main():
    parser = argparse.ArgumentParser(description="Demonstrações dos conceitos de PDSI aplicados aos upcalls.")
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--continuous", type=Path, default=Path("artifacts/continuo"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/novos_testes/disciplina"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spectral, tensor = spectral_demos(args.output)
    results = {"le02": sampling_demo(args.output), "le03": convolution_demo(args.prepared, args.continuous, args.output),
               "le06": spectral, "le11": tensor}
    save_json(args.output / "medicoes.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
