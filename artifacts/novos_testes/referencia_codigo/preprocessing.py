import numpy as np
from scipy import fft, signal


SAMPLE_RATE = 1_000
CLIP_SAMPLES = 3 * SAMPLE_RATE
FRAME_SAMPLES = 256
HOP_SAMPLES = 32
BAND = (40, 400)
SPECTROGRAM_SHAPE = (92, 86)

WINDOW = signal.windows.hamming(FRAME_SAMPLES, sym=False).astype(np.float32)
FILTER = signal.butter(4, BAND, btype="bandpass", fs=SAMPLE_RATE, output="sos")
FREQUENCIES = fft.rfftfreq(FRAME_SAMPLES, 1 / SAMPLE_RATE)
BAND_MASK = (FREQUENCIES >= BAND[0]) & (FREQUENCIES <= BAND[1])
CONTOUR_MASK = (FREQUENCIES[BAND_MASK] >= 50) & (FREQUENCIES[BAND_MASK] <= 250)
FRAMES_SECONDS = np.arange(SPECTROGRAM_SHAPE[1]) * HOP_SAMPLES / SAMPLE_RATE


def mono_float(samples):
    """Convert PCM or float audio to a one-dimensional float signal."""
    audio = np.asarray(samples)
    if audio.dtype.kind == "u":
        midpoint = (np.iinfo(audio.dtype).max + 1) / 2
        audio = (audio.astype(np.float32) - midpoint) / midpoint
    elif audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / -float(np.iinfo(audio.dtype).min)
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono or multichannel WAV samples")
    return audio


def resample(samples, sample_rate):
    if sample_rate == SAMPLE_RATE:
        return samples
    divisor = np.gcd(sample_rate, SAMPLE_RATE)
    return signal.resample_poly(samples, SAMPLE_RATE // divisor, sample_rate // divisor)


def stft_power(windows):
    """Compute clip STFT power with shape (clips, frequency, time)."""
    windows = np.atleast_2d(np.asarray(windows, dtype=np.float32))
    if windows.ndim != 2 or windows.shape[1] != CLIP_SAMPLES:
        raise ValueError(f"Expected clips with {CLIP_SAMPLES} samples")
    centered = windows - windows.mean(axis=1, keepdims=True)
    filtered = signal.sosfiltfilt(FILTER, centered, axis=1).astype(np.float32)
    peak = np.max(np.abs(filtered), axis=1, keepdims=True)
    normalized = filtered / np.maximum(peak, 1e-7)
    frames = np.lib.stride_tricks.sliding_window_view(normalized, FRAME_SAMPLES, axis=1)
    spectrum = fft.rfft(frames[:, ::HOP_SAMPLES, :] * WINDOW, axis=2)
    return (np.abs(spectrum) ** 2).transpose(0, 2, 1)


def denoised_spectrogram(power):
    band = (10 * np.log10(np.maximum(power[:, BAND_MASK, :], 1e-10))).astype(np.float32)
    return band - np.median(band, axis=2, keepdims=True)


def _mel_bank():
    edges_mel = np.linspace(*(2595 * np.log10(1 + np.array(BAND) / 700)), 34)
    edges = 700 * (10 ** (edges_mel / 2595) - 1)
    f = FREQUENCIES[np.newaxis, :]
    left = (f - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    right = (edges[2:, None] - f) / (edges[2:, None] - edges[1:-1, None])
    bank = np.maximum(0, np.minimum(left, right))
    return (bank / bank.sum(axis=1, keepdims=True)).astype(np.float32)


MEL_BANK = _mel_bank()


def baseline_features(power, spectrograms):
    """Summarize 32 log-mel bands and 10 MFCCs, plus three contour features."""
    log_mel = (10 * np.log10(np.maximum(MEL_BANK @ power, 1e-10))).astype(np.float32)
    mfcc = fft.dct(log_mel, type=2, axis=1, norm="ortho")[:, :10, :]
    features = np.concatenate((log_mel, mfcc), axis=1)
    strongest_bin = spectrograms[:, CONTOUR_MASK, :].argmax(axis=1)
    contour = FREQUENCIES[BAND_MASK][CONTOUR_MASK][strongest_bin]
    centered_time = FRAMES_SECONDS - FRAMES_SECONDS.mean()
    centered_contour = contour - contour.mean(axis=1, keepdims=True)
    slope = (centered_contour @ centered_time) / (centered_time @ centered_time)
    fitted = slope[:, None] * centered_time
    unexplained = np.sum((centered_contour - fitted) ** 2, axis=1)
    variation = np.sum(centered_contour**2, axis=1)
    r_squared = np.where(variation > 0, 1 - unexplained / np.maximum(variation, 1e-12), 0)
    shift = np.median(contour[:, -16:], axis=1) - np.median(contour[:, :16], axis=1)
    return np.column_stack(
        (features.mean(axis=2), features.std(axis=2), np.percentile(features, 90, axis=2),
         slope, r_squared, shift)
    ).astype(np.float32)
