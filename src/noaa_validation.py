import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from .cnn import load_checkpoint, predict
from .continuous import STEP
from .evaluate import save_json
from .operating_curve import VARIANTS, aggregate_at_threshold, build_curve, load_window_scores, moving_average, operating_points
from .preprocessing import CLIP_SAMPLES, SAMPLE_RATE, denoised_spectrogram, mono_float, resample, stft_power
from .template_detector import score_spectrograms


BUCKET = "noaa-passive-bioacoustic"
PREFIX = "nefsc/products/detections/nefsc_baleen/"
DEPLOYMENTS = (
    ("calibration", "Nantucket", "nefsc_ma-ri_201512_ch4_20160218_narw"),
    ("test", "Hatteras", "nefsc_nc_201712_hatteras_ch2_20180121_narw"),
    ("test", "Brunswick", "nefsc_ga_201510_ch1_20160108_narw"),
)
HOURS = (0, 1, 6, 7, 12, 13, 18, 19)
OFFSET = timezone(timedelta(hours=-5))
BUDGET = 25


def list_objects(prefix):
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urlencode({'prefix': prefix})}"
    with urlopen(url, timeout=30) as source:
        page = json.load(source)
    if page.get("nextPageToken"):
        raise ValueError(f"Unexpected multi-page listing for {prefix}")
    return page["items"]


def download_object(item, directory):
    path = directory / Path(item["name"]).name
    size = int(item["size"])
    if path.is_file() and path.stat().st_size == size:
        return path
    url = f"https://storage.googleapis.com/{BUCKET}/{quote(item['name'])}"
    pending = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urlopen(url, timeout=40) as source:
                data = source.read(size + 1)
            if len(data) != size:
                raise OSError(f"Incomplete file: {url}")
            pending.write_bytes(data)
            pending.replace(path)
            return path
        except OSError:
            if attempt == 2:
                raise


def download_data(directory):
    directory.mkdir(parents=True, exist_ok=True)
    recordings, annotations = [], []
    for split, site, deployment in DEPLOYMENTS:
        folder = PREFIX + deployment + "/"
        log, = [item for item in list_objects(folder + "data/") if item["name"].endswith(".csv")]
        selected = []
        for item in list_objects(folder + "ancillary/source-audio/"):
            name = Path(item["name"]).name
            date = re.search(r"_(\d{8}_\d{6})_CH\d+\.wav$", name)
            if date:
                stamp = datetime.strptime(date[1], "%Y%m%d_%H%M%S").replace(tzinfo=OFFSET)
                if stamp.hour in HOURS:
                    selected.append((item, pd.Timestamp(stamp)))
        if len(selected) != 32:
            raise ValueError(f"Expected 32 quarter-hour recordings at {site}, found {len(selected)}")
        raw = pd.read_csv(download_object(log, directory)).drop_duplicates(
            subset=["Start_DateTime_ISO8601", "End_DateTime_ISO8601", "Detection_Confidence"],
        )
        confidence = raw["Detection_Confidence"]
        if not set(confidence) <= {"Detected", "Possibly_Detected"} or not raw["Species"].eq("RIWH").all():
            raise ValueError(f"Unexpected annotation values in {site}")
        start = pd.to_datetime(raw["Start_DateTime_ISO8601"], utc=True, format="ISO8601")
        end = pd.to_datetime(raw["End_DateTime_ISO8601"], utc=True, format="ISO8601")
        centers = start + (end - start) / 2
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda entry: download_object(entry[0], directory), selected))
        for path, (item, stamp) in zip(paths, selected):
            rate, samples = wavfile.read(path, mmap=True)
            duration = len(samples) / rate
            if samples.ndim != 1 or duration != 900 or "UTCm5" not in path.name:
                raise ValueError(f"Unexpected audio format or duration: {path}")
            offset = (centers - stamp).dt.total_seconds()
            in_file = (offset >= 0) & (offset < duration)
            annotations.append(pd.DataFrame({
                "filename": path.name, "timestamp": offset[in_file], "snr": np.nan,
                "uncertain": confidence[in_file].eq("Possibly_Detected"),
            }))
            recordings.append({"filename": path.name, "split": split, "site": site, "object": item["name"],
                               "start_iso": stamp.isoformat(), "duration_s": duration,
                               "windows": int((duration - CLIP_SAMPLES / SAMPLE_RATE) / STEP) + 1})
        print(f"NOAA {site} ({split}): 32 arquivos, 8 h", flush=True)
    pd.DataFrame(recordings).to_csv(directory / "recordings.csv", index=False)
    pd.concat(annotations, ignore_index=True).to_csv(directory / "annotations.csv", sep=";", index=False)


def score_data(directory, cnn, output, batch_size):
    files = pd.read_csv(directory / "recordings.csv")
    candidates = json.loads((cnn / "frozen.json").read_text(encoding="utf-8"))
    frozen = max(candidates, key=lambda run: (run["val_auc"], -run["seed"]))
    model = load_checkpoint(cnn / frozen["checkpoint"])
    cnn_tables, template_tables = [], []
    for index, record in enumerate(files.itertuples(index=False), start=1):
        rate, raw = wavfile.read(directory / record.filename, mmap=True)
        audio = resample(mono_float(raw), rate)
        windows = np.lib.stride_tricks.sliding_window_view(audio, CLIP_SAMPLES)[::round(SAMPLE_RATE * STEP)]
        if len(windows) != record.windows:
            raise ValueError(f"Window grid differs from the manifest in {record.filename}")
        cnn_scores, template_scores = [], []
        for start in range(0, len(windows), batch_size):
            spectra = denoised_spectrogram(stft_power(windows[start:start + batch_size]))
            cnn_scores.append(predict(model, spectra, batch_size))
            template_scores.append(score_spectrograms(spectra, batch_size))
        table = pd.DataFrame({"filename": record.filename,
                              "center_s": CLIP_SAMPLES / (2 * SAMPLE_RATE) + np.arange(len(windows)) * STEP})
        cnn_tables.append(table.assign(score=np.concatenate(cnn_scores)))
        template_tables.append(table.assign(score=np.concatenate(template_scores)))
        if index % 8 == 0:
            print(f"NOAA: {index}/{len(files)} gravações com scores", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(cnn_tables, ignore_index=True).to_csv(output / "cnn_janelas.csv", index=False)
    pd.concat(template_tables, ignore_index=True).to_csv(output / "molde_janelas.csv", index=False)
    save_json(output / "cnn_original.json", frozen)


def compare_data(directory, output, template):
    recordings = pd.read_csv(directory / "recordings.csv")
    annotations = pd.read_csv(directory / "annotations.csv", sep=";")
    if recordings["filename"].duplicated().any():
        raise ValueError("A NOAA recording occurs more than once in the partition manifest")
    truth = {name: annotations.loc[annotations["filename"] == name] for name in recordings["filename"]}
    scores = {key: load_window_scores(output / f"{key}_janelas.csv", recordings) for key in ("cnn", "molde")}
    original = json.loads((output / "cnn_original.json").read_text(encoding="utf-8"))["threshold"]
    template_threshold = json.loads(template.read_text(encoding="utf-8"))["threshold"]

    calibration = recordings.loc[recordings["split"] == "calibration"]
    validation_scores = {model: {name: values[name] for name in calibration["filename"]}
                         for model, values in scores.items()}
    validation = build_curve(validation_scores, truth, calibration["duration_s"].sum() / 3600,
                             original, template_threshold)
    validation.to_csv(output / "curva_calibracao.csv", index=False)
    candidates = operating_points(validation)
    picks = candidates.loc[candidates["budget_fp_h"] == BUDGET].sort_values(
        ["recall", "false_alarms_per_hour", "variant"], ascending=[False, True, True],
    )
    selected = picks.iloc[0]["variant"]
    if picks.iloc[0]["tp"] == 0:
        raise ValueError("No positive detections at the prespecified calibration budget")
    picks.to_csv(output / "pontos_calibracao.csv", index=False)
    frozen = picks[["variant", "threshold"]].to_dict(orient="records")
    save_json(output / "regra_congelada.json", {"budget_fp_h": BUDGET, "selected": selected, "variants": frozen})

    frozen.append({"variant": "cnn_original_congelada_ab", "threshold": original})
    test = recordings.loc[recordings["split"] == "test"]
    rows = []
    for variant in frozen:
        name, threshold = variant["variant"], variant["threshold"]
        base = "cnn_original" if name == "cnn_original_congelada_ab" else name
        _, smooth, consecutive = VARIANTS[base]
        model = "molde" if base == "molde_2d" else "cnn"
        for site, block in [("total", test), *test.groupby("site")]:
            score_arrays = {filename: scores[model][filename] for filename in block["filename"]}
            if smooth:
                score_arrays = {filename: moving_average(values) for filename, values in score_arrays.items()}
            hours = block["duration_s"].sum() / 3600
            rows.append({"site": site, "variant": name, "selected": name == selected, "threshold": threshold,
                         "hours": hours, **aggregate_at_threshold(score_arrays, truth, threshold, consecutive, hours)})
    measured = pd.DataFrame(rows)
    measured.to_csv(output / "teste_independente.csv", index=False)
    recordings.to_csv(output / "manifesto_gravacoes.csv", index=False)
    print(measured.loc[measured["site"] == "total"].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Validação externa com áudio anotado da NOAA NEFSC.")
    parser.add_argument("stage", choices=("download", "score", "compare"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/novos_testes/noaa"))
    parser.add_argument("--cnn", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--template", type=Path, default=Path("artifacts/novos_testes/molde/frozen.json"))
    parser.add_argument("--batch-size", type=int, default=160)
    args = parser.parse_args()
    if args.stage == "download":
        download_data(args.data)
    elif args.stage == "score":
        torch.set_num_threads(4)
        score_data(args.data, args.cnn, args.output, args.batch_size)
    else:
        compare_data(args.data, args.output, args.template)


if __name__ == "__main__":
    main()
