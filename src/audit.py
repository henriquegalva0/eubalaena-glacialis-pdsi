import argparse
import hashlib
import wave
from pathlib import Path

import pandas as pd

from .evaluate import metrics, save_json
from .prepare import PARTITIONS, read_annotations


def pcm_digest(path):
    with wave.open(str(path), "rb") as audio:
        return hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()


def audit(data_root, runs, output):
    tables, unlabelled = {}, []
    for split, partitions in PARTITIONS.items():
        tables[split] = pd.concat([
            read_annotations(data_root, dataset, partition) for dataset, partition in partitions
        ], ignore_index=True)
        tables[split]["pcm_sha256"] = tables[split]["audio_path"].map(pcm_digest)
        for _, group in tables[split].groupby(["dataset", "partition"]):
            folder = group["audio_path"].iloc[0].parent
            candidates = set(folder.glob("*.wav")) - set(group["audio_path"])
            unlabelled.extend(path.relative_to(data_root).as_posix() for path in sorted(candidates))

    test = tables["test"]
    splits = {split: set(table["pcm_sha256"]) for split, table in tables.items()}
    duplicated = test["pcm_sha256"].duplicated(keep=False)
    duplicate_table = test.loc[duplicated, ["clip_id", "label", "source_filename", "pcm_sha256"]]
    output.mkdir(parents=True, exist_ok=True)
    duplicate_table.sort_values(["pcm_sha256", "clip_id"]).to_csv(output / "duplicatas_c.csv", index=False)
    save_json(output / "auditoria_audio.json", {
        "a_b_train_val_pcm_overlap": len(splits["train"] & splits["val"]),
        "c_train_pcm_overlap": len(splits["test"] & splits["train"]),
        "c_val_pcm_overlap": len(splits["test"] & splits["val"]),
        "unlabelled_wavs": unlabelled,
        "c_clips": len(test), "c_unique_pcm": len(splits["test"]),
        "c_repeated_rows": len(test) - len(splits["test"]),
        "c_conflicting_duplicate_labels": int(
            (duplicate_table.groupby("pcm_sha256")["label"].nunique() > 1).sum()
        ),
    })

    by_collection, sensitivity = [], []
    annotations = test[["clip_id", "pcm_sha256"]]
    for model in ("baseline", "cnn"):
        predictions = pd.read_csv(runs / model / "test_predictions.csv")
        for seed, table in predictions.groupby("seed"):
            enriched = table.merge(annotations, validate="one_to_one", on="clip_id").sort_values("clip_id")
            if len(enriched) != len(annotations) or len(enriched) != len(table):
                raise ValueError(f"Incomplete test predictions for {model}, seed {seed}")
            collection = enriched["source_filename"].str.extract(r"^(NOPP6_EST|SBNMS|GSC)_", expand=False)
            for name, group in enriched.groupby(collection.fillna("other")):
                by_collection.append({"model": model, "seed": seed, "collection": name, **metrics(group)})
            unique = enriched.drop_duplicates("pcm_sha256")
            sensitivity.append({"model": model, "seed": seed, **metrics(unique)})

    pd.DataFrame(by_collection).to_csv(output / "metricas_por_colecao_c.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "sensibilidade_c_sem_pcm_repetido.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Audita as partições e os resultados congelados em C.")
    parser.add_argument("--data", type=Path, required=True, help="Pasta narw-data")
    parser.add_argument("--runs", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/avaliacao"))
    args = parser.parse_args()
    audit(args.data, args.runs, args.output)


if __name__ == "__main__":
    main()
