import argparse
from pathlib import Path

import pandas as pd

from .evaluate import METRICS, export_comparison


def main():
    parser = argparse.ArgumentParser(description="Compara ablações da CNN na validação A+B.")
    parser.add_argument("--reference", type=Path, default=Path("artifacts/cnn/val_predictions.csv"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/novos_testes/ablacoes"))
    args = parser.parse_args()
    files = {"completa": args.reference}
    files.update({name: args.runs / name / "val_predictions.csv" for name in ("sem_butterworth", "sem_mediana")})
    tables = []
    for name, path in files.items():
        table = pd.read_csv(path).sort_values(["seed", "clip_id"]).reset_index(drop=True)
        identifiers = table[["seed", "clip_id", "label", "dataset", "snr"]]
        if tables and not identifiers.equals(reference):
            raise ValueError(f"Different validation clips, seeds or metadata in {name}")
        if not tables:
            reference = identifiers
        tables.append(table.assign(model=name, split="val"))
    export_comparison(pd.concat(tables, ignore_index=True), args.runs)
    metrics = pd.read_csv(args.runs / "metricas_por_seed.csv")
    total = metrics.loc[metrics["dataset"] == "total"]
    control = total.loc[total["model"] == "completa"].set_index("seed")[list(METRICS)]
    deltas = []
    for name, group in total.loc[total["model"] != "completa"].groupby("model"):
        difference = group.set_index("seed")[list(METRICS)] - control
        deltas.append(difference.assign(model=name))
    pd.concat(deltas).to_csv(args.runs / "deltas_pareados.csv")


if __name__ == "__main__":
    main()
