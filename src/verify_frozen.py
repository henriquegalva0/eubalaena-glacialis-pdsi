import argparse
import hashlib
import json
from pathlib import Path


def verify(artifacts):
    protocol = json.loads((artifacts / "protocolo_teste.json").read_text(encoding="utf-8"))
    failures = 0
    for model, runs in protocol["runs"].items():
        frozen = {run["seed"]: run for run in json.loads((artifacts / model / "frozen.json").read_text(encoding="utf-8"))}
        for run in runs:
            path = artifacts / model / run["checkpoint"]
            ok = (path.is_file()
                  and hashlib.sha256(path.read_bytes()).hexdigest() == run["sha256"]
                  and frozen.get(run["seed"], {}).get("threshold") == run["threshold"])
            failures += not ok
            print(f"{model} seed={run['seed']}: {'OK' if ok else 'DIFERENTE'}", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description="Confere checkpoints e limiares com o protocolo congelado.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    raise SystemExit(1 if verify(parser.parse_args().artifacts) else 0)


if __name__ == "__main__":
    main()
