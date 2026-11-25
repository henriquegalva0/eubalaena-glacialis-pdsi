import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .evaluate import SEEDS, best_threshold, prediction_table, save_json
from .prepare import load_partition


class UpcallCNN(nn.Sequential):
    def __init__(self):
        blocks = []
        for incoming, outgoing in ((1, 8), (8, 16), (16, 24)):
            blocks.extend((
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
        super().__init__(
            *blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(24, 1),
        )
        self.register_buffer("train_mean", torch.tensor(0.0))
        self.register_buffer("train_scale", torch.tensor(1.0))

    def forward(self, samples):
        return super().forward((samples - self.train_mean) / self.train_scale).squeeze(1)


def as_tensor(spectrograms):
    return torch.from_numpy(np.ascontiguousarray(spectrograms, dtype=np.float32))[:, None]


@torch.inference_mode()
def predict(model, spectrograms, batch_size):
    model.eval()
    return np.concatenate([
        model(batch.contiguous(memory_format=torch.channels_last)).sigmoid().numpy()
        for batch in as_tensor(spectrograms).split(batch_size)
    ])


def load_checkpoint(path):
    model = UpcallCNN().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def train(prepared, output, seeds, epochs, batch_size):
    x_train, train_clips = load_partition(prepared, "train", "spectrograms")
    x_val, val_clips = load_partition(prepared, "val", "spectrograms")
    training = as_tensor(x_train)
    labels = torch.tensor(train_clips["label"].to_numpy(), dtype=torch.float32)
    val_labels = val_clips["label"].to_numpy()
    positive_weight = (len(labels) - labels.sum()) / labels.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    mean, scale = float(x_train.mean()), max(float(x_train.std()), 1e-6)
    output.mkdir(parents=True, exist_ok=True)
    tables, histories, manifest = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = UpcallCNN().to(memory_format=torch.channels_last)
        model.train_mean.fill_(mean)
        model.train_scale.fill_(scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        checkpoint = f"seed_{seed}.pt"
        best_auc, best_epoch = -1.0, 0
        for epoch in range(1, epochs + 1):
            start = time.monotonic()
            model.train()
            total_loss = 0.0
            for indices in torch.randperm(len(training)).split(batch_size):
                batch = training[indices].contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch), labels[indices])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
            scores = predict(model, x_val, batch_size)
            auc = float(roc_auc_score(val_labels, scores))
            train_loss = total_loss / len(training)
            elapsed = time.monotonic() - start
            histories.append({"seed": seed, "epoch": epoch, "loss": train_loss,
                              "val_auc": auc, "seconds": elapsed})
            print(f"CNN seed={seed} época={epoch:02d}: loss={train_loss:.4f}, "
                  f"val_auc={auc:.4f}, {elapsed:.1f}s", flush=True)
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                pending = output / f"seed_{seed}.tmp"
                torch.save(model.state_dict(), pending)
                pending.replace(output / checkpoint)
            if epoch - best_epoch >= 4:
                break
        model = load_checkpoint(output / checkpoint)
        scores = predict(model, x_val, batch_size)
        threshold = best_threshold(val_labels, scores)
        tables.append(prediction_table(val_clips, scores, seed, threshold))
        manifest.append({"seed": seed, "checkpoint": checkpoint, "threshold": threshold,
                         "val_auc": best_auc, "best_epoch": best_epoch})
        pd.DataFrame(histories).to_csv(output / "training.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(output / "val_predictions.csv", index=False)
    save_json(output / "frozen.json", manifest)


def score_partition(prepared, output, split, batch_size):
    values, clips = load_partition(prepared, split, "spectrograms")
    config = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
    tables = []
    for run in config:
        model = load_checkpoint(output / run["checkpoint"])
        scores = predict(model, values, batch_size)
        tables.append(prediction_table(clips, scores, run["seed"], run["threshold"]))
    pd.concat(tables, ignore_index=True).to_csv(output / f"{split}_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Treina ou aplica a CNN dos espectrogramas.")
    parser.add_argument("stage", choices=("train", "val", "test"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage == "train":
        train(args.prepared, args.output, args.seeds, args.epochs, args.batch_size)
    else:
        score_partition(args.prepared, args.output, args.stage, args.batch_size)


if __name__ == "__main__":
    main()
