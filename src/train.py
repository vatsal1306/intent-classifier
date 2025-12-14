"""
Fine-tunes a small Transformer for 2-class intent classification.

Features:
    - AMP (mixed precision) on GPU
    - gradient accumulation
    - class-weighted loss (optional)
    - early stopping
    - saves best model + tokenizer
    - writes metrics + confusion matrix + misclassified examples
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.Logging import logger


# -----------------------------------
# ⚙️ Config Dataclass
# -----------------------------------
@dataclass
class TrainConfig:
    model_name: str = "google/mobilebert-uncased"
    max_length: int = 32
    batch_size: int = 64
    eval_batch_size: int = 256
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 8
    cosine_tmax: int = 8
    cosine_eta_min: float = 1e-6
    label_smoothing: float = 0.0
    use_class_weights: bool = True
    early_stopping_patience: int = 2
    metric_for_best: str = "accuracy"  # "accuracy" or "macro_f1"
    seed: int = 42
    num_workers: int = 2


# -----------------------------------
# 🧠 Utility Functions
# -----------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QueryDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer: AutoTokenizer,
            max_length: int,
    ) -> None:
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_class_weights(label_ids: np.ndarray, num_labels: int) -> torch.Tensor:
    """
    Computes normalized inverse-frequency class weights.
    """
    counts = np.bincount(label_ids, minlength=num_labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv * (counts.sum() / inv.sum())
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        id2label: Dict[int, str],
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = logits.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=sorted(id2label.keys()),
    )
    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label.keys()))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            id2label[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
            for i in range(len(id2label))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in sorted(id2label.keys())],
            digits=4,
            zero_division=0,
        ),
    }

    probs = softmax_np(logits)
    return metrics, y_true, y_pred, probs


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def save_confusion_matrix_csv(cm: np.ndarray, id2label: Dict[int, str], out_path: Path) -> None:
    labels = [id2label[i] for i in sorted(id2label.keys())]
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    df.to_csv(out_path, index=True)


# -----------------------------------
# 🏁 Main Training Script
# -----------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-description", type=str, default="")
    parser.add_argument("--model-name", type=str, default="google/mobilebert-uncased")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cosine-tmax", type=int, default=8)
    parser.add_argument("--cosine-eta-min", type=float, default=1e-6)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--metric-for-best",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro_f1"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = TrainConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        cosine_tmax=args.cosine_tmax,
        cosine_eta_min=args.cosine_eta_min,
        use_class_weights=args.use_class_weights,
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best=args.metric_for_best,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    data_dir = Path(args.data_dir)
    label_map_path = data_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing {label_map_path}. Run prepare_data.py first.")

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label2id = {k: int(v) for k, v in label_map["label2id"].items()}
    id2label = {int(k): str(v) for k, v in label_map["id2label"].items()}

    num_labels = len(label2id)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Datasets
    train_ds = QueryDataset(
        texts=train_df["text"].astype(str).tolist(),
        labels=train_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    val_ds = QueryDataset(
        texts=val_df["text"].astype(str).tolist(),
        labels=val_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    test_ds = QueryDataset(
        texts=test_df["text"].astype(str).tolist(),
        labels=test_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )

    # Loss & class weights
    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_df["label_id"].values, num_labels).to(device)
        logger.info("Class weights: %s", class_weights.cpu().numpy().tolist())

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels, weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.cosine_tmax, eta_min=cfg.cosine_eta_min
    )

    # Artifacts dir
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.artifacts_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config file
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=4), encoding="utf-8")

    # W&B Init
    wandb.init(project="intent-classification", config=asdict(cfg), mode='online', name=run_name,
               notes=args.run_description)
    # wandb.watch(model, log="all", log_freq=100)

    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0

    def get_current_metric(metrics: Dict) -> float:
        return float(metrics[cfg.metric_for_best])

    # Training loop (FP32)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, id2label)

        logger.info(
            "Epoch %d | train_loss=%.6f | val_acc=%.6f | val_macro_f1=%.6f",
            epoch,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )

        # Log to W&B
        wandb.log(
            {
                "train/loss": train_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
        )

        # Save latest model
        latest_dir = out_dir / f"latest_model_epoch_{epoch}"
        latest_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(latest_dir)
        tokenizer.save_pretrained(latest_dir)

        # Evaluate & decide best
        current_val = get_current_metric(val_metrics)
        if current_val > best_metric:
            best_metric = current_val
            best_epoch = epoch
            bad_epochs = 0

            best_dir = out_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            (out_dir / "best_val_metrics.json").write_text(
                json.dumps(val_metrics, indent=4), encoding="utf-8"
            )

        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (best_epoch=%d %s=%.4f)",
                    epoch,
                    best_epoch,
                    cfg.metric_for_best,
                    best_metric,
                )
                break

    # Final test evaluation
    if (out_dir / "best_model").exists():
        model = AutoModelForSequenceClassification.from_pretrained(out_dir / "best_model").to(device)
    test_metrics, y_true, y_pred, probs = evaluate(model, test_loader, device, id2label)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=4), encoding="utf-8")
    save_confusion_matrix_csv(np.array(test_metrics["confusion_matrix"]), id2label, out_dir / "confusion_matrix.csv")

    logger.info("Final TEST | acc=%.6f | macro_f1=%.6f", test_metrics["accuracy"], test_metrics["macro_f1"])
    wandb.log({"test/accuracy": test_metrics["accuracy"], "test/macro_f1": test_metrics["macro_f1"]})

    return 0


if __name__ == "__main__":
    # os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
