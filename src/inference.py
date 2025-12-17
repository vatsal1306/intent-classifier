"""
To run for labeled data, to save misclassified queries
python -m src.inference --ckpt runs/<run>/best_model --data-csv dataset/splits/test.csv --labeled

To run for unlabelled data, to save all predictions
python -m src.inference --ckpt runs/<run>/best_model --data-csv new_queries.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import clean_text_for_model


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.inference_mode()
def predict_all(model, tokenizer, texts: List[str], device: torch.device, max_length: int, batch_size: int = 256):
    model.eval()
    probs_out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits.detach().cpu().numpy()
        probs_out.append(softmax_np(logits))
    return np.vstack(probs_out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint dir")
    ap.add_argument("--data-csv", type=str, required=True, help="CSV with at least a text column")
    ap.add_argument("--text-col", type=str, default="text")
    ap.add_argument("--label-col", type=str, default="label_id")
    ap.add_argument(
        "--labeled",
        action="store_true",
        help="If set, data-csv is labeled and must contain label-col. If not set, runs unlabeled inference.",
    )
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    data = pd.read_csv(args.data_csv)

    if args.text_col not in data.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in {args.data_csv}")

    texts_raw = data[args.text_col].astype(str).tolist()
    texts = [clean_text_for_model(t) for t in texts_raw]

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    probs = predict_all(model, tokenizer, texts, device, args.max_length, args.batch_size)
    y_pred = probs.argmax(axis=1)

    # Resolve labels in a stable order
    id2label: Dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    num_labels = probs.shape[1]

    if args.labeled:
        if args.label_col not in data.columns:
            raise ValueError(f"--labeled was set but missing label column '{args.label_col}' in {args.data_csv}")

        y_true = data[args.label_col].astype(int).to_numpy()

        misclassified: List[Dict[str, Any]] = []
        for idx, (text, true_id, pred_id, prob) in enumerate(zip(texts, y_true, y_pred, probs)):
            if int(true_id) != int(pred_id):
                misclassified.append(
                    {
                        "row_idx": idx,
                        "text": text,
                        "true_label_id": int(true_id),
                        "pred_label_id": int(pred_id),
                        "true_label": id2label.get(int(true_id), f"unknown_{int(true_id)}"),
                        "pred_label": id2label.get(int(pred_id), f"unknown_{int(pred_id)}"),
                        "pred_confidence": float(prob[int(pred_id)]),
                        **{f"prob_{id2label.get(j, str(j))}": float(prob[j]) for j in range(num_labels)},
                    }
                )

        out_df = pd.DataFrame(misclassified)
        out_csv = Path(args.out_csv) if args.out_csv else Path(args.ckpt) / "misclassified.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote misclassified to {out_csv}")
        return 0

    # Unlabeled inference: write predictions for ALL rows (no ground-truth columns)
    preds: List[Dict[str, Any]] = []
    for idx, (text, pred_id, prob) in enumerate(zip(texts, y_pred, probs)):
        preds.append(
            {
                "row_idx": idx,
                "text": text,
                "pred_label_id": int(pred_id),
                "pred_label": id2label.get(int(pred_id), f"unknown_{int(pred_id)}"),
                "pred_confidence": float(prob[int(pred_id)]),
                **{f"prob_{id2label.get(j, str(j))}": float(prob[j]) for j in range(num_labels)},
            }
        )

    out_df = pd.DataFrame(preds)
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.ckpt) / "predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
