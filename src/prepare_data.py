"""
- Reads a labeled CSV (16K rows).
- Cleans text and canonicalizes labels to {product, outfit}.
- Builds a group_key to reduce split leakage (duplicate / near-duplicate queries).
- Creates train/val/test splits with group-aware shuffling and approximate stratification.
- Writes:
    data/splits/train.csv, val.csv, test.csv
    data/splits/label_map.json
    data/splits/data_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.Logging import logger
from src.utils import normalize_whitespace, clean_text_for_model

LABEL_CANONICAL_MAP = {
    "product query": "product",
    "outfit idea query": "outfit",
}


def normalize_for_grouping(text: str, group_digits: bool = True) -> str:
    """
    More aggressive normalization used only for grouping to avoid split leakage:
    - lowercasing
    - collapse whitespace
    - optional digit normalization
    - strips most punctuation to catch near-duplicates
    - `dress under 500` to `dress under <num>`
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = normalize_whitespace(text)

    if group_digits:
        # Replace runs of digits with a placeholder.
        text = re.sub(r"\d+", "<num>", text)

    # Keep letters/numbers/placeholder and spaces; drop punctuation.
    text = re.sub(r"[^a-z0-9<>\s]+", " ", text)
    text = normalize_whitespace(text)
    return text


def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to detect text/label columns. Supports your sample header:
    - "Search Queries Dump"
    - "Query Type"
    """
    cols = {c.lower().strip(): c for c in df.columns}

    text_candidates = [
        "search queries dump",
        "search_query",
        "query",
        "text",
        "search",
        "search_queries",
    ]
    label_candidates = [
        "query type",
        "label",
        "intent",
        "class",
        "query_type",
    ]

    text_col = next((cols[c] for c in text_candidates if c in cols), None)
    label_col = next((cols[c] for c in label_candidates if c in cols), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer columns. Found columns={list(df.columns)}. "
            "Please rename or pass --text-col and --label-col."
        )
    return text_col, label_col


def canonicalize_label(raw_label: str) -> str:
    if raw_label is None or (isinstance(raw_label, float) and math.isnan(raw_label)):
        raise ValueError("Encountered null label.")
    key = str(raw_label).strip().lower()
    if key not in LABEL_CANONICAL_MAP:
        # pass
        raise ValueError(f"Unknown label value: {raw_label!r}")
    return LABEL_CANONICAL_MAP[key]


def label_to_id_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {"product": 0, "outfit": 1}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.10
    val_size: float = 0.10
    seed: int = 42
    max_tries: int = 200
    tolerance: float = 0.03  # allowed max deviation from global label ratio per split
    group_digits: bool = True


def label_distribution(df: pd.DataFrame) -> Dict[str, float]:
    counts = df["label"].value_counts().to_dict()
    total = max(1, len(df))
    return {k: v / total for k, v in counts.items()}


def max_ratio_deviation(global_dist: Dict[str, float], split_dist: Dict[str, float]) -> float:
    keys = set(global_dist) | set(split_dist)
    return max(abs(split_dist.get(k, 0.0) - global_dist.get(k, 0.0)) for k in keys)


def group_aware_train_val_test_split(
        df: pd.DataFrame,
        cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    GroupShuffleSplit twice (train/temp then temp->val/test) and tries multiple seeds
    until label distribution drift is within cfg.tolerance for both val and test.
    """
    global_dist = label_distribution(df)
    groups = df["group_key"].values

    best = None
    best_score = float("inf")
    best_meta = {}

    for t in range(cfg.max_tries):
        seed = cfg.seed + t

        gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size + cfg.val_size, random_state=seed)
        train_idx, temp_idx = next(gss1.split(df, groups=groups))

        temp_df = df.iloc[temp_idx].copy()
        temp_groups = temp_df["group_key"].values

        # split temp into val and test
        test_frac_of_temp = cfg.test_size / (cfg.test_size + cfg.val_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac_of_temp, random_state=seed + 10_000)
        val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_groups))

        train_df = df.iloc[train_idx].copy()
        val_df = temp_df.iloc[val_idx_rel].copy()
        test_df = temp_df.iloc[test_idx_rel].copy()

        train_dist = label_distribution(train_df)
        val_dist = label_distribution(val_df)
        test_dist = label_distribution(test_df)

        score = (
                max_ratio_deviation(global_dist, train_dist)
                + max_ratio_deviation(global_dist, val_dist)
                + max_ratio_deviation(global_dist, test_dist)
        )

        ok = (
                max_ratio_deviation(global_dist, val_dist) <= cfg.tolerance
                and max_ratio_deviation(global_dist, test_dist) <= cfg.tolerance
        )

        meta = {
            "seed_used": seed,
            "global_dist": global_dist,
            "train_dist": train_dist,
            "val_dist": val_dist,
            "test_dist": test_dist,
            "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "tolerance": cfg.tolerance,
            "ok": ok,
            "score": score,
        }

        if ok:
            return train_df, val_df, test_df, meta

        if score < best_score:
            best_score = score
            best = (train_df, val_df, test_df)
            best_meta = meta

    logger.warning(
        "Could not meet tolerance after %d tries; using best attempt. best_score=%.6f",
        cfg.max_tries,
        best_score,
    )
    assert best is not None
    return best[0], best[1], best[2], best_meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='dataset/processed/search_queries_july_2025.csv',
                        help="Path to labeled CSV")
    parser.add_argument("--outdir", type=str, default="dataset/splits", help="Output directory")
    parser.add_argument("--text-col", type=str, default=None, help="Optional explicit text column name")
    parser.add_argument("--label-col", type=str, default=None, help="Optional explicit label column name")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-digits", action="store_true", help="Normalize digits in group_key")
    parser.add_argument("--no-group-digits", action="store_true", help="Do not normalize digits in group_key")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    group_digits = True
    if args.no_group_digits:
        group_digits = False
    elif args.group_digits:
        group_digits = True

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.text_col and args.label_col:
        text_col, label_col = args.text_col, args.label_col
    else:
        text_col, label_col = infer_columns(df)

    logger.info("Using text_col=%r label_col=%r", text_col, label_col)

    df = df[[text_col, label_col]].rename(columns={text_col: "text_raw", label_col: "label_raw"}).copy()
    df["text"] = df["text_raw"].map(clean_text_for_model)
    df["label"] = df["label_raw"].map(canonicalize_label)

    # Drop empties
    before = len(df)
    df = df[df["text"].astype(str).str.len() > 0].copy()
    after = len(df)
    if after < before:
        logger.warning("Dropped %d rows with empty text.", before - after)

    # Group key to reduce leakage
    df["group_key"] = df["text"].map(lambda s: normalize_for_grouping(s, group_digits=group_digits))

    # Basic report
    label2id, id2label = label_to_id_maps()
    df["label_id"] = df["label"].map(label2id)

    # Duplicates (exact and grouping-based)
    exact_dup_count = int(df.duplicated(subset=["text"]).sum())
    group_dup_count = int(df.duplicated(subset=["group_key"]).sum())

    cfg = SplitConfig(
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        seed=int(args.seed),
        group_digits=group_digits,
    )

    train_df, val_df, test_df, split_meta = group_aware_train_val_test_split(df, cfg)

    # Save outputs
    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"

    keep_cols = ["text", "label", "label_id", "group_key"]
    train_df[keep_cols].to_csv(train_path, index=False)
    val_df[keep_cols].to_csv(val_path, index=False)
    test_df[keep_cols].to_csv(test_path, index=False)

    (outdir / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, indent=2),
        encoding="utf-8",
    )

    report = {
        "input_rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "exact_duplicate_text_rows": exact_dup_count,
        "group_duplicate_rows": group_dup_count,
        "split": split_meta,
        "columns": {"text_col": text_col, "label_col": label_col},
        "group_digits": group_digits,
    }
    (outdir / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Wrote: %s", train_path)
    logger.info("Wrote: %s", val_path)
    logger.info("Wrote: %s", test_path)
    logger.info("Wrote: %s", outdir / "label_map.json")
    logger.info("Wrote: %s", outdir / "data_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
