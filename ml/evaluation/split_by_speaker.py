#!/usr/bin/env python3
"""Build deterministic speaker/duplicate-group-safe train/validation/test splits.

Manifest must be a CSV with speaker_id and label columns. Optional
duplicate_group_id links confirmed duplicate or near-duplicate recordings,
even where speaker metadata differs. All linked recordings stay in one split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Input CSV manifest")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV with split labels")
    parser.add_argument("--speaker-col", default="speaker_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--duplicate-col", default="duplicate_group_id",
                        help="Optional column; ignored if not present")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _split_indices(frame: pd.DataFrame, groups: pd.Series, seed: int) -> tuple[list[int], list[int], str]:
    """Return 80/20 train/holdout indices where feasible, retaining group isolation."""
    labels = frame["__split_label"].astype(str).to_numpy()
    group_values = groups.to_numpy()
    unique_groups = len(set(group_values))
    if unique_groups < 2:
        raise ValueError("At least two independent speaker/duplicate groups are required.")
    try:
        folds = min(5, unique_groups)
        if folds >= 2:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            train_pos, holdout_pos = next(splitter.split(frame, labels, group_values))
            return frame.index[train_pos].tolist(), frame.index[holdout_pos].tolist(), "StratifiedGroupKFold"
    except (ValueError, IndexError):
        pass
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_pos, holdout_pos = next(splitter.split(frame, labels, group_values))
    return frame.index[train_pos].tolist(), frame.index[holdout_pos].tolist(), "GroupShuffleSplit (non-stratified fallback)"


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    for col in (args.speaker_col, args.label_col):
        if col not in frame.columns:
            raise SystemExit(f"Required column missing from manifest: {col}")
    if frame.empty:
        raise SystemExit("The manifest contains no rows.")
    if (frame[args.speaker_col].str.strip() == "").any():
        raise SystemExit("Every row must have a non-empty speaker_id; missing IDs cannot be made leakage-safe.")
    if (frame[args.label_col].str.strip() == "").any():
        raise SystemExit("Every row must have a non-empty target label.")

    uf = UnionFind()
    for i, row in frame.iterrows():
        speaker_token = "speaker:" + row[args.speaker_col].strip()
        uf.find(speaker_token)
        duplicate = row.get(args.duplicate_col, "").strip() if args.duplicate_col in frame.columns else ""
        if duplicate:
            uf.union(speaker_token, "duplicate:" + duplicate)
    frame["__group"] = [
        uf.find("speaker:" + value.strip()) for value in frame[args.speaker_col].tolist()
    ]
    frame["__split_label"] = frame[args.label_col].astype(str)

    train_val_idx, test_idx, outer_method = _split_indices(frame, frame["__group"], args.seed)
    train_val = frame.loc[train_val_idx].copy()
    if len(set(train_val["__group"])) < 2:
        raise SystemExit("Not enough independent groups remain to create train and validation splits.")
    _, val_idx, inner_method = _split_indices(train_val, train_val["__group"], args.seed + 1)
    val_set = set(val_idx)
    test_set = set(test_idx)
    train_set = set(train_val_idx) - val_set
    frame["split"] = ""
    frame.loc[list(train_set), "split"] = "train"
    frame.loc[list(val_set), "split"] = "validation"
    frame.loc[list(test_set), "split"] = "test"

    # Defensive leakage assertions apply to all connected speaker/duplicate groups.
    split_sets = {name: set(frame.loc[frame["split"] == name, "__group"]) for name in ("train", "validation", "test")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_sets[a] & split_sets[b]
        if overlap:
            raise RuntimeError(f"Leakage error: groups overlap between {a} and {b}: {len(overlap)}")
    if (frame["split"] == "").any():
        raise RuntimeError("One or more manifest rows were not assigned to a split.")

    frame.drop(columns=["__group", "__split_label"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Rows written: {len(frame)}")
    print(f"Train/validation/test rows: {(frame['split']=='train').sum()} / {(frame['split']=='validation').sum()} / {(frame['split']=='test').sum()}")
    print(f"Unique speakers: {frame[args.speaker_col].nunique()}")
    print(f"Split methods: outer={outer_method}; validation={inner_method}")
    print("Leakage assertion: PASS (speaker and declared duplicate groups are mutually exclusive across partitions)")
    print("Reminder: identify near-duplicate recordings and populate duplicate_group_id before splitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
