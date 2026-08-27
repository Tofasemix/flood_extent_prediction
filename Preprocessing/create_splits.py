import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT_DIR / "clean_tabular_data.csv"
FLOOD_DIR = ROOT_DIR / "Dataset" / "FLOOD"
OUTPUT_DIR = ROOT_DIR / "splits"

RANDOM_SEED = 42
TRAIN_SIZE = 117
VAL_SIZE = 25
TEST_SIZE = 25


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_flood(cyclone_id):
    flood_path = FLOOD_DIR / f"FLOOD_{cyclone_id}.png"

    if not flood_path.exists():
        raise FileNotFoundError(
            f"Missing flood mask for '{cyclone_id}': {flood_path}"
        )

    flood_img = cv2.imread(str(flood_path), cv2.IMREAD_GRAYSCALE)

    if flood_img is None:
        raise RuntimeError(
            f"OpenCV could not read flood mask for '{cyclone_id}': {flood_path}"
        )

    # Must match CycloneFloodDataset target definition:
    # flood_mask = (flood_img > 0)
    return int(np.any(flood_img > 0))


def validate_source_dataframe(df):
    required_columns = {"name_date", "pressure", "vmax", "wind"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Source CSV is missing required columns: {sorted(missing)}"
        )

    if df["name_date"].isna().any():
        raise ValueError("Source CSV contains missing name_date values.")

    if df["name_date"].duplicated().any():
        duplicates = df.loc[df["name_date"].duplicated(), "name_date"].tolist()
        raise ValueError(
            f"Duplicate name_date identifiers found: {duplicates}"
        )

    if len(df) != TRAIN_SIZE + VAL_SIZE + TEST_SIZE:
        raise ValueError(
            f"Expected {TRAIN_SIZE + VAL_SIZE + TEST_SIZE} rows for the "
            f"configured split sizes, but found {len(df)}."
        )


def stratified_split(df):
    labels = df["flood_positive"].to_numpy()

    # First split: exact 117 train / 50 temporary.
    first_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=TRAIN_SIZE,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    train_idx, temp_idx = next(first_split.split(df, labels))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Second split: exact 25 validation / 25 test.
    second_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    val_relative_idx, test_relative_idx = next(
        second_split.split(
            temp_df,
            temp_df["flood_positive"].to_numpy(),
        )
    )

    val_df = temp_df.iloc[val_relative_idx].copy()
    test_df = temp_df.iloc[test_relative_idx].copy()

    return train_df, val_df, test_df


def verify_split_integrity(source_df, train_df, val_df, test_df):
    source_ids = set(source_df["name_date"])
    train_ids = set(train_df["name_date"])
    val_ids = set(val_df["name_date"])
    test_ids = set(test_df["name_date"])

    if train_ids & val_ids:
        raise RuntimeError("Train and validation splits overlap.")
    if train_ids & test_ids:
        raise RuntimeError("Train and test splits overlap.")
    if val_ids & test_ids:
        raise RuntimeError("Validation and test splits overlap.")

    combined_ids = train_ids | val_ids | test_ids

    if combined_ids != source_ids:
        missing = sorted(source_ids - combined_ids)
        extra = sorted(combined_ids - source_ids)
        raise RuntimeError(
            f"Split membership does not match source data. "
            f"Missing={missing}, Extra={extra}"
        )

    if len(train_df) != TRAIN_SIZE:
        raise RuntimeError(f"Train split has {len(train_df)} rows.")
    if len(val_df) != VAL_SIZE:
        raise RuntimeError(f"Validation split has {len(val_df)} rows.")
    if len(test_df) != TEST_SIZE:
        raise RuntimeError(f"Test split has {len(test_df)} rows.")


def print_split_summary(name, df):
    positive = int(df["flood_positive"].sum())
    dry = int(len(df) - positive)
    positive_pct = 100.0 * positive / len(df)

    print(
        f"{name:<10} "
        f"total={len(df):>3} | "
        f"flood-positive={positive:>3} | "
        f"dry={dry:>2} | "
        f"positive={positive_pct:>6.2f}%"
    )


def main():
    print("=" * 72)
    print("Creating deterministic stratified cyclone splits")
    print("=" * 72)
    print(f"Source CSV : {SOURCE_CSV}")
    print(f"Flood masks: {FLOOD_DIR}")
    print(f"Seed       : {RANDOM_SEED}")
    print(
        f"Target size: train={TRAIN_SIZE}, "
        f"val={VAL_SIZE}, test={TEST_SIZE}"
    )
    print()

    df = pd.read_csv(SOURCE_CSV)
    validate_source_dataframe(df)

    print("Reading ground-truth flood masks...")
    df["flood_positive"] = [
        has_flood(cyclone_id)
        for cyclone_id in df["name_date"]
    ]

    print()
    print_split_summary("Full", df)

    train_df, val_df, test_df = stratified_split(df)
    verify_split_integrity(df, train_df, val_df, test_df)

    print_split_summary("Train", train_df)
    print_split_summary("Validation", val_df)
    print_split_summary("Test", test_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_columns = [
        column
        for column in df.columns
        if column != "flood_positive"
    ]

    # Sort files after membership has been assigned so Git diffs remain stable.
    train_out = train_df.sort_values("name_date")
    val_out = val_df.sort_values("name_date")
    test_out = test_df.sort_values("name_date")

    train_out[original_columns].to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )
    val_out[original_columns].to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )
    test_out[original_columns].to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    manifest = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    manifest = manifest[
        ["name_date", "split", "flood_positive"]
    ].sort_values(["split", "name_date"])
    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)

    metadata = {
        "source_csv": str(SOURCE_CSV.relative_to(ROOT_DIR)),
        "source_csv_sha256": sha256_file(SOURCE_CSV),
        "random_seed": RANDOM_SEED,
        "stratification_target": "flood_positive",
        "flood_positive_definition": "any pixel > 0 in FLOOD_<name_date>.png",
        "split_sizes": {
            "train": TRAIN_SIZE,
            "val": VAL_SIZE,
            "test": TEST_SIZE,
        },
        "counts": {
            "full": {
                "total": int(len(df)),
                "flood_positive": int(df["flood_positive"].sum()),
                "dry": int(len(df) - df["flood_positive"].sum()),
            },
            "train": {
                "total": int(len(train_df)),
                "flood_positive": int(train_df["flood_positive"].sum()),
                "dry": int(len(train_df) - train_df["flood_positive"].sum()),
            },
            "val": {
                "total": int(len(val_df)),
                "flood_positive": int(val_df["flood_positive"].sum()),
                "dry": int(len(val_df) - val_df["flood_positive"].sum()),
            },
            "test": {
                "total": int(len(test_df)),
                "flood_positive": int(test_df["flood_positive"].sum()),
                "dry": int(len(test_df) - test_df["flood_positive"].sum()),
            },
        },
    }

    with open(OUTPUT_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print()
    print("Integrity checks: PASSED")
    print(f"Saved splits to: {OUTPUT_DIR}")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print("  - split_manifest.csv")
    print("  - split_metadata.json")


if __name__ == "__main__":
    main()
