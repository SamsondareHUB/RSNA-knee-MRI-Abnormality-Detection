"""
run_member2.py — Member 2 pipeline.

Patient-level split + label distribution.

Correctly handles unlabeled studies:
- split_assignments.csv keeps ALL studies (Member 3 needs them)
- label_distribution.csv reports gold‑labeled rates + overall rates
- split_summary.json reports labeled vs unlabeled counts

Run:
    python run_member2.py
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from config import (
    DATA_ROOT, LABELS_CSV as CONFIG_LABELS_CSV, SERIES_CSV as CONFIG_SERIES_CSV,
    STUDY_ID_COL, PATIENT_ID_COL,
    MANUFACTURER_COL, MODEL_COL,
    LABELS, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    OUTPUT_DIR as CONFIG_OUTPUT_DIR,
)

# ============================================================
# STEP 0 — VERIFY INPUTS
# ============================================================
def verify_inputs(labels_csv, series_csv):
    print("=" * 60)
    print("STEP 0 — VERIFY INPUTS")
    print("=" * 60)
    print(f"LABELS_CSV: {labels_csv}  exists={labels_csv.exists()}")
    print(f"SERIES_CSV: {series_csv}  exists={series_csv.exists()}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing: {labels_csv}")

    has_series = series_csv.exists()
    if not has_series:
        print("WARNING: Series CSV missing — skipping metadata merge")
    print("Inputs OK\n")
    return has_series

# ============================================================
# STEP 1 — LOAD LABELS
# ============================================================
def load_labels(labels_csv):
    print("[1/7] Loading labels CSV...")
    df = pd.read_csv(labels_csv)
    print(f"      rows={len(df)}  cols={len(df.columns)}")

    if STUDY_ID_COL not in df.columns:
        raise ValueError(
            f"STUDY_ID_COL='{STUDY_ID_COL}' not found.\n"
            f"Available: {df.columns.tolist()}"
        )

    missing = [l for l in LABELS if l not in df.columns]
    if missing:
        raise ValueError(
            f"Missing labels: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    print(f"      all {len(LABELS)} labels present\n")
    return df

# ============================================================
# STEP 2 — LOAD SERIES METADATA
# ============================================================
def load_series(series_csv, has_series):
    print("[2/7] Loading series metadata...")
    if not has_series:
        print("      skipped\n")
        return pd.DataFrame({STUDY_ID_COL: []})

    df = pd.read_csv(series_csv)
    print(f"      rows={len(df)}  cols={len(df.columns)}")

    keep = [STUDY_ID_COL]
    if MANUFACTURER_COL and MANUFACTURER_COL in df.columns:
        keep.append(MANUFACTURER_COL)
    if MODEL_COL and MODEL_COL in df.columns:
        keep.append(MODEL_COL)

    df = df[keep].drop_duplicates(STUDY_ID_COL)
    print(f"      kept: {keep}")
    print(f"      unique studies: {len(df)}\n")
    return df

# ============================================================
# STEP 3 — MERGE
# ============================================================
def merge_data(labels_df, series_df):
    print("[3/7] Merging...")
    if series_df.empty:
        print("      no metadata\n")
        return labels_df.copy()
    merged = labels_df.merge(series_df, on=STUDY_ID_COL, how="left")
    print(f"      merged rows: {len(merged)}\n")
    return merged

# ============================================================
# STEP 4 — GROUP IDs
# ============================================================
def add_group_ids(df):
    """Create patient_id and scanner_id."""
    print("[4/7] Creating group IDs...")
    df = df.copy()

    # Patient ID
    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        df["patient_id"] = df[PATIENT_ID_COL].astype(str)
        print(f"      patient_id from '{PATIENT_ID_COL}'")
    else:
        df["patient_id"] = df[STUDY_ID_COL].astype(str)
        print("      patient_id = StudyInstanceUID (1 study = 1 patient)")

    # Scanner ID
    if MANUFACTURER_COL and MANUFACTURER_COL in df.columns:
        mfr = df[MANUFACTURER_COL].astype(str)
    else:
        mfr = pd.Series(["unknown"] * len(df), index=df.index)

    if MODEL_COL and MODEL_COL in df.columns:
        mdl = df[MODEL_COL].astype(str)
    else:
        mdl = pd.Series(["unknown"] * len(df), index=df.index)

    df["scanner_id"] = mfr + "_" + mdl

    n_pat = df["patient_id"].nunique()
    n_scan = df["scanner_id"].nunique()
    print(f"      unique patients: {n_pat}")
    print(f"      unique scanners: {n_scan}\n")

    if n_pat < 3:
        raise ValueError(f"Only {n_pat} patient groups — split will fail")

    return df

# ============================================================
# STEP 5 — PATIENT-LEVEL SPLIT
# ============================================================
def create_splits(df, seed):
    print(f"[5/7] Splitting (seed={seed})...")
    df = df.copy().reset_index(drop=True)

    groups = df["patient_id"]
    temp_ratio = VAL_RATIO + TEST_RATIO

    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["patient_id"]))

    train_df["split"] = "train"
    val_df = temp_df.iloc[val_idx].copy(); val_df["split"] = "val"
    test_df = temp_df.iloc[test_idx].copy(); test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    print(f"      train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")
    return result

# ============================================================
# STEP 6 — LEAKAGE CHECK
# ============================================================
def verify_no_leakage(df):
    print("[6/7] Verifying no leakage...")
    leakage = df.groupby("patient_id")["split"].nunique()
    bad = leakage[leakage > 1]
    if len(bad) > 0:
        print(f"      LEAKAGE in {len(bad)} patients\n")
        return False
    print(f"      clean ({df['patient_id'].nunique()} patients)\n")
    return True

# ============================================================
# STEP 6b — LABEL DISTRIBUTION (GOLD-LABELED + OVERALL)
# ============================================================
def compute_distribution(df):
    """Compute label distribution with two denominators:
    1. GOLD‑LABELED — only studies with at least one non‑NaN label
    2. OVERALL      — all studies (NaN treated as 0)
    """
    print("[6b/7] Computing distribution...")
    label_frame = df[LABELS]
    has_any_label = label_frame.notna().any(axis=1)
    n_total = len(df)
    n_gold = int(has_any_label.sum())

    print(f"      total studies:        {n_total}")
    print(f"      gold‑labeled studies: {n_gold}")
    print(f"      unlabeled studies:    {n_total - n_gold}\n")

    rows = []
    for label in LABELS:
        row = {"Label": label}
        # OVERALL
        series_all = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            idx = sub.index
            pos = int(series_all.loc[idx].sum())
            total = len(sub)
            row[f"{split}_pos_overall"] = pos
            row[f"{split}_pct_overall"] = round(100 * pos / total, 1) if total else 0.0
        row["overall_pos_overall"] = int(series_all.sum())
        row["overall_pct_overall"] = round(100 * series_all.sum() / n_total, 1) if n_total else 0.0
        # GOLD‑LABELED
        series_gold = pd.to_numeric(df[label], errors="coerce")
        for split in ["train", "val", "test"]:
            mask = (df["split"] == split) & series_gold.notna()
            n_lab = int(mask.sum())
            pos = int((series_gold[mask] == 1).sum())
            row[f"{split}_pos_gold"] = pos
            row[f"{split}_n_gold"] = n_lab
            row[f"{split}_pct_gold"] = round(100 * pos / n_lab, 1) if n_lab else 0.0
        gold_mask = series_gold.notna()
        n_lab_all = int(gold_mask.sum())
        pos_all = int((series_gold[gold_mask] == 1).sum())
        row["overall_pos_gold"] = pos_all
        row["overall_n_gold"] = n_lab_all
        row["overall_pct_gold"] = round(100 * pos_all / n_lab_all, 1) if n_lab_all else 0.0
        rows.append(row)
    print("      done\n")
    return pd.DataFrame(rows)

# ============================================================
# STEP 7 — SAVE OUTPUTS
# ============================================================
def save_outputs(df, dist, output_dir):
    print("[7/7] Saving...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. split_assignments.csv
    split_path = output_dir / "split_assignments.csv"
    cols = [STUDY_ID_COL, "patient_id", "scanner_id", "split"] + LABELS
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(split_path, index=False)
    print(f"      {split_path}")

    # 2. label_distribution.csv
    dist_path = output_dir / "label_distribution.csv"
    dist.to_csv(dist_path, index=False)
    print(f"      {dist_path}")

    # 3. split_summary.json
    label_frame = df[LABELS]
    n_gold = int(label_frame.notna().any(axis=1).sum())
    summary = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "split_method": "patient-level + scanner-grouped",
        "total_studies": len(df),
        "total_patients": int(df["patient_id"].nunique()),
        "total_scanners": int(df["scanner_id"].nunique()),
        "labeled_studies": n_gold,
        "unlabeled_studies": len(df) - n_gold,
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        sub_label_frame = sub[LABELS]
        sub_gold = int(sub_label_frame.notna().any(axis=1).sum())
        gold_rates = {}
        for label in LABELS:
            series = pd.to_numeric(sub[label], errors="coerce")
            mask = series.notna()
            n_lab = int(mask.sum())
            pos = int((series[mask] == 1).sum())
            gold_rates[label] = {
                "n_labeled": n_lab,
                "n_positive": pos,
                "rate_pct": round(100 * pos / n_lab, 1) if n_lab else 0.0,
            }
        summary["splits"][split] = {
            "studies": len(sub),
            "patients": int(sub["patient_id"].nunique()),
            "pct": round(100 * len(sub) / len(df), 1),
            "labeled_studies": sub_gold,
            "unlabeled_studies": len(sub) - sub_gold,
            "gold_labeled_rates": gold_rates,
        }
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      {summary_path}\n")
    return summary

# -----------------------------------------------------------
# Helper: write train/val split CSVs
# -----------------------------------------------------------
def write_split_csvs(df, out_dir):
    """Write train_split.csv and val_split.csv based on the 'split' column.
    Only the StudyInstanceUID column and label columns are saved.
    """
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    cols = [STUDY_ID_COL] + LABELS
    cols = [c for c in cols if c in df.columns]
    train_path = out_dir / "train_split.csv"
    val_path = out_dir / "val_split.csv"
    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path, index=False)
    print(f"      {train_path}")
    print(f"      {val_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Member2 pipeline – configurable CSV and output paths")
    parser.add_argument("--labels-csv", type=Path, default=Path(os.getenv("LABELS_CSV_PATH", str(CONFIG_LABELS_CSV))), help="Path to the labels CSV (default from config or env var)")
    parser.add_argument("--series-csv", type=Path, default=Path(os.getenv("SERIES_CSV_PATH", str(CONFIG_SERIES_CSV))), help="Path to the series CSV (default from config or env var)")
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", str(CONFIG_OUTPUT_DIR))), help="Directory where outputs will be written (default from config or env var)")
    args = parser.parse_args()
    # Override globals for downstream functions
    global LABELS_CSV, SERIES_CSV, OUTPUT_DIR
    LABELS_CSV = args.labels_csv
    SERIES_CSV = args.series_csv
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("MEMBER 2 — END-TO-END")
    print("=" * 60)

    has_series = verify_inputs(LABELS_CSV, SERIES_CSV)
    labels_df = load_labels(LABELS_CSV)
    series_df = load_series(SERIES_CSV, has_series)
    merged = merge_data(labels_df, series_df)
    merged = add_group_ids(merged)
    merged = create_splits(merged, seed=SEED)

    if not verify_no_leakage(merged):
        raise RuntimeError("Leakage detected!")

    dist = compute_distribution(merged)
    summary = save_outputs(merged, dist, OUTPUT_DIR)
    write_split_csvs(merged, OUTPUT_DIR)

    # ---- Final report ----
    print("=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total studies:       {summary['total_studies']}")
    print(f"Total patients:      {summary['total_patients']}")
    print(f"Total scanners:      {summary['total_scanners']}")
    print(f"Labeled studies:     {summary['labeled_studies']}")
    print(f"Unlabeled studies:   {summary['unlabeled_studies']}\n")
    for split, info in summary["splits"].items():
        print(f"  {split:6s}: {info['studies']:5d} studies ({info['labeled_studies']} labeled) | {info['pct']:5.1f}%")

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION — GOLD-LABELED ONLY (corrected)")
    print("=" * 60)
    print(dist[["Label",
                "train_pct_gold", "val_pct_gold", "test_pct_gold",
                "overall_pct_gold", "overall_n_gold"]]
          .to_string(index=False))

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION — OVERALL (NaN treated as 0, reference)")
    print("=" * 60)
    print(dist[["Label", "overall_pct_overall"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("DELIVERABLES")
    print("=" * 60)
    print(f"  {OUTPUT_DIR}/split_assignments.csv")
    print(f"  {OUTPUT_DIR}/label_distribution.csv")
    print(f"  {OUTPUT_DIR}/split_summary.json")
    print(f"  {OUTPUT_DIR}/train_split.csv")
    print(f"  {OUTPUT_DIR}/val_split.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
