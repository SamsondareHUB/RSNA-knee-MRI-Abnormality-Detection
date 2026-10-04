"""
run_member2.py — Member 2 pipeline.

Patient-level split + label distribution.

Run:
python run_member2.py
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from config import (
    DATA_ROOT, LABELS_CSV, SERIES_CSV,
    STUDY_ID_COL, PATIENT_ID_COL,
    MANUFACTURER_COL, MODEL_COL,
    LABELS, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    OUTPUT_DIR,
)

# STEP 0 — VERIFY INPUTS

def verify_inputs():
    print("=" * 60)
    print("STEP 0 — VERIFY INPUTS")
    print("=" * 60)
    print(f"LABELS_CSV: {LABELS_CSV}  exists={LABELS_CSV.exists()}")
    print(f"SERIES_CSV: {SERIES_CSV}  exists={SERIES_CSV.exists()}")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing: {LABELS_CSV}")

    has_series = SERIES_CSV.exists()
    if not has_series:
        print("WARNING: Series CSV missing — skipping metadata merge")
    print("Inputs OK\n")
    return has_series

# STEP 1 — LOAD LABELS

def load_labels():
    print("[1/7] Loading labels CSV...")
    df = pd.read_csv(LABELS_CSV)
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


# STEP 2 — LOAD SERIES METADATA

def load_series(has_series):
    print("[2/7] Loading series metadata...")
    if not has_series:
        print("      skipped\n")
        return pd.DataFrame({STUDY_ID_COL: []})

    df = pd.read_csv(SERIES_CSV)
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


# STEP 3 — MERGE

def merge_data(labels_df, series_df):
    print("[3/7] Merging...")
    if series_df.empty:
        print("      no metadata\n")
        return labels_df.copy()
    merged = labels_df.merge(series_df, on=STUDY_ID_COL, how="left")
    print(f"      merged rows: {len(merged)}\n")
    return merged


# STEP 4 — GROUP IDs (FIXED)

def add_group_ids(df):
    """
    Create patient_id and scanner_id.

    FIXED: When PATIENT_ID_COL is None, use StudyInstanceUID directly
    (RSNA Knee: 1 study = 1 patient). Do NOT use rsplit on the UID,
    because all StudyInstanceUIDs share the same prefix.
    """
    print("[4/7] Creating group IDs...")
    df = df.copy()

    # ---- Patient ID ----
    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        df["patient_id"] = df[PATIENT_ID_COL].astype(str)
        print(f"      patient_id from '{PATIENT_ID_COL}'")
    else:
        # RSNA Knee: each StudyInstanceUID represents one patient
        df["patient_id"] = df[STUDY_ID_COL].astype(str)
        print("      patient_id = StudyInstanceUID (1 study = 1 patient)")

    # ---- Scanner ID ----
    if MANUFACTURER_COL and MANUFACTURER_COL in df.columns:
        mfr = df[MANUFACTURER_COL].astype(str)
    else:
        mfr = pd.Series(["unknown"] * len(df), index=df.index)

    if MODEL_COL and MODEL_COL in df.columns:
        mdl = df[MODEL_COL].astype(str)
    else:
        mdl = pd.Series(["unknown"] * len(df), index=df.index)

    df["scanner_id"] = mfr + "_" + mdl

    n_pat  = df["patient_id"].nunique()
    n_scan = df["scanner_id"].nunique()
    print(f"      unique patients: {n_pat}")
    print(f"      unique scanners: {n_scan}\n")

    if n_pat < 3:
        raise ValueError(
            f"Only {n_pat} patient groups — split will fail.\n"
            f"Sample IDs: {df['patient_id'].head(5).tolist()}"
        )

    return df

# STEP 5 — PATIENT-LEVEL SPLIT

def create_splits(df, seed):
    print(f"[5/7] Splitting (seed={seed})...")
    df = df.copy().reset_index(drop=True)

    groups     = df["patient_id"]
    temp_ratio = VAL_RATIO + TEST_RATIO

    gss1 = GroupShuffleSplit(n_splits=1,
                            test_size=temp_ratio,
                            random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    temp_df  = df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1,
                            test_size=0.5,
                            random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df,
                                        groups=temp_df["patient_id"]))

    train_df["split"] = "train"
    val_df  = temp_df.iloc[val_idx].copy();  val_df["split"]  = "val"
    test_df = temp_df.iloc[test_idx].copy(); test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    print(f"      train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n")
    return result



# STEP 6 — LEAKAGE CHECK + DISTRIBUTION

def verify_no_leakage(df):
    print("[6/7] Verifying no leakage...")
    leakage = df.groupby("patient_id")["split"].nunique()
    bad = leakage[leakage > 1]
    if len(bad) > 0:
        print(f"      LEAKAGE in {len(bad)} patients\n")
        return False
    print(f"      clean ({df['patient_id'].nunique()} patients)\n")
    return True


def compute_distribution(df):
    print("[6b/7] Computing distribution...")
    rows = []
    for label in LABELS:
        row = {"Label": label}
        for split in ["train", "val", "test"]:
            sub   = df[df["split"] == split]
            pos   = int(sub[label].sum())
            total = len(sub)
            row[f"{split}_pos"] = pos
            row[f"{split}_neg"] = total - pos
            row[f"{split}_pct"] = round(100 * pos / total, 1) if total else 0.0
        row["overall_pos"] = int(df[label].sum())
        row["overall_pct"] = round(100 * df[label].sum() / len(df), 1)
        rows.append(row)
    print("      done\n")
    return pd.DataFrame(rows)


# STEP 7 — SAVE OUTPUTS

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
    summary = {
        "generated_at":   datetime.now().isoformat(),
        "seed":           SEED,
        "split_method":   "patient-level + scanner-grouped",
        "total_studies":  len(df),
        "total_patients": int(df["patient_id"].nunique()),
        "total_scanners": int(df["scanner_id"].nunique()),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        summary["splits"][split] = {
            "studies":  len(sub),
            "patients": int(sub["patient_id"].nunique()),
            "pct":      round(100 * len(sub) / len(df), 1),
            "label_positive_rates": {
                label: round(100 * sub[label].sum() / len(sub), 1)
                for label in LABELS
            },
        }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      {summary_path}\n")
    return summary



# MAIN

def main():
    print("=" * 60)
    print("MEMBER 2 — END-TO-END")
    print("=" * 60)

    has_series = verify_inputs()
    labels_df  = load_labels()
    series_df  = load_series(has_series)
    merged     = merge_data(labels_df, series_df)
    merged     = add_group_ids(merged)
    merged     = create_splits(merged, seed=SEED)

    if not verify_no_leakage(merged):
        raise RuntimeError("Leakage detected!")

    dist    = compute_distribution(merged)
    summary = save_outputs(merged, dist, OUTPUT_DIR)

    print("=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total studies:  {summary['total_studies']}")
    print(f"Total patients: {summary['total_patients']}")
    print(f"Total scanners: {summary['total_scanners']}\n")
    for split, info in summary["splits"].items():
        print(f"  {split:6s}: {info['studies']:5d} studies | "
            f"{info['patients']:5d} patients | {info['pct']:5.1f}%")

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)
    print(dist[["Label", "train_pct", "val_pct", "test_pct"]]
        .to_string(index=False))

    print("\n" + "=" * 60)
    print("DELIVERABLES")
    print("=" * 60)
    print(f"  {OUTPUT_DIR}/split_assignments.csv")
    print(f"  {OUTPUT_DIR}/label_distribution.csv")
    print(f"  {OUTPUT_DIR}/split_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()