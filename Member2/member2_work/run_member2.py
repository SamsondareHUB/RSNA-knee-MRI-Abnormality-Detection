"""
run_member2.py — Member 2 pipeline.

Patient-level split + label distribution.

Correctly handles unlabeled studies:
- split_assignments.csv keeps ALL studies (Member 3 needs them)
- label_distribution.csv reports gold-labeled rates + overall rates
- split_summary.json reports labeled vs unlabeled counts

Run:
    python run_member2.py
"""

import json
import sys
import os

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:
    MultilabelStratifiedKFold = None


from config import (
    DATA_ROOT, LABELS_CSV, SERIES_CSV,
    STUDY_ID_COL, PATIENT_ID_COL,
    MANUFACTURER_COL, MODEL_COL,
    LABELS, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    OUTPUT_DIR,
    N_FOLDS,
    USE_CV,
)


# ============================================================
# STEP 0 — VERIFY INPUTS
# ============================================================
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


# ============================================================
# STEP 1 — LOAD LABELS
# ============================================================
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

# ---------------------------------------------------------------------------
# Helper functions for cross‑validation (Option C)
# ---------------------------------------------------------------------------

def get_gold_mask(df):
    """Return a boolean Series where a row has at least one non‑NaN label."""
    return df[LABELS].notna().any(axis=1)

def create_cv_folds(gold_df, n_folds, seed):
    """Create stratified CV folds on the gold‑labeled subset.
    Returns a list of (train_idx, test_idx) tuples.
    Falls back to random splits if MultilabelStratifiedKFold is unavailable.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for CV")
    # Build multilabel binary matrix for stratification
    y = gold_df[LABELS].notna().astype(int).values
    if MultilabelStratifiedKFold:
        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(mskf.split(gold_df, y))
        return [(train_idx, test_idx) for train_idx, test_idx in splits]
    else:
        # Simple random shuffling fallback
        rng = np.random.RandomState(seed)
        indices = np.arange(len(gold_df))
        rng.shuffle(indices)
        fold_sizes = (len(gold_df) // n_folds) * np.ones(n_folds, dtype=int)
        fold_sizes[:len(gold_df) % n_folds] += 1
        folds = []
        current = 0
        for size in fold_sizes:
            start, stop = current, current + size
            test_idx = indices[start:stop]
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            folds.append((train_idx, test_idx))
            current = stop
        return folds

def process_fold(fold_id, train_df, test_df, output_root):
    """Process a single CV fold and write artefacts.
    - `fold_id` is an integer (0‑based).
    - `train_df` and `test_df` are DataFrames containing the respective rows.
    - `output_root` is the base OUTPUT_DIR (e.g., OUTPUT_DIR / "cv_folds").
    """
    # Assign split identifiers
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Combine and annotate with fold number
    combined = pd.concat([train_df, test_df])
    combined["fold"] = fold_id

    # Verify no patient leakage across splits
    if not verify_no_leakage(combined):
        raise RuntimeError(f"Leakage detected in CV fold {fold_id}")

    # Compute distribution and save outputs
    dist = compute_distribution(combined)
    fold_dir = output_root / f"fold_{fold_id}"
    save_outputs(combined, dist, fold_dir)
    print(f"Finished CV fold {fold_id}, outputs in {fold_dir}\n")

# ============================================================
# STEP 2 — LOAD SERIES METADATA
# ============================================================
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

# ---------------------------------------------------------------------------
# Helper functions for cross‑validation (Option C)
# ---------------------------------------------------------------------------

def get_gold_mask(df):
    """Return a boolean Series where a row has at least one non‑NaN label."""
    return df[LABELS].notna().any(axis=1)

def create_cv_folds(gold_df, n_folds, seed):
    """Create stratified CV folds on the gold‑labeled subset.
    Returns a list of (train_idx, test_idx) tuples.
    Falls back to random splits if MultilabelStratifiedKFold is unavailable.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for CV")
    # Build multilabel binary matrix for stratification
    y = gold_df[LABELS].notna().astype(int).values
    if MultilabelStratifiedKFold:
        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(mskf.split(gold_df, y))
        return [(train_idx, test_idx) for train_idx, test_idx in splits]
    else:
        # Simple random shuffling fallback
        rng = np.random.RandomState(seed)
        indices = np.arange(len(gold_df))
        rng.shuffle(indices)
        fold_sizes = (len(gold_df) // n_folds) * np.ones(n_folds, dtype=int)
        fold_sizes[:len(gold_df) % n_folds] += 1
        folds = []
        current = 0
        for size in fold_sizes:
            start, stop = current, current + size
            test_idx = indices[start:stop]
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            folds.append((train_idx, test_idx))
            current = stop
        return folds

def process_fold(fold_id, train_df, test_df, output_root):
    """Process a single CV fold and write artefacts.
    - `fold_id` is an integer (0‑based).
    - `train_df` and `test_df` are DataFrames containing the respective rows.
    - `output_root` is the base OUTPUT_DIR (e.g., OUTPUT_DIR / "cv_folds").
    """
    # Assign split identifiers
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Combine and annotate with fold number
    combined = pd.concat([train_df, test_df])
    combined["fold"] = fold_id

    # Verify no patient leakage across splits
    if not verify_no_leakage(combined):
        raise RuntimeError(f"Leakage detected in CV fold {fold_id}")

    # Compute distribution and save outputs
    dist = compute_distribution(combined)
    fold_dir = output_root / f"fold_{fold_id}"
    save_outputs(combined, dist, fold_dir)
    print(f"Finished CV fold {fold_id}, outputs in {fold_dir}\n")

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

    n_pat  = df["patient_id"].nunique()
    n_scan = df["scanner_id"].nunique()
    print(f"      unique patients: {n_pat}")
    print(f"      unique scanners: {n_scan}\n")

    if n_pat < 3:
        raise ValueError(f"Only {n_pat} patient groups — split will fail")

    return df

# ---------------------------------------------------------------------------
# Helper functions for cross‑validation (Option C)
# ---------------------------------------------------------------------------

def get_gold_mask(df):
    """Return a boolean Series where a row has at least one non‑NaN label."""
    return df[LABELS].notna().any(axis=1)

def create_cv_folds(gold_df, n_folds, seed):
    """Create stratified CV folds on the gold‑labeled subset.
    Returns a list of (train_idx, test_idx) tuples.
    Falls back to random splits if MultilabelStratifiedKFold is unavailable.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for CV")
    # Build multilabel binary matrix for stratification
    y = gold_df[LABELS].notna().astype(int).values
    if MultilabelStratifiedKFold:
        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(mskf.split(gold_df, y))
        return [(train_idx, test_idx) for train_idx, test_idx in splits]
    else:
        # Simple random shuffling fallback
        rng = np.random.RandomState(seed)
        indices = np.arange(len(gold_df))
        rng.shuffle(indices)
        fold_sizes = (len(gold_df) // n_folds) * np.ones(n_folds, dtype=int)
        fold_sizes[:len(gold_df) % n_folds] += 1
        folds = []
        current = 0
        for size in fold_sizes:
            start, stop = current, current + size
            test_idx = indices[start:stop]
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            folds.append((train_idx, test_idx))
            current = stop
        return folds

def process_fold(fold_id, train_df, test_df, output_root):
    """Process a single CV fold and write artefacts.
    - `fold_id` is an integer (0‑based).
    - `train_df` and `test_df` are DataFrames containing the respective rows.
    - `output_root` is the base OUTPUT_DIR (e.g., OUTPUT_DIR / "cv_folds").
    """
    # Assign split identifiers
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Combine and annotate with fold number
    combined = pd.concat([train_df, test_df])
    combined["fold"] = fold_id

    # Verify no patient leakage across splits
    if not verify_no_leakage(combined):
        raise RuntimeError(f"Leakage detected in CV fold {fold_id}")

    # Compute distribution and save outputs
    dist = compute_distribution(combined)
    fold_dir = output_root / f"fold_{fold_id}"
    save_outputs(combined, dist, fold_dir)
    print(f"Finished CV fold {fold_id}, outputs in {fold_dir}\n")

# ============================================================
# STEP 5 — PATIENT-LEVEL SPLIT
# ============================================================
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
    """
    Compute label distribution with two denominators:

    1. GOLD-LABELED  — only studies with at least one non-NaN label
    2. OVERALL       — all studies (NaN treated as 0)
    """
    print("[6b/7] Computing distribution...")

    # Identify gold-labeled studies
    label_frame   = df[LABELS]
    has_any_label = label_frame.notna().any(axis=1)
    n_total = len(df)
    n_gold  = int(has_any_label.sum())

    print(f"      total studies:        {n_total}")
    print(f"      gold-labeled studies: {n_gold}")
    print(f"      unlabeled studies:    {n_total - n_gold}\n")

    rows = []
    for label in LABELS:
        row = {"Label": label}

        # ---- OVERALL (NaN → 0) ----
        series_all = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
        for split in ["train", "val", "test"]:
            sub   = df[df["split"] == split]
            idx   = sub.index
            pos   = int(series_all.loc[idx].sum())
            total = len(sub)
            row[f"{split}_pos_overall"] = pos
            row[f"{split}_pct_overall"] = round(100 * pos / total, 1) if total else 0.0
        row["overall_pos_overall"] = int(series_all.sum())
        row["overall_pct_overall"] = round(
            100 * series_all.sum() / n_total, 1
        ) if n_total else 0.0

        # ---- GOLD-LABELED ONLY ----
        series_gold = pd.to_numeric(df[label], errors="coerce")
        for split in ["train", "val", "test"]:
            mask  = (df["split"] == split) & series_gold.notna()
            n_lab = int(mask.sum())
            pos   = int((series_gold[mask] == 1).sum())
            row[f"{split}_pos_gold"] = pos
            row[f"{split}_n_gold"]   = n_lab
            row[f"{split}_pct_gold"] = round(100 * pos / n_lab, 1) if n_lab else 0.0

        gold_mask = series_gold.notna()
        n_lab_all = int(gold_mask.sum())
        pos_all   = int((series_gold[gold_mask] == 1).sum())
        row["overall_pos_gold"] = pos_all
        row["overall_n_gold"]   = n_lab_all
        row["overall_pct_gold"] = round(
            100 * pos_all / n_lab_all, 1
        ) if n_lab_all else 0.0

        rows.append(row)

    print("      done\n")
    return pd.DataFrame(rows)


# ============================================================
# STEP 7 — SAVE OUTPUTS
# ============================================================
def save_outputs(df, dist, output_dir):
    print("[7/7] Saving...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. split_assignments.csv ----
    # All studies (labeled + unlabeled) — Member 3 needs the full list
    split_path = output_dir / "split_assignments.csv"
    cols = [STUDY_ID_COL, "patient_id", "scanner_id", "split"] + LABELS
    if "fold" in df.columns:
        cols = ["fold"] + cols
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(split_path, index=False)
    print(f"      {split_path}")

    # ---- 2. label_distribution.csv ----
    dist_path = output_dir / "label_distribution.csv"
    dist.to_csv(dist_path, index=False)
    print(f"      {dist_path}")

    # ---- 3. split_summary.json ----
    label_frame = df[LABELS]
    n_gold = int(label_frame.notna().any(axis=1).sum())

    summary = {
        "generated_at":      datetime.now().isoformat(),
        "seed":              SEED,
        "split_method":      "patient-level + scanner-grouped",
        "total_studies":     len(df),
        "total_patients":    int(df["patient_id"].nunique()),
        "total_scanners":    int(df["scanner_id"].nunique()),
        "labeled_studies":   n_gold,
        "unlabeled_studies": len(df) - n_gold,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        sub             = df[df["split"] == split]
        sub_label_frame = sub[LABELS]
        sub_gold        = int(sub_label_frame.notna().any(axis=1).sum())

        # Gold-only rates per label
        gold_rates = {}
        for label in LABELS:
            series = pd.to_numeric(sub[label], errors="coerce")
            mask   = series.notna()
            n_lab  = int(mask.sum())
            pos    = int((series[mask] == 1).sum())
            gold_rates[label] = {
                "n_labeled":  n_lab,
                "n_positive": pos,
                "rate_pct":   round(100 * pos / n_lab, 1) if n_lab else 0.0,
            }

        summary["splits"][split] = {
            "studies":            len(sub),
            "patients":           int(sub["patient_id"].nunique()),
            "pct":                round(100 * len(sub) / len(df), 1),
            "labeled_studies":    sub_gold,
            "unlabeled_studies":  len(sub) - sub_gold,
            "gold_labeled_rates": gold_rates,
        }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      {summary_path}\n")
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("MEMBER 2 — END-TO-END")
    print("=" * 60)

    has_series = verify_inputs()
    labels_df  = load_labels()
    series_df  = load_series(has_series)
    merged     = merge_data(labels_df, series_df)
    merged     = add_group_ids(merged)
    # Decide whether to run cross‑validation (Option C) or the standard single split
    use_cv = ("--cross-validation" in sys.argv) or (os.getenv("USE_CV", "0").lower() in ("1", "true", "yes"))
    if use_cv:
        print("[5/7] Performing 5‑fold cross‑validation (Option C)…")
        # Identify gold‑labeled rows
        gold_mask = get_gold_mask(merged)
        gold_df = merged[gold_mask]
        # Create folds
        cv_folds = create_cv_folds(gold_df, N_FOLDS, SEED)
        # Base directory for CV outputs
        cv_root = OUTPUT_DIR / "cv_folds"
        for i, (train_idx, test_idx) in enumerate(cv_folds):
            # Gold‑labeled training and test sets
            train_gold = gold_df.iloc[train_idx]
            test_gold = gold_df.iloc[test_idx]
            # Full training set includes all unlabeled rows + training gold rows
            train_full = pd.concat([merged[~gold_mask], train_gold])
            # Process this fold
            process_fold(i, train_full, test_gold, cv_root)
        # After CV, exit (no further single‑split reporting)
        return
    else:
        merged = create_splits(merged, seed=SEED)

        if not verify_no_leakage(merged):
            raise RuntimeError("Leakage detected!")

        dist    = compute_distribution(merged)
        summary = save_outputs(merged, dist, OUTPUT_DIR)

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
        print(f"  {split:6s}: {info['studies']:5d} studies "
            f"({info['labeled_studies']} labeled) | "
            f"{info['pct']:5.1f}%")

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
    print("=" * 60)


if __name__ == "__main__":
    main()