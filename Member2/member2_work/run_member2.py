"""
run_member2.py — Member 2 pipeline (redone split, corrected counts).

Enforces EXACT split counts:
    gold:      Train 30    | Val 14    | Test 14    = 58
    non-gold:  Train 3055  | Val 647   | Test 647   = 4349
    total:     Train 3085  | Val 661   | Test 661   = 4407

Outputs:
    outputs/split.csv               study_id, split, is_gold, 12 labels
    outputs/train_split.csv         train rows only
    outputs/val_split.csv           val rows only
    outputs/test_split.csv          test rows only
    outputs/label_distribution.csv  positives/negatives per label per split
    outputs/split_summary.json      full summary

Run:
    python run_member2.py
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from config import (
    DATA_ROOT, LABELS_CSV, SERIES_CSV,
    STUDY_ID_COL, PATIENT_ID_COL,
    MANUFACTURER_COL, MODEL_COL,
    LABELS, SEED,
    GOLD_TRAIN, GOLD_VAL, GOLD_TEST,
    NONGOLD_TRAIN, NONGOLD_VAL, NONGOLD_TEST,
    TOTAL_STUDIES, TOTAL_GOLD, TOTAL_NONGOLD,
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
        print("⚠ Series CSV missing — skipping metadata merge")
    print("✓ Inputs OK\n")
    return has_series



# STEP 1 — LOAD LABELS

def load_labels():
    print("[1/7] Loading labels CSV...")
    df = pd.read_csv(LABELS_CSV)
    print(f"      rows={len(df)}  cols={len(df.columns)}")
    print(f"      columns: {df.columns.tolist()}")

    if STUDY_ID_COL not in df.columns:
        raise ValueError(f"STUDY_ID_COL='{STUDY_ID_COL}' not found.")

    missing = [l for l in LABELS if l not in df.columns]
    if missing:
        raise ValueError(f"Missing labels: {missing}")

    print(f"      ✓ all {len(LABELS)} labels present\n")
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


# STEP 4 — GROUP IDs + is_gold FLAG

def add_group_ids(df):
    print("[4/7] Creating group IDs and is_gold flag...")
    df = df.copy()

    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        df["patient_id"] = df[PATIENT_ID_COL].astype(str)
    else:
        df["patient_id"] = df[STUDY_ID_COL].astype(str)

    if MANUFACTURER_COL and MANUFACTURER_COL in df.columns:
        mfr = df[MANUFACTURER_COL].astype(str)
    else:
        mfr = pd.Series(["unknown"] * len(df), index=df.index)

    if MODEL_COL and MODEL_COL in df.columns:
        mdl = df[MODEL_COL].astype(str)
    else:
        mdl = pd.Series(["unknown"] * len(df), index=df.index)

    df["scanner_id"] = mfr + "_" + mdl
    df["is_gold"]    = df[LABELS].notna().any(axis=1).astype(int)

    n_gold = int(df["is_gold"].sum())
    n_non  = len(df) - n_gold

    print(f"      unique patients: {df['patient_id'].nunique()}")
    print(f"      gold-labeled:    {n_gold}")
    print(f"      non-gold:        {n_non}\n")

    return df


# STEP 5 — EXACT-COUNT SPLIT

def create_splits(df, seed):
    """
    Exact-count split:
        gold:     30 / 14 / 14   (train / val / test)
        non-gold: 3055 / 647 / 647
    No leftovers. No off-by-one.
    """
    print("[5/7] Creating splits (exact counts)...")
    df = df.copy().reset_index(drop=True)

    # ---- Verify input counts match the target plan ----
    n_gold    = int(df["is_gold"].sum())
    n_nongold = len(df) - n_gold

    print(f"      found gold:     {n_gold}    (expected {TOTAL_GOLD})")
    print(f"      found non-gold: {n_nongold}    (expected {TOTAL_NONGOLD})")
    print(f"      found total:    {len(df)}    (expected {TOTAL_STUDIES})")

    if n_gold != TOTAL_GOLD:
        raise ValueError(
            f"Gold count mismatch: found {n_gold}, expected {TOTAL_GOLD}."
        )
    if n_nongold != TOTAL_NONGOLD:
        raise ValueError(
            f"Non-gold count mismatch: found {n_nongold}, expected {TOTAL_NONGOLD}."
        )

    gold_need    = GOLD_TRAIN + GOLD_VAL + GOLD_TEST
    nongold_need = NONGOLD_TRAIN + NONGOLD_VAL + NONGOLD_TEST

    if gold_need != n_gold:
        raise ValueError(f"Gold split mismatch: {gold_need} vs {n_gold}")
    if nongold_need != n_nongold:
        raise ValueError(f"Non-gold split mismatch: {nongold_need} vs {n_nongold}")

    # ---- Shuffle each group with the same seed ----
    rng = np.random.RandomState(seed)

    gold_df    = df[df["is_gold"] == 1].copy()
    nongold_df = df[df["is_gold"] == 0].copy()

    gold_order    = rng.permutation(len(gold_df))
    nongold_order = rng.permutation(len(nongold_df))

    gold_df    = gold_df.iloc[gold_order].reset_index(drop=True)
    nongold_df = nongold_df.iloc[nongold_order].reset_index(drop=True)

    # ---- Gold slices (exact) ----
    g_train = gold_df.iloc[0:GOLD_TRAIN].copy()
    g_val   = gold_df.iloc[GOLD_TRAIN:GOLD_TRAIN + GOLD_VAL].copy()
    g_test  = gold_df.iloc[GOLD_TRAIN + GOLD_VAL:GOLD_TRAIN + GOLD_VAL + GOLD_TEST].copy()

    # ---- Non-gold slices (exact) ----
    ng_train = nongold_df.iloc[0:NONGOLD_TRAIN].copy()
    ng_val   = nongold_df.iloc[NONGOLD_TRAIN:NONGOLD_TRAIN + NONGOLD_VAL].copy()
    ng_test  = nongold_df.iloc[NONGOLD_TRAIN + NONGOLD_VAL:NONGOLD_TRAIN + NONGOLD_VAL + NONGOLD_TEST].copy()

    # ---- Map StudyInstanceUID → split ----
    uid_to_split = {}
    for sub, split_name in [
        (g_train,  "train"),
        (g_val,    "val"),
        (g_test,   "test"),
        (ng_train, "train"),
        (ng_val,   "val"),
        (ng_test,  "test"),
    ]:
        for uid in sub[STUDY_ID_COL]:
            uid_to_split[uid] = split_name

    df["split"] = df[STUDY_ID_COL].map(uid_to_split)

    if df["split"].isna().any():
        n_missing = int(df["split"].isna().sum())
        raise RuntimeError(f"{n_missing} studies were not assigned a split.")

    # ---- Verify exact counts ----
    print()
    print(f"      {'split':<7s}  {'gold':>6s}  {'non-gold':>9s}  {'total':>7s}")
    print(f"      {'-'*7}  {'-'*6}  {'-'*9}  {'-'*7}")

    checks_ok = True
    expected = {
        "train": (GOLD_TRAIN, NONGOLD_TRAIN),
        "val":   (GOLD_VAL,   NONGOLD_VAL),
        "test":  (GOLD_TEST,  NONGOLD_TEST),
    }
    for split in ["train", "val", "test"]:
        sub  = df[df["split"] == split]
        n_g  = int(sub["is_gold"].sum())
        n_ng = len(sub) - n_g
        exp_g, exp_ng = expected[split]
        ok = (n_g == exp_g and n_ng == exp_ng)
        checks_ok = checks_ok and ok
        flag = "✓" if ok else "❌"
        print(f"      {flag} {split:<5s}  {n_g:>6d}  {n_ng:>9d}  {len(sub):>7d}  "
            f"(expected {exp_g} / {exp_ng})")

    if not checks_ok:
        raise RuntimeError("Split counts do not match expected values.")

    print()
    return df



# STEP 6 — LEAKAGE CHECK

def verify_no_leakage(df):
    print("[6/7] Verifying no leakage...")
    leakage = df.groupby("patient_id")["split"].nunique()
    bad = leakage[leakage > 1]
    if len(bad) > 0:
        print(f"      ❌ LEAKAGE in {len(bad)} patients\n")
        return False
    print(f"      ✓ clean ({df['patient_id'].nunique()} patients)\n")
    return True


# STEP 6b — LABEL DISTRIBUTION (gold studies only)

def compute_distribution(df):
    print("[6b/7] Computing label distribution (gold studies only)...")

    rows = []
    for label in LABELS:
        row = {"Label": label}

        for split in ["train", "val", "test"]:
            sub_gold = df[(df["split"] == split) & (df["is_gold"] == 1)]
            series   = pd.to_numeric(sub_gold[label], errors="coerce")
            mask     = series.notna()
            n_lab    = int(mask.sum())
            pos      = int((series[mask] == 1).sum())
            neg      = n_lab - pos
            row[f"{split}_pos"] = pos
            row[f"{split}_neg"] = neg
            row[f"{split}_n"]   = n_lab
            row[f"{split}_pct"] = round(100 * pos / n_lab, 1) if n_lab else 0.0

        series_all = pd.to_numeric(df.loc[df["is_gold"] == 1, label], errors="coerce")
        mask_all   = series_all.notna()
        n_lab_all  = int(mask_all.sum())
        pos_all    = int((series_all[mask_all] == 1).sum())
        row["overall_pos"] = pos_all
        row["overall_neg"] = n_lab_all - pos_all
        row["overall_n"]   = n_lab_all
        row["overall_pct"] = (
            round(100 * pos_all / n_lab_all, 1) if n_lab_all else 0.0
        )

        rows.append(row)

    print("      done\n")
    return pd.DataFrame(rows)


# STEP 7 — SAVE OUTPUTS

def save_outputs(df, dist, output_dir):
    print("[7/7] Saving...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Column order for the CSV
    cols = [STUDY_ID_COL, "split", "is_gold"] + LABELS

    # 1. split.csv (all studies)
    split_path = output_dir / "split.csv"
    df[cols].to_csv(split_path, index=False)
    print(f"      ✓ {split_path}")

    # 2. train_split.csv / val_split.csv / test_split.csv
    for split in ["train", "val", "test"]:
        sub_path = output_dir / f"{split}_split.csv"
        df[df["split"] == split][cols].to_csv(sub_path, index=False)
        print(f"      ✓ {sub_path}")

    # 3. label_distribution.csv
    dist_path = output_dir / "label_distribution.csv"
    dist.to_csv(dist_path, index=False)
    print(f"      ✓ {dist_path}")

    # 4. split_summary.json
    summary = {
        "generated_at": datetime.now().isoformat(),
        "seed":         SEED,
        "split_method": "study-level; exact counts for gold and non-gold",
        "total_studies": len(df),
        "gold_labeled":  int(df["is_gold"].sum()),
        "non_gold":      int((df["is_gold"] == 0).sum()),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        sub  = df[df["split"] == split]
        n_g  = int(sub["is_gold"].sum())
        n_ng = len(sub) - n_g
        summary["splits"][split] = {
            "gold":     n_g,
            "non_gold": n_ng,
            "total":    len(sub),
        }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      ✓ {summary_path}\n")
    return summary



# MAIN

def main():
    print("=" * 60)
    print("MEMBER 2 — REDONE SPLIT (EXACT COUNTS)")
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
    print(f"Gold-labeled:   {summary['gold_labeled']}")
    print(f"Non-gold:       {summary['non_gold']}\n")

    print(f"{'split':<7s}  {'gold':>6s}  {'non-gold':>9s}  {'total':>7s}")
    print("-" * 35)
    for split, info in summary["splits"].items():
        print(f"{split:<7s}  {info['gold']:>6d}  {info['non_gold']:>9d}  "
            f"{info['total']:>7d}")

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION (gold-labeled studies only)")
    print("=" * 60)
    print(dist[["Label",
                "train_pos", "train_neg", "train_pct",
                "val_pos",   "val_neg",   "val_pct",
                "test_pos",  "test_neg",  "test_pct"]]
        .to_string(index=False))

    print("\n" + "=" * 60)
    print("DELIVERABLES")
    print("=" * 60)
    print(f"  {OUTPUT_DIR}/split.csv")
    print(f"  {OUTPUT_DIR}/train_split.csv")
    print(f"  {OUTPUT_DIR}/val_split.csv")
    print(f"  {OUTPUT_DIR}/test_split.csv")
    print(f"  {OUTPUT_DIR}/label_distribution.csv")
    print(f"  {OUTPUT_DIR}/split_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()