"""Strict gold-study AUROC calculation for the repository's 12 labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Member2.member2_work.config import LABELS


ID = "StudyInstanceUID"


def _read_csv(path, required, kind):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{kind}: missing columns {sorted(missing)}")
    if frame[ID].eq("").any() or frame[ID].duplicated().any():
        raise ValueError(f"{kind}: empty or duplicate study IDs")
    return frame


def _binary(series, name, missing_allowed):
    allowed = ["0", "1", "0.0", "1.0"]
    bad = ~series.isin(allowed + ([""] if missing_allowed else []))
    if bad.any():
        raise ValueError(f"invalid binary values in {name}")
    return pd.to_numeric(series.replace("", np.nan))


def evaluate(gold_csv, split_csv, prediction_csv, split_summary_json, split, input_type="probabilities"):
    """Return validated per-label, macro, and integrity data; write nothing."""
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if input_type not in ("probabilities", "logits"):
        raise ValueError("input_type must be probabilities or logits")
    gold = _read_csv(gold_csv, [ID, *LABELS], "gold labels")
    assignments = _read_csv(split_csv, [ID, "split", "is_gold"], "split assignments")
    predictions = _read_csv(prediction_csv, [ID, *LABELS], "predictions")
    summary = json.loads(Path(split_summary_json).read_text())

    if not assignments["split"].isin(["train", "val", "test"]).all():
        raise ValueError("invalid split assignment")
    assignments["is_gold"] = _binary(assignments["is_gold"], "is_gold", False).astype(int)
    for label in LABELS:
        gold[label] = _binary(gold[label], f"gold {label}", True)
    gold["has_label"] = gold[LABELS].notna().any(axis=1)
    if set(gold[ID]) != set(assignments[ID]):
        raise ValueError("gold and split study ID sets differ")
    joined = assignments.merge(gold, on=ID, validate="one_to_one", suffixes=("_split", ""))
    if not (joined["is_gold"].astype(bool) == joined["has_label"]).all():
        raise ValueError("is_gold disagrees with gold label presence")
    if any(f"{label}_split" in joined for label in LABELS):
        for label in LABELS:
            column = f"{label}_split"
            if column in joined:
                source = _binary(joined[column], f"split {label}", True)
                if not source.equals(joined[label]):
                    raise ValueError(f"split and gold labels disagree: {label}")

    counts = joined.groupby(["split", "is_gold"]).size()
    if len(joined) != summary["total_studies"] or int(joined["is_gold"].sum()) != summary["gold_labeled"]:
        raise ValueError("split counts disagree with summary")
    for part, expected in summary["splits"].items():
        if (int(counts.get((part, 1), 0)) != expected["gold"] or
            int(counts.get((part, 0), 0)) != expected["non_gold"] or
            int((joined["split"] == part).sum()) != expected["total"]):
            raise ValueError("split counts disagree with summary")
    if set(predictions[ID]) - set(joined.loc[joined["split"] == split, ID]):
        raise ValueError("predictions contain IDs outside selected split")
    target = joined[(joined["split"] == split) & (joined["is_gold"] == 1)].copy()
    if set(target[ID]) - set(predictions[ID]):
        raise ValueError("gold study missing prediction")
    for label in LABELS:
        numeric = pd.to_numeric(predictions[label], errors="coerce")
        if not np.isfinite(numeric).all():
            raise ValueError(f"invalid prediction score for {label}: nonfinite")
        if input_type == "probabilities" and not numeric.between(0, 1).all():
            raise ValueError(f"invalid prediction score for {label}: outside [0,1]")
        predictions[label] = numeric if input_type == "probabilities" else 1 / (1 + np.exp(-numeric))
    scored = target.merge(predictions, on=ID, validate="one_to_one", suffixes=("_gold", "_score"))
    rows = []
    for label in LABELS:
        y = scored[f"{label}_gold"]
        score = scored[f"{label}_score"]
        present = y.notna()
        positive = int((y[present] == 1).sum())
        negative = int((y[present] == 0).sum())
        defined = positive > 0 and negative > 0
        rows.append({"label": label, "auroc": float(roc_auc_score(y[present], score[present])) if defined else np.nan,
            "n_positive": positive, "n_negative": negative, "n_usable": int(present.sum()),
            "missing_label_count": int((~present).sum()), "auroc_defined": defined,
            "undefined_reason": "" if defined else "only one class or no labeled studies"})
    per_label = pd.DataFrame(rows)
    evaluable = int(per_label["auroc_defined"].sum())
    macro = {"macro_auroc_12_label": float(per_label["auroc"].mean()) if evaluable == 12 else None,
             "evaluable_label_count": evaluable,
             "mean_evaluable_labels_only": float(per_label["auroc"].mean()) if evaluable else None}
    patient_column = next((c for c in ("PatientID", "patient_id") if c in joined), None)
    patient_audit = "unavailable: no verified patient ID column"
    if patient_column:
        patients = joined[[patient_column, "split"]]
        if patients[patient_column].eq("").any():
            raise ValueError("patient ID column contains blanks")
        overlap = int((patients.groupby(patient_column)["split"].nunique() > 1).sum())
        if overlap:
            raise ValueError(f"cross-split patient overlap: {overlap}")
        patient_audit = "passed: no cross-split patient overlap"
    integrity = {"split": split, "split_studies": int((joined["split"] == split).sum()),
        "expected_gold_studies": len(target), "scored_gold_studies": len(scored),
        "excluded_non_gold_predictions": int(predictions[ID].isin(joined.loc[(joined["split"] == split) & (joined["is_gold"] == 0), ID]).sum()),
        "non_gold_studies_without_predictions": int(((joined["split"] == split) & (joined["is_gold"] == 0)).sum()) - int(predictions[ID].isin(joined.loc[(joined["split"] == split) & (joined["is_gold"] == 0), ID]).sum()),
        "patient_overlap_audit": patient_audit}
    return {"per_label": per_label, "macro": macro, "integrity": integrity}
