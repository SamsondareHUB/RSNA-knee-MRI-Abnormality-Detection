import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from evaluation.member4_metrics import LABELS, evaluate
from evaluation.member4_evaluate import main


ID = "StudyInstanceUID"


def fixture(tmp_path, scores=None, labels=None):
    ids = ["s1", "s2", "s3", "s4", "extra"]
    truth = [0, 0, 1, 1]
    labels = truth if labels is None else labels
    scores = [0.1, 0.2, 0.8, 0.9] if scores is None else scores
    gold = pd.DataFrame({ID: ids, **{name: ["" if value is None else str(value) for value in labels] + [""] for name in LABELS}})
    split = pd.DataFrame({ID: ids, "split": ["val"] * 5, "is_gold": [1, 1, 1, 1, 0]})
    pred = pd.DataFrame({ID: ids, **{name: scores + [0.5] for name in LABELS}})
    paths = [tmp_path / name for name in ("gold.csv", "split.csv", "pred.csv")]
    for frame, path in zip((gold, split, pred), paths):
        frame.to_csv(path, index=False)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"total_studies": 5, "gold_labeled": 4,
        "splits": {"val": {"gold": 4, "non_gold": 1, "total": 5}}}))
    return paths, summary


def run(tmp_path, **changes):
    paths, summary = fixture(tmp_path, changes.pop("scores", None), changes.pop("labels", None))
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    for key, value in changes.items():
        index, column = key.split("__", 1)
        frames[{"gold": 0, "split": 1, "pred": 2}[index]][column] = value
    for frame, path in zip(frames, paths):
        frame.to_csv(path, index=False)
    return evaluate(*paths, summary, "val")


def test_perfect_and_all_twelve_labels(tmp_path):
    result = run(tmp_path)
    assert len(result["per_label"]) == 12
    assert result["macro"]["macro_auroc_12_label"] == 1
    assert result["integrity"]["excluded_non_gold_predictions"] == 1


def test_inverted_predictions(tmp_path):
    result = run(tmp_path, scores=[0.9, 0.8, 0.2, 0.1])
    assert result["macro"]["macro_auroc_12_label"] == 0


def test_ties_match_sklearn(tmp_path):
    scores = [0.5, 0.5, 0.5, 0.5]
    result = run(tmp_path, scores=scores)
    assert result["macro"]["macro_auroc_12_label"] == roc_auc_score([0, 0, 1, 1], scores)


def test_undefined_auc_is_not_imputed(tmp_path):
    result = run(tmp_path, labels=[1, 1, 1, 1])
    assert np.isnan(result["per_label"]["auroc"]).all()
    assert result["macro"]["macro_auroc_12_label"] is None
    assert result["macro"]["evaluable_label_count"] == 0
    assert "one class" in result["per_label"].iloc[0]["undefined_reason"]


def test_missing_gold_label_is_counted(tmp_path):
    result = run(tmp_path, gold__ACL=["0", "", "1", "1", ""])
    assert result["per_label"].iloc[0]["missing_label_count"] == 1
    assert result["per_label"].iloc[0]["n_usable"] == 3


@pytest.mark.parametrize("key,value", [
    ("gold__StudyInstanceUID", ["s1", "s1", "s3", "s4", "extra"]),
    ("split__StudyInstanceUID", ["s1", "s1", "s3", "s4", "extra"]),
    ("pred__StudyInstanceUID", ["s1", "s1", "s3", "s4", "extra"]),
])
def test_duplicate_ids_fail(tmp_path, key, value):
    with pytest.raises(ValueError, match="duplicate"):
        run(tmp_path, **{key: value})


def test_missing_gold_prediction_fails(tmp_path):
    paths, summary = fixture(tmp_path)
    pd.read_csv(paths[2]).iloc[[0, 1, 3, 4]].to_csv(paths[2], index=False)
    with pytest.raises(ValueError, match="missing prediction"):
        evaluate(*paths, summary, "val")


def test_row_order_is_ignored(tmp_path):
    paths, summary = fixture(tmp_path)
    pred = pd.read_csv(paths[2]).iloc[[3, 2, 1, 0, 4]]
    pred.to_csv(paths[2], index=False)
    assert evaluate(*paths, summary, "val")["macro"]["macro_auroc_12_label"] == 1


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_probabilities_fail(tmp_path, bad):
    scores = [0.1, 0.2, 0.8, bad]
    with pytest.raises(ValueError, match="prediction score"):
        run(tmp_path, scores=scores)


def test_non_gold_prediction_does_not_affect_score(tmp_path):
    result = run(tmp_path, pred__ACL=[0.1, 0.2, 0.8, 0.9, 0.99])
    assert result["per_label"].iloc[0]["n_usable"] == 4
    assert result["per_label"].iloc[0]["auroc"] == 1


def test_invalid_split_count_fails(tmp_path):
    paths, summary = fixture(tmp_path)
    data = json.loads(summary.read_text())
    data["splits"]["val"]["gold"] = 3
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="summary"):
        evaluate(*paths, summary, "val")


def test_logits_require_explicit_mode(tmp_path):
    paths, summary = fixture(tmp_path)
    pred = pd.read_csv(paths[2])
    pred[LABELS] = np.log(pred[LABELS] / (1 - pred[LABELS]))
    pred.to_csv(paths[2], index=False)
    with pytest.raises(ValueError, match="prediction score"):
        evaluate(*paths, summary, "val")
    assert evaluate(*paths, summary, "val", input_type="logits")["macro"]["macro_auroc_12_label"] == 1


def test_cli_writes_all_outputs(tmp_path):
    paths, summary = fixture(tmp_path)
    output = tmp_path / "synthetic-test-output"
    main(["--gold-csv", str(paths[0]), "--split-csv", str(paths[1]),
          "--predictions-csv", str(paths[2]), "--split-summary", str(summary),
          "--split", "val", "--output-dir", str(output)])
    assert {p.name for p in output.iterdir()} == {"per_label_auroc.csv", "macro_auroc.json",
        "label_support.csv", "evaluation_integrity.json", "evaluation_summary.md"}
    assert json.loads((output / "macro_auroc.json").read_text())["macro_auroc_12_label"] == 1


def test_test_mode_requires_fixed_selection_record(tmp_path):
    paths, summary = fixture(tmp_path)
    with pytest.raises(ValueError, match="selection record"):
        main(["--gold-csv", str(paths[0]), "--split-csv", str(paths[1]),
              "--predictions-csv", str(paths[2]), "--split-summary", str(summary),
              "--split", "test", "--output-dir", str(tmp_path / "out")])


def test_split_labels_accept_repository_decimal_binary_encoding(tmp_path):
    paths, summary = fixture(tmp_path)
    split = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    gold = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    for label in LABELS:
        split[label] = gold[label].map({"0": "0.0", "1": "1.0", "": ""})
    split.to_csv(paths[1], index=False)
    assert evaluate(*paths, summary, "val")["macro"]["macro_auroc_12_label"] == 1


def test_conflicting_split_label_fails(tmp_path):
    paths, summary = fixture(tmp_path)
    split = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    split["ACL"] = ["1.0", "0.0", "1.0", "1.0", ""]
    split.to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="disagree"):
        evaluate(*paths, summary, "val")


def test_patient_overlap_fails(tmp_path):
    paths, summary = fixture(tmp_path)
    split = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    split.loc[4, "split"] = "test"
    split["PatientID"] = ["p1", "p2", "p3", "p4", "p1"]
    split.to_csv(paths[1], index=False)
    pd.read_csv(paths[2]).iloc[:4].to_csv(paths[2], index=False)
    data = json.loads(summary.read_text())
    data["splits"]["val"] = {"gold": 4, "non_gold": 0, "total": 4}
    data["splits"]["test"] = {"gold": 0, "non_gold": 1, "total": 1}
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="patient overlap"):
        evaluate(*paths, summary, "val")


def test_final_test_with_fixed_selection_record(tmp_path):
    paths, summary = fixture(tmp_path)
    split = pd.read_csv(paths[1], dtype=str)
    split["split"] = "test"
    split.to_csv(paths[1], index=False)
    data = json.loads(summary.read_text())
    data["splits"] = {"test": {"gold": 4, "non_gold": 1, "total": 5}}
    summary.write_text(json.dumps(data))
    record = tmp_path / "fixed.json"
    record.write_text(json.dumps({"checkpoint_id": "sha256:example", "selection_protocol": "validation AUROC",
                                  "fixed_before_test": True}))
    output = tmp_path / "synthetic-test-output"
    args = ["--gold-csv", str(paths[0]), "--split-csv", str(paths[1]),
            "--predictions-csv", str(paths[2]), "--split-summary", str(summary),
            "--split", "test", "--output-dir", str(output), "--selection-record", str(record)]
    assert main(args)["macro"]["macro_auroc_12_label"] == 1
    record.write_text(json.dumps({"checkpoint_id": "sha256:example", "selection_protocol": "validation AUROC",
                                  "fixed_before_test": "false"}))
    with pytest.raises(ValueError, match="selection record"):
        main(args)
