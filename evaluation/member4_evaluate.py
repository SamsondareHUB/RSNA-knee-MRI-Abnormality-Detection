"""CLI for reproducible validation and gated final test evaluation."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy
import pandas
import sklearn

from evaluation.member4_metrics import LABELS, evaluate


def _identifier(path):
    path = Path(path)
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("gold-csv", "split-csv", "predictions-csv", "split-summary", "output-dir"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--input-type", choices=["probabilities", "logits"], default="probabilities")
    parser.add_argument("--selection-record", help="JSON record fixing checkpoint and selection protocol before final test")
    args = parser.parse_args(argv)
    selection = None
    if args.split == "test":
        if not args.selection_record:
            raise ValueError("final test requires a selection record")
        selection = json.loads(Path(args.selection_record).read_text())
        if not selection.get("checkpoint_id") or not selection.get("selection_protocol") or selection.get("fixed_before_test") is not True:
            raise ValueError("selection record must contain checkpoint_id, selection_protocol, fixed_before_test=true")
    result = evaluate(args.gold_csv, args.split_csv, args.predictions_csv,
                      args.split_summary, args.split, args.input_type)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_label = result["per_label"]
    per_label.to_csv(output / "per_label_auroc.csv", index=False, na_rep="NaN")
    per_label[["label", "n_positive", "n_negative", "n_usable", "missing_label_count"]].to_csv(output / "label_support.csv", index=False)
    inputs = {name: _identifier(path) for name, path in {
        "gold_csv": args.gold_csv, "split_csv": args.split_csv,
        "predictions_csv": args.predictions_csv, "split_summary": args.split_summary}.items()}
    if args.selection_record:
        inputs["selection_record"] = _identifier(args.selection_record)
    metadata = {"split": args.split, "input_type": args.input_type, "labels": LABELS,
        "inputs": inputs, "software_versions": {"python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pandas.__version__, "scikit_learn": sklearn.__version__}}
    (output / "macro_auroc.json").write_text(json.dumps({**metadata, **result["macro"]}, indent=2) + "\n")
    (output / "evaluation_integrity.json").write_text(json.dumps({**metadata, **result["integrity"],
        "selection_record": selection}, indent=2) + "\n")
    lines = [f"# {args.split} evaluation summary", "", f"Gold studies scored: {result['integrity']['scored_gold_studies']}",
        f"Non-gold predictions excluded: {result['integrity']['excluded_non_gold_predictions']}",
        f"Evaluable labels: {result['macro']['evaluable_label_count']}/12",
        f"Complete 12-label macro AUROC: {result['macro']['macro_auroc_12_label'] if result['macro']['macro_auroc_12_label'] is not None else 'UNDEFINED'}",
        f"Mean over evaluable labels only: {result['macro']['mean_evaluable_labels_only'] if result['macro']['mean_evaluable_labels_only'] is not None else 'UNDEFINED'}",
        "", "These small gold sets produce statistically uncertain estimates and are not clinically validated.",
        "", "| Label | AUROC | Positive | Negative | Missing |", "|---|---:|---:|---:|---:|"]
    for row in per_label.itertuples():
        auc = "UNDEFINED" if not row.auroc_defined else f"{row.auroc:.6f}"
        lines.append(f"| {row.label} | {auc} | {row.n_positive} | {row.n_negative} | {row.missing_label_count} |")
    lines.extend(["", "## Inputs", ""])
    for name, item in inputs.items():
        lines.append(f"- {name}: `{item['name']}` (SHA-256 `{item['sha256']}`)")
    lines.extend(["", "## Software", "",
        f"- Python {platform.python_version()}; NumPy {numpy.__version__}; pandas {pandas.__version__}; scikit-learn {sklearn.__version__}."])
    (output / "evaluation_summary.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    main()
