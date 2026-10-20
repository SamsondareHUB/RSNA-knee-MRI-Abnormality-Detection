# val evaluation summary

Gold studies scored: 14
Non-gold predictions excluded: 0
Evaluable labels: 12/12
Complete 12-label macro AUROC: 0.5058396464646465
Mean over evaluable labels only: 0.5058396464646465

These small gold sets produce statistically uncertain estimates and are not clinically validated.

| Label | AUROC | Positive | Negative | Missing |
|---|---:|---:|---:|---:|
| ACL | 0.500000 | 6 | 8 | 0 |
| MCL | 0.590909 | 3 | 11 | 0 |
| Medial Meniscus | 0.500000 | 7 | 7 | 0 |
| Lateral Meniscus | 0.500000 | 6 | 8 | 0 |
| Medial OA | 0.500000 | 3 | 11 | 0 |
| Lateral OA | 0.500000 | 3 | 11 | 0 |
| PF OA | 0.500000 | 5 | 9 | 0 |
| Effusion | 0.479167 | 8 | 6 | 0 |
| Synovitis | 0.500000 | 8 | 6 | 0 |
| Baker's | 0.500000 | 3 | 11 | 0 |
| Contusion | 0.500000 | 3 | 11 | 0 |
| Fracture | 0.500000 | 4 | 10 | 0 |

## Inputs

- gold_csv: `train.csv` (SHA-256 `e9bad94012ac6dcc5b85142fc91b1d3af3a8a3418f6b516fa5b96ddc6060fc91`)
- split_csv: `split.csv` (SHA-256 `d0367d6a132b397cea3e60fd4a7963451904f5e69b736bdc6b1c1c691e93936a`)
- predictions_csv: `experiment001_val_predictions.csv` (SHA-256 `fce1bd98827b574c1dd32357c29b465b5a59caec92d621471979144e432e22eb`)
- split_summary: `split_summary.json` (SHA-256 `51a68dfe3d4fcae60f678e89f6fba4c025abe8142c999224432002648852d7a6`)

## Software

- Python 3.11.9; NumPy 2.4.6; pandas 3.0.5; scikit-learn 1.9.0.
