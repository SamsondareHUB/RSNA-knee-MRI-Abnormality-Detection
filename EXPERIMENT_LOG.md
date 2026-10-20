# Experiment Log

## Experiment 001 — ResNet50 slice baseline

- Status: Complete
- Model: ResNet50 slice encoder + 12-output classification head
- Training data: 30 gold-labeled studies
- Validation data: 14 gold-labeled studies
- Epochs: 10
- Training loss: 0.6783 -> 0.6048
- Validation macro AUROC: 0.5058
- Interpretation: near-chance performance; pipeline-validation experiment
- Checkpoint: Member3/outputs/experiment001_resnet50.pth (not committed)
- Predictions: Member3/outputs/experiment001_val_predictions.csv (not committed)
- Evaluation: Member3/outputs/evaluation_val/