# MNIST: lightweight CNN with data augmentation

A course replication study comparing identical CNN architectures trained with and without random affine augmentation. The weights and logs are from actual runs (seed=42).

## Experiment actually run
Training: 10,000 samples selected from a fixed split of MNIST training partition (55,000 available for training and 5,000 validation); 10 epochs; batch size 128; Adam learning rate 0.001; CPU, workers=0. Augmentation only for the augmented training variant: `RandomAffine(degrees=10, translate=(0.08,0.08))`. Both test evaluations use the unchanged 10,000-image official MNIST test set. Best checkpoint chosen by validation accuracy.

| Model | best validation accuracy | test accuracy | test loss |
|---|---:|---:|---:|
| Baseline | 97.72% | 97.96% | 0.065375 |
| Augmented | 97.78% | 98.33% | 0.054576 |

## Reproduction
```bash
python -m pip install -r requirements.txt
python train.py --variant baseline --epochs 10 --limit-train 10000 --workers 0 --seed 42
python train.py --variant augmented --epochs 10 --limit-train 10000 --workers 0 --seed 42
python evaluate.py --variant baseline
python evaluate.py --variant augmented
python plot_results.py
```
The dataset is downloaded automatically under `data/` (not committed). Pre-trained course-experiment checkpoints: `results/baseline/best.pth`, `results/augmented/best.pth`. Accuracy, loss and confusion matrices are in results. Re-running training overwrites the existing logs and weights: back up first. Exact numeric reproduction can depend on hardware and library versions.

## Notes
The architecture is identical across groups, so this is an evaluation of an existing data augmentation strategy, not an original CNN architecture. The test set was evaluated after the validation-selected models were trained. No repeated seeds, generalization to other datasets, or statistical significance are claimed.
