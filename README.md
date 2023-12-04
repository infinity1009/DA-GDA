# DA-GDA: Differentiable Automatic Graph Data Augmentation for Semi-Supervised Node Classification

This repository is the official implementation of DA-GDA.

## Requirements

To install requirements:

`pip install -r requirements.txt`

### Training

To train and eval the model in the paper, run commands:

- `python train_search.py`

## Results

Our model achieves the following performance on:

1. Node classification accuracy / micro $F_1$ scorecomparison:

   <img src="pics/accuracy.png" alt="accuracy" style="zoom:30%;" />

2. Efficiency evaluation:

   <img src="pics/effi.png" alt="effi" style="zoom: 50%;" />

3. Robustness evaluation:

   <img src="pics/r1.png" alt="r1" style="zoom:50%;" />

4. Hyperparameter sensitivity evaluation:

   <img src="pics/hyp.png" alt="hyp" style="zoom:50%;" />

5. Ablation study:

   <img src="pics/abs.png" alt="abs" style="zoom:50%;" />

