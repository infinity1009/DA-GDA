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

   <img src="accuracy.png" alt="accuracy" style="zoom:30%;" />

2. Efficiency evaluation:

   <img src="effi.png" alt="effi" style="zoom: 50%;" />

3. Robustness evaluation:

   <img src="r1.png" alt="r1" style="zoom:50%;" />

4. Hyperparameter sensitivity evaluation:

   <img src="hyp.png" alt="hyp" style="zoom:50%;" />

5. Ablation study:

   <img src="abs.png" alt="abs" style="zoom:50%;" />

