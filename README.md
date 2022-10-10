# DA-GDA: Differentiable Automatic Data Augmentation for Graph Neural Network

This repository is the official implementation of DA-GDA.

## Requirements

To install requirements:

`pip install -r requirements.txt`

### Training

To train and eval the model in the paper, run these commands:

- `cd code`
- `python train.py`

## Results

Our model achieves the following performance on:

1. Node classification accuracy comparison:

   <img src="accuracy.png" alt="accuracy" style="zoom:33%;" />

2. Efficiency evaluation:

   <img src="effi.png" alt="effi" style="zoom:33%;" />

3. Robustness evaluation:

   <img src="r1.png" alt="r1" style="zoom:50%;" />

   <img src="r2.png" alt="r2" style="zoom:50%;" />

4. Hyperparameter sensitivity evaluation:

   <img src="hyp.png" alt="hyp" style="zoom:50%;" />

5. Ablation study:

   <img src="abs.png" alt="abs" style="zoom:50%;" />

