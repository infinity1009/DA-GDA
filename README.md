# DA-GDA: Differentiable Automatic Graph Data Augmentation for Semi-Supervised Node Classification

This repository is the official implementation of DA-GDA.

## Requirements

To install requirements:

`pip install -r requirements.txt`

### Training

To train and eval the model in the paper, run commands:

- `python train_search.py`
- `python train_search_twitch.py`

### Configurations

We provide model configurations for each dataset under the `configs` directory.

## Results

Our model achieves the following performance on:

1. Node classification accuracy / micro $F_1$ scorecomparison:

   <img src="https://github.com/infinity1009/DA-GDA/blob/master/pics/accuracy.png?raw=true" alt="effectiveness" style="zoom:33%;" />

2. Efficiency evaluation:

   <img src="https://github.com/infinity1009/DA-GDA/blob/master/pics/effi.png?raw=true" alt="r1" title="effi" style="zoom:50%;" />

3. Robustness evaluation:

   <img src="https://github.com/infinity1009/DA-GDA/blob/master/pics/r1.png?raw=true" alt="r1" title="robustness" style="zoom:50%;" />

4. Hyperparameter sensitivity evaluation:

   <img src="https://github.com/infinity1009/DA-GDA/blob/master/pics/hyp.png?raw=true" alt="hyp" title="hyp" style="zoom:50%;" />

5. Ablation study:

<img src="https://github.com/infinity1009/DA-GDA/blob/master/pics/abs.png?raw=true" alt="ablation study" title="ablation study" style="zoom:50%;" />
