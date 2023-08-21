# Learning Training Dynamics for Active Learning

## Prerequisites:   
- Linux or macOS
- Python 3.5/3.6
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.
- pytorch 0.4.1
- cuda 8.0
- Anaconda3

## Requirements

To install virtual enviornment for requirements:

```setup
conda env create -f TiDAL.yaml
```

> 📋if you already conda, you can activate virtual experiment settings

To activate virtual enviornment:

```activate
conda activate TiDAL
```

## Running code

To train the model(s) and evaluate in the paper, run this command:

```train
python main.py -d cifar10 -i 1 -m TiDAL -q Entropy
```
